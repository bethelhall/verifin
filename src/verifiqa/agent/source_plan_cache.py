"""Reusable, candidate-independent plans backed by current source selections.

The on-disk JSON is a reproducibility artifact, never an authority. Every lookup
independently reconstructs and validates the selection from the current sources
before an entry can be reused. No answer, model, dataset ID, or gold is an input.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from verifiqa.agent.types import ClaimSpec
from verifiqa.formulas.evaluator import formula_variables


_CACHE_VERSION = 1


def _json_value(value):
    """Preserve exact decimals and tuple-keyed filing records in stable JSON."""
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _json_value(item) for key, item in sorted(value.items())}
        pairs = [[_json_value(key), _json_value(item)] for key, item in value.items()]
        return {"mapping": sorted(pairs, key=lambda pair: _json_text(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Unsupported or non-finite source-plan cache value")


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _implementation_sha256():
    """Conservatively invalidate after any local verifier/policy code change."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json"}:
            digest.update(str(path.relative_to(package)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _source_inputs(question, evidence_text, source_pdf_path, source_xbrl_bundle):
    pdf_sha = _sha(Path(source_pdf_path).read_bytes()) if source_pdf_path else ""
    bundle = _json_value(source_xbrl_bundle)
    # XBRL schema planning can recover report units from a pinned report in the
    # local manifest. Bind that metadata too; reconstruction verifies its bytes.
    report_manifest_sha = ""
    artifacts_dir = getattr(source_xbrl_bundle, "artifacts_dir", None)
    if artifacts_dir is not None:
        manifest_path = Path(artifacts_dir) / "manifest.json"
        if manifest_path.exists():
            report_manifest_sha = _sha(manifest_path.read_bytes())
    return {
        "question": question,
        "evidence_sha256": _sha(evidence_text.encode("utf-8")),
        "source_pdf_sha256": pdf_sha,
        "source_xbrl_sha256": _sha(_json_text(bundle).encode("utf-8")),
        "report_manifest_sha256": report_manifest_sha,
        "implementation_sha256": _implementation_sha256(),
    }


def _complete(spec):
    if not isinstance(spec, ClaimSpec):
        return False
    if (not spec.metric or not spec.formula or not spec.roles or not spec.claim_unit
            or not spec.planning_source.startswith("deterministic_")
            or spec.formula_source == "no_formula"
            or spec.claimed_value is not None or spec.claimed_unit):
        return False
    names = [role.name for role in spec.roles]
    if len(names) != len(set(names)) or not all(name.isidentifier() for name in names):
        return False
    try:
        return (math.isfinite(spec.tolerance) and spec.tolerance >= 0
                and formula_variables(spec.formula) == set(names))
    except (SyntaxError, TypeError, ValueError):
        return False


def _reconstruct(question, evidence_text, source_pdf_path, source_xbrl_bundle):
    from verifiqa.agent.source_question_planner import plan_from_source
    from verifiqa.agent.source_selection import attach_source_selection, validate_source_selection

    spec = plan_from_source(question, evidence_text, source_pdf_path=source_pdf_path)
    if spec is None and source_xbrl_bundle is not None:
        from verifiqa.agent.source_xbrl_schema_planner import plan_from_source_xbrl
        spec = plan_from_source_xbrl(question, evidence_text, source_xbrl_bundle)
    if not _complete(spec):
        return None
    kwargs = {"source_pdf_path": source_pdf_path, "source_xbrl_bundle": source_xbrl_bundle}
    try:
        selected = attach_source_selection(spec, question, evidence_text, **kwargs)
        if _complete(selected) and validate_source_selection(selected, question, evidence_text, **kwargs):
            return selected
    except (OSError, TypeError, ValueError):
        # Incomplete/ambiguous source references never become authoritative.
        return None
    return None


def _atomic_write(path, payload):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".source-plan-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_json_text(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def resolve_source_plan(
    question: str, evidence_text: str, *, source_pdf_path: str = "",
    source_xbrl_bundle=None, cache_dir=None,
) -> ClaimSpec | None:
    """Resolve a fresh, validated source selection, optionally reusing its JSON.

    Reusing a cache entry does not skip source reconstruction. Identical question
    and source inputs select the same plan for every answer model; each caller
    receives a deep copy so later claim parsing cannot contaminate another run.
    Unsupported source questions return ``None`` for the caller's usual fallback.
    Cache corruption or unavailable storage cannot authorize a plan.
    """
    spec = _reconstruct(question, evidence_text, source_pdf_path, source_xbrl_bundle)
    if spec is None:
        return None
    if cache_dir is None:
        return deepcopy(spec)
    try:
        inputs = _source_inputs(question, evidence_text, source_pdf_path, source_xbrl_bundle)
        key = _sha(_json_text({"version": _CACHE_VERSION, "inputs": inputs}).encode("utf-8"))
        path = Path(cache_dir) / (key + ".json")
        expected = {"version": _CACHE_VERSION, "key": key, "inputs": inputs,
                    "plan": _json_value(spec)}
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            cached = None
        if cached != expected:
            # A stale/tampered entry is replaced only with the independently
            # reconstructed plan, never partially merged or trusted by key alone.
            _atomic_write(path, expected)
    except (OSError, TypeError, ValueError):
        # A cache is optional; source validation has already succeeded.
        pass
    return deepcopy(spec)
