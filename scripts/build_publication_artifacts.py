#!/usr/bin/env python3
"""Build and validate the locked paper datasets and result artifacts.

The builder never mutates source artifacts and refuses to overwrite an existing
destination. It copies exact source bytes, selects one canonical SMT file per
VeriFin result row using ``<id>_<status>.smt2``, and emits deterministic
per-run provenance manifests. Stale status variants in a working run directory
are deliberately ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "publication_artifacts.json"
ARTIFACT_MANIFEST = "ARTIFACT_MANIFEST.json"
CHECKSUMS = "SHA256SUMS"
MANAGED_ROOTS = {"data", "datasets", "results"}
ALLOWED_STATUSES = {
    "VERIFIED",
    "VIOLATED",
    "ABSTAIN",
    "UNVERIFIED_FORMULA",
}
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CANDIDATE_PROTOCOL = "claim_input_answer_v2"
SCALAR_JUDGE_PROTOCOL = "scalar_quantity_v2"
FULL_ANSWER_JUDGE_PROTOCOL = "full_answer_one_word_v1"
EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9.-])"
)
SAFE_EMAIL_DOMAINS = {"example.com", "example.org", "example.net"}
CONTENT_RULES = {
    "absolute macOS home path": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "absolute Linux home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "Anthropic-style credential": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "OpenAI-style credential": re.compile(
        r"(?<![A-Za-z0-9])sk-(?!ant-)[A-Za-z0-9_-]{20,}"
    ),
    "Hugging Face-style credential": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "GitHub-style credential": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access-key identifier": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


class ArtifactError(RuntimeError):
    """The locked artifact set failed a publication-safety check."""


@dataclass(frozen=True)
class FileItem:
    source: Optional[Path]
    destination: PurePosixPath
    sha256: str
    size: int
    kind: str
    dataset: str
    artifact: str
    content: Optional[bytes] = None


@dataclass
class PreparedArtifacts:
    source_lock_sha256: str
    files: Dict[str, FileItem] = field(default_factory=dict)
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    support_files: List[Dict[str, Any]] = field(default_factory=list)
    verifin_runs: List[Dict[str, Any]] = field(default_factory=list)
    baselines: List[Dict[str, Any]] = field(default_factory=list)
    smt_count: int = 0
    run_manifest_count: int = 0

    def add(self, item: FileItem) -> None:
        key = item.destination.as_posix()
        if key in {ARTIFACT_MANIFEST, CHECKSUMS}:
            raise ArtifactError(f"payload path is reserved: {key}")
        if key in self.files:
            raise ArtifactError(f"duplicate payload destination: {key}")
        self.files[key] = item


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactError(f"{label} must be a non-negative integer")
    return value


def _expected_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ArtifactError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ArtifactError(f"{label} must be a non-empty relative path")
    if "\\" in value or "\x00" in value or "\n" in value or "\r" in value:
        raise ArtifactError(f"{label} contains an unsafe character: {value!r}")
    if value.startswith("/") or "//" in value:
        raise ArtifactError(f"{label} is not a safe relative path: {value!r}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ArtifactError(f"{label} contains an unsafe path component: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ArtifactError(f"{label} must be relative: {value!r}")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_source(root: Path, relative: PurePosixPath, *, directory: bool = False) -> Path:
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactError(f"source symlink is not allowed: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactError(f"locked source does not exist: {relative}") from exc
    if not _is_within(resolved, root):
        raise ArtifactError(f"source escapes repository root: {relative}")
    if directory and not resolved.is_dir():
        raise ArtifactError(f"expected source directory: {relative}")
    if not directory and not resolved.is_file():
        raise ArtifactError(f"expected source file: {relative}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scan_text_value(text: str, display_path: str) -> None:
    if "\x00" in text:
        raise ArtifactError(f"NUL byte found in text artifact: {display_path}")
    for label, pattern in CONTENT_RULES.items():
        if pattern.search(text):
            raise ArtifactError(f"{label} found in {display_path}")
    for match in EMAIL_RE.finditer(text):
        domain = match.group(0).rsplit("@", 1)[1].lower()
        if domain not in SAFE_EMAIL_DOMAINS:
            raise ArtifactError(f"non-placeholder email found in {display_path}")


def _scan_text(path: Path, display_path: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"artifact is not valid UTF-8: {display_path}") from exc
    _scan_text_value(text, display_path)


def _make_item(
    source: Path,
    destination: PurePosixPath,
    expected_sha256: Optional[str],
    *,
    kind: str,
    dataset: str,
    artifact: str,
    max_file_bytes: int,
) -> FileItem:
    size = source.stat().st_size
    if size > max_file_bytes:
        raise ArtifactError(
            f"file exceeds {max_file_bytes:,} bytes: {destination} ({size:,})"
        )
    digest = _sha256(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ArtifactError(
            f"SHA-256 mismatch for {source}: expected {expected_sha256}, got {digest}"
        )
    _scan_text(source, destination.as_posix())
    return FileItem(
        source=source,
        destination=destination,
        sha256=digest,
        size=size,
        kind=kind,
        dataset=dataset,
        artifact=artifact,
    )


def _make_generated_json_item(
    payload: Mapping[str, Any],
    destination: PurePosixPath,
    *,
    kind: str,
    dataset: str,
    artifact: str,
    max_file_bytes: int,
) -> FileItem:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(content) > max_file_bytes:
        raise ArtifactError(
            f"generated file exceeds {max_file_bytes:,} bytes: {destination}"
        )
    _scan_text_value(content.decode("utf-8"), destination.as_posix())
    return FileItem(
        source=None,
        destination=destination,
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        kind=kind,
        dataset=dataset,
        artifact=artifact,
        content=content,
    )


def _load_jsonl(path: Path, id_field: str, expected_records: int) -> Tuple[List[Dict[str, Any]], Set[str]]:
    rows: List[Dict[str, Any]] = []
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ArtifactError(f"blank JSONL record at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ArtifactError(f"JSONL row is not an object at {path}:{line_number}")
            value = row.get(id_field)
            if not isinstance(value, (str, int)) or isinstance(value, bool):
                raise ArtifactError(f"missing or invalid {id_field!r} at {path}:{line_number}")
            record_id = str(value)
            if not ID_RE.fullmatch(record_id):
                raise ArtifactError(f"unsafe record ID {record_id!r} at {path}:{line_number}")
            if record_id in ids:
                raise ArtifactError(f"duplicate record ID {record_id!r} in {path}")
            ids.add(record_id)
            rows.append(row)
    if len(rows) != expected_records:
        raise ArtifactError(
            f"record-count mismatch for {path}: expected {expected_records}, got {len(rows)}"
        )
    return rows, ids


def _baseline_protocol(entry: Mapping[str, Any]) -> str:
    """Historical reference-text runs are not valid publication candidates."""
    baseline_id = entry.get("id", "unknown")
    if entry.get("candidate_protocol") != CANDIDATE_PROTOCOL:
        raise ArtifactError(
            f"baseline {baseline_id} uses a legacy or missing candidate protocol; "
            "historical raw_answer artifacts may contain reference answers. "
            "Rerun with claim_input_answer_v2 in a new directory and update the lock."
        )
    return _expected_hash(entry.get("run_fingerprint"), f"baseline {baseline_id} run_fingerprint")


def _validate_execution_manifest(
    document: Mapping[str, Any], entry: Mapping[str, Any],
    *, dataset_sha256: str, answers_sha256: str, expected_ids: Set[str],
) -> None:
    fingerprint = _baseline_protocol(entry)
    if document.get("schema_version") != 2 or document.get("candidate_protocol") != CANDIDATE_PROTOCOL:
        raise ArtifactError(f"baseline {entry['id']} requires the original v2 execution manifest")
    config = document.get("config")
    if not isinstance(config, dict):
        raise ArtifactError(f"baseline {entry['id']} execution manifest is missing config")
    actual = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if document.get("run_fingerprint") != fingerprint or actual != fingerprint:
        raise ArtifactError(f"baseline {entry['id']} execution manifest fingerprint mismatch")
    protocol = _judge_protocol(config.get("judge_protocol", SCALAR_JUDGE_PROTOCOL), entry["id"])
    locked_protocol = _judge_protocol(entry.get("judge_protocol", SCALAR_JUDGE_PROTOCOL), entry["id"])
    if protocol != locked_protocol:
        raise ArtifactError(f"baseline {entry['id']} execution judge_protocol differs from the lock")
    if protocol == FULL_ANSWER_JUDGE_PROTOCOL and (
        config.get("baseline") != "judge" or config.get("with_formula") is not False
        or config.get("cot") is not False
    ):
        raise ArtifactError(f"baseline {entry['id']} full-answer protocol requires plain judge mode")
    expected_config = {
        "candidate_protocol": CANDIDATE_PROTOCOL,
        "dataset_sha256": dataset_sha256,
        "answers_sha256": answers_sha256,
        "baseline": "judge" if entry.get("baseline_mode") == "judge-with-formula" else entry.get("baseline_mode"),
        "with_formula": entry.get("baseline_mode") == "judge-with-formula",
    }
    if entry.get("evaluator_model") is not None:
        expected_config["model"] = entry["evaluator_model"]
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ArtifactError(f"baseline {entry['id']} execution config {key} differs from the lock")
    for field in ("prompt_sha256", "implementation_sha256"):
        _expected_hash(config.get(field), f"baseline {entry['id']} execution {field}")
    if config.get("llm_mode") not in {"claude", "vllm", "openai"} or not isinstance(config.get("model"), str) or not config["model"]:
        raise ArtifactError(f"baseline {entry['id']} execution model configuration is missing")
    selected = config.get("selected_ids")
    if not isinstance(selected, list) or set(selected) != expected_ids or len(selected) != len(expected_ids):
        raise ArtifactError(f"baseline {entry['id']} execution candidate IDs differ from the dataset")


def _judge_protocol(protocol: Any, context: str) -> str:
    if protocol not in (SCALAR_JUDGE_PROTOCOL, FULL_ANSWER_JUDGE_PROTOCOL):
        raise ArtifactError(f"baseline {context} has an unknown judge_protocol")
    return protocol


def _validate_full_answer_verdict(row: Mapping[str, Any], context: str) -> None:
    """Only an exact saved one-word response can authorize a full-answer decision."""
    if (row.get("baseline") != "judge" or row.get("with_formula") is not False
            or row.get("cot") is not False):
        raise ArtifactError(f"baseline {context} full-answer protocol requires plain judge mode")
    responses = row.get("judge_responses")
    if (not isinstance(responses, list) or len(responses) > 2
            or any(not isinstance(response, str) for response in responses)):
        raise ArtifactError(f"baseline {context} has invalid saved judge responses")
    if "judge_verdict" not in row:
        raise ArtifactError(f"baseline {context} is missing its saved judge verdict")
    verdict = row.get("judge_verdict")
    status = row.get("status")
    if status in {"VERIFIED", "VIOLATED"}:
        expected_verdict = "CORRECT" if status == "VERIFIED" else "INCORRECT"
        if (verdict != expected_verdict or not responses
                or responses[-1].strip() != expected_verdict):
            raise ArtifactError(f"baseline {context} decision lacks a matching exact one-word verdict")
        if not row.get("claim_input_answer", "").strip():
            raise ArtifactError(f"baseline {context} decides an empty candidate")
        if any(response.strip() in {"CORRECT", "INCORRECT"} for response in responses[:-1]):
            raise ArtifactError(f"baseline {context} retries an already valid one-word verdict")
    elif status == "ABSTAIN":
        if verdict is not None or any(response.strip() in {"CORRECT", "INCORRECT"} for response in responses):
            raise ArtifactError(f"baseline {context} abstention conflicts with its saved judge verdict")
    else:
        raise ArtifactError(f"baseline {context} has an invalid full-answer status")


def _validate_baseline_candidates(
    rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any],
) -> int:
    """Validate the candidate actually judged, including stored and display units."""
    fingerprint = _baseline_protocol(entry)
    judge_protocol = _judge_protocol(entry.get("judge_protocol", SCALAR_JUDGE_PROTOCOL), entry["id"])
    # Use the same deterministic normalization as the runner, never parse gold text.
    try:
        from verifiqa.eval.baseline_scoring import normalize_candidate
    except ImportError as exc:
        raise ArtifactError("candidate publication validation requires verifiqa.eval.baseline_scoring") from exc

    source_by_id = {str(row["id"]): row for row in source_rows}
    seen: Set[str] = set()
    for row in rows:
        rid = str(row.get("id", ""))
        source = source_by_id.get(rid)
        if source is None or rid in seen:
            raise ArtifactError(f"baseline {entry['id']} has missing or duplicate candidate {rid}")
        seen.add(rid)
        if row.get("candidate_protocol") != CANDIDATE_PROTOCOL or row.get("run_fingerprint") != fingerprint:
            raise ArtifactError(f"baseline {entry['id']}/{rid} mixes legacy or different runs")
        row_protocol = _judge_protocol(row.get("judge_protocol", SCALAR_JUDGE_PROTOCOL), f"{entry['id']}/{rid}")
        if row_protocol != judge_protocol:
            raise ArtifactError(f"baseline {entry['id']}/{rid} judge_protocol differs from its manifest")
        if "claim_input_answer" not in source:
            raise ArtifactError(f"baseline {entry['id']}/{rid} source lacks claim_input_answer")
        text = source["claim_input_answer"]
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ArtifactError(f"baseline {entry['id']}/{rid} has non-text claim_input_answer")
        text = text.strip()
        if row.get("claim_input_answer") != text or row.get("raw_answer") != text:
            raise ArtifactError(f"baseline {entry['id']}/{rid} does not preserve generated candidate text")
        original_value = source.get("candidate_source_claimed_value", source.get("claimed_value"))
        if "candidate_source_claimed_value" not in row or row["candidate_source_claimed_value"] != original_value:
            raise ArtifactError(f"baseline {entry['id']}/{rid} does not preserve the original source claim value")
        if "candidate_quantity_review" not in row or row["candidate_quantity_review"] != source.get("candidate_quantity_review"):
            raise ArtifactError(f"baseline {entry['id']}/{rid} does not preserve candidate review metadata")
        expected = normalize_candidate(source)
        fields = {
            "claimed_value": "stored_value",
            "claimed_value_unit": "stored_unit",
            "candidate_value": "value",
            "candidate_unit": "unit",
            "candidate_quantity_status": "status",
            "candidate_quantity_reason": "reason",
            "candidate_value_source": "value_source",
            "candidate_unit_source": "unit_source",
            "absolute_tolerance": "tolerance",
            "tolerance_source": "tolerance_source",
        }
        for saved, normalized in fields.items():
            if saved not in row or row[saved] != expected[normalized]:
                raise ArtifactError(f"baseline {entry['id']}/{rid} candidate {saved} differs from its source")
        for field in ("claimed_value", "candidate_value", "absolute_tolerance"):
            value = row[field]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ArtifactError(f"baseline {entry['id']}/{rid} has invalid {field}")
        if judge_protocol == FULL_ANSWER_JUDGE_PROTOCOL:
            if entry.get("baseline_mode") != "judge":
                raise ArtifactError(f"baseline {entry['id']} full-answer protocol requires plain judge mode")
            _validate_full_answer_verdict(row, f"{entry['id']}/{rid}")
        elif expected["status"] != "resolved" and row.get("status") != "ABSTAIN":
            raise ArtifactError(f"baseline {entry['id']}/{rid} decides an unresolved candidate")
    if seen != set(source_by_id):
        raise ArtifactError(f"baseline {entry['id']} does not preserve the complete candidate pool")
    if entry.get("candidate_answer_matches") != len(rows):
        raise ArtifactError(f"baseline {entry['id']} candidate_answer_matches does not cover every row")
    return len(rows)


def _tree_hash(entries: Iterable[Tuple[str, str]]) -> str:
    lines = sorted(f"{digest}  {relative}\n" for digest, relative in entries)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _unique_ids(entries: Sequence[Mapping[str, Any]], label: str) -> None:
    seen: Set[str] = set()
    for entry in entries:
        value = entry.get("id")
        if not isinstance(value, str) or not ID_RE.fullmatch(value):
            raise ArtifactError(f"invalid {label} ID: {value!r}")
        if value in seen:
            raise ArtifactError(f"duplicate {label} ID: {value}")
        seen.add(value)


def _load_lock(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid artifact lock JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ArtifactError("artifact lock must use schema_version 1")
    for key in ("datasets", "support_files", "verifin_runs", "baselines"):
        if not isinstance(data.get(key), list):
            raise ArtifactError(f"artifact lock {key!r} must be a list")
    if not isinstance(data.get("limits"), dict):
        raise ArtifactError("artifact lock 'limits' must be an object")
    _unique_ids(data["datasets"], "dataset")
    _unique_ids(data["verifin_runs"], "VeriFin run")
    _unique_ids(data["baselines"], "baseline")
    return data


def prepare_source(root: Path, lock_path: Path, lock: Mapping[str, Any]) -> PreparedArtifacts:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ArtifactError(f"source root is not a directory: {root}")
    lock_path = lock_path.resolve(strict=True)
    if not _is_within(lock_path, root) or lock_path.is_symlink():
        raise ArtifactError("artifact lock must be a regular file inside the source root")

    limits = lock["limits"]
    max_file_bytes = _as_int(limits.get("max_file_bytes"), "limits.max_file_bytes")
    max_total_bytes = _as_int(limits.get("max_total_bytes"), "limits.max_total_bytes")
    expected_dataset_count = _as_int(limits.get("expected_dataset_count"), "limits.expected_dataset_count")
    expected_support_count = _as_int(limits.get("expected_support_file_count"), "limits.expected_support_file_count")
    expected_run_count = _as_int(limits.get("expected_verifin_run_count"), "limits.expected_verifin_run_count")
    expected_baseline_count = _as_int(limits.get("expected_baseline_count"), "limits.expected_baseline_count")
    expected_run_manifest_count = _as_int(
        limits.get("expected_run_manifest_count"),
        "limits.expected_run_manifest_count",
    )
    expected_smt_count = _as_int(limits.get("expected_smt_count"), "limits.expected_smt_count")
    expected_file_count = _as_int(limits.get("expected_payload_file_count"), "limits.expected_payload_file_count")
    if len(lock["datasets"]) != expected_dataset_count:
        raise ArtifactError("locked dataset count does not match limits")
    if len(lock["support_files"]) != expected_support_count:
        raise ArtifactError("locked support-file count does not match limits")
    if len(lock["verifin_runs"]) != expected_run_count:
        raise ArtifactError("locked VeriFin run count does not match limits")
    if len(lock["baselines"]) != expected_baseline_count:
        raise ArtifactError("locked baseline count does not match limits")
    # Reject invalid historical experiments before copying any result bytes.
    for entry in lock["baselines"]:
        _baseline_protocol(entry)

    source_package = str(root / "src")
    if source_package not in sys.path:
        sys.path.insert(0, source_package)

    prepared = PreparedArtifacts(source_lock_sha256=_sha256(lock_path))
    dataset_ids: Dict[str, Set[str]] = {}
    dataset_records: Dict[str, int] = {}
    dataset_hashes: Dict[str, str] = {}

    for entry in lock["datasets"]:
        dataset_id = entry["id"]
        source_rel = _safe_relative(entry.get("source"), f"dataset {dataset_id} source")
        destination = _safe_relative(entry.get("destination"), f"dataset {dataset_id} destination")
        records = _as_int(entry.get("records"), f"dataset {dataset_id} records")
        id_field = entry.get("id_field")
        if not isinstance(id_field, str) or not ID_RE.fullmatch(id_field):
            raise ArtifactError(f"invalid ID field for dataset {dataset_id}")
        source = _resolve_source(root, source_rel)
        item = _make_item(
            source,
            destination,
            _expected_hash(entry.get("sha256"), f"dataset {dataset_id} sha256"),
            kind="dataset",
            dataset=dataset_id,
            artifact=dataset_id,
            max_file_bytes=max_file_bytes,
        )
        _, ids = _load_jsonl(source, id_field, records)
        prepared.add(item)
        dataset_ids[dataset_id] = ids
        dataset_records[dataset_id] = records
        dataset_hashes[dataset_id] = item.sha256
        prepared.datasets.append(
            {
                "id": dataset_id,
                "label": entry.get("label", dataset_id),
                "path": destination.as_posix(),
                "records": records,
                "id_field": id_field,
                "sha256": item.sha256,
            }
        )

    for entry in lock["support_files"]:
        source_rel = _safe_relative(entry.get("source"), "support-file source")
        destination = _safe_relative(entry.get("destination"), "support-file destination")
        kind = entry.get("kind")
        if kind not in {
            "dataset-documentation",
            "result-documentation",
            "test-fixture",
        }:
            raise ArtifactError(f"invalid support-file kind for {source_rel}: {kind!r}")
        source = _resolve_source(root, source_rel)
        item = _make_item(
            source,
            destination,
            _expected_hash(entry.get("sha256"), f"support-file {source_rel} sha256"),
            kind=kind,
            dataset="",
            artifact="publication-documentation",
            max_file_bytes=max_file_bytes,
        )
        prepared.add(item)
        prepared.support_files.append(
            {
                "path": destination.as_posix(),
                "kind": kind,
                "sha256": item.sha256,
            }
        )

    scoring_policy_item = prepared.files.get("results/scoring_policy.json")
    if scoring_policy_item is None:
        raise ArtifactError("support files must include results/scoring_policy.json")
    baseline_implementation = _resolve_source(
        root, PurePosixPath("scripts/baselines.py")
    )
    baseline_implementation_sha256 = _sha256(baseline_implementation)

    for entry in lock["verifin_runs"]:
        run_id = entry["id"]
        dataset_id = entry.get("dataset")
        if dataset_id not in dataset_ids:
            raise ArtifactError(f"run {run_id} names unknown dataset {dataset_id!r}")
        records = _as_int(entry.get("records"), f"run {run_id} records")
        if records != dataset_records[dataset_id]:
            raise ArtifactError(f"run {run_id} count differs from dataset {dataset_id}")
        source_base_rel = _safe_relative(entry.get("source"), f"run {run_id} source")
        destination_base = _safe_relative(entry.get("destination"), f"run {run_id} destination")
        _resolve_source(root, source_base_rel, directory=True)

        results_rel = source_base_rel / "results.jsonl"
        summary_rel = source_base_rel / "summary.json"
        results_source = _resolve_source(root, results_rel)
        summary_source = _resolve_source(root, summary_rel)
        results_item = _make_item(
            results_source,
            destination_base / "results.jsonl",
            _expected_hash(entry.get("results_sha256"), f"run {run_id} results_sha256"),
            kind="verifin-results",
            dataset=dataset_id,
            artifact=run_id,
            max_file_bytes=max_file_bytes,
        )
        summary_item = _make_item(
            summary_source,
            destination_base / "summary.json",
            _expected_hash(entry.get("summary_sha256"), f"run {run_id} summary_sha256"),
            kind="verifin-summary",
            dataset=dataset_id,
            artifact=run_id,
            max_file_bytes=max_file_bytes,
        )
        rows, result_ids = _load_jsonl(results_source, "id", records)
        if result_ids != dataset_ids[dataset_id]:
            missing = len(dataset_ids[dataset_id] - result_ids)
            extra = len(result_ids - dataset_ids[dataset_id])
            raise ArtifactError(
                f"run {run_id} ID set differs from {dataset_id}: {missing} missing, {extra} extra"
            )
        try:
            summary = json.loads(summary_source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"invalid summary JSON for {run_id}: {exc}") from exc
        if not isinstance(summary, dict) or summary.get("total") != records:
            raise ArtifactError(f"summary total mismatch for run {run_id}")

        prepared.add(results_item)
        prepared.add(summary_item)
        tree_entries: List[Tuple[str, str]] = [
            (results_item.sha256, "results.jsonl"),
            (summary_item.sha256, "summary.json"),
        ]
        status_counts = {status: 0 for status in sorted(ALLOWED_STATUSES)}
        smt_count = _as_int(entry.get("smt_count"), f"run {run_id} smt_count")
        if smt_count != records:
            raise ArtifactError(f"run {run_id} must lock exactly one SMT per result row")
        for row in rows:
            record_id = str(row["id"])
            status = row.get("status")
            if status not in ALLOWED_STATUSES:
                raise ArtifactError(f"unsupported status {status!r} in run {run_id}, ID {record_id}")
            status_counts[status] += 1
            filename = f"{record_id}_{status}.smt2"
            smt_source = _resolve_source(root, source_base_rel / "smt" / filename)
            smt_item = _make_item(
                smt_source,
                destination_base / "smt" / filename,
                None,
                kind="smt",
                dataset=dataset_id,
                artifact=run_id,
                max_file_bytes=max_file_bytes,
            )
            prepared.add(smt_item)
            tree_entries.append((smt_item.sha256, f"smt/{filename}"))
        if sum(status_counts.values()) != smt_count:
            raise ArtifactError(f"SMT count mismatch for run {run_id}")
        for status, summary_key in (
            ("VERIFIED", "verified"),
            ("VIOLATED", "violated"),
            ("ABSTAIN", "abstain"),
            ("UNVERIFIED_FORMULA", "unverified_formula"),
        ):
            if summary_key in summary and summary[summary_key] != status_counts[status]:
                raise ArtifactError(f"summary {summary_key} mismatch for run {run_id}")
        expected_tree = _expected_hash(entry.get("tree_sha256"), f"run {run_id} tree_sha256")
        actual_tree = _tree_hash(tree_entries)
        if actual_tree != expected_tree:
            raise ArtifactError(
                f"canonical tree hash mismatch for run {run_id}: expected {expected_tree}, got {actual_tree}"
            )
        model_identity_status = (
            "checkpoint-identity-unresolved"
            if "identity-pending" in run_id
            else "reported-label-only; immutable revision not embedded"
        )
        known_gaps = [
            "Exact immutable model revision, inference provider, and decoding "
            "configuration are not embedded in the source result files.",
            "A single run-level verifier configuration hash is unavailable; "
            "inspect the per-question result records and released SMT files.",
        ]
        if "identity-pending" in run_id:
            known_gaps.insert(
                0,
                "The Fin-o1 artifact has conflicting historical size labels; "
                "the public release intentionally makes no 8B or 30B claim.",
            )
        run_manifest = {
            "schema_version": 1,
            "artifact_type": "verifin-run",
            "artifact_id": run_id,
            "label": entry.get("label", run_id),
            "dataset": {
                "id": dataset_id,
                "records": records,
                "sha256": dataset_hashes[dataset_id],
            },
            "source_path": source_base_rel.as_posix(),
            "release_path": destination_base.as_posix(),
            "model": {
                "reported_label": entry.get("label", run_id),
                "identity_status": model_identity_status,
                "immutable_checkpoint": None,
                "revision": None,
                "inference_provider": None,
                "decoding_configuration": None,
            },
            "verifier": {
                "configuration_hash": None,
                "configuration_status": (
                    "not embedded as one immutable run-level configuration"
                ),
            },
            "files": {
                "results": {
                    "path": (destination_base / "results.jsonl").as_posix(),
                    "sha256": results_item.sha256,
                },
                "summary": {
                    "path": (destination_base / "summary.json").as_posix(),
                    "sha256": summary_item.sha256,
                },
                "smt_directory": (destination_base / "smt").as_posix(),
                "smt_count": smt_count,
                "canonical_tree_sha256": actual_tree,
            },
            "statuses": status_counts,
            "scoring": {
                "policy_path": "results/scoring_policy.json",
                "policy_sha256": scoring_policy_item.sha256,
                "scorer_path": "scripts/paper_scoring.py",
            },
            "known_metadata_gaps": known_gaps,
        }
        run_manifest_item = _make_generated_json_item(
            run_manifest,
            destination_base / "run_manifest.json",
            kind="run-manifest",
            dataset=dataset_id,
            artifact=run_id,
            max_file_bytes=max_file_bytes,
        )
        prepared.add(run_manifest_item)
        prepared.run_manifest_count += 1
        prepared.smt_count += smt_count
        prepared.verifin_runs.append(
            {
                "id": run_id,
                "label": entry.get("label", run_id),
                "dataset": dataset_id,
                "results": (destination_base / "results.jsonl").as_posix(),
                "summary": (destination_base / "summary.json").as_posix(),
                "records": records,
                "statuses": status_counts,
                "smt_directory": (destination_base / "smt").as_posix(),
                "smt_count": smt_count,
                "tree_sha256": actual_tree,
                "run_manifest": (
                    destination_base / "run_manifest.json"
                ).as_posix(),
            }
        )

    for entry in lock["baselines"]:
        baseline_id = entry["id"]
        dataset_id = entry.get("dataset")
        if dataset_id not in dataset_ids:
            raise ArtifactError(f"baseline {baseline_id} names unknown dataset {dataset_id!r}")
        records = _as_int(entry.get("records"), f"baseline {baseline_id} records")
        if records != dataset_records[dataset_id]:
            raise ArtifactError(f"baseline {baseline_id} count differs from dataset {dataset_id}")
        source_rel = _safe_relative(entry.get("source"), f"baseline {baseline_id} source")
        destination = _safe_relative(entry.get("destination"), f"baseline {baseline_id} destination")
        source = _resolve_source(root, source_rel)
        item = _make_item(
            source,
            destination,
            _expected_hash(entry.get("sha256"), f"baseline {baseline_id} sha256"),
            kind="baseline-results",
            dataset=dataset_id,
            artifact=baseline_id,
            max_file_bytes=max_file_bytes,
        )
        baseline_rows, ids = _load_jsonl(source, "id", records)
        if ids != dataset_ids[dataset_id]:
            missing = len(dataset_ids[dataset_id] - ids)
            extra = len(ids - dataset_ids[dataset_id])
            raise ArtifactError(
                f"baseline {baseline_id} ID set differs from {dataset_id}: {missing} missing, {extra} extra"
            )
        baseline_mode = entry.get("baseline_mode")
        if baseline_mode not in {
            "none",
            "judge",
            "judge-with-formula",
            "pot",
        }:
            raise ArtifactError(
                f"baseline {baseline_id} has invalid baseline_mode {baseline_mode!r}"
            )
        answers_from_rel = _safe_relative(
            entry.get("answers_from"), f"baseline {baseline_id} answers_from"
        )
        answers_from_source = _resolve_source(root, answers_from_rel)
        answers_from_sha256 = _expected_hash(
            entry.get("answers_from_sha256"),
            f"baseline {baseline_id} answers_from_sha256",
        )
        actual_answers_from_sha256 = _sha256(answers_from_source)
        if actual_answers_from_sha256 != answers_from_sha256:
            raise ArtifactError(
                f"baseline {baseline_id} candidate-source hash mismatch"
            )
        answer_rows, answer_ids = _load_jsonl(
            answers_from_source, "id", records
        )
        if answer_ids != ids:
            raise ArtifactError(
                f"baseline {baseline_id} candidate-source IDs differ"
            )
        candidate_matches = _validate_baseline_candidates(baseline_rows, answer_rows, entry)
        candidate_sources = [
            file.destination for file in prepared.files.values()
            if file.source == answers_from_source and file.sha256 == answers_from_sha256
        ]
        if len(candidate_sources) != 1:
            raise ArtifactError(f"baseline {baseline_id} candidate source must be included exactly once in the bundle")
        execution_source_rel = source_rel.parent / "run_manifest.json"
        execution_source = _resolve_source(root, execution_source_rel)
        execution_item = _make_item(
            execution_source, destination.parent / "execution_manifest.json",
            _expected_hash(entry.get("execution_manifest_sha256"), f"baseline {baseline_id} execution_manifest_sha256"),
            kind="baseline-execution-manifest", dataset=dataset_id, artifact=baseline_id,
            max_file_bytes=max_file_bytes,
        )
        execution_manifest = json.loads(execution_source.read_text(encoding="utf-8"))
        if not isinstance(execution_manifest, dict):
            raise ArtifactError(f"baseline {baseline_id} execution manifest must be an object")
        _validate_execution_manifest(
            execution_manifest, entry,
            dataset_sha256=dataset_hashes[dataset_id], answers_sha256=answers_from_sha256,
            expected_ids=ids,
        )
        evaluator_model = entry.get("evaluator_model")
        if evaluator_model is not None and (
            not isinstance(evaluator_model, str) or not evaluator_model
        ):
            raise ArtifactError(
                f"baseline {baseline_id} has invalid evaluator_model"
            )
        if not isinstance(entry.get("resume"), bool):
            raise ArtifactError(f"baseline {baseline_id} resume must be boolean")
        prepared.add(item)
        prepared.add(execution_item)
        baseline_manifest = {
            "schema_version": 1,
            "artifact_type": "baseline-run",
            "artifact_id": baseline_id,
            "label": entry.get("label", baseline_id),
            "dataset": {
                "id": dataset_id,
                "records": records,
                "sha256": dataset_hashes[dataset_id],
            },
            "source_path": source_rel.as_posix(),
            "release_path": destination.parent.as_posix(),
            "method": {
                "reported_label": entry.get("label", baseline_id),
                "implementation_path": "scripts/baselines.py",
                "implementation_sha256": baseline_implementation_sha256,
                "baseline_mode": baseline_mode,
                "evaluator_model": evaluator_model,
                "llm_mode": execution_manifest["config"]["llm_mode"],
                "resume": entry["resume"],
                "run_fingerprint": entry["run_fingerprint"],
                "execution_manifest": execution_item.destination.as_posix(),
            },
            "candidate_pool": {
                "answers_from": answers_from_rel.as_posix(),
                "answers_from_sha256": answers_from_sha256,
                "packaged_answers_from": candidate_sources[0].as_posix(),
                "candidate_protocol": CANDIDATE_PROTOCOL,
                "candidate_answer_field": "claim_input_answer",
                "candidate_answer_matches": candidate_matches,
                "records": records,
                "claimed_value_note": (
                    "Stored numeric claims and normalized display quantities are checked "
                    "against the generated candidate source. Scalar reparsing is explicit; "
                    "unresolved quantities must abstain."
                ),
            },
            "files": {
                "results": {
                    "path": destination.as_posix(),
                    "sha256": item.sha256,
                }
            },
            "scoring": {
                "policy_path": "results/scoring_policy.json",
                "policy_sha256": scoring_policy_item.sha256,
                "scorer_path": "scripts/paper_scoring.py",
            },
            "known_metadata_gaps": [
                "The dependency lock and service-side provider configuration "
                "are not embedded in the result file."
            ],
        }
        baseline_manifest_item = _make_generated_json_item(
            baseline_manifest,
            destination.parent / "run_manifest.json",
            kind="run-manifest",
            dataset=dataset_id,
            artifact=baseline_id,
            max_file_bytes=max_file_bytes,
        )
        prepared.add(baseline_manifest_item)
        prepared.run_manifest_count += 1
        prepared.baselines.append(
            {
                "id": baseline_id,
                "label": entry.get("label", baseline_id),
                "dataset": dataset_id,
                "path": destination.as_posix(),
                "records": records,
                "sha256": item.sha256,
                "candidate_protocol": CANDIDATE_PROTOCOL,
                "run_fingerprint": entry["run_fingerprint"],
                "candidate_answer_matches": candidate_matches,
                "baseline_mode": baseline_mode,
                "evaluator_model": evaluator_model,
                "execution_manifest": execution_item.destination.as_posix(),
                "execution_manifest_sha256": execution_item.sha256,
                "run_manifest": (
                    destination.parent / "run_manifest.json"
                ).as_posix(),
            }
        )

    if prepared.smt_count != expected_smt_count:
        raise ArtifactError(
            f"total SMT count mismatch: expected {expected_smt_count}, got {prepared.smt_count}"
        )
    if prepared.run_manifest_count != expected_run_manifest_count:
        raise ArtifactError(
            "run-manifest count mismatch: expected "
            f"{expected_run_manifest_count}, got {prepared.run_manifest_count}"
        )
    if len(prepared.files) != expected_file_count:
        raise ArtifactError(
            f"payload file count mismatch: expected {expected_file_count}, got {len(prepared.files)}"
        )
    total_bytes = sum(item.size for item in prepared.files.values())
    if total_bytes > max_total_bytes:
        raise ArtifactError(
            f"payload exceeds {max_total_bytes:,} bytes: {total_bytes:,}"
        )
    return prepared


def _inventory(prepared: PreparedArtifacts) -> Dict[str, Any]:
    files = [
        {
            "path": path,
            "bytes": item.size,
            "sha256": item.sha256,
            "kind": item.kind,
            "dataset": item.dataset,
            "artifact": item.artifact,
        }
        for path, item in sorted(prepared.files.items())
    ]
    return {
        "schema_version": 1,
        "source_lock": {
            "name": "publication_artifacts.json",
            "sha256": prepared.source_lock_sha256,
        },
        "tree_hash_algorithm": "sha256-of-lexicographically-sorted-sha256sum-lines-v1",
        "counts": {
            "datasets": len(prepared.datasets),
            "support_files": len(prepared.support_files),
            "verifin_runs": len(prepared.verifin_runs),
            "baselines": len(prepared.baselines),
            "run_manifests": prepared.run_manifest_count,
            "smt_files": prepared.smt_count,
            "payload_files": len(files),
            "payload_bytes": sum(entry["bytes"] for entry in files),
        },
        "datasets": prepared.datasets,
        "support_files": prepared.support_files,
        "verifin_runs": prepared.verifin_runs,
        "baselines": prepared.baselines,
        "files": files,
    }


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle, length=1024 * 1024)


def _write_bundle(destination: Path, prepared: PreparedArtifacts) -> None:
    """Add a locked bundle to an existing directory without overwriting files."""

    if not destination.is_dir() or destination.is_symlink():
        raise ArtifactError(f"destination is not a regular directory: {destination}")
    planned = set(prepared.files) | {ARTIFACT_MANIFEST, CHECKSUMS}
    conflicts = [
        relative
        for relative in sorted(planned)
        if (destination.joinpath(*PurePosixPath(relative).parts)).exists()
        or (destination.joinpath(*PurePosixPath(relative).parts)).is_symlink()
    ]
    if conflicts:
        raise ArtifactError(
            "refusing to overwrite existing artifact paths: "
            + ", ".join(conflicts[:5])
        )

    for relative, item in sorted(prepared.files.items()):
        output = destination.joinpath(*PurePosixPath(relative).parts)
        if item.source is not None:
            _copy_exclusive(item.source, output)
        elif item.content is not None:
            _write_exclusive(output, item.content)
        else:
            raise ArtifactError(f"payload has neither source nor content: {relative}")
        if _sha256(output) != item.sha256:
            raise ArtifactError(f"copied bytes failed verification: {relative}")

    inventory = _inventory(prepared)
    manifest_bytes = (
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_path = destination / ARTIFACT_MANIFEST
    _write_exclusive(manifest_path, manifest_bytes)

    checksum_entries = [(item.sha256, relative) for relative, item in prepared.files.items()]
    checksum_entries.append((hashlib.sha256(manifest_bytes).hexdigest(), ARTIFACT_MANIFEST))
    checksum_text = "".join(
        f"{digest}  {relative}\n"
        for digest, relative in sorted(checksum_entries, key=lambda pair: pair[1])
    )
    _write_exclusive(destination / CHECKSUMS, checksum_text.encode("utf-8"))


def add_bundle(root: Path, destination: Path, prepared: PreparedArtifacts) -> None:
    """Add artifacts to an existing release tree and validate managed roots."""

    root = root.resolve(strict=True)
    destination = destination.resolve(strict=True)
    if destination == root or _is_within(destination, root):
        raise ArtifactError("destination must be outside the source repository")
    _write_bundle(destination, prepared)
    check_bundle(destination, allow_other_files=True)


def build_bundle(root: Path, destination: Path, prepared: PreparedArtifacts) -> None:
    root = root.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise ArtifactError(f"destination already exists; refusing to overwrite: {destination}")
    destination = destination.resolve(strict=False)
    if destination == root or _is_within(destination, root):
        raise ArtifactError("destination must be outside the source repository")
    destination.mkdir(parents=True, exist_ok=False)
    _write_bundle(destination, prepared)
    check_bundle(destination)


def _bundle_files(root: Path) -> Set[str]:
    files: Set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(f"bundle symlink is not allowed: {path.relative_to(root)}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _check_bundle_baselines(bundle: Path, manifest: Mapping[str, Any]) -> None:
    baselines = manifest.get("baselines", [])
    if not isinstance(baselines, list):
        raise ArtifactError("bundle baselines must be a list")
    datasets = {entry["id"]: entry for entry in manifest.get("datasets", [])}
    package = bundle / "src"
    if package.is_dir() and str(package) not in sys.path:
        sys.path.insert(0, str(package))
    for entry in baselines:
        _baseline_protocol(entry)
        records = _as_int(entry.get("records"), f"baseline {entry['id']} records")
        publication_path = _resolve_source(bundle, _safe_relative(entry.get("run_manifest"), "baseline run manifest"))
        publication = json.loads(publication_path.read_text(encoding="utf-8"))
        pool = publication.get("candidate_pool", {})
        if pool.get("candidate_protocol") != CANDIDATE_PROTOCOL or pool.get("candidate_answer_field") != "claim_input_answer":
            raise ArtifactError(f"baseline {entry['id']} publication manifest uses a legacy candidate pool")
        source_path = _resolve_source(bundle, _safe_relative(pool.get("packaged_answers_from"), "baseline candidate source"))
        source_hash = _expected_hash(pool.get("answers_from_sha256"), "baseline candidate source hash")
        if _sha256(source_path) != source_hash:
            raise ArtifactError(f"baseline {entry['id']} candidate-source hash mismatch")
        source_rows, source_ids = _load_jsonl(source_path, "id", records)
        result_path = _resolve_source(bundle, _safe_relative(entry.get("path"), "baseline results"))
        result_rows, result_ids = _load_jsonl(result_path, "id", records)
        if result_ids != source_ids:
            raise ArtifactError(f"baseline {entry['id']} candidate-source IDs differ")
        _validate_baseline_candidates(result_rows, source_rows, entry)
        execution_path = _resolve_source(bundle, _safe_relative(entry.get("execution_manifest"), "baseline execution manifest"))
        if _sha256(execution_path) != _expected_hash(entry.get("execution_manifest_sha256"), "baseline execution manifest hash"):
            raise ArtifactError(f"baseline {entry['id']} execution manifest hash mismatch")
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        dataset = datasets.get(entry.get("dataset"))
        if dataset is None:
            raise ArtifactError(f"baseline {entry['id']} refers to an unknown dataset")
        dataset_path = _resolve_source(bundle, _safe_relative(dataset.get("path"), "baseline dataset"))
        _, dataset_ids = _load_jsonl(dataset_path, dataset["id_field"], records)
        if result_ids != dataset_ids:
            raise ArtifactError(f"baseline {entry['id']} IDs differ from its dataset")
        _validate_execution_manifest(
            execution, entry, dataset_sha256=_sha256(dataset_path),
            answers_sha256=source_hash, expected_ids=result_ids,
        )


def check_bundle(bundle: Path, *, allow_other_files: bool = False) -> None:
    if bundle.is_symlink():
        raise ArtifactError(f"bundle root cannot be a symlink: {bundle}")
    bundle = bundle.resolve(strict=True)
    if not bundle.is_dir():
        raise ArtifactError(f"bundle is not a directory: {bundle}")
    manifest_path = bundle / ARTIFACT_MANIFEST
    checksum_path = bundle / CHECKSUMS
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise ArtifactError(f"bundle requires {ARTIFACT_MANIFEST} and {CHECKSUMS}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid {ARTIFACT_MANIFEST}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ArtifactError(f"unsupported {ARTIFACT_MANIFEST} schema")
    inventory = manifest.get("files")
    if not isinstance(inventory, list):
        raise ArtifactError(f"{ARTIFACT_MANIFEST} files must be a list")

    expected: Dict[str, str] = {}
    for entry in inventory:
        if not isinstance(entry, dict):
            raise ArtifactError("invalid file inventory entry")
        relative = _safe_relative(entry.get("path"), "inventory path").as_posix()
        digest = _expected_hash(entry.get("sha256"), f"inventory hash for {relative}")
        if relative in expected:
            raise ArtifactError(f"duplicate inventory path: {relative}")
        path = bundle.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise ArtifactError(f"missing or unsafe bundle file: {relative}")
        size = _as_int(entry.get("bytes"), f"inventory bytes for {relative}")
        if path.stat().st_size != size:
            raise ArtifactError(f"size mismatch in bundle: {relative}")
        actual = _sha256(path)
        if actual != digest:
            raise ArtifactError(f"SHA-256 mismatch in bundle: {relative}")
        _scan_text(path, relative)
        expected[relative] = digest

    controls = {ARTIFACT_MANIFEST, CHECKSUMS}
    actual_files = _bundle_files(bundle)
    expected_files = set(expected) | controls
    if allow_other_files:
        actual_managed = {
            relative
            for relative in actual_files
            if relative in controls
            or PurePosixPath(relative).parts[0] in MANAGED_ROOTS
        }
    else:
        actual_managed = actual_files
    if actual_managed != expected_files:
        extra = sorted(actual_managed - expected_files)
        missing = sorted(expected_files - actual_managed)
        raise ArtifactError(f"bundle file set mismatch; extra={extra[:5]}, missing={missing[:5]}")

    checksum_map = dict(expected)
    checksum_map[ARTIFACT_MANIFEST] = _sha256(manifest_path)
    expected_text = "".join(
        f"{digest}  {relative}\n" for relative, digest in sorted(checksum_map.items())
    )
    actual_text = checksum_path.read_text(encoding="utf-8")
    if actual_text != expected_text:
        raise ArtifactError(f"{CHECKSUMS} does not match the bundle inventory")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ArtifactError("bundle counts must be an object")
    if counts.get("payload_files") != len(expected):
        raise ArtifactError("bundle payload file count is inconsistent")
    manifest_files = sum(
        entry.get("kind") == "run-manifest" for entry in inventory
    )
    if counts.get("run_manifests") != manifest_files:
        raise ArtifactError("bundle run-manifest count is inconsistent")
    if counts.get("payload_bytes") != sum(
        _as_int(entry.get("bytes"), "inventory bytes") for entry in inventory
    ):
        raise ArtifactError("bundle payload byte count is inconsistent")
    source_lock = manifest.get("source_lock")
    public_lock = bundle / "publication_artifacts.json"
    if public_lock.is_file():
        if not isinstance(source_lock, dict):
            raise ArtifactError("artifact manifest source_lock must be an object")
        locked_hash = _expected_hash(
            source_lock.get("sha256"), "artifact manifest source-lock hash"
        )
        if _sha256(public_lock) != locked_hash:
            raise ArtifactError(
                "publication_artifacts.json differs from the lock used to build the bundle"
            )
    _check_bundle_baselines(bundle, manifest)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-source", action="store_true", help="validate the locked source set without writing")
    mode.add_argument("--output", type=Path, help="new artifact-bundle directory to create")
    mode.add_argument("--check", type=Path, metavar="DIRECTORY", help="validate an existing artifact bundle")
    mode.add_argument(
        "--check-release",
        type=Path,
        metavar="DIRECTORY",
        help="validate artifacts embedded in a larger publication tree",
    )
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        if args.check is not None or args.check_release is not None:
            target = args.check if args.check is not None else args.check_release
            check_bundle(target, allow_other_files=args.check_release is not None)
            print(f"Publication artifact bundle passed: {target.resolve()}")
            return 0
        source_root = args.source_root.resolve(strict=True)
        manifest_path = args.manifest
        if manifest_path.is_symlink():
            raise ArtifactError("artifact lock cannot be a symlink")
        manifest_path = manifest_path.resolve(strict=True)
        lock = _load_lock(manifest_path)
        prepared = prepare_source(source_root, manifest_path, lock)
        total_bytes = sum(item.size for item in prepared.files.values())
        if args.check_source:
            print(
                "Locked publication artifacts passed: "
                f"{len(prepared.datasets)} datasets, "
                f"{len(prepared.verifin_runs)} VeriFin runs, "
                f"{len(prepared.baselines)} baselines, "
                f"{prepared.smt_count} SMT files, "
                f"{prepared.run_manifest_count} run manifests, "
                f"{len(prepared.files)} payload files, {total_bytes:,} bytes"
            )
            return 0
        build_bundle(source_root, args.output, prepared)
        print(
            f"Created publication artifact bundle with {len(prepared.files)} payload files: "
            f"{args.output.resolve()}"
        )
        return 0
    except (ArtifactError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
