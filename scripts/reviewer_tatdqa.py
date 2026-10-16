"""Calculate the latest TADQA reviewer presentation, offline.

The current view uses the shared CLI scorer on complete result logs. The legacy
snapshot loader remains available for reproducing archived presentations.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import textwrap
from typing import Any

from scripts.submission_runs import RUNS as SUBMISSION_RUNS


SNAPSHOT = "notebooks/data/tatdqa_review_runs/manifest.json"
INPUTS = {
    "dataset": "data/tatdqa_403_question_handoff_v1/dataset/financebench_xbrl_augmented.jsonl",
    "gold": "data/tatdqa_403_question_handoff_v1/evaluator_only/tatdqa_arithmetic_pilot_gold.jsonl",
    "unit_conversions": "notebooks/tatdqa_scoring_normalizations.json",
    "source_adjudications": "notebooks/tatdqa_scoring_adjudications.json",
}
OUTCOMES = ("TA", "FA", "TR", "FR", "AB")
VIEWS = ("literal_benchmark", "benchmark", "source_reviewed")
VIEW = "source_reviewed"
SNAPSHOT_MODELS = ("Claude Haiku 4.5", "Qwen2.5 7B")
RUNS = {
    "Claude Haiku 4.5": SUBMISSION_RUNS["tatdqa"]["claude-haiku-4.5"],
    "Qwen2.5 7B": SUBMISSION_RUNS["tatdqa"]["qwen2.5-7b"],
    "Qwen3 30B": SUBMISSION_RUNS["tatdqa"]["qwen3-30b"],
    "Llama 3.1 8B": SUBMISSION_RUNS["tatdqa"]["llama-3.1-8b"],
    "GPT-5.5": SUBMISSION_RUNS["tatdqa"]["gpt-5.5"],
    "Fin-o1 8B": SUBMISSION_RUNS["tatdqa"]["fino1-identity-pending"],
}
MODELS = tuple(RUNS)
N = 403
COLORS = {"TA": "#3569A8", "FA": "#C84444", "TR": "#278D88", "FR": "#D99343", "AB": "#A1A7AF"}


class SnapshotError(ValueError):
    """The pinned inputs or their reported aggregates do not reconcile."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise SnapshotError("Snapshot paths must be repository-relative")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SnapshotError("Snapshot path escapes repository") from exc
    return resolved


def _checked(root: Path, relative: str, expected: str) -> Path:
    path = _relative(root, relative)
    if not path.is_file() or sha256(path) != expected:
        raise SnapshotError(f"Hash mismatch or missing pinned input: {relative}")
    return path


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _index(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    ids = [row.get(key) for row in rows]
    if len(rows) != N or any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != N:
        raise SnapshotError(f"{label}: expected exactly {N} unique, nonempty IDs")
    return dict(zip(ids, rows))


def _rates(counts: dict[str, int]) -> dict[str, float]:
    decided = sum(counts[key] for key in ("TA", "FA", "TR", "FR"))
    accepted = counts["TA"] + counts["FA"]
    rejected = counts["TR"] + counts["FR"]
    if not accepted or not decided or not rejected:
        raise SnapshotError("The pinned runs require nonzero acceptance, decision, and rejection denominators")
    return {
        "precision": counts["TA"] / accepted,
        "decision_accuracy": (counts["TA"] + counts["TR"]) / decided,
        "coverage": decided / N,
        "rejection_precision": counts["TR"] / rejected,
        "abstention_rate": counts["AB"] / N,
    }


def load_snapshot(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate fixed input hashes, populations, questions, outcomes, and rates."""
    root = Path(root).resolve()
    manifest_path = root / SNAPSHOT
    manifest = json.loads(manifest_path.read_text())
    if set(manifest["runs"]) != set(SNAPSHOT_MODELS):
        raise SnapshotError("Unexpected model set in completed snapshot")
    snapshot_inputs = manifest["input_files"]
    if set(snapshot_inputs) != set(INPUTS):
        raise SnapshotError("Unexpected input set in completed snapshot")
    tables = []
    provenance: dict[str, Any] = {
        "dataset": "TADQA", "scoring_view": VIEW, "n_per_model": N,
        "input_mode": "pinned_completed_snapshot",
        "scope": "Aggregate existing post hoc source-reviewed scores; not an independent audit.",
        "review_scope": "Flagged benchmark mismatches; other accepted claims have not all been independently source-audited.",
        "original_primary_view": manifest["scoring_view"],
        "detailed_audit_notebook": "notebooks/tatdqa_result_metrics.ipynb",
        "manifest": {"path": SNAPSHOT, "sha256": sha256(manifest_path)},
        "matching_detailed_notebook": manifest["matching_detailed_notebook"],
        "snapshot_method": manifest["snapshot_method"],
        "historical_scoring_sources": manifest["scoring_sources"],
        "input_files": {}, "runs": {},
    }
    for model in SNAPSHOT_MODELS:
        spec = manifest["runs"][model]
        files = spec["files"]
        paths = {name: _checked(root, file["path"], file["sha256"]) for name, file in files.items()}
        metrics = json.loads(paths["metrics.json"].read_text())
        historical_hashes = metrics["provenance"]["sha256"]
        for name, source in manifest["scoring_sources"].items():
            _relative(root, source["path"])
            if historical_hashes[name] != source["sha256"]:
                raise SnapshotError(f"{model}: scoring-source hash differs between metadata and manifest")
        if metrics["primary_view"] != manifest["scoring_view"]:
            raise SnapshotError(f"{model}: primary scoring view differs between metadata and manifest")
        if historical_hashes["results"] != spec["original_results_sha256"]:
            raise SnapshotError(f"{model}: original result hash differs between metadata and manifest")
        for name, source in snapshot_inputs.items():
            if historical_hashes[name] != source["sha256"]:
                raise SnapshotError(f"{model}: input hash differs between metadata and manifest")
        input_paths = {
            name: _checked(root, source["path"], source["sha256"])
            for name, source in snapshot_inputs.items()
        }
        dataset = _index(_jsonl(input_paths["dataset"]), "financebench_id", "Dataset")
        gold = _index(_jsonl(input_paths["gold"]), "financebench_id", "Gold")
        if set(gold) != set(dataset):
            raise SnapshotError("Gold IDs differ from dataset IDs")
        with paths["scored_rows.csv"].open(newline="") as handle:
            records = list(csv.DictReader(handle))
        indexed = _index(records, "id", model)
        if set(indexed) != set(dataset):
            raise SnapshotError(f"{model}: scored IDs differ from dataset IDs")
        for rid, record in indexed.items():
            if record.get("question") != dataset[rid].get("question"):
                raise SnapshotError(f"{model}: question alignment differs for {rid}")
            if not dataset[rid].get("question"):
                raise SnapshotError(f"Dataset has missing question: {rid}")
        for view in VIEWS:
            counts = Counter(record[f"{view}_outcome"] for record in records)
            if set(counts) - set(OUTCOMES):
                raise SnapshotError(f"{model}: unknown {view} outcome")
            expected = metrics["views"][view]
            if expected["n"] != N or any(counts[label] != expected[label] for label in OUTCOMES):
                raise SnapshotError(f"{model}: {view} counts do not match pinned metrics")
            rates = _rates(counts)
            for key, value in rates.items():
                if not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-12):
                    raise SnapshotError(f"{model}: {view} {key} does not match counts")
            for record in records:
                outcome = record[f"{view}_outcome"]
                status = record["status"]
                if (outcome in {"TA", "FA"} and status != "VERIFIED") or (outcome in {"TR", "FR"} and status != "VIOLATED"):
                    raise SnapshotError(f"{model}: outcome contradicts saved status for {record['id']}")
            if view == VIEW:
                tables.append({
                    "Model": model, "N": N, **{key: counts[key] for key in OUTCOMES},
                    "Precision (%)": rates["precision"] * 100,
                    "Decision accuracy (%)": rates["decision_accuracy"] * 100,
                    "Coverage (%)": rates["coverage"] * 100,
                })
        provenance["runs"][model] = {
            "original_run": str(_relative(root, spec["original_run"]).relative_to(root)),
            "original_results_sha256": spec["original_results_sha256"],
            "original_live_results_read": False,
            "snapshot_files": files,
            "historical_scoring_hashes": historical_hashes,
            "historical_comparison": metrics["provenance"]["comparison"],
            "counts": {key: tables[-1][key] for key in ("N", *OUTCOMES)},
        }
        provenance["input_files"] = {
            name: {"path": snapshot_inputs[name]["path"], "sha256": sha256(path)}
            for name, path in input_paths.items()
        }
    return tables, provenance


def latest_run_state(
    root: Path, run_dir: Path, expected_ids: set[str], *, allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate a live log's IDs before deciding whether it can be scored.

    A missing log or valid subset is pending, never a partial benchmark score.
    Conflicting duplicates and unknown IDs remain errors during generation.
    """
    from scripts.paper_scoring import validate_run_ids
    from verifiqa.eval.tatdqa_scoring import result_status, scoring_signature

    path = Path(run_dir) / "results.jsonl"
    missing = False
    try:
        contents = path.read_bytes()
    except FileNotFoundError:
        if not allow_incomplete:
            raise
        contents = b""
        missing = True
    physical_rows = []
    partial_last_row = False
    lines = contents.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            physical_rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not allow_incomplete or index != len(lines) - 1 or contents.endswith(b"\n"):
                raise
            partial_last_row = True
    indexed = {}
    for raw in physical_rows:
        row = result_status(raw)
        rid = row.get("id")
        if rid in indexed and scoring_signature(row) != scoring_signature(indexed[rid]):
            raise ValueError(f"Conflicting duplicate result: {rid}")
        indexed[rid] = row
    ids_complete = set(indexed) == expected_ids
    complete = ids_complete and not partial_last_row
    if ids_complete or set(indexed) - expected_ids:
        validate_run_ids(list(indexed.values()), expected_ids, "tatdqa", Path(run_dir).name)
    return {
        "state": "Complete" if complete else "Pending",
        "progress": f"{len(indexed)}/{len(expected_ids)}",
        "sha256": None if missing else hashlib.sha256(contents).hexdigest(),
        "input_profile": {
            "Questions": len(expected_ids), "Physical log rows": len(physical_rows),
            "Unique received rows": len(indexed),
            "Repeated rows removed": len(physical_rows) - len(indexed),
            "Partial last row": int(partial_last_row),
        },
    }


def replacement_provenance(root: Path, result_hashes: dict[str, str]) -> dict[str, Any]:
    """Bind optional targeted-replay records to the exact complete logs scored."""
    root = Path(root).resolve()
    replacements = {}
    for model, result_hash in result_hashes.items():
        relative = f"{RUNS[model]}/result_replacements.json"
        path = root / relative
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        if record.get("after_results_sha256") != result_hash:
            raise ValueError(f"{model}: replacement provenance does not match the result log")
        replacements[model] = {
            "path": relative, "sha256": sha256(path),
            "scope": record["scope"], "question_id": record["question_id"],
            "after_results_sha256": result_hash,
        }
    return replacements


def load_latest(root: Path, *, allow_incomplete: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score each complete live log with the CLI's shared scoring policy.

    No cached CSV, summary, or archived snapshot supplies any displayed score.
    Missing or incomplete runs fail by default. Notebook callers may opt into
    pending rows with blank counts and rates; invalid or changing logs still
    fail explicitly.
    """
    from scripts.score_tatdqa import evaluate_run
    from verifiqa.eval.tatdqa_scoring import DEFAULT_SCORING_VIEW

    root = Path(root).resolve()
    paths = {model: root / relative / "results.jsonl" for model, relative in RUNS.items()}
    original_hashes = {
        model: None if allow_incomplete and not path.exists() else sha256(path)
        for model, path in paths.items()
    }
    expected_ids = set(_index(_jsonl(root / INPUTS["dataset"]), "financebench_id", "Dataset"))
    tables = []
    provenance: dict[str, Any] = {
        "dataset": "TADQA", "scoring_view": DEFAULT_SCORING_VIEW, "n_per_model": N,
        "input_mode": "latest_complete_results",
        "scope": "Score saved candidate answers and verification decisions; no model or network calls.",
        "source_reviewed_is_post_hoc": True,
        "review_scope": "Flagged benchmark mismatches; other accepted claims have not all been independently source-audited.",
        "detailed_audit_notebook": "notebooks/tatdqa_result_metrics.ipynb",
        "input_files": {}, "runs": {},
    }
    for model, relative in RUNS.items():
        state = latest_run_state(root, root / relative, expected_ids, allow_incomplete=allow_incomplete)
        if state["sha256"] != original_hashes[model] and (state["state"] == "Complete" or not allow_incomplete):
            raise ValueError(f"{model}'s result log changed during loading; rerun the notebook")
        if state["state"] == "Pending" and allow_incomplete:
            tables.append({
                "Model": model, "State": "Pending", "Progress": state["progress"],
                **{key: None for key in ("N", "VERIFIED", "VIOLATED", *OUTCOMES,
                                        "Precision (%)", "Decision accuracy (%)", "Coverage (%)")},
            })
            provenance["runs"][model] = {
                "state": "Pending", "progress": state["progress"],
                "results": {"path": f"{relative}/results.jsonl", "sha256": state["sha256"]},
                "input_profile": state["input_profile"], "counts": None,
                "reason": "Incomplete latest run; no benchmark metrics calculated.",
            }
            continue
        evaluation = evaluate_run(root, root / relative)
        recorded_hash = evaluation["provenance"]["sha256"]["results"]
        if recorded_hash != original_hashes[model] or sha256(paths[model]) != recorded_hash:
            raise ValueError(f"{model}'s result log changed during scoring; rerun the notebook")
        records = evaluation["scored_rows"]
        _index(records, "id", model)
        counts = Counter(record[f"{DEFAULT_SCORING_VIEW}_outcome"] for record in records)
        if set(counts) - set(OUTCOMES):
            raise ValueError(f"{model}: unknown scored outcome")
        accepted = counts["TA"] + counts["FA"]
        rejected = counts["TR"] + counts["FR"]
        decided = accepted + rejected
        table = {
            "Model": model, "State": "Complete", "Progress": state["progress"],
            "N": len(records), "VERIFIED": accepted, "VIOLATED": rejected,
            **{key: counts[key] for key in OUTCOMES},
            "Precision (%)": 100 * counts["TA"] / accepted if accepted else None,
            "Decision accuracy (%)": 100 * (counts["TA"] + counts["TR"]) / decided if decided else None,
            "Coverage (%)": 100 * decided / len(records),
        }
        expected = evaluation["views"][DEFAULT_SCORING_VIEW]
        if any(table[key] != expected[key] for key in OUTCOMES):
            raise ValueError(f"{model}: question counts differ from the canonical scorer")
        tables.append(table)
        provenance["runs"][model] = {
            "state": "Complete", "progress": state["progress"],
            "results": {"path": f"{relative}/results.jsonl", "sha256": recorded_hash},
            "input_profile": evaluation["input_profile"],
            "scoring_hashes": evaluation["provenance"]["sha256"],
            "comparison": evaluation["provenance"]["comparison"],
            "counts": {key: table[key] for key in ("N", "VERIFIED", "VIOLATED", *OUTCOMES)},
        }
        provenance["input_files"] = {
            name: {"path": source, "sha256": evaluation["provenance"]["sha256"][name]}
            for name, source in INPUTS.items()
        }
    if any(sha256(paths[model]) != original_hashes[model] for model in RUNS
           if provenance["runs"][model]["state"] == "Complete"):
        raise ValueError("A result log changed during scoring; rerun the notebook")
    provenance["result_replacements"] = replacement_provenance(root, {
        model: run["results"]["sha256"] for model, run in provenance["runs"].items()
        if run["state"] == "Complete"
    })
    return tables, provenance


def plot_outcomes(rows: list[dict[str, Any]]):
    """One full-denominator outcome-composition figure."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    rows = [row for row in rows if row.get("State", "Complete") == "Complete"]

    with plt.rc_context({"font.size": 11, "font.family": "DejaVu Sans"}):
        height = 3.7 + 0.45 * max(0, len(rows) - 3)
        fig, ax = plt.subplots(figsize=(10.5, height))
        left = [0.0] * len(rows)
        for label in OUTCOMES:
            values = [100 * row[label] / row["N"] for row in rows]
            ax.barh(range(len(rows)), values, left=left, height=0.52,
                    color=COLORS[label], edgecolor="white", linewidth=0.8, label=label)
            for index, value in enumerate(values):
                if value >= 5:
                    ax.text(left[index] + value / 2, index, f"{value:.1f}%", ha="center", va="center",
                            fontsize=10, color="white" if label in {"TA", "TR"} else "#1F2937")
            left = [start + value for start, value in zip(left, values)]
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row["Model"] for row in rows])
        if not rows:
            ax.text(0.5, 0.5, "Latest runs pending · no complete benchmark scores",
                    ha="center", va="center", transform=ax.transAxes)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("Share of all 403 questions")
        ax.set_title("TADQA · latest completed runs", loc="left", weight="bold", pad=42)
        ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False)
        for name in ("top", "right", "left"):
            ax.spines[name].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.grid(axis="x", color="#E6E8EB", linewidth=0.6)
        ax.set_axisbelow(True)
        false_accepts = "; ".join(f"{row['Model']}: {row['FA']} FA" for row in rows)
        fig.text(0.02, 0.025,
                 "Post hoc source-reviewed scoring · not a full source audit\n"
                 + textwrap.fill(false_accepts, width=110),
                 fontsize=9, color="#4B5563", va="bottom")
        fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def export(root: Path, rows: list[dict[str, Any]], provenance: dict[str, Any], fig) -> dict[str, str]:
    """Write only compact presentation outputs; original artifacts stay untouched."""
    root = Path(root).resolve()
    outputs = {"table": "paper/tables/tatdqa.csv", "png": "paper/figures/tatdqa.png", "pdf": "paper/figures/tatdqa.pdf"}
    for relative in outputs.values():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    with (root / outputs["table"]).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig.savefig(root / outputs["png"], dpi=180, bbox_inches="tight")
    fig.savefig(root / outputs["pdf"], bbox_inches="tight", metadata={"Creator": "VeriFin reviewer notebook", "CreationDate": None, "ModDate": None})
    record = {**provenance, "presentation_code": {"path": "scripts/reviewer_tatdqa.py", "sha256": sha256(root / "scripts/reviewer_tatdqa.py")},
              "outputs": {name: {"path": relative, "sha256": sha256(root / relative)} for name, relative in outputs.items()}}
    provenance_path = root / "paper/provenance/tatdqa.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return {**outputs, "provenance": "paper/provenance/tatdqa.json"}
