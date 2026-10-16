"""Offline TADQA scoring with preserved benchmark and source-reviewed views."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.paper_scoring import (
    load_jsonl, score_rows, validate_run_ids,
)
from scripts.submission_runs import RUNS as SUBMISSION_RUNS
from verifiqa.eval.tatdqa_scoring import (
    DEFAULT_SCORING_VIEW, SCORING_VIEWS, TatdqaScoringContext,
    result_status, scoring_signature, score_tatdqa_rows,
    validate_annotation_sources, summary_for_view,
)


def evaluate_run(
    repo_root: Path = ROOT, run_dir: Path | None = None,
    handoff_dir: Path | None = None, policy_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    run_dir = (run_dir or repo_root / SUBMISSION_RUNS["tatdqa"]["claude-haiku-4.5"]).resolve()
    handoff_dir = (handoff_dir or repo_root / "data/tatdqa_403_question_handoff_v1").resolve()
    context = TatdqaScoringContext.load(
        handoff_dir / "dataset/financebench_xbrl_augmented.jsonl",
        handoff_dir, repo_root, policy_path,
    )
    paths = {
        **context.input_paths,
        "results": run_dir / "results.jsonl",
        "published_scorer": repo_root / "scripts/paper_scoring.py",
        "tatdqa_scorer": Path(__file__).resolve(),
    }
    expected_ids = context.expected_ids
    dataset_rows = context.dataset_rows
    gold_by_id = context.gold_by_id
    policy = context.policy
    physical_rows = load_jsonl(paths["results"])
    rows_by_id = {}
    for raw_row in physical_rows:
        row = result_status(raw_row)
        rid = row["id"]
        if rid in rows_by_id and scoring_signature(row) != scoring_signature(rows_by_id[rid]):
            raise ValueError(f"Conflicting duplicate result: {rid}")
        rows_by_id[rid] = row
    validate_run_ids(list(rows_by_id.values()), expected_ids, "tatdqa", run_dir.name)
    rows = [rows_by_id[entry["financebench_id"]] for entry in dataset_rows]
    evaluated = context.score_rows(rows)
    literal_rows = [{**row, "gold_answer": gold_by_id[row["id"]]["raw_answer"]}
                    for row in rows]
    if score_rows(literal_rows, "tatdqa", policy, {}) != {
        key: evaluated["views"]["literal_benchmark"][key]
        for key in ("TA", "FA", "TR", "FR", "AB")
    }:
        raise AssertionError("Literal benchmark view differs from published scorer")
    failure_counts = Counter()
    for row in rows:
        if row["status"] == "ABSTAIN":
            reason = row.get("failure_reason", "")
            category = (row.get("diagnostics") or {}).get("abstention_category")
            failure_counts["Anthropic credit balance exhausted" if "credit balance is too low" in reason
                           else category or reason.split(":")[0] or "unspecified"] += 1
    evaluated.update({
        "run_dir": str(run_dir), "rows": rows, "gold_by_id": gold_by_id,
        "reviewed_units": context.reviewed_units, "annotations": context.annotations,
        "abstention_reasons": dict(failure_counts),
        "input_profile": {"Questions": len(expected_ids), "Physical log rows": len(physical_rows),
                          "Unique scored rows": len(rows), "Repeated rows removed": len(physical_rows) - len(rows)},
        "provenance": {
            **context.provenance,
            "published_scorer": "scripts/paper_scoring.py",
            "normalizer": "verifiqa.eval.tatdqa_scoring",
            "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
        },
    })
    return evaluated


def write_outputs(evaluated: dict[str, Any], primary_view: str = DEFAULT_SCORING_VIEW) -> None:
    if primary_view not in SCORING_VIEWS:
        raise ValueError(f"Unknown TADQA scoring view: {primary_view}")
    run_dir = Path(evaluated["run_dir"])
    metrics = {
        **evaluated["views"][primary_view], "views": evaluated["views"],
        "primary_view": primary_view,
        "input_profile": evaluated["input_profile"],
        "abstention_reasons": evaluated["abstention_reasons"],
        "provenance": {**evaluated["provenance"], "primary_view": primary_view},
    }
    (run_dir / "notebook_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    records = evaluated["scored_rows"]
    for filename, selected in (
        ("notebook_scored_rows.csv", records),
        ("notebook_unit_normalizations.csv", [r for r in records if r["gold_conversion_factor"] is not None or "conversion" in r["normalization_reason"].lower()]),
    ):
        with (run_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            for record in selected:
                writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                                 for key, value in record.items()})
    # Refresh the displayed summary using the primary view, without rewriting
    # any verifier claims/statuses or their historical correctness annotations.
    summary_path = run_dir / "summary.json"
    existing_summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary = summary_for_view(evaluated, primary_view, existing_summary)
    temporary = run_dir / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(summary_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--scoring-view", choices=SCORING_VIEWS, default=DEFAULT_SCORING_VIEW,
                        help="Primary summary scoring (default: source_reviewed, post hoc)")
    args = parser.parse_args(argv)
    evaluated = evaluate_run(args.repo_root, args.run, args.handoff, args.policy)
    if not args.no_write:
        write_outputs(evaluated, args.scoring_view)
    print(json.dumps(evaluated["views"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
