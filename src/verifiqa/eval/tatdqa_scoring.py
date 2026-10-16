"""Shared, evaluation-only TADQA scoring for the CLI and review notebook.

Original annotations, reviewed unit conversions, and post hoc source targets
are retained as separate views. These inputs never participate in verification.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from verifiqa.eval.candidate_scoring import numeric_claim_comparison

SCORING_VIEWS = ("literal_benchmark", "benchmark", "source_reviewed")
DEFAULT_SCORING_VIEW = "source_reviewed"


def numeric_targets(text: Any, pattern: str) -> list[float]:
    values = []
    for token in re.findall(pattern, str(text or "")):
        value = float(token.replace(",", ""))
        if math.isfinite(value):
            values.append(value)
    return values


def result_status(row: dict) -> dict:
    row = deepcopy(row)
    if row.get("status") == "UNVERIFIED_FORMULA":
        row["status"] = "ABSTAIN"
        row["diagnostics"] = row.get("diagnostics") or {}
        row["diagnostics"]["abstention_category"] = "unverified_formula"
        row["failure_reason"] = row.get("failure_reason") or "formula_unverified"
    return row


def scoring_signature(row: dict) -> dict:
    """Only collapse repeats that agree on every input used by this evaluator."""
    return {key: row.get(key) for key in (
        "status", "claimed_value", "question", "gold_answer", "gold_scale",
        "answer", "claim_input_answer", "facts", "diagnostics",
        "claimed_value_unit", "claimed_unit", "claim_unit", "gold_unit",
        "claim", "verification_ir",
    )}


def outcome(row: dict, correct: bool | None) -> str:
    if correct is None or row.get("status") not in {"VERIFIED", "VIOLATED"}:
        return "AB"
    if row["status"] == "VERIFIED":
        return "TA" if correct else "FA"
    return "FR" if correct else "TR"


def metric_values(counts: dict[str, int]) -> dict:
    n = sum(counts.values())
    accepted = counts["TA"] + counts["FA"]
    rejected = counts["TR"] + counts["FR"]
    decided = accepted + rejected
    return {
        "n": n, **counts,
        "precision": counts["TA"] / accepted if accepted else None,
        "decision_accuracy": (counts["TA"] + counts["TR"]) / decided if decided else None,
        "coverage": decided / n if n else None,
        "rejection_precision": counts["TR"] / rejected if rejected else None,
        "abstention_rate": counts["AB"] / n if n else None,
    }


def rounded_metric(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _correct_for_targets(
    row: dict, targets: list[float], policy: dict,
    absolute_tolerances: list[float | None] | None = None,
) -> bool | None:
    if row.get("claimed_value") is None:
        return None
    # Empty targets deliberately mean no compatible quantity. No gold-token fallback.
    candidate = row.get("claimed_value")
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return None
    try:
        candidate = float(candidate)
    except OverflowError:
        return None
    if not math.isfinite(candidate):
        return None
    magnitudes = bool(policy["candidate_correctness"].get("compare_magnitudes", True))
    if absolute_tolerances is not None and len(absolute_tolerances) != len(targets):
        raise ValueError("Each target requires its own tolerance")
    for index, target in enumerate(targets):
        value = abs(candidate) if magnitudes else candidate
        comparable_target = abs(target) if magnitudes else target
        absolute = absolute_tolerances[index] if absolute_tolerances is not None else None
        if abs(value - comparable_target) <= max(1.0 if absolute is None else absolute,
                                                 0.01 * abs(comparable_target)):
            return True
    return False


def _index_records(records: list[dict], name: str) -> dict[str, dict]:
    index = {}
    for record in records:
        rid = record["id"]
        if rid in index:
            raise ValueError(f"Duplicate {name} record: {rid}")
        index[rid] = record
    return index


def score_tatdqa_rows(
    rows: list[dict], gold_by_id: dict[str, dict], policy: dict,
    reviewed_units: dict[str, dict], annotations: dict[str, dict],
) -> dict[str, Any]:
    """Use source-defined numeric targets; an annotation never forces correctness."""
    evaluated = []
    counts = {view: {key: 0 for key in ("TA", "FA", "TR", "FR", "AB")}
              for view in SCORING_VIEWS}
    pattern = policy["candidate_correctness"]["numeric_token_pattern"]
    for row in rows:
        row = result_status(row)
        rid = row["id"]
        gold = gold_by_id[rid]
        raw_gold = gold["raw_answer"]
        literal_targets = numeric_targets(raw_gold, pattern)
        comparison = numeric_claim_comparison(
            row, raw_gold, gold_scale=gold.get("scale", ""),
            reviewed_conversion=reviewed_units.get(rid),
        )
        normalized_targets = comparison["targets"] if comparison["units_compatible"] else []
        normalized_tolerances = [comparison["absolute_tolerance_in_claim_units"]] * len(normalized_targets)
        reviewed_targets = normalized_targets
        reviewed_tolerances = normalized_tolerances
        review = annotations.get(rid)
        if review:
            if str(raw_gold) != review["raw_answer"]:
                raise ValueError(f"Gold drift in source adjudication: {rid}")
            reviewed_targets = []
            reviewed_tolerances = []
            for target in review["targets"]:
                target_conversion = None
                if "absolute_tolerance" in target or "claim_unit_qualifier" in target:
                    target_conversion = {
                        "factor": 1.0, "comparison_unit": target["unit"],
                        **{key: target[key] for key in (
                            "absolute_tolerance", "tolerance_unit", "claim_unit_qualifier",
                        ) if key in target},
                    }
                typed_target = numeric_claim_comparison(
                    row, str(target["value"]), gold_scale=target["unit"],
                    reviewed_conversion=target_conversion,
                )
                if typed_target["units_compatible"]:
                    reviewed_targets.extend(typed_target["targets"])
                    reviewed_tolerances.extend(
                        [typed_target["absolute_tolerance_in_claim_units"]] * len(typed_target["targets"]))
        targets_by_view = {
            "literal_benchmark": literal_targets,
            "benchmark": normalized_targets,
            "source_reviewed": reviewed_targets,
        }
        record = {
            "id": rid, "question": row["question"], "status": row["status"],
            "answer": row.get("answer", ""), "claimed_value": row.get("claimed_value"),
            "gold_answer": gold["answer"], "gold_raw_answer": raw_gold,
            "gold_scale": gold.get("scale", ""),
            "claimed_value_unit": comparison["claim_unit"],
            "claim_unit_source": comparison["claim_unit_source"],
            "declared_claim_unit": comparison["declared_claim_unit"],
            "gold_target_unit": comparison["gold_unit"],
            "units_compatible": comparison["units_compatible"],
            "gold_conversion_factor": reviewed_units.get(rid, {}).get("factor"),
            "normalization_reason": comparison["conversion_reason"],
            "normalized_gold_targets": normalized_targets,
            "source_reviewed_targets": reviewed_targets,
            "normalized_absolute_tolerances": normalized_tolerances,
            "source_reviewed_absolute_tolerances": reviewed_tolerances,
            "source_review_classification": review.get("classification", "") if review else "",
            "source_review_reason": review.get("reason", "") if review else "",
            "failure_reason": row.get("failure_reason", ""),
            "abstention_category": (row.get("diagnostics") or {}).get("abstention_category", ""),
        }
        for view, targets in targets_by_view.items():
            tolerances = {"benchmark": normalized_tolerances,
                          "source_reviewed": reviewed_tolerances}.get(view)
            correct = _correct_for_targets(row, targets, policy, tolerances)
            label = outcome(row, correct)
            counts[view][label] += 1
            record[f"{view}_candidate_correct"] = correct
            record[f"{view}_outcome"] = label
        # Backward-compatible CSV names refer to the unit-normalized benchmark view.
        record["candidate_correct"] = record["benchmark_candidate_correct"]
        record["outcome"] = record["benchmark_outcome"]
        evaluated.append(record)
    return {
        "views": {view: metric_values(value) for view, value in counts.items()},
        "scored_rows": evaluated,
    }


def validate_annotation_sources(repo_root: Path, entry: dict) -> None:
    for source in entry.get("sources", []):
        path = repo_root / source["path"]
        if not path.is_file():
            raise ValueError(f"Missing reviewed source: {source['path']}")
        expected_hash = source.get("sha256")
        if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Reviewed source changed: {source['path']}")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object in {path}:{number}")
        rows.append(row)
    return rows


def _index_dataset(rows: list[dict], name: str) -> dict[str, dict]:
    index = {}
    for row in rows:
        rid = row.get("financebench_id")
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"Missing question ID in {name}")
        if rid in index:
            raise ValueError(f"Duplicate question ID in {name}: {rid}")
        index[rid] = row
    return index


@dataclass
class TatdqaScoringContext:
    """Validated evaluator inputs, loaded once for a complete or partial run.

    The canonical handoff has 403 questions. ``score_rows`` also accepts a
    subset, so an interrupted or limited CLI run reports its observed results.
    The offline notebook separately requires a complete benchmark run.
    """

    repo_root: Path
    handoff_dir: Path
    dataset_rows: list[dict]
    gold_by_id: dict[str, dict]
    reviewed_units: dict[str, dict]
    annotations: dict[str, dict]
    policy: dict
    input_paths: dict[str, Path]
    provenance: dict[str, Any]

    @property
    def expected_ids(self) -> set[str]:
        return set(self.gold_by_id)

    @classmethod
    def from_dataset(
        cls, data_path: Path, repo_root: Path,
        policy_path: Path | None = None,
    ) -> TatdqaScoringContext | None:
        """Discover a TADQA handoff; return None for other dataset layouts."""
        data_path = Path(data_path).resolve()
        handoff = next((parent for parent in data_path.parents
                        if (parent / "evaluator_only/tatdqa_arithmetic_pilot_gold.jsonl").is_file()), None)
        if handoff is None:
            return None
        return cls.load(data_path, handoff, repo_root, policy_path)

    @classmethod
    def load(
        cls, data_path: Path, handoff_dir: Path, repo_root: Path,
        policy_path: Path | None = None,
    ) -> TatdqaScoringContext:
        repo_root = Path(repo_root).resolve()
        handoff_dir = Path(handoff_dir).resolve()
        if policy_path is None:
            policy_path = repo_root / "results/scoring_policy.json"
            if not policy_path.is_file():
                policy_path = repo_root / "publication_assets_source/results/scoring_policy.json"
        paths = {
            "dataset": Path(data_path).resolve(),
            "gold": handoff_dir / "evaluator_only/tatdqa_arithmetic_pilot_gold.jsonl",
            "policy": Path(policy_path).resolve(),
            "unit_conversions": repo_root / "notebooks/tatdqa_scoring_normalizations.json",
            "source_adjudications": repo_root / "notebooks/tatdqa_scoring_adjudications.json",
            "quantity_normalization": Path(__file__).with_name("candidate_scoring.py"),
            "tatdqa_scoring_core": Path(__file__).resolve(),
        }
        policy = json.loads(paths["policy"].read_text())
        correctness = policy.get("candidate_correctness")
        if not isinstance(correctness, dict) or "numeric_token_pattern" not in correctness:
            raise ValueError("TADQA scoring policy requires numeric candidate correctness rules")
        if any("force_correct" in key for key in correctness):
            raise ValueError("ID-wide force-correct rules are forbidden")
        dataset_rows = _load_jsonl(paths["dataset"])
        dataset_by_id = _index_dataset(dataset_rows, "dataset")
        gold_by_id = _index_dataset(_load_jsonl(paths["gold"]), "gold")
        if len(dataset_by_id) != 403 or set(dataset_by_id) != set(gold_by_id):
            raise ValueError("TADQA requires 403 matching dataset and gold question IDs")
        for rid, example in dataset_by_id.items():
            if example.get("answer") != gold_by_id[rid].get("answer"):
                raise ValueError(f"Question/gold drift: {rid}")
        reviewed_units = _index_records(
            json.loads(paths["unit_conversions"].read_text())["records"], "unit conversion")
        annotations = _index_records(
            json.loads(paths["source_adjudications"].read_text())["records"], "source adjudication")
        if (set(reviewed_units) | set(annotations)) - set(gold_by_id):
            raise ValueError("Evaluator annotation contains unknown question IDs")
        for entry in [*reviewed_units.values(), *annotations.values()]:
            rid = entry["id"]
            if str(gold_by_id[rid]["raw_answer"]) != entry["raw_answer"]:
                raise ValueError(f"Gold drift in evaluator annotation: {rid}")
            if entry.get("question", dataset_by_id[rid]["question"]) != dataset_by_id[rid]["question"]:
                raise ValueError(f"Question drift in evaluator annotation: {rid}")
            validate_annotation_sources(repo_root, entry)
        provenance = {
            "scorer": "verifiqa.eval.tatdqa_scoring",
            "comparison": correctness.get("match_rule", "max(1.0, 0.01 * abs(target))"),
            "manual_correctness_overrides": False,
            "annotation_rule": "Match source-defined numeric targets in compatible units; never force correctness by question ID.",
            "reviewed_tolerance_rule": "An explicit reviewed absolute tolerance is expressed in its independently documented source unit and converted with the target; unrelated records retain the published numeric-unit policy.",
            "primary_view": DEFAULT_SCORING_VIEW,
            "source_reviewed_is_post_hoc": True,
            "review_scope": "Flagged benchmark mismatches; other accepted claims have not all been independently source-audited.",
            "sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
        }
        return cls(repo_root, handoff_dir, dataset_rows, gold_by_id, reviewed_units,
                   annotations, policy, paths, provenance)

    def score_row(self, row: dict) -> dict:
        return self.score_rows([row])["scored_rows"][0]

    def score_rows(self, rows: list[dict]) -> dict[str, Any]:
        examples = {row["financebench_id"]: row for row in self.dataset_rows}
        seen = set()
        for row in rows:
            rid = row.get("id")
            if rid not in self.gold_by_id:
                raise ValueError(f"Unknown TADQA question ID: {rid}")
            if rid in seen:
                raise ValueError(f"Duplicate TADQA result: {rid}")
            seen.add(rid)
            display_gold = row.get("gold_answer", row.get("raw_answer"))
            if (row.get("question") != examples[rid]["question"]
                    or display_gold != self.gold_by_id[rid]["answer"]):
                raise ValueError(f"Question/gold drift: {rid}")
        return score_tatdqa_rows(rows, self.gold_by_id, self.policy,
                                self.reviewed_units, self.annotations)


def summary_for_view(
    evaluated: dict[str, Any], primary_view: str = DEFAULT_SCORING_VIEW,
    existing_summary: dict[str, Any] | None = None, elapsed_s: float | None = None,
) -> dict[str, Any]:
    """Map the shared per-question evaluation into CLI summary field names."""
    if primary_view not in SCORING_VIEWS:
        raise ValueError(f"Unknown TADQA scoring view: {primary_view}")
    primary = evaluated["views"][primary_view]
    ta, fa, tr, fr = (primary[key] for key in ("TA", "FA", "TR", "FR"))
    decided, correct, incorrect = ta + fa + tr + fr, ta + tr, fa + fr
    correctness = [record[f"{primary_view}_candidate_correct"] for record in evaluated["scored_rows"]]
    provenance = evaluated.get("provenance", {})
    summary = {
        **(existing_summary or {}),
        "total": primary["n"], "gold_evaluable": primary["n"],
        "all_verifications": decided, "solver_verdicts": decided,
        "correct_verifications": correct, "correct_solver_verdicts": correct,
        "incorrect_verifications": incorrect, "incorrect_solver_verdicts": incorrect,
        "expected_sat": sum(value is True for value in correctness),
        "expected_unsat": sum(value is False for value in correctness),
        "actual_sat": ta + fa, "actual_unsat": tr + fr,
        "verified": ta + fa, "violated": tr + fr, "abstain": primary["AB"],
        "abstain_reasons": evaluated.get("abstention_reasons", {}),
        "verified_correct": ta, "verified_incorrect": fa,
        "violated_correct": tr, "violated_incorrect": fr,
        "correct_decisions": correct, "incorrect_decisions": incorrect,
        "accuracy": rounded_metric(primary["decision_accuracy"]),
        "solver_accuracy": rounded_metric(primary["decision_accuracy"]),
        "decision_accuracy": rounded_metric(primary["decision_accuracy"]),
        "coverage": rounded_metric(primary["coverage"]),
        "overall_accuracy": round(correct / primary["n"], 4) if primary["n"] else None,
        "sat_precision": rounded_metric(primary["precision"]),
        "verification_precision": rounded_metric(primary["precision"]),
        "unsat_precision": rounded_metric(primary["rejection_precision"]),
        "violation_precision": rounded_metric(primary["rejection_precision"]),
        "scoring": {
            "primary_view": primary_view,
            "scorer": "verifiqa.eval.tatdqa_scoring",
            "source_reviewed_is_post_hoc": primary_view == "source_reviewed",
            "result_sha256": provenance.get("sha256", {}).get("results"),
        },
        "scoring_views": evaluated["views"],
    }
    if elapsed_s is not None:
        summary["elapsed_s"] = elapsed_s
    return summary


