#!/usr/bin/env python3
"""Compute TA/FA/TR/FR/AB counts from source artifacts.

"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUTCOMES = ("TA", "FA", "TR", "FR", "AB")
DECISION_STATUSES = {"VERIFIED", "VIOLATED"}


class ScoringValidationError(ValueError):
    """Raised when policy, data, or result artifacts are inconsistent."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ScoringValidationError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScoringValidationError(
                    f"Invalid JSON in {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ScoringValidationError(
                    f"Expected an object in {path}:{line_number}"
                )
            rows.append(row)
    return rows


def load_policy(path: Path) -> dict[str, Any]:
    policy = load_json(path)
    if "datasets" not in policy or "canonical_runs" not in policy:
        raise ScoringValidationError(
            f"Policy {path} must define datasets and canonical_runs"
        )
    correctness = policy.get("candidate_correctness", {})
    protocol = correctness.get("protocol")
    if protocol not in (None, "legacy_magnitude_any_token_v1", "strict_scalar_v2"):
        raise ScoringValidationError(f"Unknown candidate correctness protocol: {protocol}")
    if protocol == "strict_scalar_v2" and (
        correctness.get("compare_magnitudes") or correctness.get("match_any_gold_numeric_token")
    ):
        raise ScoringValidationError(
            "strict_scalar_v2 forbids magnitude-only and any-gold-token matching"
        )
    if any("force_correct" in key for key in correctness):
        raise ScoringValidationError(
            "ID-wide force-correct rules are forbidden; use candidate-specific "
            "allowed_targets in manual_adjudications.json"
        )
    return policy


def _manual_dataset_key(document: Mapping[str, Any]) -> str:
    explicit = document.get("dataset_key")
    if isinstance(explicit, str) and explicit:
        return explicit
    label = document.get("dataset")
    if not isinstance(label, str) or not label:
        raise ScoringValidationError(
            "Manual adjudications require a dataset or dataset_key"
        )
    return re.sub(r"-\d+$", "", label.lower())


def load_manual_adjudications(
    path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    document = load_json(path)
    rows = document.get("records", document.get("adjudications"))
    if not isinstance(rows, list):
        raise ScoringValidationError(
            f"Adjudication file {path} must contain a records list"
        )
    default_dataset = _manual_dataset_key(document)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ScoringValidationError(
                f"Adjudication {position} in {path} must be an object"
            )
        dataset = row.get("dataset", default_dataset)
        record_id_value = row.get("id")
        raw_targets = row.get("allowed_targets")
        if not isinstance(dataset, str) or not isinstance(record_id_value, str):
            raise ScoringValidationError(
                f"Adjudication {position} requires string dataset and id"
            )
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ScoringValidationError(
                f"Adjudication {dataset}/{record_id_value} requires allowed_targets"
            )

        targets: list[float] = []
        for raw_target in raw_targets:
            value = (
                raw_target.get("value")
                if isinstance(raw_target, dict)
                else raw_target
            )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ScoringValidationError(
                    f"Non-numeric target in adjudication "
                    f"{dataset}/{record_id_value}"
                )
            value = float(value)
            if not math.isfinite(value):
                raise ScoringValidationError(
                    f"Non-finite target in adjudication "
                    f"{dataset}/{record_id_value}"
                )
            targets.append(value)

        key = (dataset, record_id_value)
        if key in index:
            raise ScoringValidationError(
                f"Duplicate adjudication for {dataset}/{record_id_value}"
            )
        index[key] = {**row, "allowed_target_values": targets}
    return index


def record_id(row: Mapping[str, Any], id_field: str | None = None) -> str:
    value = row.get(id_field) if id_field else None
    if value is None:
        value = row.get("id") or row.get("financebench_id")
    if not isinstance(value, str) or not value:
        raise ScoringValidationError("A row is missing its benchmark ID")
    return value


def numeric_targets(text: Any, pattern: str) -> list[float]:
    targets: list[float] = []
    for token in re.findall(pattern, str(text or "")):
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(value):
            targets.append(value)
    return targets


def _matches_target(
    candidate: float,
    target: float,
    *,
    compare_magnitudes: bool,
    absolute_floor: float = 1.0,
    relative_tolerance: float = 0.01,
) -> bool:
    if compare_magnitudes:
        candidate, target = abs(candidate), abs(target)
    tolerance = max(absolute_floor, relative_tolerance * abs(target))
    return abs(candidate - target) <= tolerance


def candidate_is_correct(
    row: Mapping[str, Any],
    dataset_key: str,
    policy: Mapping[str, Any],
    adjudications: Mapping[tuple[str, str], Mapping[str, Any]],
) -> bool | None:
    correctness = policy["candidate_correctness"]
    if correctness.get("protocol") == "strict_scalar_v2":
        return _strict_candidate_is_correct(row, dataset_key, adjudications)
    if (row.get("candidate_protocol") == "claim_input_answer_v2"
            and not correctness.get("allow_corrected_candidate_protocol", False)):
        raise ScoringValidationError(
            "Corrected candidate v2 results require candidate_correctness.protocol="
            "strict_scalar_v2; the historical magnitude/any-token policy is not valid."
        )
    claimed_field = correctness.get("claimed_value_field", "claimed_value")
    claimed_value = row.get(claimed_field)
    if claimed_value is None:
        return None
    if isinstance(claimed_value, bool) or not isinstance(
        claimed_value, (int, float)
    ):
        raise ScoringValidationError(
            f"Non-numeric {claimed_field} for {record_id(row)}"
        )
    candidate = float(claimed_value)
    if not math.isfinite(candidate):
        raise ScoringValidationError(
            f"Non-finite {claimed_field} for {record_id(row)}"
        )

    rid = record_id(row)
    manual = adjudications.get((dataset_key, rid))
    if manual is not None:
        targets = list(manual["allowed_target_values"])
    else:
        gold_text = None
        for field in correctness.get(
            "gold_text_fields_in_priority_order", ["gold_answer", "answer"]
        ):
            if row.get(field) not in (None, ""):
                gold_text = row[field]
                break
        targets = numeric_targets(
            gold_text,
            correctness.get("numeric_token_pattern", r"-?\d[\d,]*\.?\d*"),
        )
    if not targets:
        return False

    return any(
        _matches_target(
            candidate,
            target,
            compare_magnitudes=bool(correctness.get("compare_magnitudes", True)),
        )
        for target in targets
    )


def _strict_candidate_is_correct(
    row: Mapping[str, Any],
    dataset_key: str,
    adjudications: Mapping[tuple[str, str], Mapping[str, Any]],
) -> bool | None:
    """Score explicit quantities, without silently migrating historical labels."""
    from verifiqa.eval.baseline_scoring import (
        normalize_candidate, parse_quantity, quantities_match, requested_unit,
    )
    from verifiqa.eval.candidate_scoring import unit_info

    rid = record_id(row)
    baseline = "baseline" in row or "candidate_protocol" in row
    if baseline and row.get("candidate_protocol") != "claim_input_answer_v2":
        raise ScoringValidationError(
            f"{rid}: legacy baseline verdict cannot be scored under the corrected protocol; rerun it"
        )
    quantity = normalize_candidate(row)
    if quantity["status"] != "resolved":
        return None
    if baseline:
        if row.get("candidate_quantity_status") != "resolved":
            return None
        for key, expected in (("candidate_value", quantity["value"]),
                              ("candidate_unit", quantity["unit"]),
                              ("claimed_value_unit", quantity["stored_unit"]),
                              ("absolute_tolerance", quantity["tolerance"])):
            if row.get(key) != expected:
                raise ScoringValidationError(f"{rid}: inconsistent corrected candidate metadata: {key}")
    manual = adjudications.get((dataset_key, rid))
    if manual is not None:
        if manual.get("scoring_protocol") != "strict_scalar_v2":
            raise ScoringValidationError(
                f"{rid}: historical adjudication requires review under strict_scalar_v2"
            )
        targets = manual.get("allowed_targets")
        if not isinstance(targets, list) or not targets:
            raise ScoringValidationError(f"{rid}: missing reviewed quantity targets")
        gold = []
        for target in targets:
            if not isinstance(target, Mapping) or not target.get("unit"):
                raise ScoringValidationError(f"{rid}: reviewed targets require explicit value and unit")
            scalar = parse_quantity(f"{target.get('value')} {target['unit']}")
            if scalar is None:
                raise ScoringValidationError(f"{rid}: invalid reviewed quantity target")
            gold.append(scalar)
    else:
        gold_text = row.get("gold_answer", row.get("answer"))
        # Gold units can use the question or explicit benchmark metadata, never
        # the candidate magnitude, candidate unit, or operand values.
        unit = row.get("gold_unit") or requested_unit(row.get("question"))
        scalar = parse_quantity(gold_text, unit)
        if scalar is None or not scalar["unit"]:
            raise ScoringValidationError(
                f"{rid}: gold answer is not an explicit scalar quantity; add a reviewed target"
            )
        requested = unit_info(unit)
        if (scalar["family"] == requested["family"] == "money"
                and scalar["currency"] == requested["currency"]
                and scalar["scale"] == 1 and requested["scale"] != 1):
            if row.get("gold_unit"):
                scalar = {**scalar, **requested}
            else:
                raise ScoringValidationError(
                    f"{rid}: gold currency omits the scale requested by the question; "
                    "declare gold_unit or add a reviewed target"
                )
        gold = [scalar]
    matches = [quantities_match(quantity["value"], quantity["unit"],
                                target["value"], target["unit"], quantity["tolerance"])
               for target in gold]
    if any(match is None for match in matches):
        raise ScoringValidationError(f"{rid}: unresolved gold/candidate units")
    return any(matches)


def validate_dataset_ids(
    rows: Sequence[Mapping[str, Any]],
    dataset_key: str,
    dataset_spec: Mapping[str, Any],
) -> set[str]:
    denominator = dataset_spec.get("denominator")
    if not isinstance(denominator, int) or denominator <= 0:
        raise ScoringValidationError(
            f"Dataset {dataset_key} has an invalid denominator"
        )
    ids = [record_id(row, dataset_spec.get("id_field")) for row in rows]
    duplicates = [rid for rid, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ScoringValidationError(
            f"Dataset {dataset_key} has {len(duplicates)} duplicate IDs"
        )
    if len(ids) != denominator:
        raise ScoringValidationError(
            f"Dataset {dataset_key} has {len(ids)} rows; expected {denominator}"
        )
    return set(ids)


def validate_run_ids(
    rows: Sequence[Mapping[str, Any]],
    expected_ids: set[str],
    dataset_key: str,
    run_label: str,
) -> None:
    ids = [record_id(row) for row in rows]
    counts = Counter(ids)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    actual_ids = set(ids)
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if duplicates or missing or extra or len(rows) != len(expected_ids):
        raise ScoringValidationError(
            f"Run {dataset_key}/{run_label} violates the benchmark denominator: "
            f"rows={len(rows)}, unique={len(actual_ids)}, duplicates={duplicates}, "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"expected={len(expected_ids)}"
        )


def score_rows(
    rows: Iterable[Mapping[str, Any]],
    dataset_key: str,
    policy: Mapping[str, Any],
    adjudications: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, int]:
    counts = {name: 0 for name in OUTCOMES}
    claimed_field = policy["candidate_correctness"].get(
        "claimed_value_field", "claimed_value"
    )
    for row in rows:
        status = row.get("status")
        strict = policy["candidate_correctness"].get("protocol") == "strict_scalar_v2"
        if strict and ("baseline" in row or "candidate_protocol" in row):
            if row.get("candidate_protocol") != "claim_input_answer_v2":
                raise ScoringValidationError(
                    f"{record_id(row)}: legacy baseline verdict requires a fresh corrected run"
                )
        elif (not strict and row.get("candidate_protocol") == "claim_input_answer_v2"
              and not policy["candidate_correctness"].get("allow_corrected_candidate_protocol", False)):
            raise ScoringValidationError(
                "Corrected candidate v2 results require strict_scalar_v2 scoring"
            )
        if row.get(claimed_field) is None or status not in DECISION_STATUSES:
            if strict and status in DECISION_STATUSES:
                raise ScoringValidationError(
                    f"{record_id(row)}: decision with no numeric candidate"
                )
            counts["AB"] += 1
            continue
        correct = candidate_is_correct(row, dataset_key, policy, adjudications)
        if strict and correct is None:
            raise ScoringValidationError(
                f"{record_id(row)}: decision on an unresolved candidate; review it or abstain"
            )
        if status == "VERIFIED":
            counts["TA" if correct else "FA"] += 1
        else:
            counts["FR" if correct else "TR"] += 1
    return counts


def score_all(
    repo_root: Path,
    policy: Mapping[str, Any],
    adjudications: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    allow_legacy_baselines: bool = False,
) -> list[dict[str, Any]]:
    expected_by_dataset: dict[str, set[str]] = {}
    for dataset_key, dataset_spec in policy["datasets"].items():
        dataset_rows = load_jsonl(repo_root / dataset_spec["source_path"])
        expected_by_dataset[dataset_key] = validate_dataset_ids(
            dataset_rows, dataset_key, dataset_spec
        )

    output: list[dict[str, Any]] = []
    for dataset_key, dataset_runs in policy["canonical_runs"].items():
        expected_ids = expected_by_dataset[dataset_key]
        for group in ("answer_models", "baselines", "ablations"):
            for run_name, run_spec in dataset_runs.get(group, {}).items():
                if run_spec.get("status") == "missing":
                    output.append(
                        {
                            "dataset": dataset_key,
                            "group": group,
                            "run": run_name,
                            "status": "missing",
                            **{name: None for name in OUTCOMES},
                        }
                    )
                    continue
                rows = load_jsonl(
                    repo_root / run_spec["source_path"] / "results.jsonl"
                )
                if (group == "baselines"
                        and policy["candidate_correctness"].get("protocol") != "strict_scalar_v2"
                        and not allow_legacy_baselines):
                    raise ScoringValidationError(
                        "Historical baseline comparisons are invalid because candidate text and "
                        "scored values can differ. Rerun under claim_input_answer_v2 and use "
                        "strict_scalar_v2 scoring. For historical reproduction only, explicitly "
                        "set allow_legacy_baselines=True or --allow-legacy-baselines."
                    )
                validate_run_ids(rows, expected_ids, dataset_key, run_name)
                counts = score_rows(rows, dataset_key, policy, adjudications)
                declared = run_spec.get("counts")
                if declared is not None and counts != declared:
                    raise ScoringValidationError(
                        f"Computed counts for {dataset_key}/{run_name} are "
                        f"{counts}, but the manifest declares {declared}"
                    )
                output.append(
                    {
                        "dataset": dataset_key,
                        "group": group,
                        "run": run_name,
                        "status": "available",
                        "scoring_protocol": policy["candidate_correctness"].get(
                            "protocol", "legacy_magnitude_any_token_v1"
                        ),
                        "comparison_valid": not (
                            group == "baselines"
                            and policy["candidate_correctness"].get("protocol") != "strict_scalar_v2"
                        ),
                        **counts,
                    }
                )
    return output


def _print_tsv(results: Sequence[Mapping[str, Any]]) -> None:
    columns = ("dataset", "group", "run", *OUTCOMES)
    print("\t".join(columns))
    for result in results:
        print("\t".join(str(result.get(column, "")) for column in columns))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--format", choices=("tsv", "json"), default="tsv")
    parser.add_argument(
        "--allow-legacy-baselines", action="store_true",
        help="Reproduce invalid historical baseline counts for auditing only; not comparison evidence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.policy is not None:
        policy_path = args.policy
    else:
        source_policy = (
            repo_root / "data/scoring/scoring_policy.json"
        )
        published_policy = repo_root / "results/scoring_policy.json"
        policy_path = source_policy if source_policy.is_file() else published_policy
    policy = load_policy(policy_path)
    if policy["candidate_correctness"].get("protocol") != "strict_scalar_v2":
        print(
            "WARNING: reproducing historical magnitude/any-token scores. Legacy baseline "
            "verdicts have a known candidate mismatch and are not valid comparison evidence. "
            "Corrected runs require a reviewed strict_scalar_v2 policy and fresh verdicts.",
            file=sys.stderr,
        )
    manual_name = policy["candidate_correctness"].get(
        "manual_adjudications", "manual_adjudications.json"
    )
    manual_path = args.adjudications or (policy_path.parent / manual_name)
    adjudications = load_manual_adjudications(manual_path)
    results = score_all(repo_root, policy, adjudications,
                        allow_legacy_baselines=args.allow_legacy_baselines)
    if args.format == "json":
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        _print_tsv(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
