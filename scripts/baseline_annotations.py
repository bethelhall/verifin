"""Validate evaluation-only reference and candidate quantities against sources.

Reference targets never depend on a candidate. Candidate quantities never depend
on a reference or a verdict. Both annotations are bound to immutable source text.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from scripts import paper_scoring
from verifiqa.eval.baseline_scoring import candidate_review_fingerprint, convert_value, normalize_candidate, parse_quantity

REFERENCE_PATH = "notebooks/baseline_reference_quantities.json"
CANDIDATE_PATH = "notebooks/baseline_candidate_quantities.json"


def reference_fingerprint(dataset: str, row: dict) -> str:
    payload = {"dataset": dataset, "id": paper_scoring.record_id(row),
               "question": row["question"], "gold_answer": row["answer"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _field(row: dict, field: str) -> Any:
    value: Any = row
    if not re.fullmatch(r"\w+(?:\[\d+\])?(?:\.\w+(?:\[\d+\])?)*", field):
        raise ValueError(f"Invalid evidence field: {field}")
    for part in field.split("."):
        match = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
        value = value[match[1]]
        if match[2] is not None:
            value = value[int(match[2])]
    return value


def load_reference_quantities(root: Path, datasets: dict, source_hashes: dict,
                             scoring_view: str = "source_reviewed") -> dict:
    if scoring_view not in {"source_reviewed", "benchmark_normalized"}:
        raise ValueError(f"Unknown baseline scoring view: {scoring_view}")
    path = root / REFERENCE_PATH
    source_hashes[REFERENCE_PATH] = hashlib.sha256(path.read_bytes()).hexdigest()
    document = paper_scoring.load_json(path)
    if document.get("schema_version") != 1 or document.get("scoring_protocol") != "strict_scalar_v2":
        raise ValueError("Invalid reference quantity protocol")
    targets = {}
    for record in document["records"]:
        key = (record["dataset"], record["id"])
        if key in targets:
            raise ValueError(f"Duplicate reference quantity: {key}")
        row = datasets[key[0]][key[1]]
        if (record["question"] != row["question"] or record["gold_answer"] != row["answer"]
                or record["reference_sha256"] != reference_fingerprint(key[0], row)):
            raise ValueError(f"Stale reference quantity: {key}")
        if not record.get("normalization_rule") or not record.get("rationale") or not record.get("evidence"):
            raise ValueError(f"Unsubstantiated reference quantity: {key}")
        if not record.get("allowed_targets"):
            raise ValueError(f"Missing reference target: {key}")
        for target in record["allowed_targets"]:
            if isinstance(target.get("value"), bool) or not isinstance(target.get("value"), (int, float)):
                raise ValueError(f"Invalid reference value: {key}")
            scalar = parse_quantity(f"{target['value']} {target.get('unit', '')}")
            if scalar is None or not scalar["unit"]:
                raise ValueError(f"Invalid reference quantity: {key}")
        for evidence in record["evidence"]:
            relative = evidence["path"]
            if relative not in source_hashes or source_hashes[relative] != evidence["sha256"]:
                raise ValueError(f"Reference evidence hash differs: {key}/{relative}")
            if evidence.get("id") != key[1]:
                raise ValueError(f"Reference evidence ID differs: {key}")
            quote = evidence["quote"]
            if not quote or quote not in _field(row, evidence["field"]):
                raise ValueError(f"Reference evidence quote differs: {key}")
        selected = copy.deepcopy(record["allowed_targets"])
        if scoring_view == "source_reviewed" and record.get("source_quantity"):
            source_quantity = record["source_quantity"]
            if not record.get("source_derivation"):
                raise ValueError(f"Missing source quantity derivation: {key}")
            if isinstance(source_quantity.get("value"), bool) or not isinstance(source_quantity.get("value"), (int, float)):
                raise ValueError(f"Invalid source reference value: {key}")
            if parse_quantity(f"{source_quantity['value']} {source_quantity.get('unit', '')}") is None:
                raise ValueError(f"Invalid source reference quantity: {key}")
            selected = [copy.deepcopy(source_quantity)]
            # Retain an original benchmark target only when it is independently
            # reproducible by rounding the source quantity in the same units.
            # This is a finite explicit target set, not a wider error tolerance.
            for benchmark in record["allowed_targets"]:
                converted = convert_value(source_quantity["value"], source_quantity["unit"], benchmark["unit"])
                if converted is None:
                    continue
                benchmark_value = Decimal(str(benchmark["value"]))
                quantum = Decimal(1).scaleb(min(0, benchmark_value.normalize().as_tuple().exponent))
                rounded = Decimal(str(converted)).quantize(quantum, rounding=ROUND_HALF_UP)
                if rounded == benchmark_value and benchmark not in selected:
                    selected.append(copy.deepcopy(benchmark))
        targets[key] = {**copy.deepcopy(record), "scoring_protocol": "strict_scalar_v2",
                        "benchmark_targets": copy.deepcopy(record["allowed_targets"]),
                        "allowed_targets": selected, "selected_view": scoring_view,
                        "selection_rule": "Source quantity plus benchmark targets reproducible by decimal rounding" if scoring_view == "source_reviewed" else "Original benchmark quantity with explicit units"}
    return targets


def load_candidate_quantities(root: Path, candidates: dict, candidate_paths: dict, source_hashes: dict) -> tuple[dict, list]:
    path = root / CANDIDATE_PATH
    source_hashes[CANDIDATE_PATH] = hashlib.sha256(path.read_bytes()).hexdigest()
    document = paper_scoring.load_json(path)
    if document.get("version") != 1 or document.get("scoring_protocol") != "strict_scalar_v2":
        raise ValueError("Invalid candidate quantity protocol")
    quantities = {}
    seen = set()
    for record in document["records"] + document["unresolved_records"]:
        key = (record["dataset"], record["id"])
        if key in seen:
            raise ValueError(f"Duplicate candidate quantity: {key}")
        seen.add(key)
        if (record["source_path"] != candidate_paths[key[0]]
                or record["source_sha256"] != source_hashes[record["source_path"]]):
            raise ValueError(f"Candidate source hash differs: {key}")
        row = candidates[key[0]][key[1]]
        review = record.get("candidate_quantity_review")
        bound = review or record
        if (bound.get("claim_input_answer") != row["claim_input_answer"].strip()
                or bound.get("claimed_value") != row.get("claimed_value")
                or bound.get("sha256") != candidate_review_fingerprint(row)):
            raise ValueError(f"Stale candidate quantity: {key}")
        if review:
            if not record.get("matched_quote") or record["matched_quote"] not in row["claim_input_answer"]:
                raise ValueError(f"Candidate quotation differs: {key}")
            evaluated = {**row, "candidate_quantity_review": review}
            if normalize_candidate(evaluated)["status"] != "resolved":
                raise ValueError(f"Candidate quantity fails unit/value checks: {key}")
            quantities[key] = copy.deepcopy(record)
        elif not record.get("reason"):
            raise ValueError(f"Unexplained unresolved candidate: {key}")
    return quantities, copy.deepcopy(document["unresolved_records"])


def scoring_row(row: dict, dataset: str, quantities: dict) -> dict:
    """Return an annotated copy; numeric claim, text, and saved verdict persist."""
    result = copy.deepcopy(row)
    result["source_status"] = row["status"]
    if result["status"] == "UNVERIFIED_FORMULA":
        result["status"] = "ABSTAIN"
    key = (dataset, paper_scoring.record_id(row))
    if key in quantities:
        review = quantities[key]["candidate_quantity_review"]
        if candidate_review_fingerprint(row) != review["sha256"]:
            raise ValueError(f"Candidate text or stored claim differs across methods: {key}")
        result["candidate_quantity_review"] = copy.deepcopy(review)
    return result
