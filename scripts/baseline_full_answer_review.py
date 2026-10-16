"""Source-bound correctness reviews for answers without a scoreable scalar.

These annotations classify the saved answer against the canonical benchmark.
They never infer correctness from a judge's verdict and never modify source
answers, references, quantities, or decisions. Callers must explicitly opt in.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts import paper_scoring

REVIEW_PATH = "notebooks/baseline_full_answer_reviews.json"
PROTOCOL = "full_answer_correctness_review_v1"
REQUIRED_FIELDS = frozenset({
    "candidate_model", "dataset", "id", "source_path", "source_sha256", "source_row",
    "candidate_answer", "claimed_value", "question", "gold_answer", "dataset_path",
    "dataset_sha256", "correct", "rationale", "evidence",
})
OPTIONAL_FIELDS = frozenset({"kind", "reviewer", "rule", "reference_basis"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact(left: Any, right: Any) -> bool:
    """Keep JSON booleans distinct from numbers and preserve scalar types."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def _field(document: Any, path: list[Any], key: tuple[str, str, str]) -> Any:
    if not isinstance(path, list) or not path:
        raise ValueError(f"{key}: evidence field_path must be a nonempty list")
    value = document
    for part in path:
        if isinstance(value, dict) and isinstance(part, str) and part in value:
            value = value[part]
        elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
            value = value[part]
        else:
            raise ValueError(f"{key}: evidence field_path does not identify a canonical field")
    return value


def load_reviews(root: Path, *, candidate_model: str, loaded_runs: list[dict],
                 policy: dict, source_hashes: dict[str, str],
                 eligible: set[tuple[str, str]]) -> tuple[dict, dict]:
    """Validate annotations and return selected-model candidate correctness.

    ``eligible`` is computed independently from the unchanged scorer and contains
    only currently unscored, decided full-answer judge rows. The same review is
    then available to both methods, regardless of their saved decisions.
    """
    root = Path(root).resolve()
    review_path = root / REVIEW_PATH
    if not review_path.is_file():
        raise ValueError(f"Full-answer review was requested but is missing: {REVIEW_PATH}")
    digest = _sha256(review_path)
    document = paper_scoring.load_json(review_path)
    if (type(document.get("schema_version")) is not int or document["schema_version"] != 1
            or document.get("protocol") != PROTOCOL):
        raise ValueError("Unsupported full-answer review schema version or protocol")
    if document.get("reference_basis", "canonical_benchmark") != "canonical_benchmark":
        raise ValueError("Full-answer reviews must retain the canonical benchmark reference")
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError("Full-answer review records must be a list")
    selected, seen = [], set()
    for record in records:
        if not isinstance(record, dict) or not REQUIRED_FIELDS <= record.keys():
            raise ValueError("Full-answer review is missing required source-binding fields")
        if record.keys() - REQUIRED_FIELDS - OPTIONAL_FIELDS:
            raise ValueError("Full-answer review has unsupported fields (verdicts are not correctness evidence)")
        for name in ("candidate_model", "dataset", "id", "source_path", "source_sha256",
                     "candidate_answer", "question", "gold_answer", "dataset_path", "dataset_sha256",
                     "rationale"):
            if not isinstance(record[name], str) or not record[name].strip():
                raise ValueError(f"Full-answer review {name} must be a nonempty string")
        key = (record["candidate_model"], record["dataset"], record["id"])
        if key in seen:
            raise ValueError(f"Duplicate full-answer review: {key}")
        seen.add(key)
        if type(record["correct"]) is not bool:
            raise ValueError(f"{key}: reviewed correctness must be a boolean")
        if type(record["source_row"]) is not int or record["source_row"] < 1:
            raise ValueError(f"{key}: source_row must be a positive integer")
        if not isinstance(record["evidence"], list):
            raise ValueError(f"{key}: evidence must be a list")
        for evidence in record["evidence"]:
            if (not isinstance(evidence, dict) or not {"field_path", "quote"} <= evidence.keys()
                    or evidence.keys() - {"field_path", "quote", "record_id"}
                    or not isinstance(evidence["field_path"], list) or not evidence["field_path"]
                    or not isinstance(evidence["quote"], str) or not evidence["quote"]):
                raise ValueError(f"{key}: evidence requires field_path and a nonempty exact quote")
            if "record_id" in evidence and (not isinstance(evidence["record_id"], str)
                                             or not evidence["record_id"]):
                raise ValueError(f"{key}: evidence record_id must be a nonempty string")
        if record.get("reference_basis", "canonical_benchmark") != "canonical_benchmark":
            raise ValueError(f"{key}: review changes the canonical reference basis")
        if record["candidate_model"] == candidate_model:
            selected.append(record)

    verifin = {run["dataset"]: run for run in loaded_runs if run["method"] == "VeriFin"}
    candidates, references = {}, {}
    for dataset, run in verifin.items():
        candidates[dataset] = {paper_scoring.record_id(row): (position, row)
                               for position, row in enumerate(run["original_records"], 1)}
        dataset_path = policy["datasets"][dataset]["source_path"]
        references[dataset] = {paper_scoring.record_id(row): row
                               for row in paper_scoring.load_jsonl(root / dataset_path)}
    reviews = {}
    for record in selected:
        dataset, rid = record["dataset"], record["id"]
        key = (dataset, rid)
        if key not in eligible:
            raise ValueError(f"{candidate_model}/{dataset}/{rid}: review is not an unscored full-answer judge decision")
        run = verifin[dataset]
        source_row, candidate = candidates[dataset][rid]
        question = references[dataset][rid]
        dataset_path = policy["datasets"][dataset]["source_path"]
        expected = dict(
            source_path=run["result_path"], source_sha256=source_hashes[run["result_path"]],
            source_row=source_row, candidate_answer=candidate["claim_input_answer"],
            claimed_value=candidate.get("claimed_value"), question=question["question"],
            gold_answer=question["answer"], dataset_path=dataset_path,
            dataset_sha256=source_hashes[dataset_path],
        )
        for name, value in expected.items():
            if not _exact(record[name], value):
                raise ValueError(f"{candidate_model}/{dataset}/{rid}: full-answer review {name} differs from source")
        for evidence in record["evidence"]:
            evidence_rid = evidence.get("record_id", rid)
            if evidence_rid not in references[dataset]:
                raise ValueError(f"{candidate_model}/{dataset}/{rid}: unknown canonical evidence record_id")
            evidence_question = references[dataset][evidence_rid]
            value = _field(evidence_question, evidence["field_path"], (candidate_model, dataset, rid))
            if not isinstance(value, str) or evidence["quote"] not in value:
                raise ValueError(f"{candidate_model}/{dataset}/{rid}: evidence quote is absent from canonical field")
        reviews[key] = record
    if _sha256(review_path) != digest:
        raise ValueError("Full-answer review changed during loading")
    source_hashes[REVIEW_PATH] = digest
    provenance = dict(
        enabled=True, path=REVIEW_PATH, sha256=digest, protocol=PROTOCOL, schema_version=1,
        reference_basis="canonical_benchmark", candidate_model=candidate_model,
        unique_reviewed_candidates=len(reviews), eligible_unscored_judge_candidates=len(eligible),
        remaining_unreviewed_candidates=len(eligible - reviews.keys()), records=selected,
        scope="Previously unscored full-answer judge candidates; shared correctness for both methods",
        saved_decisions_preserved=True,
        document_metadata={key: value for key, value in document.items() if key != "records"},
    )
    return reviews, provenance


def outcome(correct: bool, saved_status: str) -> str:
    """Assign a confusion category without changing the saved decision."""
    if type(correct) is not bool:
        raise ValueError("Reviewed correctness must be a boolean")
    if saved_status == "VERIFIED":
        return "TA" if correct else "FA"
    if saved_status == "VIOLATED":
        return "FR" if correct else "TR"
    if saved_status == "ABSTAIN":
        return "AB"
    raise ValueError(f"Unexpected saved status: {saved_status}")
