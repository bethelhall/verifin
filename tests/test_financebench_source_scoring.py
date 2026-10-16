"""Source targets repair evaluation metadata without repairing model answers.

All model responses in this module are synthetic. Frozen quantities and statuses
come from a checksum-identified run; the tests never read a running results file.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from verifiqa.eval import financebench_scoring


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/numerical_questions.jsonl"
REVIEW = ROOT / "data/financebench_scoring_review.json"
FIXTURE = ROOT / "tests/fixtures/financebench_notebook_scoring_cases.json"
RESOLVED_MILLIONS = (
    ("03029", 1577), ("04672", 8738), ("03882", 1615.9),
    ("04417", 5409), ("07661", 381.603), ("10285", 12645),
    ("04209", 59268), ("04854", 3215.4), ("03031", 5818),
    ("04171", 302.578), ("03282", 5466.312), ("03531", 16525),
    ("04980", 4625), ("03620", 9068), ("02024", 1959), ("00705", 400),
)
SCALES = ((1e6, "USD"), (1e3, "USD thousands"), (1, "USD millions"),
          (.001, "USD billions"))


def canonical_rows():
    return {row["financebench_id"]: row for row in
            map(json.loads, DATASET.read_text().splitlines())}


def candidate(rid, amount, unit="USD millions", *, status="VERIFIED", model="any-model"):
    if not rid.startswith("financebench_id_"):
        rid = "financebench_id_" + rid
    example = canonical_rows()[rid]
    return {"id": rid, "question": example["question"], "raw_answer": example["answer"],
            "gold_answer": example["answer"], "gold_scale": "", "status": status,
            "answer": f"Synthetic candidate: {amount} {unit}",
            "claim_input_answer": f"Synthetic candidate: {amount} {unit}",
            "claimed_value": amount, "claimed_value_unit": unit,
            "llm_run": {"model": model, "mode": "synthetic_test_fixture"}, "facts": {}}


def frozen_records():
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["kind"] == "synthetic_notebook_scoring_fixture"
    assert fixture["provenance"]["canonical_dataset_sha256"] == hashlib.sha256(
        DATASET.read_bytes()).hexdigest()
    rows = []
    for case in fixture["cases"]:
        row = candidate(case["id"], case["claimed_value"], case["claimed_value_unit"],
                        status=case["status"], model="gpt-6.1-sol")
        if case.get("display_unit"):
            prose = f"Synthetic candidate: {case['claimed_value']} {case['display_unit']}"
            row["answer"] = row["claim_input_answer"] = prose
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def scorer():
    return financebench_scoring.FinanceBenchScorer.from_root(ROOT)


@pytest.mark.parametrize("rid,amount", RESOLVED_MILLIONS)
@pytest.mark.parametrize("factor,unit", SCALES)
def test_reviewed_source_target_accepts_only_equivalent_physical_quantities(
        scorer, rid, amount, factor, unit):
    row = candidate(rid, amount * factor, unit)
    original = deepcopy(row)
    scored = scorer.score_row(row)
    assert scored["candidate_correct"] is True
    assert scored["outcome"] == "TA"
    assert row == original


@pytest.mark.parametrize("rid,amount", RESOLVED_MILLIONS)
@pytest.mark.parametrize("multiplier", [-1, .001, 1.2, 1000])
def test_reviewed_targets_do_not_rescue_wrong_sign_scale_or_magnitude(
        scorer, rid, amount, multiplier):
    scored = scorer.score_row(candidate(rid, amount * multiplier))
    assert scored["candidate_correct"] is False
    assert scored["outcome"] == "FA"


@pytest.mark.parametrize("rid,amount", RESOLVED_MILLIONS)
@pytest.mark.parametrize("unit", ["EUR millions", "percent", "count millions", "USD per share"])
def test_source_corrections_preserve_currency_and_quantity_dimensions(scorer, rid, amount, unit):
    scored = scorer.score_row(candidate(rid, amount, unit))
    assert scored["candidate_correct"] is False
    assert scored["outcome"] == "FA"


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "claude-haiku-4-5", "qwen2.5-7b", "unknown-model"])
@pytest.mark.parametrize("status,outcome", [("VERIFIED", "TA"), ("VIOLATED", "FR"), ("ABSTAIN", "AB")])
def test_source_targets_are_independent_of_model_and_verifier_status(scorer, model, status, outcome):
    row = candidate("03029", 1577, model=model, status=status)
    scored = scorer.score_row(row)
    assert scored["candidate_correct"] is True
    assert scored["outcome"] == outcome


@pytest.mark.parametrize("amount", [0, 411])
@pytest.mark.parametrize("status,outcome", [("VERIFIED", "UNSCORED_VERIFIED"), ("VIOLATED", "UNSCORED_REJECTED"), ("ABSTAIN", "AB")])
def test_disputed_statement_vs_footnote_scope_is_explicitly_unscored(
        scorer, amount, status, outcome):
    scored = scorer.score_row(candidate("01328", amount, status=status))
    assert scored["candidate_correct"] is None
    assert scored["outcome"] == outcome


@pytest.mark.parametrize("amount", [-411, 999, 411000])
def test_disputed_question_does_not_exempt_arbitrary_or_wrong_scale_answers(scorer, amount):
    scored = scorer.score_row(candidate("01328", amount))
    assert scored["candidate_correct"] is False
    assert scored["outcome"] == "FA"


def test_composite_reference_uses_the_requested_total_not_any_component(scorer):
    for amount, correct in [(1959, True), (1097, False), (862, False)]:
        scored = scorer.score_row(candidate("02024", amount))
        assert scored["candidate_correct"] is correct


@pytest.mark.parametrize("invalid", ["unknown_id", "question", "raw_gold", "cached_gold"])
def test_raw_alignment_is_checked_before_using_any_cached_correctness(scorer, invalid):
    row = candidate("03029", 1577)
    row.update(answer_matches_gold=True, verified_correct=True, decision_correct=True,
               expected_solver_status="SAT", solver_correct=True)
    if invalid == "unknown_id":
        row["id"] = "financebench_id_not_in_dataset"
    elif invalid == "question":
        row["question"] += " A different question."
    elif invalid == "raw_gold":
        row["raw_answer"] = "Different reference"
    else:
        row["gold_answer"] = "Different cached reference"
    with pytest.raises(ValueError):
        scorer.score_row(row)


def test_duplicate_result_ids_are_rejected(scorer):
    row = candidate("03029", 1577)
    with pytest.raises(ValueError):
        scorer.summarize_rows([row, deepcopy(row)])


def test_stale_cached_grade_cannot_make_wrong_candidate_correct(scorer):
    row = candidate("03029", 12345)
    row.update(answer_matches_gold=True, verified_correct=True, decision_correct=True,
               expected_solver_status="SAT", solver_correct=True)
    scored = scorer.score_row(row)
    assert scored["candidate_correct"] is False
    assert scored["outcome"] == "FA"


def test_full_frozen_run_exposes_unresolved_accept_in_denominators(scorer):
    rows = frozen_records()
    original = deepcopy(rows)
    summary = scorer.summarize_rows(rows)
    assert rows == original
    assert (summary["total"], summary["verified"], summary["violated"], summary["abstain"]) == (67, 38, 8, 21)
    assert (summary["TA"], summary["FA"], summary["UNSCORED_VERIFIED"],
            summary["UNSCORED_REJECTED"]) == (37, 0, 1, 0)
    assert sum(summary[key] for key in
               ("TA", "FA", "TR", "FR", "AB", "UNSCORED_VERIFIED", "UNSCORED_REJECTED")) == 67
    assert (summary["TR"], summary["FR"]) == (2, 6)
    assert summary["precision"] == pytest.approx(37 / 37)
    assert summary["scoring"]["graded_precision_denominator"] == 37
    assert summary["scored_decisions"] == 45
    assert summary["raw_decisions"] == 46
    assert summary["decision_accuracy"] == pytest.approx((37 + summary["TR"]) / 45)
    assert summary["coverage"] == pytest.approx(46 / 67)
    assert summary["scored_coverage"] == pytest.approx(45 / 67)
    assert summary["precision_lower_bound"] == pytest.approx(37 / 38)
    assert summary["precision_upper_bound"] == 1


def test_annotation_returns_copy_without_repairing_candidate_or_protocol(scorer):
    row = candidate("03029", 12345, status="VERIFIED")
    row.update(formula="source_fact", program="saved protocol", smtlib="saved SMT",
               facts={"source_fact": {"value": 1577, "unit": "USD millions"}},
               diagnostics={"evidence": "unchanged"}, answer_matches_gold=True)
    original = deepcopy(row)
    annotated = scorer.annotate_row(row)
    assert row == original
    assert annotated is not row
    for key in ("status", "answer", "claim_input_answer", "claimed_value", "claimed_value_unit",
                "formula", "program", "smtlib", "facts", "diagnostics"):
        assert annotated[key] == original[key]
    assert annotated["answer_matches_gold"] is False
    assert annotated["financebench_scoring"]["outcome"] == "FA"


def test_frozen_fixture_is_not_a_live_run_and_has_exact_canonical_ids(scorer):
    rows = frozen_records()
    assert len(rows) == len({row["id"] for row in rows}) == 67
    assert set(scorer.dataset) == {row["id"] for row in rows}
    assert all(row["llm_run"]["mode"] == "synthetic_test_fixture" for row in rows)


@pytest.fixture
def copied_review(tmp_path):
    """Small mutable metadata with read-only hardlinks to immutable evidence."""
    import os
    review = json.loads(REVIEW.read_text())
    dataset = tmp_path / review["dataset"]["path"]
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(DATASET.read_bytes())
    destination = tmp_path / "data/financebench_scoring_review.json"
    destination.write_text(json.dumps(review))
    for record in review["records"]:
        for source in record.get("sources", []):
            target = tmp_path / source["path"]
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(ROOT / source["path"], target)
    return tmp_path, destination


@pytest.mark.parametrize("invalid", ["duplicate_review_id", "unknown_review_id", "question",
                                     "raw_gold", "question_hash", "gold_hash"])
def test_review_identity_and_hash_drift_fail_closed(copied_review, invalid):
    root, path = copied_review
    review = json.loads(path.read_text())
    record = review["records"][0]
    if invalid == "duplicate_review_id":
        review["records"].append(deepcopy(record))
    elif invalid == "unknown_review_id":
        record["id"] = "financebench_unknown"
    elif invalid == "question":
        record["question"] = "Different question"
    elif invalid == "raw_gold":
        record["raw_answer"] = "Different gold"
    elif invalid == "question_hash":
        record["question_sha256"] = "0" * 64
    else:
        record["raw_answer_sha256"] = "0" * 64
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        financebench_scoring.FinanceBenchScorer.from_root(root)


@pytest.mark.parametrize("invalid", ["hash", "duplicate_id", "missing_id"])
def test_canonical_dataset_hash_and_complete_identity_are_validated(copied_review, invalid):
    root, review_path = copied_review
    review = json.loads(review_path.read_text())
    dataset = root / review["dataset"]["path"]
    lines = dataset.read_text().splitlines()
    if invalid == "duplicate_id":
        lines[-1] = lines[0]
    elif invalid == "missing_id":
        lines.pop()
    else:
        lines.append(" ")
    dataset.write_text("\n".join(lines) + "\n")
    if invalid != "hash":
        review["dataset"]["sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
        review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        financebench_scoring.FinanceBenchScorer.from_root(root)


@pytest.mark.parametrize("invalid", ["missing_sources", "source_hash", "source_quote",
                                     "source_record", "source_path"])
def test_reviewed_source_evidence_identity_hash_and_quote_are_required(copied_review, invalid):
    root, path = copied_review
    review = json.loads(path.read_text())
    record = review["records"][0]
    source = record["sources"][0]
    if invalid == "missing_sources":
        record["sources"] = []
    elif invalid == "source_hash":
        source["field_sha256"] = "0" * 64
    elif invalid == "source_quote":
        source["quote"] = "A fabricated source excerpt that does not occur in the evidence."
    elif invalid == "source_record":
        source["record_id"] = "financebench_id_wrong_record"
    else:
        source["path"] = "data/nonexistent_proof.jsonl"
    path.write_text(json.dumps(review))
    with pytest.raises((ValueError, FileNotFoundError)):
        financebench_scoring.FinanceBenchScorer.from_root(root)


@pytest.mark.parametrize("rid,amount,unit", [
    ("05718", 389, "USD millions"), ("01491", 13200, "USD millions"),
    ("00283", 70, "USD millions"), ("01346", 3, "percentage points"),
    ("01351", -3, "percentage points"),
])
def test_reference_repairs_also_identify_correct_rejected_answers(scorer, rid, amount, unit):
    row = candidate(rid, amount, unit, status="VIOLATED")
    scored = scorer.score_row(row)
    assert scored["candidate_correct"] is True
    assert scored["outcome"] == "FR"


def test_source_targets_leave_unreviewed_scalar_or_non_scalar_questions_alone(scorer):
    # The review explicitly declines to turn the evidence numbers for these
    # questions into new scalar targets for a different requested answer type.
    for suffix in ("00394", "02119", "01936"):
        assert "financebench_id_" + suffix not in scorer.records


@pytest.mark.parametrize("amount", [0, 411])
@pytest.mark.parametrize("unit", ["EUR millions", "percent", "count millions", "USD per share"])
def test_scope_dispute_does_not_exempt_incompatible_candidate_units(scorer, amount, unit):
    scored = scorer.score_row(candidate("01328", amount, unit))
    assert scored["candidate_correct"] is False
    assert scored["outcome"] == "FA"


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "qwen2.5-7b", "unrecognized-model"])
def test_entire_frozen_run_has_identical_grades_for_every_backbone(scorer, model):
    rows = frozen_records()
    expected = scorer.score_rows(rows)
    for row in rows:
        row["llm_run"]["model"] = model
    assert scorer.score_rows(rows) == expected


def test_authority_defect_stays_visible_for_immutable_audited_snapshot(scorer):
    # The archived snapshot is checksum-bound review evidence, never a live
    # results file. Its grounding finding must not disappear when grading is fixed.
    review = json.loads(REVIEW.read_text())
    snapshot = ROOT / review["authority_audit"]["snapshot_path"]
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == review["authority_audit"]["results_sha256"]
    authority = scorer.authority_audit(snapshot)
    assert authority["status"] == "audited"
    assert authority["matching_snapshot"] is True
    assert authority["known_grounding_defects"] == 1
    assert authority["findings"][0]["id"] == "financebench_id_01328"
    # A proof failure is disclosed without pretending that either plausible
    # numeric interpretation has been established as a false financial answer.
    assert scorer.score_row(candidate("01328", 0))["outcome"] == "UNSCORED_VERIFIED"


def test_frozen_display_unit_review_preserves_stored_claim_and_never_passes_an_override():
    fixture = json.loads(FIXTURE.read_text())
    expected = {"financebench_id_04784": .2, "financebench_id_01346": 3,
                "financebench_id_01351": -3, "financebench_id_01476": 1}
    reviewed = {case["id"]: case for case in fixture["cases"] if case.get("display_unit")}
    assert set(reviewed) == set(expected)
    rows = {row["id"]: row for row in frozen_records()}
    for rid, value in expected.items():
        row = rows[rid]
        assert row["claimed_value"] == value
        assert row["claimed_value_unit"] == "percent"
        assert "display_unit" not in row
        assert "percentage points" in row["answer"]
        assert row["answer"] == row["claim_input_answer"]
    review = fixture["provenance"]["explicit_display_unit_review"]
    archived = ROOT / review["source_snapshot"]
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == review["source_sha256"]
    original = {row["id"]: row for row in map(json.loads, archived.read_text().splitlines())}
    assert all("percentage point" in original[rid]["answer"] for rid in expected)
