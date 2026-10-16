"""Full-answer reviews bind independent correctness to immutable saved answers."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import baseline_full_answer_review as review
from scripts import baselines, reviewer_baselines
from scripts.submission_runs import RUNS

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5.5"
RUN_ROOT = "results/new_submissions/baselines/review_test"
NARRATIVE = "Revenue was 25 USD millions; prior-year revenue was 10 USD millions."


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(tmp_path, monkeypatch, *, candidate=NARRATIVE, claimed_value=None,
            correct=True, verdict="CORRECT", candidate_status="ABSTAIN", gold="25 USD millions",
            protocol="full_answer_one_word_v1", extra_evidence_record=False,
            missing_financebench_unit=False):
    """Produce real writer/manifests for two tiny datasets, without any API."""
    policy = json.loads((ROOT / reviewer_baselines.ORIGINAL_POLICY_PATH).read_text())
    execution = json.loads((ROOT / reviewer_baselines.EXECUTION_POLICY_PATH).read_text())
    for document in (policy, execution):
        for specification in document["datasets"].values():
            specification["denominator"] = 2 if extra_evidence_record else 1
    write_json(tmp_path / reviewer_baselines.ORIGINAL_POLICY_PATH, policy)
    write_json(tmp_path / reviewer_baselines.EXECUTION_POLICY_PATH, execution)
    write_json(tmp_path / reviewer_baselines.ORIGINAL_ADJUDICATIONS_PATH,
               {"dataset": "financebench", "records": []})
    monkeypatch.setattr(baselines, "build_llm", lambda args: SimpleNamespace(
        chat=lambda *args, **kwargs: verdict))
    annotations = []
    for dataset, spec in reviewer_baselines.DATASETS.items():
        rid = f"{dataset}_one"
        question = {"financebench_id": rid, "question": "What is revenue in USD millions? Round to the nearest million.",
                    "answer": gold, "evidence": [{"evidence_text": f"Reported revenue is {gold}."}],
                    "ground_truth": {"value": float(gold.split()[0]), "unit": "USD millions",
                                     "operands": [{"concept": "Revenue", "weight": 1,
                                                   "value_millions": float(gold.split()[0])}]}}
        if missing_financebench_unit and dataset == "financebench":
            question["question"] = "What is revenue?"
        row = dict(id=rid, question=question["question"], gold_answer=gold,
                   claim_input_answer=candidate, claimed_value=claimed_value,
                   claimed_value_unit=None if missing_financebench_unit else "USD millions",
                   status=candidate_status)
        dataset_path = policy["datasets"][dataset]["source_path"]
        candidate_path = f"{RUNS[dataset][MODEL]}/results.jsonl"
        write_json(tmp_path / dataset_path, question)
        write_json(tmp_path / candidate_path, row)
        if extra_evidence_record:
            other_question = copy.deepcopy(question)
            other_question["financebench_id"] = f"{dataset}_two"
            other_question["evidence"][0]["evidence_text"] = "An independently stored canonical evidence page."
            other_candidate = dict(row, id=f"{dataset}_two")
            with (tmp_path / dataset_path).open("a") as stream:
                stream.write(json.dumps(other_question) + "\n")
            with (tmp_path / candidate_path).open("a") as stream:
                stream.write(json.dumps(other_candidate) + "\n")
        args = SimpleNamespace(
            out=tmp_path / RUN_ROOT / f"{spec['prefix']}_judge",
            data=tmp_path / dataset_path, answers_from=tmp_path / candidate_path,
            baseline="judge", with_formula=False, cot=False, resume=False,
            ids=None, limit=0, max_tokens=2048, k=5, threshold=.5,
            llm_mode="openai", model=MODEL, judge_protocol=protocol,
            vllm_url="http://localhost:11434/v1", preflight=False, verbose=False,
        )
        baselines.run(args)
        annotations.append(dict(
            candidate_model=MODEL, dataset=dataset, id=rid, source_path=candidate_path,
            source_sha256=sha(tmp_path / candidate_path), source_row=1,
            candidate_answer=candidate, claimed_value=claimed_value, question=question["question"],
            gold_answer=gold, dataset_path=dataset_path, dataset_sha256=sha(tmp_path / dataset_path),
            correct=correct, rationale="Compare the full candidate answer to the canonical benchmark.",
            evidence=[{"field_path": ["evidence", 0, "evidence_text"], "quote": gold}],
        ))
    document = dict(schema_version=1, protocol=review.PROTOCOL, reference_basis="canonical_benchmark",
                    reviewer="AI-assisted post hoc review", records=annotations)
    write_json(tmp_path / review.REVIEW_PATH, document)
    return document


def report(root, **kwargs):
    return reviewer_baselines.load_report(root, RUN_ROOT, candidate_model=MODEL,
                                         methods=("judge",), **kwargs)


def test_correct_zero_is_false_reject_without_changing_verifin_abstention(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch,
            candidate="Revenue was 0 USD millions; prior-year revenue was 10 USD millions.",
            gold="0 USD millions", verdict="INCORRECT")
    source_paths = [path for path in tmp_path.rglob("*.jsonl")]
    before = {path: path.read_bytes() for path in source_paths}
    original = report(tmp_path)
    result = report(tmp_path, review_full_answers=True)
    judge = [row for row in result["rows"] if row["Method"] == "LLM Judge"]
    verifin = [row for row in result["rows"] if row["Method"] == "VeriFin"]
    assert all(row["FR"] == row["VIOLATED"] == 1 and row["Unscored"] == 0 for row in judge)
    assert all(row["ABSTAIN"] == 1 and row["FR"] == row["FA"] == 0 for row in verifin)
    assert all(row["Full-answer reviewed"] == 1 for row in result["rows"])
    judge_outcomes = [row for row in result["outcomes"] if row["method"] == "LLM Judge"]
    assert all(row["benchmark_outcome"] == "UNSCORED" and row["outcome"] == "FR"
               and row["review_correct"] is True for row in judge_outcomes)
    assert result["provenance"]["full_answer_review"]["record_count"] == 2
    assert result["provenance"]["full_answer_review"]["method_records_reviewed"] == 4
    assert len(result["full_answer_reviews"]) == 2
    assert not result["unscored"]
    assert before == {path: path.read_bytes() for path in source_paths}
    assert report(tmp_path) == original
    assert "full_answer_review" not in original["provenance"]
    assert all("Full-answer reviewed" not in row for row in original["rows"])


def test_missing_unit_scalar_zero_matches_canonical_zero(tmp_path, monkeypatch):
    document = prepare(tmp_path, monkeypatch, candidate="0", gold="0", verdict="INCORRECT",
                       missing_financebench_unit=True)
    # XBRLFiling's compulsory units in the question already resolve its scalar;
    # only FinanceBench's bare zero qualifies for this targeted review.
    document["records"] = [row for row in document["records"] if row["dataset"] == "financebench"]
    write_json(tmp_path / review.REVIEW_PATH, document)
    original = report(tmp_path)
    original_judge, = [row for row in original["rows"]
                       if row["Dataset"] == "FinanceBench" and row["Method"] == "LLM Judge"]
    assert original_judge["Unscored"] == 1
    result = report(tmp_path, review_full_answers=True)
    judge, = [row for row in result["rows"]
              if row["Dataset"] == "FinanceBench" and row["Method"] == "LLM Judge"]
    assert judge["FR"] == 1 and judge["Unscored"] == 0


@pytest.mark.parametrize("correct,verdict,candidate_status,judge_outcome,verifin_outcome", [
    (True, "CORRECT", "VIOLATED", "TA", "FR"),
    (True, "INCORRECT", "VERIFIED", "FR", "TA"),
    (False, "CORRECT", "VIOLATED", "FA", "TR"),
    (False, "INCORRECT", "VERIFIED", "TR", "FA"),
])
def test_correctness_shared_across_methods_and_independent_of_verdict(
        tmp_path, monkeypatch, correct, verdict, candidate_status, judge_outcome, verifin_outcome):
    prepare(tmp_path, monkeypatch, candidate=NARRATIVE if correct else "[answer unavailable]",
            correct=correct, verdict=verdict, candidate_status=candidate_status)
    result = report(tmp_path, review_full_answers=True)
    for row in result["rows"]:
        expected = judge_outcome if row["Method"] == "LLM Judge" else verifin_outcome
        assert row[expected] == 1
        assert row["Unscored"] == 0


@pytest.mark.parametrize("field,value,match", [
    ("source_path", "some/other/results.jsonl", "source_path differs"),
    ("source_sha256", "0" * 64, "source_sha256 differs"),
    ("source_row", 2, "source_row differs"),
    ("candidate_answer", "25 million", "candidate_answer differs"),
    ("claimed_value", 25, "claimed_value differs"),
    ("question", "Different question", "question differs"),
    ("gold_answer", "0 USD millions", "gold_answer differs"),
    ("dataset_path", "different.jsonl", "dataset_path differs"),
    ("dataset_sha256", "0" * 64, "dataset_sha256 differs"),
    ("correct", 1, "must be a boolean"),
    ("correct", "false", "must be a boolean"),
    ("rationale", " ", "rationale must be a nonempty string"),
    ("source_row", True, "positive integer"),
    ("reference_basis", "judge_verdict", "canonical reference basis"),
    ("id", "another_question", "not an unscored full-answer judge decision"),
])
def test_source_or_review_mismatch_fails_closed(tmp_path, monkeypatch, field, value, match):
    document = prepare(tmp_path, monkeypatch)
    document["records"][0][field] = value
    write_json(tmp_path / review.REVIEW_PATH, document)
    with pytest.raises(ValueError, match=match):
        report(tmp_path, review_full_answers=True)


@pytest.mark.parametrize("change,match", [
    (lambda doc: doc.update(schema_version=True), "schema version or protocol"),
    (lambda doc: doc.update(protocol="unrecognized"), "schema version or protocol"),
    (lambda doc: doc["records"].append(copy.deepcopy(doc["records"][0])), "Duplicate"),
    (lambda doc: doc["records"][0].update(judge_verdict="CORRECT"), "unsupported fields"),
    (lambda doc: doc["records"][0]["evidence"][0].update(quote="fabricated quote"), "quote is absent"),
    (lambda doc: doc["records"][0]["evidence"][0].update(field_path=["evidence", 9]), "canonical field"),
])
def test_invalid_document_or_evidence_rejected(tmp_path, monkeypatch, change, match):
    document = prepare(tmp_path, monkeypatch)
    change(document)
    write_json(tmp_path / review.REVIEW_PATH, document)
    with pytest.raises(ValueError, match=match):
        report(tmp_path, review_full_answers=True)


def test_partial_review_keeps_missing_cases_unscored_and_metrics_withheld(tmp_path, monkeypatch):
    document = prepare(tmp_path, monkeypatch)
    document["records"].pop()
    write_json(tmp_path / review.REVIEW_PATH, document)
    result = report(tmp_path, review_full_answers=True)
    assert len(result["unscored"]) == 1
    row, = [row for row in result["rows"] if row["Unscored"]]
    assert row["Precision (%)"] is None and row["Decision accuracy (%)"] is None
    assert row["Coverage (%)"] == 100
    assert result["provenance"]["full_answer_review"]["remaining_unreviewed_candidates"] == 1


def test_already_graded_answer_cannot_be_overridden(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch, candidate="25 USD millions", claimed_value=25)
    with pytest.raises(ValueError, match="not an unscored full-answer judge decision"):
        report(tmp_path, review_full_answers=True)


def test_full_answer_review_requires_file_and_compatible_protocol(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    (tmp_path / review.REVIEW_PATH).unlink()
    with pytest.raises(ValueError, match="requested but is missing"):
        report(tmp_path, review_full_answers=True)
    with pytest.raises(ValueError, match="original scoring and an explicit candidate_model"):
        report(tmp_path, scoring_view="source_reviewed", review_full_answers=True)
    with pytest.raises(ValueError, match="original scoring and an explicit candidate_model"):
        reviewer_baselines.load_report(tmp_path, review_full_answers=True)


def test_scalar_protocol_cannot_use_full_answer_review(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch, candidate="25 USD millions", claimed_value=25,
            protocol="scalar_quantity_v2")
    with pytest.raises(ValueError, match="only strict one-word full-answer judge"):
        report(tmp_path, review_full_answers=True)


def test_review_changed_during_scoring_is_detected(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch)
    original_outcome = review.outcome

    def changing_outcome(correct, status):
        path = tmp_path / review.REVIEW_PATH
        path.write_text(path.read_text() + " ")
        return original_outcome(correct, status)

    monkeypatch.setattr(review, "outcome", changing_outcome)
    with pytest.raises(ValueError, match="Source changed during scoring"):
        report(tmp_path, review_full_answers=True)


def test_evidence_can_bind_another_record_in_the_same_hashed_dataset(tmp_path, monkeypatch):
    document = prepare(tmp_path, monkeypatch, extra_evidence_record=True)
    document["records"][0]["evidence"] = [{
        "record_id": "financebench_two", "field_path": ["evidence", 0, "evidence_text"],
        "quote": "independently stored canonical evidence page",
    }]
    write_json(tmp_path / review.REVIEW_PATH, document)
    result = report(tmp_path, review_full_answers=True)
    assert len(result["full_answer_reviews"]) == 2
    assert len(result["unscored"]) == 2


def test_unknown_canonical_evidence_record_is_rejected(tmp_path, monkeypatch):
    document = prepare(tmp_path, monkeypatch)
    document["records"][0]["evidence"][0]["record_id"] = "absent_record"
    write_json(tmp_path / review.REVIEW_PATH, document)
    with pytest.raises(ValueError, match="unknown canonical evidence record_id"):
        report(tmp_path, review_full_answers=True)
