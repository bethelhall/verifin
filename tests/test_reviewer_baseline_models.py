"""Selected-model comparisons bind judge verdicts to that model's saved answers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import baselines, reviewer_baselines
from scripts.build_publication_artifacts import ArtifactError
from scripts.submission_runs import RUNS


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = "results/new_submissions/baselines/test_model"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n")


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        assert any("25 USD millions" in message.content for message in messages)
        return "CORRECT"


def prepare_comparison(tmp_path, monkeypatch, model, *, judge_protocol="scalar_quantity_v2",
                       candidate_text="25 USD millions", claimed_value=25, llm_mode="claude",
                       candidate_changes=None, candidate_drop=()):
    """Run the real writer with one synthetic question per dataset and no API."""
    policy = json.loads((ROOT / reviewer_baselines.ORIGINAL_POLICY_PATH).read_text())
    execution_policy = json.loads((ROOT / reviewer_baselines.EXECUTION_POLICY_PATH).read_text())
    for document in (policy, execution_policy):
        for definition in document["datasets"].values():
            definition["denominator"] = 1
    write_json(tmp_path / reviewer_baselines.ORIGINAL_POLICY_PATH, policy)
    write_json(tmp_path / reviewer_baselines.EXECUTION_POLICY_PATH, execution_policy)
    write_json(tmp_path / reviewer_baselines.ORIGINAL_ADJUDICATIONS_PATH,
               {"dataset": "financebench", "records": []})
    llm = FakeLLM()
    monkeypatch.setattr(baselines, "build_llm", lambda args: llm)
    for dataset, spec in reviewer_baselines.DATASETS.items():
        question = "What is revenue in USD millions? Round to the nearest million."
        rid = f"{dataset}_question_1"
        row = {
            "financebench_id": rid, "question": question, "answer": "25 USD millions",
            "evidence": [{"evidence_text": "Reported revenue is 25 USD millions."}],
            "ground_truth": {"value": 25, "unit": "USD millions",
                             "operands": [{"concept": "Revenue", "weight": 1,
                                           "value_millions": 25}]},
        }
        candidate = {"id": rid, "question": question, "gold_answer": row["answer"],
                     "claim_input_answer": candidate_text, "claimed_value": claimed_value,
                     "claimed_value_unit": "USD millions", "status": "VERIFIED"}
        candidate.update(candidate_changes or {})
        for field in candidate_drop:
            candidate.pop(field, None)
        dataset_path = tmp_path / policy["datasets"][dataset]["source_path"]
        candidate_path = tmp_path / RUNS[dataset][model] / "results.jsonl"
        write_row(dataset_path, row)
        write_row(candidate_path, candidate)
        # A selected-model comparison must never fall back to this legacy pool.
        legacy = execution_policy["baseline_protocol"]["answer_generator_run_by_dataset"][dataset]
        write_json(tmp_path / legacy, {"not": "a valid candidate run"})
        args = SimpleNamespace(
            out=tmp_path / BASELINE_ROOT / f"{spec['prefix']}_judge",
            data=dataset_path, answers_from=candidate_path,
            baseline="judge", with_formula=False, cot=False, resume=False,
            ids=None, limit=0, max_tokens=2048, k=5, threshold=.5,
            llm_mode=llm_mode, model=f"judge-{model}", judge_protocol=judge_protocol,
            vllm_url="http://localhost:11434/v1", preflight=False, verbose=False,
        )
        baselines.run(args)
    assert llm.calls == 2
    return tmp_path


@pytest.mark.parametrize("model", sorted(reviewer_baselines.CANDIDATE_MODELS))
def test_model_comparison_uses_only_selected_submission_candidates(tmp_path, monkeypatch, model):
    root = prepare_comparison(tmp_path, monkeypatch, model)
    inputs = reviewer_baselines.load_inputs(root, BASELINE_ROOT, methods=("judge",),
                                             candidate_model=model)
    report = reviewer_baselines.load_report(root, BASELINE_ROOT, methods=("judge",),
                                            candidate_model=model)
    assert inputs["candidate_model"] == report["provenance"]["candidate_model"] == model
    assert len(report["rows"]) == 4
    assert all(row["N"] == row["TA"] == 1 for row in report["rows"])
    assert report["provenance"]["scoring_protocol"] == "legacy_magnitude_any_token_v1"
    for dataset in reviewer_baselines.DATASETS:
        run, = [run for run in inputs["loaded_runs"]
                if run["dataset"] == dataset and run["method"] == "VeriFin"]
        assert run["result_path"] == f"{RUNS[dataset][model]}/results.jsonl"
        assert run["model"] == run["candidate_model"] == model
    assert not any("detsmt" in path for path in inputs["source_hashes"])
    assert reviewer_baselines.load_historical_precision_family(root, inputs) is None


def test_judge_from_another_candidate_pool_is_rejected(tmp_path, monkeypatch):
    model = "qwen2.5-7b"
    root = prepare_comparison(tmp_path, monkeypatch, model)
    # Even the same text and value with a changed source file cannot be swapped
    # into a completed comparison after its execution manifest was recorded.
    path = root / RUNS["financebench"][model] / "results.jsonl"
    candidate = json.loads(path.read_text())
    candidate["source_run"] = "different saved candidate pool"
    write_row(path, candidate)
    with pytest.raises(ArtifactError, match="answers_sha256 differs"):
        reviewer_baselines.load_inputs(root, BASELINE_ROOT, methods=("judge",),
                                        candidate_model=model)


def test_invalid_candidate_model_fails_before_loading_files(tmp_path):
    with pytest.raises(ValueError, match="Unknown baseline candidate model"):
        reviewer_baselines.load_inputs(tmp_path, BASELINE_ROOT, candidate_model="gpt-6.1")


def test_omitting_model_keeps_historical_metadata_and_sources():
    inputs = reviewer_baselines.load_inputs(ROOT, methods=("judge",))
    assert "candidate_model" not in inputs
    assert all("candidate_model" not in run for run in inputs["loaded_runs"])
    assert any("mc_calc_detsmt_v2_haiku" in path for path in inputs["source_hashes"])


FULL_PROTOCOL = "full_answer_one_word_v1"
NARRATIVE = "The requested revenue is 25 USD millions; the previous year was 10 USD millions."


@pytest.fixture
def full_answer_root(tmp_path, monkeypatch):
    return prepare_comparison(tmp_path, monkeypatch, "gpt-5.5", judge_protocol=FULL_PROTOCOL,
                              candidate_text=NARRATIVE, llm_mode="openai")


def test_full_answer_judge_accepts_unresolved_numeric_metadata_with_exact_verdict(full_answer_root):
    bundle = reviewer_baselines.load_inputs(full_answer_root, BASELINE_ROOT, methods=("judge",),
                                            candidate_model="gpt-5.5")
    for run in bundle["loaded_runs"]:
        if run["method"] == "LLM Judge":
            row, = run["records"]
            assert row["candidate_quantity_status"] == "unresolved"
            assert row["judge_protocol"] == run["judge_protocol"] == FULL_PROTOCOL
            assert row["status"] == "VERIFIED"
            assert row["judge_verdict"] == row["judge_responses"][-1] == "CORRECT"
            assert row["claim_input_answer"] == NARRATIVE
    report = reviewer_baselines.load_report(full_answer_root, BASELINE_ROOT, methods=("judge",),
                                            candidate_model="gpt-5.5")
    assert all(row["TA"] == 1 for row in report["rows"])
    assert all(run["judge_protocol"] == FULL_PROTOCOL for run in report["provenance"]["runs"]
               if run["method"] == "LLM Judge")


@pytest.mark.parametrize("status,verdict,responses,outcome", [
    ("VERIFIED", "CORRECT", [" \nCORRECT\n"], "TA"),
    ("VIOLATED", "INCORRECT", ["The answer is incorrect.", "INCORRECT"], "FR"),
    ("ABSTAIN", None, ["I cannot tell.", "CORRECT."], "ABSTAIN"),
])
def test_full_answer_final_verdict_and_invalid_output_abstention_are_preserved(
        full_answer_root, status, verdict, responses, outcome):
    path = full_answer_root / BASELINE_ROOT / "fb_judge/results.jsonl"
    row = json.loads(path.read_text())
    row.update(status=status, judge_verdict=verdict, judge_responses=responses)
    write_row(path, row)
    report = reviewer_baselines.load_report(full_answer_root, BASELINE_ROOT, methods=("judge",),
                                            candidate_model="gpt-5.5")
    selected, = [row for row in report["rows"]
                 if row["Dataset"] == "FinanceBench" and row["Method"] == "LLM Judge"]
    assert selected[outcome] == 1


@pytest.mark.parametrize("changes", [
    {"judge_protocol": "scalar_quantity_v2"},
    {"judge_protocol": "unrecognized_protocol"},
    {"judge_verdict": "INCORRECT"},
    {"judge_responses": ["The answer is CORRECT"]},
    {"judge_responses": ["correct"]},
    {"judge_responses": ["CORRECT", "CORRECT"]},
    {"judge_responses": []},
    {"judge_responses": [1]},
    {"judge_responses": ["invalid", "invalid", "CORRECT"]},
    {"status": "ABSTAIN"},
    {"claimed_value": 20},
])
def test_full_answer_rows_reject_protocol_verdict_and_numeric_tampering(full_answer_root, changes):
    path = full_answer_root / BASELINE_ROOT / "fb_judge/results.jsonl"
    row = json.loads(path.read_text())
    row.update(changes)
    write_row(path, row)
    with pytest.raises(ArtifactError):
        reviewer_baselines.load_inputs(full_answer_root, BASELINE_ROOT, methods=("judge",),
                                        candidate_model="gpt-5.5")


def test_full_answer_protocol_must_be_explicit_on_each_row(full_answer_root):
    path = full_answer_root / BASELINE_ROOT / "fb_judge/results.jsonl"
    row = json.loads(path.read_text())
    del row["judge_protocol"]
    write_row(path, row)
    with pytest.raises(ArtifactError, match="judge_protocol differs"):
        reviewer_baselines.load_inputs(full_answer_root, BASELINE_ROOT, methods=("judge",),
                                        candidate_model="gpt-5.5")


@pytest.mark.parametrize("changes,match", [
    ({"judge_protocol": "unrecognized_protocol"}, "unknown judge_protocol"),
    ({"cot": True}, "requires plain judge mode"),
    ({"with_formula": True}, "requires plain judge mode"),
])
def test_full_answer_manifest_rejects_invalid_protocol_even_with_recomputed_hash(full_answer_root, changes, match):
    directory = full_answer_root / BASELINE_ROOT / "fb_judge"
    path = directory / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"].update(changes)
    fingerprint = hashlib.sha256(json.dumps(manifest["config"], sort_keys=True,
                                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    manifest["run_fingerprint"] = fingerprint
    write_json(path, manifest)
    row = json.loads((directory / "results.jsonl").read_text())
    row["run_fingerprint"] = fingerprint
    write_row(directory / "results.jsonl", row)
    with pytest.raises(ArtifactError, match=match):
        reviewer_baselines.load_inputs(full_answer_root, BASELINE_ROOT, methods=("judge",),
                                        candidate_model="gpt-5.5")


def test_full_answer_decision_without_scalar_is_unscored_not_abstention(tmp_path, monkeypatch):
    root = prepare_comparison(tmp_path, monkeypatch, "gpt-5.5", judge_protocol=FULL_PROTOCOL,
                              candidate_text=NARRATIVE, claimed_value=None, llm_mode="openai")
    report = reviewer_baselines.load_report(root, BASELINE_ROOT, methods=("judge",),
                                            candidate_model="gpt-5.5")
    judge_rows = [row for row in report["rows"] if row["Method"] == "LLM Judge"]
    assert len(judge_rows) == 2
    assert all(row["VERIFIED"] == row["Unscored"] == 1 for row in judge_rows)
    assert all(row["ABSTAIN"] == row["TA"] == row["FA"] == 0 for row in judge_rows)
    assert all(row["Coverage (%)"] == 100 for row in judge_rows)
    assert all(row["Precision (%)"] is None for row in judge_rows)


def test_missing_verifin_gold_binds_exact_raw_reference_without_mutating_sources(tmp_path, monkeypatch):
    root = prepare_comparison(
        tmp_path, monkeypatch, "gpt-5.5", judge_protocol=FULL_PROTOCOL,
        candidate_changes={"raw_answer": "25 USD millions"}, candidate_drop=("gold_answer",),
    )
    paths = [root / RUNS[dataset]["gpt-5.5"] / "results.jsonl"
             for dataset in reviewer_baselines.DATASETS]
    saved_bytes = {path: path.read_bytes() for path in paths}
    inputs = reviewer_baselines.load_inputs(root, BASELINE_ROOT, methods=("judge",),
                                           candidate_model="gpt-5.5")
    report = reviewer_baselines.load_report(root, BASELINE_ROOT, methods=("judge",),
                                           candidate_model="gpt-5.5")
    assert all(path.read_bytes() == saved_bytes[path] for path in paths)
    for run in inputs["loaded_runs"]:
        if run["method"] == "VeriFin":
            original, = run["original_records"]
            prepared, = run["records"]
            assert "gold_answer" not in original
            assert prepared["gold_answer"] == original["raw_answer"] == "25 USD millions"
            assert prepared["claim_input_answer"] == original["claim_input_answer"]
            assert prepared["status"] == original["status"]
    bindings = report["provenance"]["reference_binding_provenance"]
    assert bindings == inputs["reference_binding_provenance"]
    assert len(bindings) == 2
    for binding in bindings:
        assert binding["method"] == "VeriFin"
        assert binding["source_field"] == "raw_answer"
        assert binding["target_field"] == "gold_answer"
        assert binding["match_rule"] == "exact_dataset_answer"
        assert binding["count"] == len(binding["ids"]) == 1
        assert binding["source_path"] == f"{RUNS[binding['dataset']]['gpt-5.5']}/results.jsonl"
    assert all(row["TA"] == 1 for row in report["rows"])


@pytest.mark.parametrize("changes,drop", [
    ({"raw_answer": "25 USD millions", "gold_answer": "26 USD millions"}, ()),
    ({"raw_answer": "25 USD millions", "gold_answer": None}, ()),
    ({"raw_answer": "26 USD millions"}, ("gold_answer",)),
    ({"raw_answer": "25 USD millions "}, ("gold_answer",)),
    ({}, ("gold_answer", "raw_answer")),
    ({"raw_answer": "25 USD millions",
      "question": "What is revenue in USD millions? Round to the nearest million. "}, ("gold_answer",)),
])
def test_verifin_reference_fallback_rejects_conflicting_or_missing_references(
        tmp_path, monkeypatch, changes, drop):
    root = prepare_comparison(tmp_path, monkeypatch, "gpt-5.5", judge_protocol=FULL_PROTOCOL,
                              candidate_changes=changes, candidate_drop=drop)
    with pytest.raises(ValueError, match="question or gold answer differs from dataset"):
        reviewer_baselines.load_inputs(root, BASELINE_ROOT, methods=("judge",),
                                       candidate_model="gpt-5.5")


def test_baseline_judge_cannot_use_missing_gold_reference_fallback(tmp_path, monkeypatch):
    root = prepare_comparison(tmp_path, monkeypatch, "gpt-5.5", judge_protocol=FULL_PROTOCOL)
    path = root / BASELINE_ROOT / "fb_judge/results.jsonl"
    row = json.loads(path.read_text())
    assert row["raw_answer"] == row["gold_answer"] == "25 USD millions"
    del row["gold_answer"]
    write_row(path, row)
    with pytest.raises(ValueError, match="financebench/LLM Judge/.*gold answer differs"):
        reviewer_baselines.load_inputs(root, BASELINE_ROOT, methods=("judge",),
                                       candidate_model="gpt-5.5")
