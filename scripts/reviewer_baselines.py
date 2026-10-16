"""Validate current comparison inputs and apply one declared scoring policy.

Original verdicts and claims stay immutable. The default restores the original
paper criterion; strict quantity scoring remains an explicit diagnostic view.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scripts import baseline_annotations, baseline_full_answer_review, baseline_reference_screening, paper_scoring
from scripts.build_publication_artifacts import (
    FULL_ANSWER_JUDGE_PROTOCOL,
    _validate_baseline_candidates,
    _validate_execution_manifest,
)
from scripts.submission_runs import RUNS as SUBMISSION_RUNS

POLICY_PATH = "data/scoring/scoring_policy_v2.json"
ADJUDICATIONS_PATH = "data/scoring/manual_adjudications_v2.json"
EXECUTION_POLICY_PATH = POLICY_PATH
ORIGINAL_POLICY_PATH = "data/scoring/scoring_policy.json"
ORIGINAL_ADJUDICATIONS_PATH = "data/scoring/manual_adjudications.json"
RUN_ROOT = "results/onepool_parsing_fixed"
HISTORICAL_PRECISION_PATH = "paper/provenance/baseline_precision_historical_family.json"
CANDIDATE_MODELS = frozenset({
    "qwen2.5-7b", "qwen3-30b", "llama-3.1-8b", "gpt-5.5", "claude-haiku-4.5",
})
DATASETS = {
    "financebench": {"label": "FinanceBench", "prefix": "fb"},
    "xbrlfiling": {"label": "XBRLFiling", "prefix": "xbrl"},
}
METHODS = (
    ("none", "Direct acceptance", "none"),
    ("judge", "LLM Judge", "judge"),
    ("judge_formula", "Judge + Formula", "judge-with-formula"),
    ("pot", "Program of Thought", "pot"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_historical_precision_family(root: Path, inputs: dict[str, Any]) -> dict[str, Any] | None:
    """Reuse the original correction only for the exact historical inputs.

    Excluded baseline logs are not required. Their original test results and
    hashes remain in the frozen family. A new run or changed scoring input
    returns ``None`` instead of mixing historical tests with new results.
    """
    path = Path(root) / HISTORICAL_PRECISION_PATH
    if not path.is_file():
        return None
    family = paper_scoring.load_json(path)
    comparisons = family.get("comparisons", [])
    expected = {(definition["label"], label) for definition in DATASETS.values()
                for _, label, _ in METHODS}
    actual = [(row.get("Dataset"), row.get("Baseline")) for row in comparisons]
    if (family.get("schema_version") != 1 or family.get("family_size") != 8
            or len(actual) != 8 or set(actual) != expected):
        raise ValueError("Historical precision family must contain the original eight unique comparisons")
    for row in comparisons:
        for field in ("p-value", "Holm p-value"):
            value = row.get(field)
            if value is not None and (not isinstance(value, (int, float))
                                      or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Invalid historical {field}")
    historical_hashes = family.get("source_hashes", {})
    if not inputs["source_hashes"] or any(historical_hashes.get(relative) != digest
                                         for relative, digest in inputs["source_hashes"].items()):
        return None
    return {**family, "path": HISTORICAL_PRECISION_PATH, "sha256": _sha256(path)}


def load_inputs(root: Path, run_root: str = RUN_ROOT, *, scoring_view: str = "original",
                methods: tuple[str, ...] | None = None,
                candidate_model: str | None = None) -> dict[str, Any]:
    """Load selected baselines plus VeriFin and validate their shared candidates.

    ``methods`` uses run suffixes (for example ``("judge",)``). Omitted methods
    preserve the complete historical comparison, without requiring excluded
    baseline directories when a subset is requested.

    ``candidate_model`` selects that model's consolidated submission candidates.
    Omit it to preserve the historical Haiku sources. The caller supplies the
    corresponding model's baseline ``run_root``; manifests must match the chosen
    candidate files exactly, and the declared scoring policy stays unchanged.
    """
    root = Path(root).resolve()
    if candidate_model is not None and candidate_model not in CANDIDATE_MODELS:
        raise ValueError(f"Unknown baseline candidate model: {candidate_model}")
    hashes: dict[str, str] = {}
    selected_methods = METHODS
    if methods is not None:
        if isinstance(methods, str) or not methods or len(set(methods)) != len(methods):
            raise ValueError("Baseline methods must be a nonempty sequence of unique run suffixes")
        unknown = set(methods) - {suffix for suffix, _, _ in METHODS}
        if unknown:
            raise ValueError(f"Unknown baseline methods: {sorted(unknown)}")
        selected_methods = tuple(method for method in METHODS if method[0] in methods)

    def source(relative: str) -> Path:
        path = (root / relative).resolve()
        path.relative_to(root)
        hashes[relative] = _sha256(path)
        return path

    if scoring_view not in {"original", "source_reviewed", "benchmark_normalized"}:
        raise ValueError(f"Unknown baseline scoring view: {scoring_view}")
    # The candidate pool and corrected execution protocol are independent of
    # the policy used to judge candidate correctness. Always validate execution
    # against its original v2 metadata, even when restoring the paper's v1 rule.
    execution_policy = paper_scoring.load_policy(source(EXECUTION_POLICY_PATH))
    if execution_policy["candidate_correctness"].get("protocol") != "strict_scalar_v2":
        raise ValueError("Corrected baseline execution requires its v2 policy metadata")
    original_scoring = scoring_view == "original"
    policy_path = ORIGINAL_POLICY_PATH if original_scoring else POLICY_PATH
    adjudications_path = ORIGINAL_ADJUDICATIONS_PATH if original_scoring else ADJUDICATIONS_PATH
    policy = paper_scoring.load_policy(source(policy_path))
    if original_scoring:
        policy["candidate_correctness"]["protocol"] = "legacy_magnitude_any_token_v1"
        policy["candidate_correctness"]["allow_corrected_candidate_protocol"] = True
    adjudications = paper_scoring.load_manual_adjudications(source(adjudications_path))
    datasets, candidates, candidate_paths, loaded_runs = {}, {}, {}, []
    for dataset, spec in DATASETS.items():
        dataset_spec = policy["datasets"][dataset]
        if dataset_spec != execution_policy["datasets"][dataset]:
            raise ValueError(f"{dataset}: scoring and execution benchmark definitions differ")
        dataset_relative = dataset_spec["source_path"]
        questions = paper_scoring.load_jsonl(source(dataset_relative))
        ids = paper_scoring.validate_dataset_ids(questions, dataset, dataset_spec)
        datasets[dataset] = {paper_scoring.record_id(row): row for row in questions}
        candidate_relative = (
            f"{SUBMISSION_RUNS[dataset][candidate_model]}/results.jsonl"
            if candidate_model is not None else
            execution_policy["baseline_protocol"]["answer_generator_run_by_dataset"][dataset]
        )
        candidate_paths[dataset] = candidate_relative
        candidate_rows = paper_scoring.load_jsonl(source(candidate_relative))
        paper_scoring.validate_run_ids(candidate_rows, ids, dataset, "shared candidate source")
        candidates[dataset] = {paper_scoring.record_id(row): row for row in candidate_rows}
        for suffix, label, mode in selected_methods:
            directory = f"{run_root}/{spec['prefix']}_{suffix}"
            result_path = f"{directory}/results.jsonl"
            records = paper_scoring.load_jsonl(source(result_path))
            paper_scoring.validate_run_ids(records, ids, dataset, label)
            manifest_path = f"{directory}/run_manifest.json"
            manifest = paper_scoring.load_json(source(manifest_path))
            config = manifest.get("config", {})
            entry = dict(id=f"{dataset}/{suffix}", candidate_protocol="claim_input_answer_v2",
                         run_fingerprint=manifest.get("run_fingerprint"), baseline_mode=mode,
                         evaluator_model=config.get("model"), candidate_answer_matches=len(records))
            if "judge_protocol" in config:
                entry["judge_protocol"] = config["judge_protocol"]
            _validate_execution_manifest(manifest, entry, dataset_sha256=hashes[dataset_relative],
                                         answers_sha256=hashes[candidate_relative], expected_ids=ids)
            _validate_baseline_candidates(records, candidate_rows, entry)
            for row in records:
                if any(row.get(field) != config.get(field) for field in ("model", "baseline", "with_formula", "cot")):
                    raise ValueError(f"{dataset}/{label}/{row['id']}: execution metadata differs from manifest")
            loaded_runs.append(dict(dataset=dataset, dataset_label=spec["label"], method=label,
                                    original_records=records, result_path=result_path,
                                    manifest_path=manifest_path, manifest=manifest, model=config["model"]))
            if "judge_protocol" in config:
                loaded_runs[-1]["judge_protocol"] = config["judge_protocol"]
        loaded_runs.append(dict(dataset=dataset, dataset_label=spec["label"], method="VeriFin",
                                original_records=candidate_rows, result_path=candidate_relative,
                                manifest_path=None, manifest=None,
                                model=candidate_model or "Not recorded in source log"))
    if candidate_model is not None:
        for run in loaded_runs:
            run["candidate_model"] = candidate_model
    # References and candidate quantities are checked independently. Neither
    # annotations nor normalized copies enter execution-manifest validation.
    references, quantities, unresolved = {}, {}, []
    if not original_scoring:
        references = baseline_annotations.load_reference_quantities(root, datasets, hashes, scoring_view)
        overlapping = references.keys() & adjudications.keys()
        if overlapping:
            raise ValueError(f"Conflicting reference annotations: {sorted(overlapping)}")
        adjudications.update(references)
        quantities, unresolved = baseline_annotations.load_candidate_quantities(root, candidates, candidate_paths, hashes)
    reference_bindings = []
    for run in loaded_runs:
        dataset = run["dataset"]
        run["records"] = []
        bound_ids = []
        for original in run["original_records"]:
            rid = paper_scoring.record_id(original)
            question = datasets[dataset][rid]
            # Some VeriFin writers store the benchmark reference in raw_answer
            # and omit gold_answer. Bind only an exact reference match, keeping
            # the saved candidate and original record untouched. An explicit
            # gold_answer (including None) must still match the dataset.
            reference_copy = original
            if (run["method"] == "VeriFin" and "gold_answer" not in original
                    and original.get("raw_answer") == question["answer"]):
                reference_copy = {**original, "gold_answer": question["answer"]}
                bound_ids.append(rid)
            if (original.get("question") != question["question"]
                    or reference_copy.get("gold_answer") != question["answer"]):
                raise ValueError(f"{dataset}/{run['method']}/{rid}: question or gold answer differs from dataset")
            if run["method"] != "VeriFin" and original.get("answer") != question["answer"]:
                raise ValueError(f"{dataset}/{run['method']}/{rid}: answer differs from dataset")
            prepared = baseline_annotations.scoring_row(reference_copy, dataset, quantities)
            if prepared["status"] not in {"VERIFIED", "VIOLATED", "ABSTAIN"}:
                raise ValueError(f"{dataset}/{run['method']}/{rid}: unexpected saved status")
            run["records"].append(prepared)
        if bound_ids:
            reference_bindings.append(dict(
                dataset=dataset, method=run["method"], source_path=run["result_path"],
                source_field="raw_answer", target_field="gold_answer",
                reference_path=policy["datasets"][dataset]["source_path"],
                match_rule="exact_dataset_answer", count=len(bound_ids), ids=bound_ids,
            ))
    for relative, digest in hashes.items():
        if _sha256(root / relative) != digest:
            raise ValueError(f"Source changed during loading: {relative}; rerun the notebook")
    reference_exclusions = baseline_reference_screening.screen_reference_consistency(datasets)
    result = dict(policy=policy, adjudications=adjudications, loaded_runs=loaded_runs, scoring_view=scoring_view,
                scoring_policy_path=policy_path, adjudications_path=adjudications_path,
                execution_policy_path=EXECUTION_POLICY_PATH,
                reference_exclusions=reference_exclusions,
                reference_binding_provenance=reference_bindings,
                reference_screening_policy=dict(baseline_reference_screening.POLICY),
                source_hashes=hashes, normalization_provenance=dict(
                    references=list(references.values()), candidates=list(quantities.values()),
                    unresolved_candidates=unresolved,
                    reference_path=None if original_scoring else baseline_annotations.REFERENCE_PATH,
                    candidate_path=None if original_scoring else baseline_annotations.CANDIDATE_PATH))
    if candidate_model is not None:
        result["candidate_model"] = candidate_model
    return result


def _benchmark_outcome(row: dict, dataset: str, policy: dict, adjudications: dict) -> tuple[str, str]:
    """Keep the original scorer and its unresolved-candidate rules in one place."""
    rid = row["id"]
    try:
        if (row.get("judge_protocol") == FULL_ANSWER_JUDGE_PROTOCOL
                and row["status"] in {"VERIFIED", "VIOLATED"} and row.get("claimed_value") is None):
            raise paper_scoring.ScoringValidationError(f"{rid}: decision with no numeric candidate")
        scored = paper_scoring.score_rows([row], dataset, policy, adjudications)
        return next(name for name, count in scored.items() if count), ""
    except paper_scoring.ScoringValidationError as error:
        allowed_errors = {f"{rid}: decision on an unresolved candidate; review it or abstain"}
        if row.get("judge_protocol") == FULL_ANSWER_JUDGE_PROTOCOL:
            allowed_errors.add(f"{rid}: decision with no numeric candidate")
        if str(error) not in allowed_errors:
            raise
        return "UNSCORED", str(error)


def load_report(root: Path, run_root: str = RUN_ROOT, *, scoring_view: str = "original",
                screen_references: bool = False, methods: tuple[str, ...] | None = None,
                candidate_model: str | None = None,
                review_full_answers: bool = False) -> dict[str, Any]:
    """Score the complete saved run; reference-screened metrics are opt-in diagnostics.

    The notebook calls the same scorer explicitly. Excluding references changes
    the evaluated population, so it must never silently replace full-run metrics.

    ``review_full_answers`` explicitly adds source-bound correctness reviews to
    previously unscored full-answer judge candidates. Every saved decision stays
    intact; matching VeriFin answers receive the same correctness annotation.
    """
    if review_full_answers and (scoring_view != "original" or candidate_model is None):
        raise ValueError("Full-answer review requires original scoring and an explicit candidate_model")
    bundle = load_inputs(root, run_root, scoring_view=scoring_view, methods=methods,
                         candidate_model=candidate_model)
    reviews, review_provenance = {}, None
    if review_full_answers:
        judge_runs = [run for run in bundle["loaded_runs"] if run["method"] != "VeriFin"]
        if (not judge_runs or any(run["method"] != "LLM Judge"
                                 or run.get("judge_protocol") != FULL_ANSWER_JUDGE_PROTOCOL
                                 for run in judge_runs)):
            raise ValueError("Full-answer review requires only strict one-word full-answer judge runs")
        eligible = {(run["dataset"], row["id"]) for run in judge_runs for row in run["records"]
                    if _benchmark_outcome(row, run["dataset"], bundle["policy"],
                                          bundle["adjudications"])[0] == "UNSCORED"}
        reviews, review_provenance = baseline_full_answer_review.load_reviews(
            root, candidate_model=candidate_model, loaded_runs=bundle["loaded_runs"],
            policy=bundle["policy"], source_hashes=bundle["source_hashes"], eligible=eligible,
        )
    rows, outcomes, unscored, runs = [], [], [], []
    for run in bundle["loaded_runs"]:
        counts, statuses = Counter(), Counter()
        pending_accepts = reviewed_count = 0
        for position, row in enumerate(run["records"], 1):
            rid, status = row["id"], row["status"]
            statuses[status] += 1
            record = dict(dataset=run["dataset"], method=run["method"], id=rid, status=status,
                          source_status=row["source_status"], source_path=run["result_path"],
                          source_sha256=bundle["source_hashes"][run["result_path"]], source_row=position)
            benchmark_outcome, scoring_error = _benchmark_outcome(
                row, run["dataset"], bundle["policy"], bundle["adjudications"])
            review = reviews.get((run["dataset"], rid))
            reviewed_outcome = (baseline_full_answer_review.outcome(review["correct"], status)
                                if review else benchmark_outcome)
            if review_full_answers:
                reviewed_count += review is not None
                record.update(correctness_reviewed=review is not None,
                              review_correct=review["correct"] if review else None,
                              review_rationale=review["rationale"] if review else "",
                              review_source_path=review_provenance["path"] if review else "",
                              review_source_sha256=review_provenance["sha256"] if review else "",
                              reviewed_outcome=reviewed_outcome)
            exclusion = bundle["reference_exclusions"][run["dataset"]].get(rid) if screen_references else None
            outcome = "REFERENCE_MISMATCH" if exclusion else reviewed_outcome
            eligibility = dict(benchmark_outcome=benchmark_outcome, reference_eligible=exclusion is None,
                               reference_reason=exclusion["reason"] if exclusion else "")
            if outcome == "UNSCORED":
                pending_accepts += status == "VERIFIED"
                unscored.append(dict(record, **eligibility, outcome=outcome, reason=scoring_error,
                                     question=row["question"], gold_answer=row["gold_answer"],
                                     candidate_answer=row["claim_input_answer"]))
            counts[outcome] += 1
            outcomes.append(dict(record, **eligibility, outcome=outcome))
        accepted, rejected = statuses["VERIFIED"], statuses["VIOLATED"]
        total, decided = len(run["records"]), accepted + rejected
        scored_accepts = counts["TA"] + counts["FA"]
        scored_decisions = scored_accepts + counts["TR"] + counts["FR"]
        assert sum(counts.values()) == total
        rows.append({
            "Dataset": run["dataset_label"], "Method": run["method"], "N": total,
            **{name: statuses[name] for name in ("VERIFIED", "VIOLATED", "ABSTAIN")},
            **{name: counts[name] for name in ("TA", "FA", "TR", "FR")}, "Unscored": counts["UNSCORED"],
            "Reference excluded": counts["REFERENCE_MISMATCH"], "Eligible N": total - counts["REFERENCE_MISMATCH"],
            "Scored accepts": scored_accepts, "Scored decisions": scored_decisions,
            "Precision (%)": 100 * counts["TA"] / scored_accepts if scored_accepts and not pending_accepts else None,
            "Decision accuracy (%)": 100 * (counts["TA"] + counts["TR"]) / scored_decisions if scored_decisions and not counts["UNSCORED"] else None,
            "Coverage (%)": 100 * decided / total,
            "Abstention (%)": 100 * statuses["ABSTAIN"] / total,
        })
        if review_full_answers:
            rows[-1]["Full-answer reviewed"] = reviewed_count
        runs.append(dict(dataset=run["dataset"], method=run["method"], results=run["result_path"],
                         model=run["model"], manifest=run["manifest_path"]))
        if "judge_protocol" in run:
            runs[-1]["judge_protocol"] = run["judge_protocol"]
    for relative, digest in bundle["source_hashes"].items():
        if _sha256(Path(root) / relative) != digest:
            raise ValueError(f"Source changed during scoring: {relative}")
    sources = [dict(path=relative, sha256=digest) for relative, digest in bundle["source_hashes"].items()]
    provenance = dict(run_root=run_root, scoring_protocol=bundle["policy"]["candidate_correctness"]["protocol"],
                      scoring_view=scoring_view, publication_ready=False,
                      allow_corrected_candidate_protocol=bundle["policy"]["candidate_correctness"].get("allow_corrected_candidate_protocol", False),
                      scoring_policy_path=bundle["scoring_policy_path"],
                      adjudications_path=bundle["adjudications_path"],
                      execution_policy_path=bundle["execution_policy_path"],
                      scoring_policy_status=bundle["policy"].get("status"), runs=runs, sources=sources,
                      normalization_provenance=bundle["normalization_provenance"],
                      reference_binding_provenance=bundle["reference_binding_provenance"],
                      screen_references=screen_references, reference_screening_policy=bundle["reference_screening_policy"],
                      reference_exclusions=bundle["reference_exclusions"] if screen_references else {},
                      reference_excluded_counts={dataset: len(records) if screen_references else 0
                                                 for dataset, records in bundle["reference_exclusions"].items()},
                      reference_screening=dict(enabled=screen_references, policy=bundle["reference_screening_policy"],
                                               excluded_counts={dataset: len(records) if screen_references else 0
                                                                for dataset, records in bundle["reference_exclusions"].items()}))
    if candidate_model is not None:
        provenance["candidate_model"] = candidate_model
    result = dict(rows=rows, outcomes=outcomes, unscored=unscored, review_queue=unscored,
                  sources=sources, provenance=provenance)
    if review_full_answers:
        review_provenance["record_count"] = len(reviews)
        review_provenance["method_records_reviewed"] = sum(row["Full-answer reviewed"] for row in rows)
        provenance["full_answer_review"] = review_provenance
        provenance["effective_scoring_protocol"] = (
            provenance["scoring_protocol"] + "+" + baseline_full_answer_review.PROTOCOL)
        result["full_answer_reviews"] = list(reviews.values())
    return result
