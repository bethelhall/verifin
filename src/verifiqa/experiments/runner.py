from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from verifiqa.answerers import CotAnswerer, PotAnswerer
from verifiqa.evidence import LongContextEvidenceProvider, OracleEvidenceProvider, RagEvidenceProvider
from verifiqa.eval.metrics import numeric_answer_accuracy
from verifiqa.experiments.registry import parse_methods
from verifiqa.experiments.schemas import (
    AnswerOutcome,
    ExperimentResult,
    MethodSpec,
    SelectedEvidence,
    VerificationOutcome,
)
from verifiqa.types import ABSTENTION_MESSAGE, FinanceBenchExample, to_jsonable
from verifiqa.verifiers import FaithfulnessVerifier, LlmJudgeVerifier, NoVerifier, VerifiqaVerifier


def run_experiment(
    *,
    examples: list[FinanceBenchExample],
    method_names: str,
    llm_client,
    out_dir: Path,
    retriever=None,
    corpus_chunks=None,
    evidence_top_k: int = 10,
    long_context_max_chars: int = 200_000,
    policy_semantic_check: bool = False,
    certificate_grounding_check: bool = True,
    config: dict | None = None,
) -> list[ExperimentResult]:
    specs = parse_methods(method_names)
    _validate_inputs(specs, retriever, corpus_chunks)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results: list[ExperimentResult] = []

    providers = _build_providers(
        specs,
        retriever=retriever,
        corpus_chunks=corpus_chunks or [],
        evidence_top_k=evidence_top_k,
        long_context_max_chars=long_context_max_chars,
    )
    answerers = {
        "cot": CotAnswerer(llm_client),
        "pot": PotAnswerer(llm_client),
    }
    verifiers = {
        "none": NoVerifier(),
        "llm_judge": LlmJudgeVerifier(llm_client),
        "faithfulness": FaithfulnessVerifier(llm_client),
        "verifiqa_strict": VerifiqaVerifier(
            llm_client,
            strict=True,
            policy_semantic_check=policy_semantic_check,
            certificate_grounding_check=certificate_grounding_check,
        ),
        "verifiqa_full": VerifiqaVerifier(
            llm_client,
            strict=False,
            policy_semantic_check=policy_semantic_check,
            certificate_grounding_check=certificate_grounding_check,
        ),
    }

    with results_path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            provider = providers[spec.evidence_mode]
            answerer = answerers[spec.answerer]
            verifier = verifiers[spec.verifier]
            for example in examples:
                started = time.time()
                call_start = int(getattr(llm_client, "call_index", 0))
                evidence = provider.select(example)
                answer = answerer.answer(example, evidence)
                verification = verifier.verify(example, evidence, answer)
                call_end = int(getattr(llm_client, "call_index", call_start))
                result = _make_result(
                    spec=spec,
                    example=example,
                    evidence=evidence,
                    answer=answer,
                    verification=verification,
                    llm_calls=call_end - call_start,
                    runtime_seconds=time.time() - started,
                )
                results.append(result)
                handle.write(json.dumps(result.to_jsonable(), sort_keys=True) + "\n")
                handle.flush()

    summary = summarize_experiment_results(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "methods": [spec.name for spec in specs],
                "n_examples": len(examples),
                "evidence_top_k": evidence_top_k,
                "long_context_max_chars": long_context_max_chars,
                "policy_semantic_check": policy_semantic_check,
                **(config or {}),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return results


def load_experiment_results(path: Path) -> list[ExperimentResult]:
    results_path = path / "results.jsonl" if path.is_dir() else path
    rows = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return [_result_from_json(row) for row in rows]


def summarize_experiment_results(results: Iterable[ExperimentResult]) -> dict:
    rows = list(results)
    methods = sorted({row.method for row in rows})
    return {
        "n_results": len(rows),
        "methods": {
            method: _summarize_method([row for row in rows if row.method == method])
            for method in methods
        },
    }


def write_summary(results_path: Path, out_dir: Path) -> Path:
    rows = load_experiment_results(results_path)
    summary = summarize_experiment_results(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_providers(
    specs: list[MethodSpec],
    *,
    retriever,
    corpus_chunks,
    evidence_top_k: int,
    long_context_max_chars: int,
) -> dict[str, object]:
    needed = {spec.evidence_mode for spec in specs}
    providers = {}
    if "oracle" in needed:
        providers["oracle"] = OracleEvidenceProvider()
    if "rag" in needed:
        providers["rag"] = RagEvidenceProvider(retriever, top_k=evidence_top_k)
    if "long_context" in needed:
        providers["long_context"] = LongContextEvidenceProvider(corpus_chunks, max_chars=long_context_max_chars)
    return providers


def _validate_inputs(specs: list[MethodSpec], retriever, corpus_chunks) -> None:
    evidence_modes = {spec.evidence_mode for spec in specs}
    if "rag" in evidence_modes and retriever is None:
        raise ValueError("rag_methods_require_corpus_retriever")
    if "long_context" in evidence_modes and not corpus_chunks:
        raise ValueError("long_context_methods_require_corpus_chunks")


def _make_result(
    *,
    spec: MethodSpec,
    example: FinanceBenchExample,
    evidence: SelectedEvidence,
    answer: AnswerOutcome,
    verification: VerificationOutcome,
    llm_calls: int,
    runtime_seconds: float,
) -> ExperimentResult:
    final_answer = verification.final_answer or answer.answer or ABSTENTION_MESSAGE
    return ExperimentResult(
        method=spec.name,
        evidence_mode=spec.evidence_mode,
        answerer=spec.answerer,
        verifier=spec.verifier,
        financebench_id=example.financebench_id,
        question=example.question,
        gold_answer=example.answer,
        draft_answer=answer.answer,
        final_answer=final_answer,
        verified_answer=verification.verified_answer,
        corrected_answer=verification.corrected_answer,
        verified=verification.verified,
        abstained=verification.abstained,
        correct=_correct(final_answer, example.answer),
        draft_correct=_correct(answer.answer, example.answer),
        final_status=verification.final_status,
        evidence_status=evidence.status,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in evidence.chunks],
        evidence_pages=[
            {
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name,
                "page": chunk.page,
                "source_type": chunk.source_type,
            }
            for chunk in evidence.chunks
        ],
        retrieval_plan=evidence.retrieval_plan,
        llm_calls=llm_calls,
        runtime_seconds=runtime_seconds,
        failure_reason=verification.failure_reason or answer.error,
        verification_checks=verification.verification_checks,
    )


def _summarize_method(rows: list[ExperimentResult]) -> dict:
    n = len(rows)
    if not rows:
        return {}
    answered = [row for row in rows if not row.abstained]
    verified = [row for row in rows if row.verified]
    verified_correct = [row for row in verified if row.correct is True]
    verified_incorrect = [row for row in verified if row.correct is False]
    total_llm_calls = sum(row.llm_calls for row in rows)
    total_runtime = sum(row.runtime_seconds for row in rows)
    out: dict = {
        "n": n,
        "answer_accuracy": _pct(row.correct for row in rows),
        "selective_accuracy": _pct(row.correct for row in answered),
        "verified_coverage": _round(len(verified) / n if n else 0.0),
        "verification_precision": _round(len(verified_correct) / len(verified) if verified else 0.0),
        "false_verification_rate": _round(len(verified_incorrect) / len(verified) if verified else 0.0),
        "abstention_rate": _round(sum(1 for row in rows if row.abstained) / n),
        "by_status": _counts(row.final_status for row in rows),
        "by_verification_check": _summarize_verification_checks(rows),
        "failure_reasons": _counts(row.failure_reason for row in rows if row.failure_reason),
    }
    # Include draft_accuracy only when it differs from answer_accuracy (experiment framework sets
    # a separate draft; for `verifiqa run` results they're identical so it's just noise)
    draft_acc = _pct(row.draft_correct for row in rows)
    if draft_acc != out["answer_accuracy"]:
        out["draft_accuracy"] = draft_acc
    # Include per-evidence-status breakdown only when there are meaningful (non-empty) labels
    ev_status = _counts(row.evidence_status for row in rows)
    if set(ev_status.keys()) - {""}:
        out["by_evidence_status"] = ev_status
    # Include timing/call counts only when they were actually tracked
    if total_llm_calls:
        out["mean_llm_calls"] = _round(total_llm_calls / n)
    if total_runtime:
        out["mean_runtime_seconds"] = _round(total_runtime / n)
    return out


def _result_from_json(row: dict) -> ExperimentResult:
    gold = row.get("gold_answer", "")
    # RavResult rows store the answer in "answer"; ExperimentResult rows use "final_answer"
    final_answer = row.get("final_answer") or row.get("answer", "")
    draft_answer = row.get("draft_answer") or final_answer
    # Compute correctness when absent (e.g. results from `verifiqa run`)
    correct = row["correct"] if "correct" in row else _correct(final_answer, gold)
    draft_correct = row["draft_correct"] if "draft_correct" in row else _correct(draft_answer, gold)
    return ExperimentResult(
        method=row.get("method") or "verifiqa",
        evidence_mode=row.get("evidence_mode", ""),
        answerer=row.get("answerer", ""),
        verifier=row.get("verifier", ""),
        financebench_id=row.get("financebench_id", ""),
        question=row.get("question", ""),
        gold_answer=gold,
        draft_answer=draft_answer,
        final_answer=final_answer,
        verified_answer=row.get("verified_answer", ""),
        corrected_answer=row.get("corrected_answer"),
        verified=bool(row.get("verified", False)),
        abstained=bool(row.get("abstained", False)),
        correct=correct,
        draft_correct=draft_correct,
        final_status=row.get("final_status", ""),
        evidence_status=row.get("evidence_status", ""),
        retrieved_chunk_ids=list(row.get("retrieved_chunk_ids") or []),
        evidence_pages=list(row.get("evidence_pages") or []),
        retrieval_plan=row.get("retrieval_plan"),
        llm_calls=int(row.get("llm_calls", 0)),
        runtime_seconds=float(row.get("runtime_seconds", 0.0)),
        failure_reason=row.get("failure_reason", ""),
        verification_checks=dict(row.get("verification_checks") or {}),
    )


def _correct(answer: str, gold: str):
    if not gold:
        return None
    return numeric_answer_accuracy(answer or "", gold)


def _mean_bool(values) -> float:
    values = [value for value in values if value is not None]
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _round(value: float, places: int = 3) -> float:
    return round(value, places)


def _pct(values) -> float:
    return _round(_mean_bool(values))


def _counts(values) -> dict:
    out = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _summarize_verification_checks(rows: list[ExperimentResult]) -> dict:
    check_names = sorted({
        name
        for row in rows
        for name in (row.verification_checks or {}).keys()
    })
    return {
        name: _counts(
            (row.verification_checks or {}).get(name, {}).get("status", "missing")
            for row in rows
        )
        for name in check_names
    }
