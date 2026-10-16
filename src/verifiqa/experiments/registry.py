from __future__ import annotations

from verifiqa.experiments.schemas import MethodSpec


METHOD_SPECS: dict[str, MethodSpec] = {
    "rag_cot": MethodSpec(
        name="rag_cot",
        evidence_mode="rag",
        answerer="cot",
        verifier="none",
        description="Retrieved evidence + CoT answer, accepted unconditionally.",
    ),
    "rag_pot": MethodSpec(
        name="rag_pot",
        evidence_mode="rag",
        answerer="pot",
        verifier="none",
        description="Retrieved evidence + executable Program/Python answer.",
    ),
    "long_context_cot": MethodSpec(
        name="long_context_cot",
        evidence_mode="long_context",
        answerer="cot",
        verifier="none",
        description="Full relevant filing/document context + CoT answer.",
    ),
    "oracle_cot": MethodSpec(
        name="oracle_cot",
        evidence_mode="oracle",
        answerer="cot",
        verifier="none",
        description="Gold evidence + CoT answer.",
    ),
    "oracle_pot": MethodSpec(
        name="oracle_pot",
        evidence_mode="oracle",
        answerer="pot",
        verifier="none",
        description="Gold evidence + executable Program/Python answer.",
    ),
    "llm_judge": MethodSpec(
        name="llm_judge",
        evidence_mode="rag",
        answerer="cot",
        verifier="llm_judge",
        description="RAG + CoT answer checked by an LLM judge.",
    ),
    "faithfulness": MethodSpec(
        name="faithfulness",
        evidence_mode="rag",
        answerer="cot",
        verifier="faithfulness",
        description="RAG + CoT answer checked by a generic evidence-faithfulness verifier.",
    ),
    "verifiqa_strict": MethodSpec(
        name="verifiqa_strict",
        evidence_mode="rag",
        answerer="cot",
        verifier="verifiqa_strict",
        description="RAG + CoT answer checked by VerifiQA without answer correction.",
    ),
    "verifiqa_full": MethodSpec(
        name="verifiqa_full",
        evidence_mode="rag",
        answerer="cot",
        verifier="verifiqa_full",
        description="RAG + CoT answer checked by VerifiQA with evidence-derived correction/repair.",
    ),
}

DEFAULT_METHODS = tuple(METHOD_SPECS)


def parse_methods(value: str) -> list[MethodSpec]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names or names == ["all"]:
        names = list(DEFAULT_METHODS)
    unknown = [name for name in names if name not in METHOD_SPECS]
    if unknown:
        known = ", ".join(sorted(METHOD_SPECS))
        raise ValueError(f"unknown_experiment_method:{unknown[0]} known={known}")
    return [METHOD_SPECS[name] for name in names]
