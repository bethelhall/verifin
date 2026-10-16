from __future__ import annotations

from verifiqa.baselines.vanilla_rag import _chunks_from_example
from verifiqa.generation.llm_client import ChatMessage
from verifiqa.retrieval.evidence_retriever import EvidenceRetriever
from verifiqa.types import FinanceBenchExample, RavResult, ABSTENTION_MESSAGE


class ReflectionBaseline:
    """CoT + self-reflection baseline (critic agent without oracle labels).

    Evidence → LLM answer → LLM self-critique → final answer.
    Mirrors Tan et al. (2025) CoT+critic setup.  Uses oracle evidence when
    ``example.evidence`` is populated; falls back to corpus retrieval otherwise.
    """

    def __init__(self, evidence_retriever: EvidenceRetriever, llm_client=None,
                 evidence_callback=None, oracle_retrieval: bool = False):
        self.evidence_retriever = evidence_retriever
        self.llm_client = llm_client
        self.evidence_callback = evidence_callback
        self.oracle_retrieval = oracle_retrieval

    def run_example(self, example: FinanceBenchExample) -> RavResult:
        if self.oracle_retrieval and example.evidence:
            chunks = _chunks_from_example(example)
        else:
            chunks = self.evidence_retriever.retrieve(
                example.question, example.company, example.doc_name
            )

        if self.evidence_callback is not None:
            self.evidence_callback(
                example, None, chunks,
                "retrieved" if chunks else "no_retrieved_evidence",
            )

        if not chunks:
            return RavResult(
                financebench_id=example.financebench_id,
                question=example.question,
                gold_answer=example.answer,
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                final_status="ABSTAIN",
                first_pass_solver_status="N/A",
            )

        evidence_text = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)

        if self.llm_client is None:
            return RavResult(
                financebench_id=example.financebench_id,
                question=example.question,
                gold_answer=example.answer,
                answer=evidence_text,
                abstained=False,
                final_status="ANSWERED",
                first_pass_solver_status="N/A",
            )

        # Pass 1: initial CoT answer
        first_prompt = (
            "Answer the following financial question using only the evidence provided. "
            "Show your calculation steps, then state the final numeric answer clearly.\n\n"
            f"Question: {example.question}\n\n"
            f"Evidence:\n{evidence_text}"
        )
        first = self.llm_client.chat(
            [ChatMessage(role="user", content=first_prompt)],
            temperature=0.0,
            stage="baseline_reflection_first",
        ).strip()

        # Pass 2: self-critique and correction
        reflection_prompt = (
            "Review your answer to the financial question below. "
            "Check: (1) are the numbers taken correctly from the evidence? "
            "(2) is the arithmetic correct? "
            "If your answer is correct, repeat it unchanged. "
            "If not, provide the corrected answer with steps.\n\n"
            f"Question: {example.question}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Your previous answer:\n{first}"
        )
        revised = self.llm_client.chat(
            [ChatMessage(role="user", content=reflection_prompt)],
            temperature=0.0,
            stage="baseline_reflection_revision",
        ).strip()

        answer = revised or first
        return RavResult(
            financebench_id=example.financebench_id,
            question=example.question,
            gold_answer=example.answer,
            answer=answer or ABSTENTION_MESSAGE,
            abstained=not bool(answer),
            final_status="ANSWERED" if answer else "ABSTAIN",
            first_pass_solver_status="N/A",
        )

    # Legacy compatibility
    def run(self, example: FinanceBenchExample) -> str:
        result = self.run_example(example)
        return result.answer
