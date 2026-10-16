from __future__ import annotations

from verifiqa.generation.llm_client import ChatMessage
from verifiqa.retrieval.evidence_retriever import EvidenceRetriever
from verifiqa.types import EvidenceChunk, FinanceBenchExample, RavResult, ABSTENTION_MESSAGE


def _chunks_from_example(example: FinanceBenchExample) -> list[EvidenceChunk]:
    """Build oracle EvidenceChunks directly from example.evidence (FinanceBench/FinQA)."""
    chunks = []
    for idx, ev in enumerate(example.evidence or []):
        if not isinstance(ev, dict):
            continue
        text = ev.get("evidence_text") or ev.get("evidence_text_full_page") or ""
        if not text:
            continue
        doc = ev.get("doc_name") or example.doc_name or ""
        page = ev.get("evidence_page_num")
        chunks.append(EvidenceChunk(
            chunk_id=f"oracle:{example.financebench_id}:{doc}:p{page}:{idx}",
            doc_name=doc,
            page=int(page) if page is not None else None,
            text=text,
            source_type="oracle_evidence",
            company=example.company,
            financebench_id=example.financebench_id,
        ))
    return chunks


class VanillaRagBaseline:
    """CoT baseline: evidence → single LLM call → answer accepted unconditionally.

    Uses oracle evidence when ``example.evidence`` is populated (FinanceBench /
    FinQA oracle setting); falls back to corpus retrieval otherwise.
    """

    def __init__(self, evidence_retriever: EvidenceRetriever, llm_client=None,
                 evidence_callback=None, oracle_retrieval: bool = False):
        self.evidence_retriever = evidence_retriever
        self.llm_client = llm_client
        self.evidence_callback = evidence_callback
        self.oracle_retrieval = oracle_retrieval

    def run_example(self, example: FinanceBenchExample) -> RavResult:
        # Retrieve evidence
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

        evidence_text = "\n\n".join(
            f"[{c.chunk_id}]\n{c.text}" for c in chunks
        )

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

        prompt = (
            "Answer the following financial question using only the evidence provided. "
            "Show your calculation steps, then state the final numeric answer clearly.\n\n"
            f"Question: {example.question}\n\n"
            f"Evidence:\n{evidence_text}"
        )
        answer = self.llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            stage="baseline_vanilla_rag",
        ).strip()

        return RavResult(
            financebench_id=example.financebench_id,
            question=example.question,
            gold_answer=example.answer,
            answer=answer or ABSTENTION_MESSAGE,
            abstained=not bool(answer),
            final_status="ANSWERED" if answer else "ABSTAIN",
            first_pass_solver_status="N/A",
        )

    # Legacy compatibility — some callers use .run() → returns raw string
    def run(self, example: FinanceBenchExample) -> str:
        result = self.run_example(example)
        return result.answer
