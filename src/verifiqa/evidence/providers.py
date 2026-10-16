from __future__ import annotations

from typing import Iterable, Optional

from verifiqa.experiments.schemas import SelectedEvidence
from verifiqa.retrieval.planner import RetrievalPlanner
from verifiqa.types import EvidenceChunk, FinanceBenchExample


class OracleEvidenceProvider:
    """Use benchmark-provided evidence passages."""

    def select(self, example: FinanceBenchExample) -> SelectedEvidence:
        chunks = chunks_from_example(example)
        return SelectedEvidence(
            chunks=chunks,
            status="retrieved" if chunks else "no_oracle_evidence",
        )


class RagEvidenceProvider:
    """Use corpus retrieval with a deterministic retrieval plan."""

    def __init__(
        self,
        retriever,
        top_k: int = 10,
        planner: Optional[RetrievalPlanner] = None,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.planner = planner or RetrievalPlanner(None)

    def select(self, example: FinanceBenchExample) -> SelectedEvidence:
        if (
            example.doc_name
            and hasattr(self.retriever, "has_document")
            and not self.retriever.has_document(example.doc_name)
        ):
            return SelectedEvidence(
                chunks=[],
                status=f"missing_document:{example.doc_name}",
            )
        plan = self.planner.plan(example.question)
        if plan.facts and hasattr(self.retriever, "retrieve_with_plan"):
            chunks = self.retriever.retrieve_with_plan(
                example.question,
                plan,
                company=example.company,
                doc_name=example.doc_name,
                top_k=self.top_k,
            )
        else:
            chunks = self.retriever.retrieve(
                example.question,
                company=example.company,
                doc_name=example.doc_name,
                top_k=self.top_k,
            )
        return SelectedEvidence(
            chunks=chunks,
            status="retrieved" if chunks else "no_retrieved_evidence",
            retrieval_plan=plan,
        )


class LongContextEvidenceProvider:
    """Return page chunks for the relevant filing/document as one long context."""

    def __init__(self, corpus_chunks: Iterable[EvidenceChunk], max_chars: int = 200_000):
        self.max_chars = max_chars
        self._by_doc: dict[str, list[EvidenceChunk]] = {}
        for chunk in corpus_chunks:
            if not chunk.doc_name:
                continue
            if chunk.source_type not in {"filing_page", "benchmark_evidence_page", "filing"}:
                continue
            self._by_doc.setdefault(chunk.doc_name, []).append(chunk)
        for doc_name, chunks in self._by_doc.items():
            self._by_doc[doc_name] = sorted(
                chunks,
                key=lambda c: (c.page is None, c.page if c.page is not None else 10**9, c.chunk_id),
            )

    def select(self, example: FinanceBenchExample) -> SelectedEvidence:
        chunks = self._by_doc.get(example.doc_name, [])
        if not chunks:
            return SelectedEvidence(
                chunks=[],
                status=f"missing_document:{example.doc_name or 'unknown'}",
            )
        selected: list[EvidenceChunk] = []
        used_chars = 0
        for chunk in chunks:
            if selected and used_chars + len(chunk.text) > self.max_chars:
                break
            selected.append(chunk)
            used_chars += len(chunk.text)
        return SelectedEvidence(
            chunks=selected,
            status="retrieved" if selected else "no_long_context_evidence",
        )


def chunks_from_example(example: FinanceBenchExample) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for idx, ev in enumerate(example.evidence or []):
        if not isinstance(ev, dict):
            continue
        text = ev.get("evidence_text") or ev.get("evidence_text_full_page") or ""
        if not text:
            continue
        doc = ev.get("doc_name") or ev.get("evidence_doc_name") or example.doc_name or ""
        page = ev.get("evidence_page_num")
        chunks.append(
            EvidenceChunk(
                chunk_id=f"oracle:{example.financebench_id}:{doc}:p{page}:{idx}",
                doc_name=doc,
                page=int(page) if page is not None else None,
                text=str(text),
                source_type="oracle_evidence",
                company=example.company,
                financebench_id=example.financebench_id,
            )
        )
    return chunks
