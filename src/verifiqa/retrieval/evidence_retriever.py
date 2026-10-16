from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from verifiqa.types import EvidenceChunk, RetrievalFact, RetrievalPlan
from verifiqa.retrieval.table_rows import attach_structured_rows, document_scale_context, extract_structured_rows

_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
_LEGACY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TABLE_ROW_CHUNK_RE = re.compile(r"^corpus:(.+):row:p(\d+):r\d+$")

# BGE-large requires this prefix on queries (not on documents) for retrieval tasks
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _cache_path(cache_dir: Path, model_name: str, n_chunks: int) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")
    return cache_dir / f"embeddings_{slug}_{n_chunks}.npy"


class EvidenceRetriever:
    def __init__(
        self,
        chunks: Iterable[EvidenceChunk],
        model_name: str = _LEGACY_MODEL,
        reranker_model: str = _RERANK_MODEL,
        rerank: bool = False,
        first_stage_k: int = 15,
        embedding_cache_dir: Optional[Path] = None,
        rerank_planned_queries: bool = False,
        retrieval_mode: str = "dense",
        hybrid_rrf_k: int = 60,
    ):
        self.chunks = list(chunks)
        self.model_name = model_name
        self.rerank = rerank
        self.first_stage_k = first_stage_k
        self.rerank_planned_queries = rerank_planned_queries
        if retrieval_mode not in {"dense", "hybrid"}:
            raise ValueError(f"unknown_retrieval_mode:{retrieval_mode}")
        if hybrid_rrf_k <= 0:
            raise ValueError("hybrid_rrf_k_must_be_positive")
        self.retrieval_mode = retrieval_mode
        self.hybrid_rrf_k = hybrid_rrf_k
        self.debug_callback: Optional[Callable[[dict], None]] = None
        self._use_bge_prefix = "bge" in model_name.lower()
        self._chunk_by_doc_page = {}
        for chunk in self.chunks:
            if (
                chunk.doc_name
                and chunk.page is not None
                and chunk.source_type in {"filing_page", "benchmark_evidence_page", "filing"}
            ):
                self._chunk_by_doc_page.setdefault((chunk.doc_name, chunk.page), chunk)
        self._doc_scale_context = document_scale_context(self._chunk_by_doc_page.values())

        if not self.chunks:
            raise ValueError("no_evidence_chunks")

        self._bm25: _Bm25Index | None = None
        self._model = SentenceTransformer(model_name)

        cache_file = (
            _cache_path(embedding_cache_dir, model_name, len(self.chunks))
            if embedding_cache_dir
            else None
        )

        if cache_file and cache_file.exists():
            print(f"Loading cached embeddings from {cache_file}...", flush=True)
            cached_embeddings = np.load(str(cache_file))
            if _valid_embedding_matrix(cached_embeddings, len(self.chunks)):
                self._embeddings = np.asarray(cached_embeddings, dtype=np.float32)
            else:
                print(f"Ignoring invalid cached embeddings at {cache_file}; rebuilding.", flush=True)
                self._embeddings = self._encode_chunks()
                np.save(str(cache_file), self._embeddings)
        else:
            self._embeddings = self._encode_chunks()
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cache_file), self._embeddings)
                print(f"Embeddings cached to {cache_file}", flush=True)

        self._reranker = CrossEncoder(reranker_model) if rerank else None

    def set_debug_callback(self, callback: Optional[Callable[[dict], None]]) -> None:
        self.debug_callback = callback

    def retrieve(
        self,
        question: str,
        company: str = "",
        doc_name: str = "",
        top_k: int = 5,
        financebench_id: str = "",
    ) -> List[EvidenceChunk]:
        candidates, candidate_embeddings, candidate_indices = self._filter(financebench_id, company, doc_name)
        if not candidates:
            return []

        ranked = self._rank_candidates(
            question,
            candidates,
            candidate_embeddings,
            candidate_indices,
            limit=top_k,
            oracle_mode=bool(financebench_id),
            use_reranker=True,
        )
        return self._with_page_context([candidates[i] for i in ranked[:top_k]])

    def retrieve_with_plan(
        self,
        question: str,
        plan: Optional[RetrievalPlan],
        company: str = "",
        doc_name: str = "",
        top_k: int = 5,
        financebench_id: str = "",
    ) -> List[EvidenceChunk]:
        """Retrieve an evidence set by searching separately for each required fact."""

        if plan is None or not plan.facts:
            return self.retrieve(question, company=company, doc_name=doc_name, top_k=top_k, financebench_id=financebench_id)

        candidates, candidate_embeddings, candidate_indices = self._filter(financebench_id, company, doc_name)
        if not candidates:
            return []

        oracle_mode = bool(financebench_id)
        fact_rankings: list[list[int]] = []
        planned_rerank_enabled = bool(self._reranker and self.rerank_planned_queries)
        planned_rank_limit = top_k if planned_rerank_enabled else max(self.first_stage_k, top_k)
        planned_rerank_pool_k = min(self.first_stage_k, max(top_k * 3, top_k))
        for fact in plan.facts:
            ranked = self._rank_candidates(
                _fact_query(question, fact),
                candidates,
                candidate_embeddings,
                candidate_indices,
                limit=planned_rank_limit,
                oracle_mode=oracle_mode,
                aliases=fact.aliases,
                period=fact.period,
                statement=fact.statement,
                use_reranker=planned_rerank_enabled,
                first_stage_limit=planned_rerank_pool_k if planned_rerank_enabled else None,
            )
            fact_rankings.append(ranked)

        question_ranked = self._rank_candidates(
            question,
            candidates,
            candidate_embeddings,
            candidate_indices,
            limit=planned_rank_limit,
            oracle_mode=oracle_mode,
            use_reranker=planned_rerank_enabled,
            first_stage_limit=planned_rerank_pool_k if planned_rerank_enabled else None,
        )

        selected: list[int] = []
        seen_chunks: set[str] = set()

        def add_index(index: int) -> bool:
            chunk_id = candidates[index].chunk_id
            if chunk_id in seen_chunks:
                return False
            selected.append(index)
            seen_chunks.add(chunk_id)
            return True

        # First guarantee coverage: at least one strong candidate per required
        # fact before filling with general question context. Prefer a page no
        # earlier fact has already claimed, so one statement page does not crowd
        # out another fact's evidence page.
        seen_pages: set[tuple[str, int | None]] = set()
        for ranked in fact_rankings:
            fallback: int | None = None
            placed = False
            for index in ranked:
                page_key = (candidates[index].doc_name, candidates[index].page)
                if fallback is None:
                    fallback = index
                if page_key in seen_pages:
                    continue
                if add_index(index):
                    seen_pages.add(page_key)
                    placed = True
                    break
            if not placed and fallback is not None:
                if add_index(fallback):
                    seen_pages.add((candidates[fallback].doc_name, candidates[fallback].page))
            if len(selected) >= top_k:
                return self._with_structured_rows(
                    self._with_page_context([candidates[i] for i in selected[:top_k]]),
                    retrieval_plan=plan,
                )

        for index in question_ranked[:1]:
            add_index(index)
            if len(selected) >= top_k:
                return self._with_structured_rows(
                    self._with_page_context([candidates[i] for i in selected[:top_k]]),
                    retrieval_plan=plan,
                )

        # Then round-robin additional candidates for multi-page or split-table
        # facts until the requested evidence budget is full.
        cursor = 1
        while len(selected) < top_k:
            added = False
            for ranked in fact_rankings:
                if cursor < len(ranked):
                    added = add_index(ranked[cursor]) or added
                    if len(selected) >= top_k:
                        break
            if not added:
                break
            cursor += 1

        for index in question_ranked:
            add_index(index)
            if len(selected) >= top_k:
                break

        return self._with_structured_rows(
            self._with_page_context([candidates[i] for i in selected[:top_k]]),
            retrieval_plan=plan,
        )

    def expand_for_facts(
        self,
        question: str,
        plan: RetrievalPlan,
        target_facts: Sequence[RetrievalFact],
        existing_chunks: Sequence[EvidenceChunk],
        company: str = "",
        doc_name: str = "",
        top_k: int = 5,
        financebench_id: str = "",
    ) -> List[EvidenceChunk]:
        """Promote targeted evidence for facts that failed grounding.

        This is an opt-in second pass used by the verifier feedback loop.  It
        keeps the final evidence budget fixed by putting newly targeted fact
        pages first, then filling from the original retrieval order.
        """

        if not target_facts:
            return list(existing_chunks)

        candidates, candidate_embeddings, candidate_indices = self._filter(financebench_id, company, doc_name)
        if not candidates:
            return list(existing_chunks)

        candidate_by_chunk_id = {chunk.chunk_id: index for index, chunk in enumerate(candidates)}
        selected: list[int] = []
        fallback_chunks: list[EvidenceChunk] = []
        seen_chunks: set[str] = set()
        seen_pages: set[tuple[str, int | None]] = set()
        existing_pages = {
            (chunk.doc_name, chunk.page)
            for chunk in existing_chunks
            if chunk.doc_name or chunk.page is not None
        }

        def add_index(index: int) -> bool:
            chunk = candidates[index]
            if chunk.chunk_id in seen_chunks:
                return False
            selected.append(index)
            seen_chunks.add(chunk.chunk_id)
            seen_pages.add((chunk.doc_name, chunk.page))
            return True

        def add_existing(chunk: EvidenceChunk) -> None:
            index = candidate_by_chunk_id.get(chunk.chunk_id)
            if index is not None:
                add_index(index)
                return
            if chunk.chunk_id in seen_chunks:
                return
            fallback_chunks.append(chunk)
            seen_chunks.add(chunk.chunk_id)

        oracle_mode = bool(financebench_id)
        planned_rerank_enabled = bool(self._reranker and self.rerank_planned_queries)
        rank_limit = max(self.first_stage_k, top_k)
        rerank_pool_k = min(self.first_stage_k, max(top_k * 3, top_k))

        ranked_by_fact: list[list[int]] = []
        for fact in target_facts:
            ranked = self._rank_fact_expansion_candidates(
                question,
                fact,
                candidates,
                candidate_embeddings,
                candidate_indices,
                limit=rank_limit,
                oracle_mode=oracle_mode,
                use_reranker=planned_rerank_enabled,
                first_stage_limit=rerank_pool_k if planned_rerank_enabled else None,
            )
            ranked_by_fact.append(ranked)
            if not ranked:
                continue
            fallback = ranked[0]
            fresh_page = next(
                (
                    index for index in ranked
                    if (candidates[index].doc_name, candidates[index].page) not in existing_pages
                    and (candidates[index].doc_name, candidates[index].page) not in seen_pages
                ),
                None,
            )
            add_index(fresh_page if fresh_page is not None else fallback)
            if len(selected) + len(fallback_chunks) >= top_k:
                return self._finalize_expanded_selection(
                    candidates,
                    selected,
                    fallback_chunks,
                    top_k=top_k,
                    retrieval_plan=plan,
                )

        for chunk in existing_chunks:
            add_existing(chunk)
            if len(selected) + len(fallback_chunks) >= top_k:
                return self._finalize_expanded_selection(
                    candidates,
                    selected,
                    fallback_chunks,
                    top_k=top_k,
                    retrieval_plan=plan,
                )

        for ranked in ranked_by_fact:
            for index in ranked:
                add_index(index)
                if len(selected) + len(fallback_chunks) >= top_k:
                    return self._finalize_expanded_selection(
                        candidates,
                        selected,
                        fallback_chunks,
                        top_k=top_k,
                        retrieval_plan=plan,
                    )

        return self._finalize_expanded_selection(
            candidates,
            selected,
            fallback_chunks,
            top_k=top_k,
            retrieval_plan=plan,
        )

    def has_document(self, doc_name: str) -> bool:
        if not doc_name:
            return True
        doc = doc_name.lower()
        if any(c.doc_name.lower() == doc for c in self.chunks):
            return True
        return any(doc in c.doc_name.lower() for c in self.chunks)

    def _with_page_context(self, chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
        """Attach local table context after ranking.

        PDF extraction often splits filing tables so page N contains the row
        values while page N-1 contains the statement title and "(in millions)"
        scale header.  Retrieval should still rank pages by their own content;
        this context is only for downstream answer/formalization/grounding.
        """

        contextualized = []
        seen: dict[str, int] = {}
        for chunk in chunks:
            expanded = self._expand_table_row_parent(chunk)
            prefix = self._previous_page_context(expanded)
            text = expanded.text
            if prefix:
                text = f"{prefix}\n\n{text}"
            contextual = EvidenceChunk(
                chunk_id=expanded.chunk_id,
                doc_name=expanded.doc_name,
                page=expanded.page,
                text=text,
                source_type=expanded.source_type,
                company=expanded.company,
                financebench_id=expanded.financebench_id,
            )
            existing_index = seen.get(contextual.chunk_id)
            if existing_index is not None:
                contextualized[existing_index] = _merge_duplicate_context(
                    contextualized[existing_index],
                    contextual,
                )
                continue
            seen[contextual.chunk_id] = len(contextualized)
            contextualized.append(contextual)
        return contextualized

    def _expand_table_row_parent(self, chunk: EvidenceChunk) -> EvidenceChunk:
        if chunk.source_type != "table_row":
            return chunk
        parent = self._parent_page_for_table_row(chunk)
        if parent is None:
            return chunk
        return EvidenceChunk(
            chunk_id=parent.chunk_id,
            doc_name=parent.doc_name,
            page=parent.page,
            text=(
                f"[retrieved table-row match from {chunk.chunk_id}]\n"
                f"{chunk.text}\n\n"
                f"PARENT FILING PAGE TEXT:\n{parent.text}"
            ),
            source_type=parent.source_type,
            company=parent.company or chunk.company,
            financebench_id=parent.financebench_id or chunk.financebench_id,
        )

    def _parent_page_for_table_row(self, chunk: EvidenceChunk) -> EvidenceChunk | None:
        parsed = _parse_table_row_chunk_id(chunk.chunk_id)
        if parsed is not None:
            doc_name, page = parsed
            parent = self._chunk_by_doc_page.get((doc_name, page))
            if parent is not None:
                return parent
        if chunk.doc_name and chunk.page is not None:
            return self._chunk_by_doc_page.get((chunk.doc_name, chunk.page))
        return None

    def _with_structured_rows(
        self,
        chunks: List[EvidenceChunk],
        *,
        retrieval_plan: RetrievalPlan | None,
    ) -> List[EvidenceChunk]:
        return attach_structured_rows(
            chunks,
            retrieval_plan,
            doc_scale_context=self._doc_scale_context,
        )

    def _rank_fact_expansion_candidates(
        self,
        question: str,
        fact: RetrievalFact,
        candidates: List[EvidenceChunk],
        candidate_embeddings: np.ndarray,
        candidate_indices: Sequence[int],
        *,
        limit: int,
        oracle_mode: bool,
        use_reranker: bool,
        first_stage_limit: int | None,
    ) -> list[int]:
        queries = [_fact_query(question, fact)]
        for alias in fact.aliases:
            alias_fact = RetrievalFact(
                name=fact.name,
                aliases=[alias],
                period=fact.period,
                statement=fact.statement,
            )
            queries.append(_fact_query(question, alias_fact))

        ranked: list[int] = []
        seen: set[int] = set()
        for query in queries:
            query_ranked = self._rank_candidates(
                query,
                candidates,
                candidate_embeddings,
                candidate_indices,
                limit=limit,
                oracle_mode=oracle_mode,
                aliases=fact.aliases,
                period=fact.period,
                statement=fact.statement,
                use_reranker=use_reranker,
                first_stage_limit=first_stage_limit,
            )
            for index in query_ranked:
                if index in seen:
                    continue
                ranked.append(index)
                seen.add(index)
        return ranked

    def _finalize_expanded_selection(
        self,
        candidates: List[EvidenceChunk],
        selected: list[int],
        fallback_chunks: list[EvidenceChunk],
        *,
        top_k: int,
        retrieval_plan: RetrievalPlan,
    ) -> List[EvidenceChunk]:
        selected_chunks = [candidates[i] for i in selected]
        selected_chunks.extend(fallback_chunks)
        return self._with_structured_rows(
            self._with_page_context(selected_chunks[:top_k]),
            retrieval_plan=retrieval_plan,
        )

    def _previous_page_context(self, chunk: EvidenceChunk) -> str:
        if chunk.page is None or not chunk.doc_name:
            return ""
        previous = self._chunk_by_doc_page.get((chunk.doc_name, chunk.page - 1))
        if previous is None:
            return ""
        snippet = _table_header_context(previous.text)
        if not snippet:
            return ""
        return f"[previous page table header/scale context from {previous.chunk_id}]\n{snippet}"

    def _encode_chunks(self):
        texts = [chunk.text for chunk in self.chunks]
        embeddings = self._model.encode(
            texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if not _valid_embedding_matrix(embeddings, len(self.chunks)):
            raise ValueError("invalid_chunk_embeddings")
        return embeddings

    def _filter(self, financebench_id: str, company: str, doc_name: str):
        if financebench_id:
            indices = [i for i, c in enumerate(self.chunks) if c.financebench_id == financebench_id]
        elif doc_name:
            doc = doc_name.lower()
            exact = [i for i, c in enumerate(self.chunks) if c.doc_name.lower() == doc]
            indices = exact or [i for i, c in enumerate(self.chunks) if doc in c.doc_name.lower()]
            if company and any(self.chunks[i].company for i in indices):
                company_text = company.lower()
                narrowed = [
                    i for i in indices
                    if company_text in self.chunks[i].company.lower()
                ]
                indices = narrowed or indices
        elif company:
            if any(c.company for c in self.chunks):
                company_text = company.lower()
                indices = [
                    i for i, c in enumerate(self.chunks)
                    if company_text in c.company.lower()
                ]
            else:
                indices = list(range(len(self.chunks)))
        else:
            indices = list(range(len(self.chunks)))

        if not indices:
            return [], self._embeddings[:0], []

        return [self.chunks[i] for i in indices], self._embeddings[indices], indices

    def _rank_candidates(
        self,
        query_text: str,
        candidates: List[EvidenceChunk],
        candidate_embeddings: np.ndarray,
        candidate_indices: Sequence[int],
        limit: int,
        oracle_mode: bool,
        aliases: Sequence[str] = (),
        period: str = "",
        statement: str = "",
        use_reranker: bool = True,
        first_stage_limit: int | None = None,
    ) -> list[int]:
        query = (_BGE_QUERY_PREFIX + query_text) if self._use_bge_prefix else query_text
        query_embedding = np.asarray(self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )[0], dtype=np.float32)
        if not np.isfinite(query_embedding).all():
            raise ValueError("non_finite_query_embedding")

        # With normalized embeddings, cosine similarity = dot product.
        # np.matmul can emit noisy BLAS warnings on some macOS NumPy builds even
        # when all inputs/outputs are finite; einsum avoids that path.
        dense_scores = np.einsum("ij,j->i", candidate_embeddings, query_embedding, optimize=True)
        if not np.isfinite(dense_scores).all():
            raise ValueError("non_finite_retrieval_scores")

        if self.retrieval_mode == "hybrid" or self.debug_callback is not None:
            bm25_scores = self._bm25_index().score(query_text, candidate_indices)
        else:
            bm25_scores = np.zeros(len(candidates), dtype=np.float32)
        lexical_scores = np.asarray([
            _lexical_score(query_text, chunk, aliases=aliases, period=period, statement=statement)
            for chunk in candidates
        ], dtype=np.float32)
        if self.retrieval_mode == "hybrid":
            scores = _hybrid_rrf_scores(
                dense_scores=dense_scores,
                bm25_scores=bm25_scores,
                lexical_scores=lexical_scores,
                rrf_k=self.hybrid_rrf_k,
            )
        else:
            scores = dense_scores + lexical_scores

        first_k = min(max(first_stage_limit or self.first_stage_k, limit), len(candidates))
        top_indices = list(_rank_indices(scores, candidates, first_k, oracle_mode))
        reranker_applied = False
        rerank_score_by_index: dict[int, float] = {}
        rerank_rank_by_index: dict[int, int] = {}
        if self._reranker and use_reranker and len(top_indices) > limit:
            first_stage = [candidates[i] for i in top_indices]
            pairs = [(query_text, c.text) for c in first_stage]
            rerank_scores = self._reranker.predict(pairs)
            reranked = sorted(zip(rerank_scores, top_indices), key=lambda x: -x[0])
            ranked = [int(i) for _, i in reranked[:limit]]
            reranker_applied = True
            rerank_score_by_index = {int(i): float(score) for score, i in reranked}
            rerank_rank_by_index = {int(i): rank for rank, (_, i) in enumerate(reranked, start=1)}
        else:
            ranked = [int(i) for i in top_indices[:limit]]
        if self.debug_callback is not None:
            self.debug_callback(_rank_debug_record(
                query_text=query_text,
                candidates=candidates,
                dense_scores=dense_scores,
                bm25_scores=bm25_scores,
                lexical_scores=lexical_scores,
                combined_scores=scores,
                first_stage_indices=top_indices,
                returned_indices=ranked,
                limit=limit,
                first_k=first_k,
                aliases=aliases,
                period=period,
                statement=statement,
                use_reranker=use_reranker,
                reranker_applied=reranker_applied,
                rerank_score_by_index=rerank_score_by_index,
                rerank_rank_by_index=rerank_rank_by_index,
                retrieval_mode=self.retrieval_mode,
                hybrid_rrf_k=self.hybrid_rrf_k,
            ))
        return ranked

    def _bm25_index(self) -> "_Bm25Index":
        if self._bm25 is None:
            self._bm25 = _Bm25Index(self.chunks)
        return self._bm25


class _Bm25Index:
    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.term_counts: list[Counter[str]] = []
        self.doc_lengths = np.zeros(len(chunks), dtype=np.float32)
        document_frequency: Counter[str] = Counter()
        for index, chunk in enumerate(chunks):
            counts = Counter(_tokens(chunk.text))
            self.term_counts.append(counts)
            self.doc_lengths[index] = float(sum(counts.values()))
            document_frequency.update(counts.keys())
        self.avg_doc_length = float(np.mean(self.doc_lengths)) if len(chunks) else 1.0
        if self.avg_doc_length <= 0:
            self.avg_doc_length = 1.0
        n_docs = max(len(chunks), 1)
        self.idf = {
            term: float(np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)))
            for term, df in document_frequency.items()
        }

    def score(self, query: str, indices: Sequence[int]) -> np.ndarray:
        query_terms = list(dict.fromkeys(_tokens(query)))
        scores = np.zeros(len(indices), dtype=np.float32)
        if not query_terms or not indices:
            return scores
        for out_index, chunk_index in enumerate(indices):
            counts = self.term_counts[chunk_index]
            doc_length = float(self.doc_lengths[chunk_index])
            if doc_length <= 0:
                continue
            score = 0.0
            norm = self.k1 * (1.0 - self.b + self.b * doc_length / self.avg_doc_length)
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1.0)) / (tf + norm)
            scores[out_index] = score
        return scores


def _hybrid_rrf_scores(
    *,
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    lexical_scores: np.ndarray,
    rrf_k: int,
) -> np.ndarray:
    return (
        _reciprocal_rank_scores(dense_scores, rrf_k=rrf_k, only_positive=False)
        + _reciprocal_rank_scores(bm25_scores, rrf_k=rrf_k, only_positive=True)
        + _reciprocal_rank_scores(lexical_scores, rrf_k=rrf_k, only_positive=True)
    )


def _reciprocal_rank_scores(
    scores: np.ndarray,
    *,
    rrf_k: int,
    only_positive: bool,
) -> np.ndarray:
    fused = np.zeros(len(scores), dtype=np.float32)
    if len(scores) == 0:
        return fused
    indices = np.argsort(scores)[::-1]
    rank = 1
    for index in indices:
        if only_positive and scores[index] <= 0:
            continue
        fused[index] = 1.0 / (float(rrf_k) + rank)
        rank += 1
    return fused


def _valid_embedding_matrix(embeddings, n_chunks: int) -> bool:
    if not isinstance(embeddings, np.ndarray):
        return False
    if embeddings.ndim != 2 or embeddings.shape[0] != n_chunks or embeddings.shape[1] == 0:
        return False
    if not np.issubdtype(embeddings.dtype, np.number):
        return False
    if not np.isfinite(embeddings).all():
        return False
    return True


def _rank_debug_record(
    *,
    query_text: str,
    candidates: List[EvidenceChunk],
    dense_scores: np.ndarray,
    bm25_scores: np.ndarray,
    lexical_scores: np.ndarray,
    combined_scores: np.ndarray,
    first_stage_indices: list[int],
    returned_indices: list[int],
    limit: int,
    first_k: int,
    aliases: Sequence[str],
    period: str,
    statement: str,
    use_reranker: bool,
    reranker_applied: bool,
    rerank_score_by_index: dict[int, float],
    rerank_rank_by_index: dict[int, int],
    retrieval_mode: str,
    hybrid_rrf_k: int,
) -> dict:
    returned_rank_by_index = {index: rank for rank, index in enumerate(returned_indices, start=1)}
    first_stage = []
    for rank, index in enumerate(first_stage_indices, start=1):
        chunk = candidates[index]
        first_stage.append({
            "rank": rank,
            "candidate_index": int(index),
            "chunk_id": chunk.chunk_id,
            "doc_name": chunk.doc_name,
            "page": chunk.page,
            "source_type": chunk.source_type,
            "dense_score": float(dense_scores[index]),
            "bm25_score": float(bm25_scores[index]),
            "lexical_score": float(lexical_scores[index]),
            "combined_score": float(combined_scores[index]),
            "rerank_score": rerank_score_by_index.get(int(index)),
            "rerank_rank": rerank_rank_by_index.get(int(index)),
            "returned_rank": returned_rank_by_index.get(int(index)),
        })
    return {
        "query_text": query_text,
        "aliases": list(aliases),
        "period": period,
        "statement": statement,
        "limit": limit,
        "first_k": first_k,
        "retrieval_mode": retrieval_mode,
        "hybrid_rrf_k": hybrid_rrf_k,
        "use_reranker": use_reranker,
        "reranker_available": bool(rerank_score_by_index) or reranker_applied,
        "reranker_applied": reranker_applied,
        "first_stage": first_stage,
        "returned_chunk_ids": [candidates[index].chunk_id for index in returned_indices],
    }


def _parse_table_row_chunk_id(chunk_id: str) -> tuple[str, int] | None:
    match = _TABLE_ROW_CHUNK_RE.match(chunk_id or "")
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _merge_duplicate_context(existing: EvidenceChunk, addition: EvidenceChunk) -> EvidenceChunk:
    if addition.text in existing.text:
        return existing
    prefix = ""
    marker = "PARENT FILING PAGE TEXT:"
    if "[retrieved table-row match" in addition.text and marker in addition.text:
        prefix = addition.text.split(marker, 1)[0].strip()
    else:
        prefix = addition.text.strip()
    if not prefix or prefix in existing.text:
        return existing
    return EvidenceChunk(
        chunk_id=existing.chunk_id,
        doc_name=existing.doc_name,
        page=existing.page,
        text=f"{prefix}\n\n{existing.text}",
        source_type=existing.source_type,
        company=existing.company,
        financebench_id=existing.financebench_id,
    )


def _rank_indices(
    scores: np.ndarray,
    candidates: List[EvidenceChunk],
    first_k: int,
    oracle_mode: bool,
) -> np.ndarray:
    ranked = list(np.argsort(scores)[::-1])
    if not oracle_mode:
        return np.asarray(ranked[:first_k])

    evidence = [
        i for i in ranked
        if candidates[i].source_type != "xbrl" and ":evidence:" in candidates[i].chunk_id
    ]
    if not evidence:
        return np.asarray(ranked[:first_k])

    evidence_set = set(evidence)
    remaining = [i for i in ranked if i not in evidence_set]
    return np.asarray((evidence + remaining)[:first_k])


def _fact_query(question: str, fact: RetrievalFact) -> str:
    parts = [question, f"required fact: {fact.name}"]
    if fact.aliases:
        parts.append("aliases: " + "; ".join(fact.aliases))
    if fact.period:
        parts.append("period: " + fact.period)
    if fact.statement:
        parts.append("statement: " + fact.statement)
    return "\n".join(parts)


def _lexical_score(
    query: str,
    chunk: EvidenceChunk,
    aliases: Sequence[str] = (),
    period: str = "",
    statement: str = "",
) -> float:
    text = _normalized_text(chunk.text)
    query_tokens = _tokens(query)
    text_tokens = set(_tokens(chunk.text))
    if not text_tokens:
        return 0.0

    score = 0.0
    if query_tokens:
        overlap = len(set(query_tokens) & text_tokens)
        score += min(1.5, overlap / max(len(set(query_tokens)), 1) * 3.0)

    for alias in aliases:
        alias_norm = _normalized_text(alias)
        if not alias_norm:
            continue
        alias_tokens = set(_tokens(alias))
        if alias_norm in text:
            score += 5.0
        elif alias_tokens:
            overlap = len(alias_tokens & text_tokens)
            score += min(3.0, overlap / len(alias_tokens) * 3.0)
        if _alias_has_table_values(chunk.text, alias, period):
            score += 7.0

    if aliases and _has_structured_row_value(chunk, aliases=aliases, period=period, statement=statement):
        score += 10.0

    for year in re.findall(r"\b20\d{2}\b", period or ""):
        if year in text:
            score += 2.0

    statement_norm = _normalized_text(statement)
    if statement_norm and statement_norm in text:
        score += 1.0
    if statement and _statement_matches(statement, text):
        score += 2.0
    if statement and aliases and _has_primary_statement_heading(statement, chunk.text):
        score += 5.0
    if aliases and _has_multi_year_table(text):
        score += 1.0

    # Notes-page penalty: notes continuation pages cite primary statement names
    # as column headers (e.g. the AOCI reclassification table), so they receive
    # false positive bonuses from _statement_matches and _has_primary_statement_heading.
    # Penalise them for any primary-statement-targeted query so the actual
    # primary statement page wins.
    if statement and _is_notes_continuation_chunk(chunk.text):
        score -= 8.0

    source = (chunk.source_type or "").lower()
    if "xbrl" in source:
        score -= 0.25
    elif source == "table_row":
        score += 1.5
    elif source:
        score += 0.25
    if ":evidence:" in chunk.chunk_id:
        score += 0.5
    return score


def _has_structured_row_value(
    chunk: EvidenceChunk,
    *,
    aliases: Sequence[str],
    period: str,
    statement: str,
) -> bool:
    # Pre-indexed table_row chunks: parse the structured text directly.
    # This is much cheaper than running the full row extractor and correctly
    # handles the TABLE ROW CHUNK text format.
    if chunk.source_type == "table_row":
        return _table_row_chunk_matches(chunk.text, aliases=aliases, period=period)
    # Filing-page chunks: extract rows on the fly.
    plan = RetrievalPlan(
        facts=[
            RetrievalFact(
                name="_retrieval_target",
                aliases=list(aliases),
                period=period,
                statement=statement,
            )
        ]
    )
    try:
        return bool(extract_structured_rows(chunk, plan))
    except Exception:
        return False


def _table_row_chunk_matches(
    text: str,
    *,
    aliases: Sequence[str],
    period: str,
) -> bool:
    """Return True when a TABLE ROW CHUNK directly matches a retrieval target.

    Hard requirements (all must hold):
    - Unit scale must be non-empty (rows from AOCI/notes tables have no scale
      declaration, making them unsafe to use as primary evidence).
    - At least one alias must appear in the row label.
    - At least one year from *period* must be present as a column.
    """
    lines = text.splitlines()
    row_label = ""
    unit_scale = ""
    column_years: set[str] = set()
    for line in lines:
        if line.startswith("Row label:"):
            row_label = line.split(":", 1)[1].strip().lower()
        elif line.startswith("Unit scale:"):
            unit_scale = line.split(":", 1)[1].strip()
        elif re.match(r"^-\s+(19|20)\d{2}:", line):
            year = line.split(":")[0].lstrip("- ").strip()
            column_years.add(year)

    if not unit_scale:
        return False
    alias_hit = any(
        _normalized_text(a) in row_label or row_label in _normalized_text(a)
        for a in aliases
        if a
    )
    if not alias_hit:
        return False
    if period:
        period_years = set(re.findall(r"(?:19|20)\d{2}", period))
        return bool(period_years & column_years)
    return True


def _alias_has_table_values(text: str, alias: str, period: str = "") -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return False
    alias_norm = _normalized_text(alias)
    alias_compact = _compact_text(alias)
    period_years = set(re.findall(r"(?:19|20)\d{2}", period or ""))
    for index, line in enumerate(lines):
        line_norm = _normalized_text(line)
        line_compact = _compact_text(line)
        alias_hit = (
            alias_norm and alias_norm in line_norm
            or alias_compact and alias_compact in line_compact
        )
        if not alias_hit:
            continue
        if not (
            _line_contains_alias_value(line, alias)
            or any(_is_numeric_table_line(item) for item in lines[index + 1:index + 7])
        ):
            continue
        if period_years and not any(year in text for year in period_years):
            continue
        return True
    return False


def _line_contains_alias_value(line: str, alias: str) -> bool:
    """Return true when a row label and value appear on the same line.

    This deliberately avoids prose such as "Shipping costs are recorded as cost
    of sales" because those accounting-policy mentions retrieve the wrong page
    for metric questions.  It accepts normal table rows and compact PDF rows
    like "Restructuring and impairment charges411247289".
    """

    alias_compact = _compact_text(alias)
    line_compact = _compact_text(line)
    if not alias_compact:
        return False
    pos = line_compact.find(alias_compact)
    if pos == -1:
        return False
    tail = line_compact[pos + len(alias_compact):]
    match = re.search(r"\d", tail)
    if not match:
        return False
    prefix = tail[:match.start()]
    # Table units may sit between label and value; prose usually has verbs.
    allowed_prefixes = (
        "", "usd", "dollar", "dollars", "million", "millions",
        "thousand", "thousands", "billion", "billions", "inmillions",
        "inthousands", "inbillions",
    )
    return any(prefix == allowed or prefix.endswith(allowed) for allowed in allowed_prefixes)


def _is_numeric_table_line(line: str) -> bool:
    stripped = line.strip()
    if not re.search(r"\d", stripped):
        return False
    without_numbers = re.sub(r"[-\d\s$,.:%()/_+]+", "", stripped)
    return without_numbers == ""


def _statement_matches(statement: str, normalized_text: str) -> bool:
    statement_norm = _normalized_text(statement)
    aliases = {
        "income statement": (
            "statement of income",
            "statements of income",
            "statement of operations",
            "statements of operations",
            "statement of earnings",
            "statements of earnings",
            "results of operations",
        ),
        "balance sheet": (
            "balance sheet",
            "balance sheets",
            "statement of financial position",
            "statements of financial position",
        ),
        "cash flow": (
            "statement of cash flows",
            "statements of cash flows",
            "cash flow statement",
            "cash flows",
        ),
    }
    candidates = aliases.get(statement_norm, (statement_norm,))
    return any(_normalized_text(candidate) in normalized_text for candidate in candidates)


def _has_primary_statement_heading(statement: str, text: str) -> bool:
    top_lines = "\n".join((text or "").splitlines()[:28])
    top = _normalized_text(top_lines)
    statement_norm = _normalized_text(statement)
    headings = {
        "income statement": (
            "consolidated statements of earnings",
            "consolidated statement of earnings",
            "consolidated statements of income",
            "consolidated statement of income",
            "consolidated statements of operations",
            "consolidated statement of operations",
        ),
        "balance sheet": (
            "consolidated balance sheets",
            "consolidated balance sheet",
            "consolidated statements of financial position",
            "consolidated statement of financial position",
        ),
        "cash flow": (
            "consolidated statements of cash flows",
            "consolidated statement of cash flows",
        ),
    }
    return any(_normalized_text(item) in top for item in headings.get(statement_norm, ()))


def _is_notes_continuation_chunk(text: str) -> bool:
    """Return True when the first line identifies this as a notes-to-FS page.

    Notes continuation pages frequently cite primary statement names as column
    headers (e.g. AOCI reclassification tables reference 'Consolidated Statements
    of Operations').  Without this guard they receive false +5 heading bonuses
    and score on par with actual primary statement pages.
    """
    first_line = (text or "").strip().split("\n")[0].lower()
    return bool(re.search(
        r"notes\s+to\s+(?:consolidated\s+)?financial\s+statements",
        first_line,
    ))


def _has_multi_year_table(normalized_text: str) -> bool:
    compact = normalized_text.replace(" ", "")
    if re.search(r"(?:19|20)\d{2}(?:19|20)\d{2}", compact):
        return True
    return len(set(re.findall(r"(?:19|20)\d{2}", normalized_text))) >= 2


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _normalized_text(value: str) -> str:
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(value: str) -> list[str]:
    normalized = _normalized_text(value)
    if not normalized:
        return []
    return [
        token for token in normalized.split()
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _table_header_context(text: str, window: int = 2) -> str:
    lines = [line.strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    selected: set[int] = set()
    for index, line in enumerate(lines):
        if _is_table_context_line(line):
            for offset in range(-window, window + 1):
                pos = index + offset
                if 0 <= pos < len(lines) and _is_context_safe(lines[pos]):
                    selected.add(pos)
    if not selected:
        return ""
    ordered = [lines[i] for i in sorted(selected)]
    return "\n".join(ordered[-24:])


def _is_table_context_line(line: str) -> bool:
    normalized = _normalized_text(line)
    if not normalized:
        return False
    return any(marker in normalized for marker in (
        "in thousands",
        "in millions",
        "in billions",
        "amounts in thousands",
        "amounts in millions",
        "amounts in billions",
        "dollars in thousands",
        "dollars in millions",
        "dollars in billions",
        "usd thousands",
        "usd millions",
        "usd billions",
        "consolidated balance sheets",
        "consolidated statements",
        "consolidated statement",
        "statements of cash flows",
        "statement of cash flows",
        "statements of operations",
        "statement of operations",
        "statements of income",
        "statement of income",
    ))


def _is_context_safe(line: str) -> bool:
    # Keep statement/scale headers and period labels, but avoid importing
    # unrelated numeric table rows from the neighboring page as evidence facts.
    if _is_table_context_line(line):
        return True
    normalized = _normalized_text(line)
    if not normalized:
        return False
    if re.fullmatch(r"(?:20\d{2}|19\d{2})(?:\s+(?:20\d{2}|19\d{2}))*", normalized):
        return True
    return any(marker in normalized for marker in (
        "for the years ended",
        "years ended",
        "at december",
        "as of",
        "table of contents",
    ))


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "if",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "we", "what",
    "which", "with", "year", "fy", "q1", "q2", "q3", "q4",
}
