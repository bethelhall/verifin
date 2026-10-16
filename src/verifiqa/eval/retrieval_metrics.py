from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from verifiqa.types import FinanceBenchExample, RavResult


def _parse_corpus_page(chunk_id: str) -> Optional[Tuple[str, int]]:
    m = re.match(r"corpus:(.+):p(\d+)(?::c\d+)?$", chunk_id)
    return (m.group(1), int(m.group(2))) if m else None


def _oracle_doc_pages(example: FinanceBenchExample) -> List[Tuple[str, int]]:
    evidence = example.evidence
    if evidence is None:
        return []
    if not isinstance(evidence, list):
        evidence = [evidence]
    result = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        doc = item.get("evidence_doc_name") or example.doc_name
        page = item.get("evidence_page_num")
        if page is None:
            continue
        try:
            result.append((str(doc), int(page)))
        except (ValueError, TypeError):
            pass
    return result


def _matches_oracle(retrieved: Tuple[str, int], oracle_set: set) -> bool:
    doc, pg = retrieved
    return any((doc, pg + d) in oracle_set for d in (-1, 0, 1))


def _exact_match(retrieved: Tuple[str, int], oracle_set: set) -> bool:
    return retrieved in oracle_set


def _retrieved_pages(chunk_ids: List[str], k: int) -> List[dict]:
    pages = []
    for cid in chunk_ids[:k]:
        parsed = _parse_corpus_page(cid)
        if parsed is None:
            continue
        doc, page = parsed
        pages.append({"chunk_id": cid, "doc_name": doc, "page": page})
    return pages


def _failure_mode(
    corpus_ids: List[str],
    retrieved_at_k: List[Tuple[str, int]],
    oracle_pages: List[Tuple[str, int]],
    hit: bool,
    exact_hit: bool,
    recall: float,
) -> str:
    if not corpus_ids:
        return "missing_or_unretrieved_document"
    if not oracle_pages:
        return "no_oracle_page_annotation"
    if hit and not exact_hit:
        return "near_page_hit"
    if recall >= 1.0:
        return "covered_all_oracle_pages"
    if hit and recall < 1.0 and len(set(oracle_pages)) > 1:
        return "partial_multi_page_coverage"
    if any(doc == odoc for doc, _ in retrieved_at_k for odoc, _ in oracle_pages):
        return "wrong_page_same_document"
    return "wrong_document_or_page"


def compute_retrieval_metrics(
    results: List[RavResult],
    examples: List[FinanceBenchExample],
    k: int = 5,
) -> dict:
    """
    Hit@k, Recall@k, and MRR@k computed by (doc_name, page) matching.

    Oracle page numbers and corpus page numbers may differ by 1 (pymupdf vs.
    PDF viewer page labels), so ±1 tolerance is applied.
    Only results with corpus:* chunk IDs (real retrieval mode) are evaluated.
    """
    example_by_id: Dict[str, FinanceBenchExample] = {e.financebench_id: e for e in examples}

    per_question = []
    n_skipped_no_corpus = 0
    n_skipped_no_oracle = 0

    for result in results:
        example = example_by_id.get(result.financebench_id)
        if not example:
            continue

        corpus_ids = [cid for cid in result.retrieved_chunk_ids if cid.startswith("corpus:")]
        oracle_pages = _oracle_doc_pages(example)
        if not oracle_pages:
            n_skipped_no_oracle += 1
            continue

        if not corpus_ids:
            n_skipped_no_corpus += 1

        retrieved_at_k: List[Tuple[str, int]] = [
            p for cid in corpus_ids[:k]
            if (p := _parse_corpus_page(cid)) is not None
        ]

        oracle_set = {(doc, pg) for doc, pg in oracle_pages}

        # Hit@k: any retrieved chunk from the right doc+page (±1 tolerance)
        hit = any(_matches_oracle(p, oracle_set) for p in retrieved_at_k)
        exact_hit = any(_exact_match(p, oracle_set) for p in retrieved_at_k)

        # Recall@k: fraction of oracle passages that have a match
        covered = sum(
            1 for doc, pg in oracle_pages
            if any(_matches_oracle(r, {(doc, pg)}) for r in retrieved_at_k)
        )
        recall = covered / len(oracle_pages)

        # MRR@k: reciprocal rank of first relevant result
        mrr = 0.0
        for rank, p in enumerate(retrieved_at_k, 1):
            if _matches_oracle(p, oracle_set):
                mrr = 1.0 / rank
                break

        per_question.append({
            "financebench_id": result.financebench_id,
            "final_status": result.final_status,
            "first_pass_solver_status": result.first_pass_solver_status,
            "hit_at_k": hit,
            "exact_hit_at_k": exact_hit,
            "recall_at_k": recall,
            "mrr_at_k": mrr,
            "n_oracle_passages": len(oracle_pages),
            "n_distinct_oracle_pages": len(set(oracle_pages)),
            "n_retrieved": len(retrieved_at_k),
            "oracle_pages": [
                {"doc_name": doc, "page": page}
                for doc, page in oracle_pages
            ],
            "retrieved_pages": _retrieved_pages(corpus_ids, k),
            "failure_mode": _failure_mode(corpus_ids, retrieved_at_k, oracle_pages, hit, exact_hit, recall),
        })

    n = len(per_question)
    if n == 0:
        return {
            "k": k,
            "n": 0,
            "note": "no results with corpus chunk IDs and oracle page annotations",
        }

    failure_counts: Dict[str, int] = {}
    for row in per_question:
        failure_counts[row["failure_mode"]] = failure_counts.get(row["failure_mode"], 0) + 1

    return {
        "k": k,
        "n": n,
        "n_skipped_no_corpus": n_skipped_no_corpus,
        "n_skipped_no_oracle_pages": n_skipped_no_oracle,
        "hit_rate_at_k": sum(q["hit_at_k"] for q in per_question) / n,
        "exact_hit_rate_at_k": sum(q["exact_hit_at_k"] for q in per_question) / n,
        "mean_recall_at_k": sum(q["recall_at_k"] for q in per_question) / n,
        "mrr_at_k": sum(q["mrr_at_k"] for q in per_question) / n,
        "multi_page_questions": sum(1 for q in per_question if q["n_distinct_oracle_pages"] > 1),
        "multi_page_full_coverage_rate": _rate(
            q["recall_at_k"] >= 1.0 for q in per_question if q["n_distinct_oracle_pages"] > 1
        ),
        "failure_mode_counts": dict(sorted(failure_counts.items())),
        "per_question": per_question,
    }


def _rate(values) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(1 for v in values if v) / len(values)
