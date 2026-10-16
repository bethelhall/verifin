from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

from verifiqa.types import EvidenceChunk, FinanceBenchExample


def _first_present(record, names, default=""):
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return default


def _evidence_items(evidence):
    if evidence is None:
        return []
    if isinstance(evidence, list):
        return evidence
    return [evidence]


def load_financebench_jsonl(path: Path, numeric_only: bool = False) -> List[FinanceBenchExample]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            answer = str(_first_present(record, ["answer", "gold_answer"], ""))
            if numeric_only and not _answer_has_number(answer):
                continue
            examples.append(
                FinanceBenchExample(
                    financebench_id=str(_first_present(record, ["financebench_id", "id", "question_id"], "")),
                    question=str(_first_present(record, ["question"], "")),
                    answer=answer,
                    evidence=record.get("evidence"),
                    justification=str(_first_present(record, ["justification"], "")),
                    question_type=str(_first_present(record, ["question_type"], "")),
                    question_reasoning=str(_first_present(record, ["question_reasoning"], "")),
                    company=str(_first_present(record, ["company"], "")),
                    doc_name=str(_first_present(record, ["doc_name", "evidence_doc_name", "document"], "")),
                    raw=record,
                )
            )
    return examples


def _answer_has_number(answer: str) -> bool:
    import re
    return bool(re.search(r'\d', answer))


def find_financebench_jsonl(financebench_dir: Path) -> Path:
    candidates = [
        financebench_dir / "data" / "financebench_xbrl_augmented.jsonl",
        financebench_dir / "financebench_xbrl_augmented.jsonl",
        financebench_dir / "data" / "financebench_metrics_numeric.jsonl",
        financebench_dir / "financebench_metrics_numeric.jsonl",
        financebench_dir / "data" / "financebench_numerical.jsonl",
        financebench_dir / "financebench_numerical.jsonl",
        financebench_dir / "data" / "financebench_open_source.jsonl",
        financebench_dir / "financebench_open_source.jsonl",
        financebench_dir / "data" / "fixture.jsonl",
        financebench_dir / "fixture.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(financebench_dir.rglob("*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No FinanceBench JSONL file found under {financebench_dir}")
    return matches[0]


def chunks_from_examples(examples: Iterable[FinanceBenchExample]) -> List[EvidenceChunk]:
    chunks = []
    seen = set()
    for example in examples:
        for index, item in enumerate(_evidence_items(example.evidence)):
            if isinstance(item, dict):
                text = str(_first_present(item, ["evidence_text", "text", "evidence_text_full_page"], ""))
                full_page = str(_first_present(item, ["evidence_text_full_page"], ""))
                if full_page and len(full_page) > len(text):
                    text = full_page
                doc_name = str(_first_present(item, ["evidence_doc_name", "doc_name"], example.doc_name))
                page = _first_present(item, ["evidence_page_num", "page", "page_num"], None)
            else:
                text = str(item)
                doc_name = example.doc_name
                page = None
            if not text:
                continue
            chunk_id = f"{example.financebench_id or 'example'}:evidence:{index}"
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    doc_name=doc_name,
                    page=int(page) if isinstance(page, (int, float)) else None,
                    text=text,
                    company=example.company,
                    financebench_id=example.financebench_id,
                )
            )
    return chunks


def chunks_from_xbrl(examples: Iterable[FinanceBenchExample]) -> List[EvidenceChunk]:
    """Convert XBRL facts stored in example.raw into text EvidenceChunks."""
    chunks = []
    facts_per_chunk = 50
    for example in examples:
        raw = getattr(example, "raw", None)
        if not raw:
            continue
        xbrl = raw.get("xbrl") or {}
        facts: dict = xbrl.get("facts") or {}
        if not facts:
            continue
        items = sorted(facts.items())
        for chunk_idx, start in enumerate(range(0, len(items), facts_per_chunk)):
            batch = items[start : start + facts_per_chunk]
            lines = [f"XBRL financial facts for {example.doc_name} (values in USD millions):"]
            for name, value in batch:
                lines.append(f"  {name}: {value} USD millions")
            text = "\n".join(lines)
            chunks.append(EvidenceChunk(
                chunk_id=f"{example.financebench_id}:xbrl:{chunk_idx}",
                doc_name=example.doc_name,
                page=None,
                text=text,
                source_type="xbrl",
                company=example.company,
                financebench_id=example.financebench_id,
            ))
    return chunks


def chunks_from_text_pages(financebench_dir: Path) -> List[EvidenceChunk]:
    chunks = []
    page_dirs = [financebench_dir / "pages", financebench_dir / "data" / "pages"]
    for page_dir in page_dirs:
        if not page_dir.exists():
            continue
        for path in sorted(page_dir.rglob("*.txt")):
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                continue
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"text_page:{path.relative_to(page_dir)}",
                    doc_name=path.stem,
                    page=None,
                    text=text,
                    source_type="filing",
                )
            )
    return chunks


def load_financebench(financebench_path: Path) -> Tuple[List[FinanceBenchExample], List[EvidenceChunk]]:
    if financebench_path.is_file():
        examples = load_financebench_jsonl(financebench_path)
        chunks = chunks_from_examples(examples)
        chunks.extend(chunks_from_xbrl(examples))
    else:
        jsonl_path = find_financebench_jsonl(financebench_path)
        examples = load_financebench_jsonl(jsonl_path)
        chunks = chunks_from_examples(examples)
        chunks.extend(chunks_from_xbrl(examples))
        chunks.extend(chunks_from_text_pages(financebench_path))
    if not chunks:
        raise ValueError(
            "No evidence chunks were built. Provide FinanceBench evidence fields or pre-extracted text pages."
        )
    return examples, chunks
