"""FinQA dataset loader.

Converts FinQA examples (Chen et al., 2021) into FinanceBenchExample objects
so they can be run through the VerifiQA oracle pipeline without any changes to
the pipeline itself.

Oracle evidence is constructed from the ``gold_inds`` field, which maps keys
like ``table_3`` or ``text_12`` to the supporting facts needed to answer each
question.  The full table is also included as a second evidence chunk so the
LLM has enough context for certificate extraction.

Usage::

    from verifiqa.dataset_finqa import load_finqa
    examples = load_finqa(Path("data/finqa"), split="dev")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from verifiqa.types import FinanceBenchExample


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_finqa(
    data_dir: Path,
    split: str = "dev",
    limit: int = 0,
    numeric_only: bool = False,
) -> List[FinanceBenchExample]:
    """Load a FinQA split and return FinanceBenchExample objects.

    Args:
        data_dir:     Directory containing ``{split}.json`` files.
        split:        One of ``"train"``, ``"dev"``, ``"test"``.
        limit:        Cap the number of examples (0 = all).
        numeric_only: If True, skip examples whose answer is not a plain number.
    """
    path = Path(data_dir) / f"{split}.json"
    raw_examples = json.loads(path.read_text(encoding="utf-8"))

    results: List[FinanceBenchExample] = []
    for item in raw_examples:
        ex = _convert(item)
        if ex is None:
            continue
        if numeric_only and not _is_numeric(ex.answer):
            continue
        results.append(ex)
        if limit and len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _convert(item: dict) -> Optional[FinanceBenchExample]:
    qa = item.get("qa", {})
    question = (qa.get("question") or "").strip()
    exe_ans = qa.get("exe_ans")
    if not question or exe_ans is None:
        return None

    gold_inds: dict = qa.get("gold_inds") or {}
    pre_text: list = item.get("pre_text") or []
    post_text: list = item.get("post_text") or []
    table: list = item.get("table") or []          # list-of-lists; row 0 = header
    filename: str = item.get("filename") or item.get("id") or ""
    fid: str = item.get("id") or filename

    doc_name = _doc_name_from_filename(filename)
    evidence = _build_evidence(gold_inds, pre_text, post_text, table, doc_name)
    company = _company_from_filename(filename)

    # FinQA programs return raw ratios for percentage questions (0.935 for 93.5%).
    # Normalize to the display value so claimed_value matches formula output (* 100).
    display_ans = str(qa.get("answer", "")).strip()
    if display_ans.endswith("%") and isinstance(exe_ans, (int, float)):
        exe_ans = round(float(exe_ans) * 100, 6)

    return FinanceBenchExample(
        financebench_id=f"finqa_{fid}",
        question=question,
        answer=str(exe_ans),
        evidence=evidence,
        company=company,
        doc_name=doc_name,
        raw=item,
    )


def _build_evidence(
    gold_inds: dict,
    pre_text: list,
    post_text: list,
    table: list,
    filename: str,
) -> list:
    """Build a list of evidence dicts (matching FinanceBench evidence format)."""
    seen_texts: set = set()
    evidence: list = []

    def add(text: str, label: str = "") -> None:
        text = text.strip()
        if text and text not in seen_texts:
            seen_texts.add(text)
            evidence.append({
                "evidence_text": text,
                "doc_name": filename,
                "evidence_page_num": None,
                "source_label": label,
            })

    all_text = pre_text + post_text  # text_ indices index into this combined list
    header = table[0] if table else []

    # 1. Gold evidence rows/sentences from gold_inds
    for key in sorted(gold_inds.keys()):
        if key.startswith("table_"):
            row_idx = _parse_idx(key)
            if row_idx is not None and row_idx < len(table):
                add(_format_table_row(header, table[row_idx]), label=key)
        elif key.startswith("text_"):
            text_idx = _parse_idx(key)
            if text_idx is not None and text_idx < len(all_text):
                add(all_text[text_idx], label=key)

    # 2. Full table as a single evidence chunk (gives the LLM layout context)
    if table:
        add(_format_full_table(table), label="full_table")

    return evidence


def _format_table_row(header: list, row: list) -> str:
    """Format a single table row with its column headers.

    Example: "company: American Express | payments volume (billions): 637 | ..."
    """
    if not header or not row:
        return " | ".join(str(v) for v in row)
    pairs = []
    for h, v in zip(header, row):
        h = str(h).strip()
        v = str(v).strip()
        pairs.append(f"{h}: {v}" if h else v)
    return " | ".join(pairs)


def _format_full_table(table: list) -> str:
    """Format an entire table as pipe-separated rows."""
    if not table:
        return ""
    lines = []
    for row in table:
        lines.append(" | ".join(str(c).strip() for c in row))
    return "\n".join(lines)


def _parse_idx(key: str) -> Optional[int]:
    """Extract the integer suffix from a key like 'table_3' or 'text_12'."""
    try:
        return int(key.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _company_from_filename(filename: str) -> str:
    """Best-effort extraction of company ticker from FinQA filename.

    FinQA filenames look like ``"AAPL/2021/page_42.pdf-3"``.
    """
    parts = filename.replace("\\", "/").split("/")
    return parts[0] if parts else ""


def _doc_name_from_filename(filename: str) -> str:
    """Return the canonical FinQA filing key used by XBRL artifacts.

    FinQA examples identify snippets by page filenames like
    ``AAPL/2021/page_42.pdf-3``. The XBRL cache is organized by filing, using
    ``AAPL_2021_10K``.
    """
    parts = [part for part in filename.replace("\\", "/").split("/") if part]
    if len(parts) >= 2 and parts[0] and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}_10K"
    return filename


def _is_numeric(value: str) -> bool:
    """Return True if the value looks like a plain number (int or float)."""
    try:
        float(str(value).replace(",", "").replace("%", ""))
        return True
    except ValueError:
        return False
