from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from verifiqa.types import EvidenceChunk, RetrievalFact, RetrievalPlan


STRUCTURED_ROWS_HEADER = "STRUCTURED TABLE ROWS"
TABLE_ROW_CHUNK_HEADER = "TABLE ROW CHUNK"


@dataclass
class StructuredTableRow:
    required_fact: str
    row_label: str
    columns: dict[str, float]
    unit_scale: str
    unit_scale_quote: str
    unit: str
    statement: str
    page: int | None
    chunk_id: str
    source_quote: str


@dataclass
class GenericTableRow:
    row_label: str
    columns: dict[str, float]
    unit_scale: str
    unit_scale_quote: str
    unit: str
    statement: str
    page: int | None
    chunk_id: str
    source_quote: str


def table_row_chunks_from_pages(
    chunks: Iterable[EvidenceChunk],
    *,
    max_rows_per_page: int = 80,
) -> list[EvidenceChunk]:
    """Create retrieval-only chunks for individual financial table rows."""

    page_chunks = [
        chunk
        for chunk in chunks
        if chunk.source_type == "filing_page" and chunk.page is not None
    ]
    doc_scale_context = document_scale_context(page_chunks)
    output: list[EvidenceChunk] = []
    for chunk in page_chunks:
        rows = extract_generic_table_rows(
            chunk,
            doc_scale_context=doc_scale_context.get(chunk.doc_name, ""),
            max_rows=max_rows_per_page,
        )
        for row_index, row in enumerate(rows):
            output.append(EvidenceChunk(
                chunk_id=f"corpus:{chunk.doc_name}:row:p{chunk.page}:r{row_index}",
                doc_name=chunk.doc_name,
                page=chunk.page,
                text=_format_table_row_chunk(row, parent_chunk_id=chunk.chunk_id),
                source_type="table_row",
                company=chunk.company,
                financebench_id=chunk.financebench_id,
            ))
    return output


def extract_generic_table_rows(
    chunk: EvidenceChunk,
    *,
    doc_scale_context: str = "",
    max_rows: int = 80,
) -> list[GenericTableRow]:
    """Extract likely table rows without needing a question-specific plan."""

    lines = _clean_lines(chunk.text)
    if not lines:
        return []

    statement = _statement_for_chunk(chunk.text)
    unit_scale, unit_scale_quote = _scale_context(chunk.text, doc_scale_context)
    columns = _column_headers(lines, "")
    rows: list[GenericTableRow] = []
    seen: set[str] = set()

    for index, line in enumerate(lines):
        row_label = _generic_row_label(line)
        if not row_label or not _looks_like_generic_row_label(line, row_label):
            continue
        window = _row_window(lines, index)
        values = _numbers_for_row(window, expected_count=len(columns))
        if not values:
            continue
        row_columns = _assign_columns(values, columns, "")
        if not row_columns:
            continue
        source_quote = _source_quote(window)
        key = _compact(f"{row_label}:{source_quote}")
        if key in seen:
            continue
        seen.add(key)
        rows.append(GenericTableRow(
            row_label=row_label,
            columns=row_columns,
            unit_scale=unit_scale,
            unit_scale_quote=unit_scale_quote,
            unit="USD millions" if unit_scale in {"ones", "thousands", "millions", "billions"} else "",
            statement=statement,
            page=chunk.page,
            chunk_id=chunk.chunk_id,
            source_quote=source_quote,
        ))
        if len(rows) >= max_rows:
            break

    return rows


def attach_structured_rows(
    chunks: Iterable[EvidenceChunk],
    plan: RetrievalPlan | None,
    *,
    doc_scale_context: dict[str, str] | None = None,
) -> list[EvidenceChunk]:
    """Prefix retrieved chunks with row-level facts extracted from their tables.

    The raw page is still included for provenance, but prompts can now consume a
    stable row representation instead of reconstructing table structure from
    flattened PDF text.
    """

    output = []
    for chunk in chunks:
        rows = extract_structured_rows(
            chunk,
            plan,
            doc_scale_context=(doc_scale_context or {}).get(chunk.doc_name, ""),
        )
        if not rows:
            output.append(chunk)
            continue
        payload = json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True)
        output.append(EvidenceChunk(
            chunk_id=chunk.chunk_id,
            doc_name=chunk.doc_name,
            page=chunk.page,
            text=f"{STRUCTURED_ROWS_HEADER}:\n{payload}\n\nRAW FILING PAGE TEXT:\n{chunk.text}",
            source_type=chunk.source_type,
            company=chunk.company,
            financebench_id=chunk.financebench_id,
        ))
    return output


def _format_table_row_chunk(row: GenericTableRow, *, parent_chunk_id: str) -> str:
    columns = "\n".join(
        f"- {column}: {value:g}"
        for column, value in row.columns.items()
    )
    parts = [
        f"{TABLE_ROW_CHUNK_HEADER}:",
        f"Parent chunk id: {parent_chunk_id}",
        f"Document: {parent_chunk_id}",
    ]
    if row.page is not None:
        parts.append(f"Page: {row.page}")
    if row.statement:
        parts.append(f"Statement: {row.statement}")
    parts.extend([
        f"Row label: {row.row_label}",
        "Columns:",
        columns,
    ])
    if row.unit_scale:
        parts.append(f"Unit scale: {row.unit_scale}")
    if row.unit_scale_quote:
        parts.append(f"Unit scale quote: {row.unit_scale_quote}")
    if row.unit:
        parts.append(f"Unit: {row.unit}")
    parts.append(f"Source quote: {row.source_quote}")
    return "\n".join(part for part in parts if part)


def extract_structured_rows(
    chunk: EvidenceChunk,
    plan: RetrievalPlan | None,
    *,
    doc_scale_context: str = "",
) -> list[StructuredTableRow]:
    if plan is None or not plan.facts:
        return []

    lines = _clean_lines(chunk.text)
    if not lines:
        return []

    statement = _statement_for_chunk(chunk.text)
    unit_scale, unit_scale_quote = _scale_context(chunk.text, doc_scale_context)
    rows: list[StructuredTableRow] = []
    seen: set[tuple[str, str]] = set()
    for fact in plan.facts:
        fact_rows = _rows_for_fact(
            chunk=chunk,
            lines=lines,
            fact=fact,
            statement=statement or fact.statement,
            unit_scale=unit_scale,
            unit_scale_quote=unit_scale_quote,
        )
        for row in fact_rows:
            key = (row.required_fact, row.source_quote)
            if key in seen:
                continue
            rows.append(row)
            seen.add(key)
            break
    return rows


def document_scale_context(chunks: Iterable[EvidenceChunk]) -> dict[str, str]:
    """Find document-level table scale declarations.

    Annual reports often state once that tabular dollars are in millions.  That
    sentence applies to MD&A tables that do not repeat "(in millions)" locally.
    """

    by_doc: dict[str, str] = {}
    for chunk in chunks:
        if not chunk.doc_name or chunk.doc_name in by_doc:
            continue
        phrase = _document_scale_phrase(chunk.text)
        if phrase:
            by_doc[chunk.doc_name] = phrase
    return by_doc


def _rows_for_fact(
    *,
    chunk: EvidenceChunk,
    lines: list[str],
    fact: RetrievalFact,
    statement: str,
    unit_scale: str,
    unit_scale_quote: str,
) -> list[StructuredTableRow]:
    aliases = fact.aliases or [fact.name.replace("_", " ")]
    columns = _column_headers(lines, fact.period)
    candidates: list[tuple[int, int, StructuredTableRow]] = []
    for index, line in enumerate(lines):
        matched_alias = _matched_alias(line, aliases)
        if not matched_alias:
            continue
        if _is_section_header_match(line, matched_alias):
            continue
        if not _looks_like_table_row_label(line, matched_alias):
            continue
        window = _row_window(lines, index)
        values = _numbers_for_row(window, expected_count=len(columns))
        if not values:
            continue
        row_columns = _assign_columns(values, columns, fact.period)
        if not row_columns:
            continue
        source_quote = _source_quote(window)
        row_label = _row_label(line, matched_alias)
        row = StructuredTableRow(
            required_fact=fact.name,
            row_label=row_label,
            columns=row_columns,
            unit_scale=unit_scale,
            unit_scale_quote=unit_scale_quote,
            unit="USD millions" if unit_scale in {"ones", "thousands", "millions", "billions"} else "",
            statement=statement,
            page=chunk.page,
            chunk_id=chunk.chunk_id,
            source_quote=source_quote,
        )
        candidates.append((_row_match_score(row.row_label, matched_alias, fact), index, row))
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    return [row for _, _, row in candidates]


def _clean_lines(text: str) -> list[str]:
    lines = []
    for raw in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def _row_window(lines: list[str], index: int) -> list[str]:
    parts = [lines[index]]
    numeric_seen = bool(_numbers_in_text(lines[index]))
    for line in lines[index + 1:index + 14]:
        if numeric_seen and _looks_like_next_row(line):
            break
        parts.append(line)
        if _numbers_in_text(line):
            numeric_seen = True
    return parts


def _looks_like_next_row(line: str) -> bool:
    if not re.search(r"[A-Za-z]", line):
        return False
    normalized = _normalized(line)
    if normalized in {"table of contents", "continued on following page"}:
        return False
    if normalized in {"operating activities", "investing activities", "financing activities"}:
        return False
    return True


def _source_quote(window: Sequence[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(window)).strip()


def _generic_row_label(line: str) -> str:
    if not re.search(r"[A-Za-z]", line or ""):
        return ""
    label = re.sub(r"[\$()0-9,.\-%]+", " ", line)
    label = re.sub(r"\s+", " ", label).strip(" :-")
    return label


def _looks_like_generic_row_label(line: str, row_label: str) -> bool:
    tokens = _tokens(row_label)
    if not tokens:
        return False
    normalized = _normalized(row_label)
    if normalized in {
        "table of contents",
        "continued on following page",
        "assets",
        "liabilities and equity",
        "operating activities",
        "investing activities",
        "financing activities",
    }:
        return False
    if len(tokens) > 14:
        return False
    if re.search(r"[.;!?]", line or "") and len(tokens) > 6:
        return False
    if ":" in (line or "") and not _numbers_in_text(line):
        return False
    return True


def _row_label(line: str, alias: str) -> str:
    if re.search(r"[A-Za-z]", line):
        label = re.sub(r"[\$()0-9,.\-%]+", " ", line)
        label = re.sub(r"\s+", " ", label).strip()
        if label:
            return label
    return alias


def _looks_like_table_row_label(line: str, alias: str) -> bool:
    """Reject prose mentions of a fact while keeping table row labels.

    PDF text extraction flattens both statement rows and MD&A prose into lines.
    A sentence like "$1 million of costs in cost of sales in fiscal 2019" should
    not become a row for the "Cost of sales" fact.  Actual row labels are short
    and reduce back to the alias after numbers/symbols are removed.
    """

    alias_norm = _normalized(alias)
    label_norm = _normalized(_row_label(line, alias))
    if not alias_norm or alias_norm not in label_norm:
        return False
    alias_tokens = set(_tokens(alias))
    label_tokens = _tokens(label_norm)
    if alias_tokens and not alias_tokens.issubset(set(label_tokens)):
        return False
    if len(label_tokens) > max(len(alias_tokens) + 4, 10):
        return False
    if re.search(r"[.;!?]", line) and len(label_tokens) > len(alias_tokens) + 2:
        return False
    return True


def _matched_alias(line: str, aliases: Sequence[str]) -> str:
    line_norm = _normalized(line)
    line_compact = _compact(line)
    line_tokens = set(_tokens(line))
    candidates: list[tuple[int, int, str]] = []
    for alias in aliases:
        alias_norm = _normalized(alias)
        if not alias_norm:
            continue
        matched = False
        if alias_norm in line_norm or _compact(alias) in line_compact:
            matched = True
        alias_tokens = set(_tokens(alias))
        if alias_tokens and alias_tokens.issubset(line_tokens):
            matched = True
        if matched:
            candidates.append((len(alias_tokens), len(alias_norm), alias))
    if not candidates:
        return ""
    return max(candidates)[2]


def _is_section_header_match(line: str, alias: str) -> bool:
    """Reject table section headers that introduce child rows.

    Balance sheets commonly have section labels such as "Current assets:" and
    "Current liabilities:" immediately followed by the first child row.  If we
    treat the header as a row, the child row's numbers get incorrectly attached
    to the section label.  A true data row like "Total current assets: $5,121.3"
    is kept because the text after the colon is numeric rather than another row
    label.
    """

    before, separator, after = line.partition(":")
    if not separator:
        return False
    if _numbers_in_text(before):
        return False
    before_norm = _normalized(before)
    alias_norm = _normalized(alias)
    if not before_norm or not alias_norm:
        return False
    alias_tokens = set(_tokens(alias))
    before_tokens = set(_tokens(before))
    if alias_norm not in before_norm and not (alias_tokens and alias_tokens.issubset(before_tokens)):
        return False
    if not _numbers_in_text(after):
        return True
    before_first_number = re.split(r"\(?[-+]?\$?\d", after, maxsplit=1)[0]
    return bool(re.search(r"[A-Za-z]", before_first_number))


def _row_match_score(row_label: str, alias: str, fact: RetrievalFact) -> int:
    label_norm = _normalized(row_label)
    alias_norm = _normalized(alias)
    score = min(len(alias_norm), 60)
    if label_norm == alias_norm:
        score += 40
    elif label_norm.startswith(alias_norm):
        score += 25
    elif alias_norm in label_norm:
        score += 15
    if _fact_prefers_total(fact):
        score += 30 if "total" in _tokens(row_label) else -30
    return score


def _fact_prefers_total(fact: RetrievalFact) -> bool:
    if "total" in _tokens(fact.name):
        return True
    return any("total" in _tokens(alias) for alias in fact.aliases)


def _column_headers(lines: list[str], period: str) -> list[str]:
    vertical_years = _vertical_year_headers(lines)
    if vertical_years:
        return vertical_years
    for line in lines[:80]:
        years = _years(line)
        if len(years) >= 2:
            return _dedupe(years)
    period_years = _years(period)
    if period_years:
        return _dedupe(period_years)
    years = []
    for line in lines[:80]:
        years.extend(_years(line))
    return _dedupe(years[:4])


def _vertical_year_headers(lines: list[str]) -> list[str]:
    for index in range(min(len(lines), 80)):
        if not re.fullmatch(r"(?:19|20)\d{2}", lines[index].strip()):
            continue
        years = []
        for line in lines[index:index + 6]:
            stripped = line.strip()
            if re.fullmatch(r"(?:19|20)\d{2}", stripped):
                years.append(stripped)
                continue
            if years:
                break
        if len(years) >= 2:
            return _dedupe(years)
    return []


def _assign_columns(values: list[float], columns: list[str], period: str) -> dict[str, float]:
    if columns:
        usable = values[:len(columns)]
        if not usable:
            return {}
        return {column: value for column, value in zip(columns, usable)}
    years = _years(period)
    if years and values:
        return {years[0]: values[0]}
    if values:
        return {"value": values[0]}
    return {}


def _numbers_for_row(window: Sequence[str], expected_count: int) -> list[float]:
    text = " ".join(window)
    values = _numbers_in_text(text)
    if expected_count > 1 and len(values) == 1:
        split = _split_compact_number_run(text, expected_count)
        if split:
            return split
    if expected_count > 1 and len(values) < expected_count:
        split = _split_compact_number_run(text, expected_count)
        if split and len(split) > len(values):
            return split
    return values


def _numbers_in_text(text: str) -> list[float]:
    values = []
    pattern = r"\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?"
    for raw in re.findall(pattern, text or ""):
        value = _parse_number(raw)
        if value is not None:
            values.append(value)
    return values


def _split_compact_number_run(text: str, expected_count: int) -> list[float]:
    if expected_count <= 1:
        return []
    matches = re.findall(r"[A-Za-z\),)]([0-9]{%d,})(?![0-9])" % max(3, expected_count * 2), text or "")
    for compact_digits in matches:
        split = _best_digit_split(compact_digits, expected_count)
        if split:
            return [float(value) for value in split]
    return []


def _best_digit_split(digits: str, count: int) -> list[int]:
    candidates: list[list[str]] = []

    def backtrack(pos: int, remaining: int, parts: list[str]) -> None:
        if remaining == 0:
            if pos == len(digits):
                candidates.append(parts[:])
            return
        min_left = remaining - 1
        max_len = min(6, len(digits) - pos - min_left)
        for width in range(1, max_len + 1):
            part = digits[pos:pos + width]
            if len(part) > 1 and part.startswith("0"):
                continue
            parts.append(part)
            backtrack(pos + width, remaining - 1, parts)
            parts.pop()

    backtrack(0, count, [])
    if not candidates:
        return []

    def score(parts: list[str]) -> tuple[float, float, int]:
        lengths = [len(part) for part in parts]
        mean = sum(lengths) / len(lengths)
        variance = sum((length - mean) ** 2 for length in lengths)
        tiny_penalty = sum(1 for part in parts if len(part) == 1)
        magnitude_span = max(float(part) for part in parts) / max(1.0, min(float(part) for part in parts))
        return variance, magnitude_span, tiny_penalty

    best = min(candidates, key=score)
    return [int(part) for part in best]


def _parse_number(raw: str) -> float | None:
    token = raw.strip()
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").replace("$", "").replace(",", "").replace("%", "").strip()
    if not token:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    # For retrieval/formalization evidence, expose the displayed magnitude.
    # The formalizer can decide whether a cash-flow outflow should be a positive
    # "amount" or a signed value in the formula.
    return abs(value) if negative else value


def _scale_context(text: str, doc_context: str) -> tuple[str, str]:
    for source in (text or "", doc_context or ""):
        phrase = _local_scale_phrase(source)
        if phrase:
            return _scale_from_phrase(phrase), phrase
    return "", ""


def _document_scale_phrase(text: str) -> str:
    patterns = [
        r"unless otherwise noted,\s+tabular dollars are (?:presented )?in (?:thousands|millions|billions)[^.\n]*",
        r"tabular dollars are (?:presented )?in (?:thousands|millions|billions)[^.\n]*",
        r"amounts in (?:thousands|millions|billions)[^.\n]*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()
    return ""


def _local_scale_phrase(text: str) -> str:
    patterns = [
        r"\(\s*in\s+(?:thousands|millions|billions)[^)]*\)",
        r"\$\s+in\s+(?:thousands|millions|billions)",
        r"dollars\s+in\s+(?:thousands|millions|billions)",
        r"tabular dollars are (?:presented )?in (?:thousands|millions|billions)[^.\n]*",
        r"unless otherwise noted,\s+tabular dollars are (?:presented )?in (?:thousands|millions|billions)[^.\n]*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()
    return ""


def _scale_from_phrase(phrase: str) -> str:
    text = (phrase or "").lower()
    if "thousand" in text:
        return "thousands"
    if "billion" in text:
        return "billions"
    if "million" in text:
        return "millions"
    return ""


def _statement_for_chunk(text: str) -> str:
    normalized = _normalized(text)
    if "cash flow" in normalized:
        return "cash flow"
    if "balance sheet" in normalized or "statement of financial position" in normalized:
        return "balance sheet"
    # Match both singular ("statement of operations") and plural ("statements of
    # operations") — PDFs typically use the plural form.
    if any(phrase in normalized for phrase in (
        "statement of income",
        "statements of income",
        "statement of operations",
        "statements of operations",
        "statement of earnings",
        "statements of earnings",
        "income statement",
    )):
        # Guard: if this is a notes-continuation page, the statement reference
        # is a column header, not a page heading — return empty so table-row
        # chunks from notes pages do not claim income-statement provenance.
        if _is_notes_continuation_page(text):
            return ""
        return "income statement"
    return ""


def _is_notes_continuation_page(text: str) -> bool:
    """Return True when the chunk is a notes-to-financial-statements page.

    Such pages reference primary statement names as column labels (e.g. the
    AOCI reclassification table column 'Affected line item in the Consolidated
    Statements of Operations'), which would otherwise falsely tag the chunk as
    an income-statement page.
    """
    first_line = (text or "").strip().split("\n")[0].lower()
    return bool(re.search(
        r"notes\s+to\s+(?:consolidated\s+)?financial\s+statements",
        first_line,
    ))


def _years(text: str) -> list[str]:
    return re.findall(r"(?:19|20)\d{2}", text or "")


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        output.append(value)
        seen.add(value)
    return output


def _normalized(value: str) -> str:
    value = (value or "").lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _tokens(value: str) -> list[str]:
    return [token for token in _normalized(value).split() if token and token not in _STOPWORDS]


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is",
    "of", "on", "or", "the", "to", "was", "were", "with",
}
