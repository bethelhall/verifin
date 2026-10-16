"""Deterministic table-based fact extraction (no LLM).

Parses pipe-separated tables from evidence_text, then looks up each role by
matching (row_label, column_year).  The caller uses these results instead of
(or before) the LLM extractor.

Outcomes per role:
  - GroundedFact        → exactly one cell matched; use it
  - None                → no cell matched; fall through to LLM extraction
  - AmbiguousTableLookup raised → multiple cells matched; caller should ABSTAIN
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from verifiqa.agent.types import GroundedFact, RoleSpec


class AmbiguousTableLookup(Exception):
    """Multiple table cells satisfy the role — caller must ABSTAIN."""

    def __init__(self, role_name: str, candidates: list["_Cell"]) -> None:
        self.role_name = role_name
        self.candidates = candidates
        vals = [f"{c.row_label!r}@{c.col_header!r}={c.raw}" for c in candidates[:4]]
        super().__init__(f"ambiguous_table_lookup:{role_name} candidates=[{', '.join(vals)}]")


@dataclass
class _Cell:
    row_label: str
    col_header: str
    raw: str
    value: float
    source_quote: str = ""
    unit: str = ""
    section: str = ""
    table_id: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def lookup_fact(role: RoleSpec, evidence_text: str, *, question: str = "") -> Optional[GroundedFact]:
    """Try to find *role*'s value deterministically in *evidence_text* tables.

    Returns a GroundedFact on an unambiguous match.
    Returns None when no table cell matches (fall through to LLM extraction).
    Raises AmbiguousTableLookup when multiple cells match (caller abstains).
    """
    segment_blocks = _segment_heading_blocks(evidence_text)
    all_cells = _pdf_cells(role, evidence_text)
    all_cells.extend(_segment_pdf_cells(role, segment_blocks))
    from verifiqa.agent.typed_column_grounding import useful_life_cells
    all_cells.extend(_Cell(**record) for record in useful_life_cells(
        role, evidence_text, question=question, row_matches=_row_label_matches))
    for table, title, raw_lines in _pipe_tables_with_context(evidence_text):
        cells = _find_cells(role, table, title=title)
        identity = "pipe:" + hashlib.sha256((title + "\n" + "\n".join(raw_lines)).encode()).hexdigest()[:16]
        for cell in cells:
            cell.source_quote = next((line for line in raw_lines
                                      if "|" in line and line.split("|")[0].strip() == cell.row_label
                                      and cell.raw in line), "")
            cell.unit = ("percent" if "%" in cell.raw else
                         (_caption_quantity_unit(table[0][0])
                          or _caption_money_unit(" ".join(table[0]))))
            cell.table_id = identity
        if _role_uses_share_quantity(role):
            cells = [cell for cell in cells if cell.unit.startswith("shares ")]
        all_cells.extend(cells)

    # A repeated chunk can contain the same source cell, but equal values in
    # different sections/tables are distinct candidates.
    seen: set[tuple] = set()
    unique: list[_Cell] = []
    for c in all_cells:
        key = (c.row_label, c.col_header, c.raw, c.unit, c.section, c.table_id)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if unique and question:
        from verifiqa.types import VerificationFact
        from verifiqa.verification.xbrl_scope import requested_scopes
        # These are mutually exclusive actuarial row groups. An omitted group
        # must not make a remaining wrong-group row look unambiguous.
        pension_scopes = [scope for scope in requested_scopes(VerificationFact(question, 0, ""))
                          if scope[0] in {"net periodic benefit cost", "benefit obligation"}]
        if len(pension_scopes) == 1:
            unique = [c for c in unique if pension_scopes[0] in requested_scopes(
                VerificationFact("", 0, "", row_label=c.row_label, source_section=c.section))]
        # A question may name a printed subsection even when the planner has
        # emitted only its short row label (e.g. two pension "Discount rate"
        # rows). Select only an explicitly named, complete subsection.
        words = _label_tokens(question)
        scoped = [c for c in unique if _question_names_section(words, c.section)]
        if scoped:
            unique = scoped

    if re.search(r"\bconsolidated\b", question, re.I):
        unique = [c for c in unique if not c.section.lower().startswith("segment results")]
    elif segment_blocks:
        from verifiqa.verification.money_units import explicit_money_currencies
        foreign_caption = any(explicit_money_currencies(caption) - {"USD"}
                              for _label, _block, caption in segment_blocks)
        if foreign_caption or any(not c.section.lower().startswith("segment results") for c in unique):
            raise AmbiguousTableLookup(role.name, unique)
    if not unique and segment_blocks and not re.search(r"\bconsolidated\b", question, re.I):
        # An explicit segment table cannot justify a silent fallback to a
        # consolidated registry fact when its cell cannot be reconstructed.
        raise AmbiguousTableLookup(role.name, [])
    if not unique:
        return None

    if len(unique) > 1:
        raise AmbiguousTableLookup(role.name, unique)

    cell = unique[0]
    if not cell.source_quote:
        return None
    return GroundedFact(
        name=role.name,
        value=cell.value,
        unit=cell.unit,
        source_quote=cell.source_quote,
        period=role.period or cell.col_header,
        row_label=cell.row_label,
        source_section=cell.section,
        source_column=cell.col_header,
        source_table_id=cell.table_id,
    )


_PDF_CELL = r"\$?\s*(?:\(\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\)|[-+]?\d[\d,]*(?:\.\d+)?|[—–-])"


def _segment_heading_blocks(evidence_text: str):
    """Locate explicit segment-results headings immediately before a table."""
    headings = list(re.finditer(
        r"\bSEGMENT\s+RESULTS\s*[:—-]?\s*(?P<label>[^\n]{1,100})\n"
        r"(?=\s*(?:Fiscal|Years?\s+Ended|(?:19|20)\d{2})\b)", evidence_text, re.I,
    ))
    blocks = []
    for index, match in enumerate(headings):
        body = evidence_text[match.end():headings[index + 1].start()
                             if index + 1 < len(headings) else len(evidence_text)]
        caption = "\n".join(evidence_text[:match.end()].splitlines()[-3:])
        caption += "\n" + (body.splitlines()[0] if body.splitlines() else "")
        blocks.append((match.group("label").strip(), body, caption))
    return blocks


def _segment_pdf_cells(role: RoleSpec, blocks) -> list[_Cell]:
    """Read explicit annual segment tables with a separate percent-change column."""
    cells = []
    for label, block, _caption in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        header = re.fullmatch(
            r"\s*(?P<years>Fiscal\s+(?:19|20)\d{2}(?:\s+Fiscal\s+(?:19|20)\d{2})+)"
            r"\s+%\s*Change\s*(?P<unit>\(\s*in\s+(?:thousands|millions|billions)\s*\))\s*",
            lines[0], re.I,
        )
        if not header:
            continue
        years = re.findall(r"(?:19|20)\d{2}", header.group("years"))
        wanted = re.findall(r"(?:19|20)\d{2}", role.period)
        if len(set(years)) != len(years) or len(wanted) != 1 or wanted[0] not in years:
            continue
        section = "Segment results: " + label
        unit = _caption_money_unit(header.group("unit"))
        table_id = "segment:" + hashlib.sha256((section + "\n" + block).encode()).hexdigest()[:16]
        numeric = _PDF_CELL + r"(?:\s*%)?"
        row = re.compile(
            r"\s*(?P<label>[A-Za-z][A-Za-z &'’/()-]*?)\s+(?P<values>"
            + numeric + r"(?:\s+" + numeric + r"){" + str(len(years)) + r"})"
            r"(?=\s*$|\s+[A-Za-z])", re.I,
        )
        for line in lines[1:]:
            remaining = line
            while remaining.strip():
                match = row.match(remaining)
                if not match:
                    break
                raws = [m.group().strip() for m in re.finditer(numeric, match.group("values"))]
                if any("%" in raw for raw in raws[:-1]):
                    break  # The annual monetary cells cannot be change percentages.
                if _row_label_matches(role, match.group("label")):
                    raw = raws[years.index(wanted[0])]
                    value = _parse_number(raw)
                    if value is not None:
                        cells.append(_Cell(match.group("label"), "Fiscal " + wanted[0], raw, value,
                                           match.group().strip(), unit, section, table_id))
                remaining = remaining[match.end():]
            if remaining.strip():
                break  # A narrative paragraph ends this grid.
    return cells


def _caption_money_unit(text: str) -> str:
    from verifiqa.verification.money_units import explicit_money_currencies, unrecognized_currency_unit
    if explicit_money_currencies(text) - {"USD"}:
        return ""  # This reader supplies USD units only; no implicit FX.
    codes = re.findall(r"\b([A-Za-z]{3})\s*\$?\s*\(?\s*(?:in\s+)?(?:thousand|million|billion)s?\b"
                       r"|\b(?:thousand|million|billion)s?\s+(?:of\s+)?([A-Za-z]{3})\b", text, re.I)
    if any(unrecognized_currency_unit(code.upper())
           for pair in codes for code in pair if code and code.lower() != "all"):
        return ""
    match = re.search(r"\b(?:in|amounts in|figures in)\s+(?:usd\s+)?(thousands|millions|billions)\b", text, re.I)
    if match:
        return "USD " + match.group(1).lower()
    if re.search(r"\bUSD\s*[‘’']?000\b", text, re.I):
        return "USD thousands"
    if re.fullmatch(r"\s*\(?\s*(?:(?:all\s+)?(?:amounts|figures)\s+)?(?:in\s+)?"
                    r"(?:USD|U\.?\s*S\.?\s+dollars?|dollars?)\s*\)?\s*", text, re.I):
        return "USD"
    return ""


def _caption_quantity_unit(text: str) -> str:
    match = re.fullmatch(r"\s*\(?(million(?:s)?|thousand(?:s)?)\s+shares\)?\s*", text, re.I)
    if match:
        scale = match.group(1).lower().rstrip("s") + "s"
        return "shares " + scale
    return ""


def _role_uses_share_quantity(role: RoleSpec) -> bool:
    labels = " ".join(role.aliases or [role.name]).lower().replace("_", " ")
    return bool(re.search(r"\bshares\b|\bshare count\b", labels)) and "per share" not in labels


def _quantity_pdf_cells(role: RoleSpec, evidence_text: str, target: str, scopes: list[str]) -> list[_Cell]:
    """Read share quantities only under an explicit share-scale caption.

    A caption may follow the dated header and a monetary row in an earnings
    table. Only complete rows or the known section heading may intervene.
    A wrapped requested label must match exactly after joining its two lines.
    """
    from verifiqa.types import VerificationFact
    from verifiqa.verification.xbrl_scope import requested_scopes

    def year_header(text: str) -> list[str]:
        if not re.fullmatch(r"\s*(?:19|20)\d{2}(?:\s+(?:19|20)\d{2})+\s*", text):
            return []
        years = re.findall(r"(?:19|20)\d{2}", text)
        return years if len(years) == len(set(years)) else []

    def complete_row(text: str, count: int):
        match = re.fullmatch(
            r"\s*(?P<label>.+?)\s+(?P<values>" + _PDF_CELL + r"(?:\s+" + _PDF_CELL
            + r"){" + str(count - 1) + r"})\s*", text,
        )
        if match and re.search(r"\d", re.sub(r"\([^)]*\)", "", match.group("label"))):
            return None
        return match

    lines = evidence_text.splitlines()
    cells = []
    for index, line in enumerate(lines):
        unit = _caption_quantity_unit(line)
        if not unit:
            continue
        headers = year_header(lines[index + 1]) if index + 1 < len(lines) else []
        start = index + 2
        if not headers:
            # The nearest preceding dated line must itself be an annual header.
            header_index = next((j for j in range(index - 1, -1, -1)
                                 if re.search(r"(?:19|20)\d{2}", lines[j])), None)
            if header_index is None:
                continue
            headers = year_header(lines[header_index])
            if not headers:
                continue
            between = lines[header_index + 1:index]
            if any(text.strip() and text.strip().upper() != "EARNINGS PER SHARE"
                   and complete_row(text, len(headers)) is None for text in between):
                continue
            start = index + 1
        if target not in headers:
            continue
        cursor = start
        while cursor < len(lines):
            quote = lines[cursor]
            match = complete_row(quote, len(headers))
            consumed = 1
            if match is None and cursor + 1 < len(lines):
                # A line wrap may extend only an exact requested row, never an
                # arbitrary heading, narrative sentence, or partial cell group.
                joined = quote + " " + lines[cursor + 1]
                candidate = complete_row(joined, len(headers))
                if candidate and _row_label_matches(role, candidate.group("label")):
                    match = candidate
                    quote += "\n" + lines[cursor + 1]
                    consumed = 2
            if match is None or re.search(r"\bUSD\b|USDm|\$|per share", match.group("label"), re.I):
                break
            row_label = match.group("label").strip()
            if _row_label_matches(role, row_label):
                row_scopes = requested_scopes(VerificationFact("", 0, "", row_label=row_label))
                if not any(scope not in row_scopes for scope in scopes):
                    raws = [m.group().strip() for m in re.finditer(_PDF_CELL, match.group("values"))]
                    raw = raws[headers.index(target)]
                    value = _parse_number(raw)
                    if value is not None:
                        cells.append(_Cell(row_label, target, raw, value, quote, unit))
            cursor += consumed
    return cells


def _inline_captioned_grid(line: str) -> Optional[tuple[list[str], str, list[tuple[str, list[str]]]]]:
    """Parse an entire annual grid flattened onto its caption/header line.

    The full grid must have a complete cell group for every named row. A bare
    numeric footer is accepted only if it reconciles to all preceding rows;
    otherwise extra columns would be indistinguishable from a total row.
    """
    header = re.fullmatch(
        r"\s*(?P<caption>\(?(?:in|amounts in|figures in)\s+(?:USD\s+)?"
        r"(?:thousands|millions|billions)(?:\s+of\s+\$)?\)?)\s+"
        r"(?P<years>(?:19|20)\d{2}(?:\s+(?:19|20)\d{2})+)\s+(?P<body>.+)",
        line, re.I,
    )
    if not header:
        return None
    years = re.findall(r"(?:19|20)\d{2}", header.group("years"))
    if len(years) != len(set(years)):
        return None
    count = len(years)
    values_pattern = _PDF_CELL + r"(?:\s+" + _PDF_CELL + r"){" + str(count - 1) + r"}"
    # Permit short printed footnote markers and "see note N" in labels. A
    # parenthesized first value remains a value when N complete cells follow
    # the label, rather than N additional cells following a footnote marker.
    label_pattern = r"[A-Za-z](?:[A-Za-z\s.,/&:'’_-]|\(\s*(?:see\s+note\s+)?\d{1,2}\s*\))*?"
    row_pattern = re.compile(
        r"\s*(?P<label>" + label_pattern + r")\s+(?P<values>" + values_pattern + r")"
        r"(?=\s*$|\s+(?:[A-Za-z]|" + values_pattern + r"\s*$))", re.I,
    )
    remaining = header.group("body")
    rows: list[tuple[str, list[str]]] = []
    while remaining.strip():
        row = row_pattern.match(remaining)
        if row is None:
            footer = re.fullmatch(r"\s*(" + values_pattern + r")\s*", remaining)
            if footer is None or len(rows) < 2:
                return None
            totals = [m.group().strip() for m in re.finditer(_PDF_CELL, footer.group(1))]

            def printed_decimal(raw: str) -> Optional[Decimal]:
                text = raw.replace("$", "").replace(",", "").strip()
                if text in {"—", "–", "-"}:
                    return None
                negative = text.startswith("(") and text.endswith(")")
                number = Decimal(text.strip("() "))
                return -abs(number) if negative else number

            # Missing cells never become facts. Summing the reported numeric
            # cells here only checks whether the bare footer is really a total.
            for col, total in enumerate(totals):
                printed_total = printed_decimal(total)
                if printed_total is None:
                    return None
                reported_sum = sum(
                    (printed_decimal(raws[col]) or Decimal(0) for _, raws in rows),
                    Decimal(0),
                )
                if reported_sum != printed_total:
                    return None
            break
        label = re.sub(r"\s*\(\s*\d{1,2}\s*\)\s*$", "", row.group("label")).strip()
        raws = [m.group().strip() for m in re.finditer(_PDF_CELL, row.group("values"))]
        if len(raws) != count:
            return None
        rows.append((label, raws))
        remaining = remaining[row.end():]
    return years, _caption_money_unit(header.group("caption")), rows


def _label_tokens(text: str) -> tuple[str, ...]:
    return tuple(t for t in re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
                 if not re.fullmatch(r"(?:19|20)\d{2}", t))


def _question_names_section(words: tuple[str, ...], section: str) -> bool:
    # The final component is an actual printed subgroup, not a paragraph whose
    # many incidental words might happen to occur somewhere in the question.
    tokens = _label_tokens(section.split(" / ")[-1])
    return len(tokens) >= 2 and any(words[i:i + len(tokens)] == tokens
                                   for i in range(len(words) - len(tokens) + 1))


def _printed_heading(text: str) -> str:
    """A prose caption is table context, but not a cell's entity/group scope."""
    text = text.strip()
    if (not text or len(text.split()) > 12 or not re.search(r"[A-Za-z]", text)
            or re.search(r"[.!?]|\b(?:following|table|reconciliation|summarizes|details|"
                         r"comprised|consisted|represents|consists|provides)\b", text, re.I)):
        return ""
    without_note = re.sub(r"^\s*(?:Note\s*)?\d+[.)]?\s*", "", text, flags=re.I)
    if re.search(r"\d", without_note):
        return ""
    return text


_MONTH = (r"January|February|March|April|May|June|July|August|September|October|"
          r"November|December")
_MONEY_CAPTION = (r"\(?(?:(?:amounts|figures|dollars)\s+)?in\s+(?:USD\s+)?"
                  r"(?:thousands|millions|billions)(?:\s+of\s+\$)?"
                  r"(?:\s*,\s*except\s+(?:for\s+)?per\s+(?:common\s+stock|share)"
                  r"\s+(?:amounts|data))?\s*\)?")
_ANNUAL_HEADER = re.compile(
    r"^\s*(?:" + _MONEY_CAPTION + r"\s*)?"
    r"(?P<prefix>(?:(?:for\s+the\s+)?(?:fiscal\s+)?years?\s+(?:ended|ending)|"
    r"as\s+of|fiscal\s+year\s+end|fiscal)?\s*(?:(?:" + _MONTH + r")\s+\d{1,2}\s*,?\s*)?)"
    r"(?P<years>(?:19|20)\d{2}(?:\s*\(\d{1,2}\))?"
    r"(?:\s+(?:19|20)\d{2}(?:\s*\(\d{1,2}\))?)+)"
    r"(?P<body>\s+.*)?$", re.I,
)


def _annual_pdf_header(line: str):
    """Return only explicit, unique annual columns and any flattened body."""
    dates = re.fullmatch(
        r"\s*(?:(?:" + _MONTH + r")\s+\d{1,2},?\s+(?:19|20)\d{2}\s*){2,}", line, re.I)
    if dates:
        years = re.findall(r"(?:19|20)\d{2}", line)
        month_days = re.findall(r"(?:" + _MONTH + r")\s+\d{1,2},?", line, re.I)
        if len(years) == len(set(years)):
            return years, dict(zip(years, [day + " " + year
                                          for day, year in zip(month_days, years)])), ""
        return None
    matched = _ANNUAL_HEADER.fullmatch(line)
    if matched is None:
        return None
    years = re.findall(r"(?:19|20)\d{2}", matched.group("years"))
    if len(years) != len(set(years)):
        return None
    return years, matched.group("prefix").strip(), (matched.group("body") or "").strip()


def _join_split_annual_headers(lines: list[str]) -> list[str]:
    """Join only consecutive printed month/day + year column header pairs."""
    out, index = [], 0
    day = re.compile(r"\s*(?:" + _MONTH + r")\s+\d{1,2},?\s*", re.I)
    year = re.compile(r"\s*(?:19|20)\d{2}\s*")
    while index < len(lines):
        end = index
        pairs = []
        while end + 1 < len(lines) and day.fullmatch(lines[end]) and year.fullmatch(lines[end + 1]):
            pairs.extend((lines[end].strip(), lines[end + 1].strip()))
            end += 2
        if len(pairs) >= 4:
            out.append(" ".join(pairs))
            index = end
        else:
            out.append(lines[index])
            index += 1
    return out


def _subsection(text: str) -> str:
    """Recognize compact accounting row groups, never narrative/table resets."""
    text = text.strip().rstrip(":").strip()
    if (not text or len(text.split()) > 9 or re.search(r"\d|[.!?]", text)
            or re.search(r"\b(?:table|following|this|these|new|undated)\b", text, re.I)):
        return ""
    if re.search(r"\b(?:assets|liabilities|operations|cost|obligation|profit|expenses|"
                 r"adjustments|securities|numerator|denominator)\b", text, re.I):
        return text
    return ""


def _annual_row_groups(text: str, count: int):
    """Parse complete rows, including several rows flattened onto one line.

    An extra/missing numeric cell invalidates the whole line. Bare footer totals
    stay unsupported here; the separate inline-caption parser reconciles those.
    """
    numeric = _PDF_CELL + r"(?:\s*%)?"
    values = numeric + r"(?:\s+" + numeric + r"){" + str(count - 1) + r"}"
    row_pattern = re.compile(
        r"\s*(?P<label>.+?)\s+(?P<values>" + values + r")"
        r"(?=\s*$|\s+(?:[A-Za-z]|\d[\d./]*\s+[A-Za-z]))", re.I,
    )
    rows = []
    remaining = text
    while remaining.strip():
        matched = row_pattern.match(remaining)
        if matched is None:
            return None
        label = matched.group("label").strip().rstrip(". ")
        # Printed footnote markers can qualify labels; they are not cells.
        label = re.sub(r"(?:\s*\((?:see\s+note\s+)?\d{1,2}\))+\s*$", "", label, flags=re.I).strip()
        # Do not reclassify an extra leading value as part of the label.
        if (not re.search(r"[A-Za-z]", label)
                or (re.search(r"\d\s*$", label) and not re.search(
                    r"\b(?:" + _MONTH + r")\s+\d{1,2}\s*$", label, re.I))):
            return None
        if re.match(r"\d", label) and not re.match(
                r"\d+(?:[./]\d+)*\s+(?:nanometers?|microns?|nm)\b", label, re.I):
            return None
        section = ""
        if ":" in label:
            prefix, suffix = label.rsplit(":", 1)
            if prefix.strip().lower() not in {"less", "add"}:
                section = _subsection(prefix)
                if not section or not suffix.strip():
                    return None
                label = suffix.strip()
        raws = [m.group().strip() for m in re.finditer(numeric, matched.group("values"))]
        if len(raws) != count or (any("%" in raw for raw in raws)
                                  and not all("%" in raw for raw in raws)):
            return None
        rows.append((section, label, raws))
        remaining = remaining[matched.end():]
    return rows or None


def _pdf_cells(role: RoleSpec, evidence_text: str) -> list[_Cell]:
    """Read complete, explicitly dated PDF rows with a declared monetary scale.

    No nearest-year selection, inferred totals, or incomplete/multi-level grids.
    Unclear layouts remain the responsibility of the LLM extractor.
    """
    # Only complete annual grids are supported here. Quarterly/date-specific
    # operands require a matching structured source or text extraction.
    period_years = re.findall(r"(?:19|20)\d{2}", role.period or role.name)
    if len(set(period_years)) != 1 or re.search(r"\bq[1-4]\b|quarter|\d{4}-\d{2}-\d{2}|january|february|march|april|may|june|july|august|september|october|november", role.period, re.I):
        return []
    from verifiqa.types import VerificationFact
    from verifiqa.verification.xbrl_scope import requested_scopes
    scopes = requested_scopes(VerificationFact(role.name, 0, "", row_label="; ".join(role.aliases)))
    target = period_years[0]
    def normalized(text):
        text = re.sub(r"\b(?:fy)?(?:19|20)\d{2}\b", "", text.lower().replace("_", " "))
        return tuple(re.findall(r"[a-z0-9]+", text))
    labels = {normalized(alias) for alias in (role.aliases or [role.name])}
    if _role_uses_share_quantity(role):
        return _quantity_pdf_cells(role, evidence_text, target, scopes)
    if any(re.search(r"per share|number of shares|weighted average|share count", " ".join(label)) for label in labels):
        return []
    headers, unit, cells = [], "", []
    title, section, column_prefix, table_id = "", "", "", ""
    page_unit = ""
    lines = _join_split_annual_headers(evidence_text.splitlines())
    for line_index, line in enumerate(lines):
        if "|" in line:
            headers = []
            continue
        inline_grid = _inline_captioned_grid(line)
        if inline_grid is not None:
            grid_years, grid_unit, grid_rows = inline_grid
            headers, unit, section = grid_years, grid_unit, ""
            column_prefix = ""
            table_id = "pdf:" + hashlib.sha256(
                (title + "\n" + " ".join(headers) + "\n" + unit).encode()).hexdigest()[:16]
            if target not in grid_years:
                continue
            for row_label, raws in grid_rows:
                if not _row_label_matches(role, row_label):
                    continue
                row_scopes = requested_scopes(VerificationFact("", 0, "", row_label=row_label))
                if any(scope not in row_scopes for scope in scopes):
                    continue
                raw = raws[grid_years.index(target)]
                value = _parse_number(raw)
                if value is not None:
                    cells.append(_Cell(row_label, target, raw, value, line, grid_unit,
                                       _printed_heading(title), table_id))
            continue
        declared = _caption_money_unit(line)
        if declared and re.fullmatch(
                r"\s*\(?Tabular\s+amounts\s+in\s+(?:thousands|millions|billions)"
                r",\s*unless\s+otherwise\s+(?:disclosed|noted|indicated)\)?\s*", line, re.I):
            page_unit = declared
        header = _annual_pdf_header(line)
        if header is not None:
            headers, prefix, body = header
            section = ""
            column_prefix = prefix
            if not prefix and line_index and re.fullmatch(
                    r"\s*(?:(?:Years?\s+Ended|As\s+of)\s+)?(?:" + _MONTH + r")\s+\d{1,2},?\s*",
                    lines[line_index - 1], re.I):
                column_prefix = lines[line_index - 1].strip()
            unit = declared or unit or page_unit
            table_id = "pdf:" + hashlib.sha256(
                (title + "\n" + " ".join(headers) + "\n" + unit).encode()).hexdigest()[:16]
            if not body:
                continue
            line_body = body
        elif declared:
            unit = declared
            # Only a standalone caption can retain an already printed header.
            # Long prose containing a scale starts a new context.
            if re.fullmatch(r"\s*" + _MONEY_CAPTION + r"\s*", line, re.I):
                continue
            if not headers or not re.match(r"\s*" + _MONEY_CAPTION, line, re.I):
                headers = []
                title = line.strip()
                continue
            line_body = re.sub(r"^\s*" + _MONEY_CAPTION + r"\s*", "", line, flags=re.I)
        else:
            line_body = line
        if not headers:
            # Captions can be followed by a date line or a compact heading.
            if line.strip() and not re.fullmatch(
                    r"\s*(?:(?:Years?\s+Ended|As\s+of)\s+)?(?:" + _MONTH + r")\s+\d{1,2},?\s*",
                    line, re.I):
                if (len(line.split()) <= 12 and not re.search(r"[.!?]", line)
                        and not re.search(r"\d{4}", line)):
                    title = line.strip()
                elif not declared:
                    unit = ""
            continue
        subsection = _subsection(line_body)
        if subsection and (line_body.strip().endswith(":") or line_body.strip() in {
                "Numerator", "Denominator", "Deferred tax assets", "Deferred tax liabilities"}):
            section = subsection
            continue
        if not line_body.strip():
            continue
        rows = _annual_row_groups(line_body, len(headers))
        if rows is None:
            headers, unit = [], ""
            if (len(line_body.split()) <= 12 and re.search(r"[A-Za-z]", line_body)
                    and not re.search(r"[.!?]|\d{4}", line_body)):
                title = line_body.strip()
            continue
        if target not in headers:
            continue
        for row_section, row_label, raw_cells in rows:
            if row_section:
                section = row_section
            full_section = " / ".join(part for part in (_printed_heading(title), section) if part)
            if not (_row_label_matches(role, row_label)
                    or (section and _row_label_matches(role, section + " " + row_label))):
                continue
            row_scopes = requested_scopes(VerificationFact(
                "", 0, "", row_label=row_label, source_section=full_section))
            if any(scope not in row_scopes for scope in scopes):
                continue
            raw = raw_cells[headers.index(target)]
            row_unit = "percent" if "%" in raw else unit
            if not row_unit:
                continue
            value = _parse_number(raw)
            if value is not None:
                column = (column_prefix.get(target, target) if isinstance(column_prefix, dict)
                          else (column_prefix + " " + target).strip())
                cells.append(_Cell(row_label, column, raw, value, line, row_unit,
                                   full_section, table_id))
    return cells


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------

def _parse_tables(evidence_text: str) -> list[list[list[str]]]:
    """Return a list of tables found in *evidence_text*.

    Each table is a list of rows; each row is a list of cell strings.
    Only blocks with ≥ 2 pipe-separated rows are treated as tables.
    """
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []

    for raw_line in (evidence_text or "").split("\n"):
        line = raw_line.strip()
        if "|" in line:
            current.append([c.strip() for c in line.split("|")])
        else:
            if len(current) >= 2:
                tables.append(current)
            current = []

    if len(current) >= 2:
        tables.append(current)

    return tables


def _pipe_tables_with_context(evidence_text: str):
    """Keep the literal row block and its preceding printed heading together."""
    rows, raw_lines, title = [], [], ""
    for line in (evidence_text or "").splitlines():
        if "|" in line:
            rows.append([cell.strip() for cell in line.strip().split("|")])
            raw_lines.append(line)
        else:
            if len(rows) >= 2:
                yield rows, title, raw_lines
            rows, raw_lines = [], []
            if line.strip():
                title = line.strip()
    if len(rows) >= 2:
        yield rows, title, raw_lines


# ---------------------------------------------------------------------------
# Cell matching
# ---------------------------------------------------------------------------

def _find_cells(role: RoleSpec, table: list[list[str]], *, title: str = "") -> list[_Cell]:
    """Return cells in *table* matching *role* by row label and column year."""
    if len(table) < 2:
        return []

    headers = table[0]          # First row = column headers
    data_rows = table[1:]       # Remaining rows = data

    # Find which column indices contain the role's requested year.
    col_indices = _matching_col_indices(role.period, headers)
    if not col_indices:
        return []

    # Find which data rows match the role's aliases.
    from verifiqa.types import VerificationFact
    from verifiqa.verification.xbrl_scope import requested_scopes

    required_scopes = requested_scopes(VerificationFact(
        role.name, 0, "", row_label="; ".join(role.aliases)))
    cells: list[_Cell] = []
    section = ""
    for row in data_rows:
        if not row:
            continue
        row_label = row[0] if row else ""
        if not any(cell.strip() for cell in row[1:]) and _subsection(row_label):
            section = _subsection(row_label)
            continue
        if not (_row_label_matches(role, row_label)
                or (section and _row_label_matches(role, section + " " + row_label))):
            continue
        full_section = " / ".join(part for part in (_printed_heading(title), section) if part)
        row_scopes = requested_scopes(VerificationFact(
            "", 0, "", row_label=row_label, source_section=full_section))
        if any(scope not in row_scopes for scope in required_scopes):
            continue
        for col_idx in col_indices:
            if col_idx >= len(row):
                continue
            raw = row[col_idx]
            value = _parse_number(raw)
            if value is None:
                continue
            col_header = headers[col_idx] if col_idx < len(headers) else ""
            cells.append(_Cell(
                row_label=row_label,
                col_header=col_header,
                raw=raw,
                value=value,
                section=full_section,
            ))

    return cells


# ---------------------------------------------------------------------------
# Column matching: period year → column indices
# ---------------------------------------------------------------------------

def _matching_col_indices(period: str, headers: list[str]) -> list[int]:
    """Return header indices whose year matches the year in *period*.

    Returns an empty list when *period* has no 4-digit year (caller falls
    through to LLM) or when no header contains that year.
    """
    from verifiqa.verification.periods import period_key, quarter_number

    target = period_key(period)
    if target is None:
        return []
    requested_quarter = quarter_number(period)

    matching = []
    for i, header in enumerate(headers):
        if i == 0:
            continue  # First cell is the row-label column, never a data column
        source = period_key(header)
        if source is None or source[0] != target[0]:
            continue
        source_quarter = quarter_number(header)
        if requested_quarter != source_quarter and (requested_quarter or source_quarter):
            continue
        if target[1] > 0 and source[1] != target[1]:
            continue
        if target[2] > 0 and source[2] != target[2]:
            continue
        matching.append(i)
    return matching


# ---------------------------------------------------------------------------
# Row matching: role aliases → row label
# ---------------------------------------------------------------------------

# Matches only 4-digit calendar years like 2013, 1998 — NOT 2-digit or 3-digit numbers.
_CALENDAR_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def _row_label_matches(role: RoleSpec, row_label: str) -> bool:
    """Match complete labels while retaining classes, totals, and qualifiers."""
    if not row_label:
        return False
    def tokens(text):
        return tuple(t for t in re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
                     if not _CALENDAR_YEAR_RE.match(t))
    label_tokens = tokens(row_label)
    if not label_tokens:
        return False

    # Prefer explicit aliases; fall back to role name only when no aliases given.
    candidates: list[str] = list(role.aliases) if role.aliases else [role.name.replace("_", " ")]

    for alias in candidates:
        alias_tokens = tokens(alias)
        if alias_tokens and alias_tokens == label_tokens:
            return True

    return False


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

def _parse_number(raw: str) -> Optional[float]:
    """Parse a numeric value from a raw cell string, or None if not parseable.

    Handles accounting formats: "$ -2207 ( 2207 )" → -2207, "( 1234 )" → -1234.
    Strategy: extract the FIRST signed numeric token; if the raw string is purely
    parenthetical (no explicit sign), treat it as negative.
    """
    # A percent suffix is a unit marker; it must not conceal accounting
    # parentheses when deciding the sign, e.g. '(15)%' is negative fifteen.
    cleaned = raw.replace("$", "").replace(",", "").replace("%", "")
    # Find the first numeric token (optional sign; handles "1.8", ".3", "2207")
    m = re.search(r"(-?\s*(?:\d[\d]*(?:\.\d+)?|\.\d+))", cleaned)
    if not m:
        return None
    token = m.group(1).replace(" ", "")
    try:
        value = float(token)
    except ValueError:
        return None
    # If the ENTIRE raw string is a bare parenthetical like "( 1234 )" or "(1234)"
    # with no leading minus, that is the standard accounting negative notation.
    if value > 0 and re.fullmatch(r"\s*\(\s*[\d,]+(?:\.\d+)?\s*\)\s*", cleaned.strip()):
        value = -value
    return value
