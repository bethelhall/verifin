"""Explicit PDF column layouts omitted by the simple annual-grid reader.

These readers retain printed column order and units.  They never inspect a
candidate answer or use a reported change as a substitute for annual operands.
Unsupported widths, missing selected cells and competing tables stay ambiguous.
"""
from __future__ import annotations

import re


_YEAR = r"(?:19|20)\d{2}"
_CELL = (r"(?:\$\s*)?(?:\(\s*-?\d[\d,]*(?:\.\d+)?\s*\)"
         r"|-?\d[\d,]*(?:\.\d+)?|[—–-])(?:\s*%)?")
_MONTH = (r"January|February|March|April|May|June|July|August|September|October|"
          r"November|December")
_DATE = r"(?:" + _MONTH + r")\s+\d{1,2},?\s+" + _YEAR


def supplemental_tables(evidence: str):
    """Return independently typed annual operands from exact source layouts."""
    tables = []
    lines = evidence.splitlines()
    for index, line in enumerate(lines):
        comparison = _comparison_table(line, lines, index)
        if comparison is not None:
            tables.append(comparison)
        directed = _dated_directive_rows(line, lines, index)
        if directed is not None:
            tables.append(directed)
        declared = _declared_scale_table(line, lines, index)
        if declared is not None:
            tables.append(declared)
    return tables


def _context(lines, index):
    """Keep the nearest caption, including scope-bearing prose, for validation."""
    from verifiqa.agent.table_fact_extractor import _caption_money_unit
    sections, unit = [], ""
    for line in lines[max(0, index - 4):index]:
        if not line.strip() or line.strip().isdigit() or line.strip() == "Table of Contents":
            continue
        declared = _caption_money_unit(line)
        if declared:
            unit = declared
        elif re.search(r"[.!?]", line):
            # A different prose paragraph cannot donate an earlier table's
            # scale to a newly encountered grid.
            unit = ""
        sections.append(line.strip())
    return " / ".join(sections), unit


def _comparison_table(line, lines, index):
    from verifiqa.agent.source_question_planner import _Table, _Row, _identity, _number, source_row_unit
    from verifiqa.agent.table_fact_extractor import _caption_money_unit

    # Examples: 'December 31, (in thousands of $) 2018 2017 Change % Change'
    # and 'Year Ended December 31, Change 2018 2017 Percent (in thousands)'.
    header = re.match(
        r"^\s*(?P<prefix>(?:(?:Years?\s+Ended|As\s+of|Fiscal)\s+)?"
        r"(?:(?:" + _MONTH + r")\s+\d{1,2},?\s*)?"
        r"(?:\(in\s+(?:thousands|millions|billions)(?:\s+of\s+\$)?\)\s*)?"
        r"(?:(?:Change|Percent)\s+)?)"
        r"(?P<years>" + _YEAR + r"(?:\s+" + _YEAR + r")+)\s+"
        r"(?P<tail>.+)$", line, re.I)
    if not header:
        return None
    prefix, tail = header.group("prefix"), header.group("tail")
    years = re.findall(_YEAR, header.group("years"))
    if len(years) != len(set(years)):
        return None
    labels = re.match(r"Change\s+%\s*Change\b", tail, re.I)
    if labels:
        extra = 2
    else:
        labels = re.match(r"(?:Percent\s+Change|Change\s+Percent|Percent|Change)\b", tail, re.I)
        extra = 1
    if not labels:
        return None
    body = tail[labels.end():].strip()
    # A third column called 'Change' without an explicit percent label is a
    # monetary difference. No such column may masquerade as another year.
    percent_last = bool(re.search(r"%|percent", prefix + " " + labels.group(), re.I))
    caption = re.match(r"\(in\s+(?:thousands|millions|billions)(?:\s+of\s+\$)?\)\s*", body, re.I)
    unit = _caption_money_unit(prefix + " " + (caption.group() if caption else ""))
    if caption:
        body = body[caption.end():]
    section, prior_unit = _context(lines, index)
    unit = unit or prior_unit
    width = len(years) + extra
    if not body:
        # A complete printed header may be followed by one complete row per
        # line. Stop at a narrative/header boundary rather than search ahead.
        source_rows = []
        for following in lines[index + 1:]:
            if not following.strip():
                continue
            if _row_groups(following, width) is None:
                break
            source_rows.append((following, following))
    else:
        source_rows = [(body, line)]
    if not source_rows:
        return None
    table = _Table(years, section=section, unit=unit,
                   identity=_identity("\n".join(quote for _, quote in source_rows), line[:header.end("years")]),
                   unit_declared=bool(unit))
    subsection = ""
    for body, quote in source_rows:
        rows = _row_groups(body, width)
        if rows is None:
            return None
        for label, raws in rows:
            # The final printed comparison column must agree with its header
            # type; mixed-width percentage subrows cannot enter this schema.
            if percent_last and "%" not in raws[-1]:
                return None
            if any("%" in raw for raw in raws[:len(years)]):
                return None
            if not percent_last and any("%" in raw for raw in raws[len(years):]):
                return None
            # These are explicit row-group headings in operating-data tables.
            # Their counts must not inherit ARPU's currency or become revenue.
            group = re.search(r"\bCustomer\s+Connections\s*[—–-]\s*end\s+of\s+period\s+", label, re.I)
            if group:
                subsection = group.group().strip()
                label = label[group.end():].strip()
            annual_raws = raws[:len(years)]
            row_unit = "count" if subsection else source_row_unit(label, "", unit, annual_raws)
            numbers = [_number(raw) for raw in annual_raws]
            table.rows.append(_Row(label, numbers, annual_raws, quote, subsection, row_unit))
    return table if table.rows else None


def _row_groups(text, width):
    """Read an entire mixed-unit row block with the explicitly declared width."""
    values = _CELL + r"(?:\s+" + _CELL + r"){" + str(width - 1) + r"}"
    row = re.compile(r"\s*(?P<label>.+?)\s+(?P<values>" + values + r")"
                     r"(?=\s*$|\s+[A-Za-z]|\s+" + values + r"\s*$)", re.I)
    result, remaining = [], text
    while remaining.strip():
        match = row.match(remaining)
        if match is None:
            # A bare table footer is unlabelled and cannot be an operand. It
            # is allowed only as exactly one complete final cell group.
            if result and re.fullmatch(r"\s*" + values + r"\s*", remaining):
                break
            return None
        label = match.group("label").strip()
        label = re.sub(r"\s*\(\s*\d{1,2}\s*\)\s*$", "", label)
        if not re.search(r"[A-Za-z]", label) or re.search(r"\d", re.sub(r"\(\s*\d{1,2}\s*\)", "", label)):
            return None
        raws = [m.group().strip() for m in re.finditer(_CELL, match.group("values"))]
        if len(raws) != width:
            return None
        result.append((label, raws))
        remaining = remaining[match.end():]
    return result or None


def _dated_directive_rows(line, lines, index):
    """Read an explicitly bounded 'Less: measure' row after exact dated columns.

    Debt descriptions earlier in the same flattened table contain interest
    rates and installment amounts. They are never used to infer row width or
    selected facts. The directive's complete adjacent cells supply that row.
    """
    from verifiqa.agent.source_question_planner import _Table, _Row, _identity, _number
    from verifiqa.agent.table_fact_extractor import _caption_money_unit
    header = re.match(r"^\s*(?P<dates>(?:" + _DATE + r"\s*){2,})(?P<body>.+)$", line, re.I)
    if not header:
        return None
    dates = [m.group().strip() for m in re.finditer(_DATE, header.group("dates"), re.I)]
    years = [re.findall(_YEAR, date)[0] for date in dates]
    if len(years) != len(set(years)):
        return None
    section, unit = _context(lines, index)
    if not unit:
        return None
    cells = _CELL + r"(?:\s+" + _CELL + r"){" + str(len(dates) - 1) + r"}"
    row = re.compile(r"\bLess:\s*(?P<label>[A-Za-z][A-Za-z /-]*?)\s+"
                     r"(?P<values>" + cells + r")(?=\s*$|\s+[A-Za-z])", re.I)
    found = []
    for match in row.finditer(header.group("body")):
        raws = [m.group().strip() for m in re.finditer(_CELL, match.group("values"))]
        if len(raws) != len(dates) or any("%" in raw for raw in raws):
            continue
        found.append(_Row(match.group("label").strip(), [_number(raw) for raw in raws],
                          raws, line, unit=unit))
    if not found:
        return None
    return _Table(dates, found, section, unit, _identity(line, section), unit_declared=True)


def _declared_scale_table(line, lines, index):
    """Read the explicit 'All figures in USD ‘000 YEAR YEAR' grid layout."""
    from verifiqa.agent.source_question_planner import _Table, _Row, _identity, _number, source_row_unit
    from verifiqa.agent.table_fact_extractor import _annual_row_groups
    header = re.fullmatch(r"\s*All\s+figures\s+in\s+USD\s*[‘’']?000\s+"
                          r"(?P<years>" + _YEAR + r"(?:\s+" + _YEAR + r")+)\s*", line, re.I)
    if not header:
        return None
    years = re.findall(_YEAR, header.group("years"))
    if len(years) != len(set(years)):
        return None
    section, _unit = _context(lines, index)
    table = _Table(years, section=section, unit="USD thousands",
                   identity=_identity(line, section + "|" + str(index)), unit_declared=True)
    for quote in lines[index + 1:]:
        parsed = _annual_row_groups(quote, len(years))
        if parsed is None:
            break
        for group, label, raws in parsed:
            # 'as of December 31' is a date qualifier on the printed total,
            # rather than part of its financial measure identity. The literal
            # qualifier remains in the quote; the header selects each year.
            if re.fullmatch(r"Total\s+as\s+of\s+(?:" + _MONTH + r")\s+\d{1,2},?", label, re.I):
                label = "Total"
            table.rows.append(_Row(label, [_number(raw) for raw in raws], raws, quote, group,
                                   source_row_unit(label, group, table.unit, raws)))
    return table if table.rows else None
