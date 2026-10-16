"""Conservative arithmetic plans selected from the question and source tables.

This module has deliberately no answer, benchmark identifier, or gold argument.
It selects labels, columns, and complete populations before constructing a
formula. Unsupported layouts and competing interpretations return ``None``.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from verifiqa.agent.types import ClaimSpec, GroundedFact, RoleSpec
from verifiqa.answer_spec import answer_spec_from_question


@dataclass
class _Row:
    label: str
    values: list[float]
    raws: list[str]
    quote: str
    section: str = ""
    unit: str = ""


@dataclass
class _Table:
    columns: list[str]
    rows: list[_Row] = field(default_factory=list)
    section: str = ""
    unit: str = ""
    identity: str = ""
    unit_declared: bool = False
    relative_scale_only: bool = False


@dataclass
class _Plan:
    spec: ClaimSpec
    facts: dict[str, GroundedFact]


def plan_from_source(question: str, evidence_text: str, *, source_pdf_path: str = "") -> ClaimSpec | None:
    """Return a uniquely source-supported arithmetic plan, otherwise ``None``."""
    plan = _resolved_plan(question, evidence_text, source_pdf_path)
    return plan.spec if plan else None


def source_plan_facts(
    spec: ClaimSpec, question: str, evidence_text: str, *, source_pdf_path: str = "",
) -> dict[str, GroundedFact] | None:
    """Reconstruct independent source facts only for the exact resolved plan.

    Callers cannot use this as a label/value lookup for an arbitrary proposed
    formula. The entire plan is regenerated from source and question, then every
    role and the formula must agree. Numeric values are never selection cues.
    """
    plan = _resolved_plan(question, evidence_text, source_pdf_path)
    if not plan or _signature(plan.spec) != _signature(spec):
        return None
    return plan.facts


def _resolved_plan(question, evidence, source_pdf_path):
    text_plan = _plan(question, evidence)
    if text_plan is not None or not source_pdf_path:
        return text_plan
    from verifiqa.agent.source_pdf_group_planner import source_pdf_plan
    return source_pdf_plan(question, evidence, source_pdf_path)


def _signature(spec):
    return (spec.metric, spec.formula, spec.operation, spec.claim_unit,
            spec.tolerance, spec.formula_source, spec.period, spec.planning_source,
            getattr(spec, "source_document_sha256", ""),
            spec.source_schema_sha256, spec.source_report_unit,
            tuple((r.name, tuple(r.aliases), r.period, r.concept, r.xbrl_context_id) for r in spec.roles))


def _slug(text):
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return "measure_" + slug if slug[:1].isdigit() else slug


def _tokens(text):
    # Plurals are normalized, but scope-bearing words such as net, total,
    # deferred, basic and diluted are intentionally retained.
    return tuple(word[:-1] if len(word) > 3 and word.endswith("s") else word
                 for word in re.findall(r"[a-z0-9]+", text.lower())
                 if word not in {"a", "an", "the", "of", "and", "in", "from", "to", "for"})


def _number(raw):
    from verifiqa.agent.table_fact_extractor import _parse_number
    # The extractor's low-level helper intentionally accepts a number inside
    # text. A planner needs an entire typed cell, not the first number of a
    # malformed multi-value cell or a note/year string.
    if not re.fullmatch(r"\s*\$?\s*(?:\(\s*-?\d[\d,]*(?:\.\d+)?\s*\)|-?\d[\d,]*(?:\.\d+)?)\s*%?\s*", str(raw)):
        return None
    return _parse_number(raw)


def source_row_unit(label: str, section: str, unit: str, raws: list[str]) -> str:
    """Apply explicit per-share and quantity types to a declared table unit.

    Monetary scales apply to monetary amounts, not per-share values. Conversely
    a share-based expense is still monetary. No scale is learned from an answer.
    """
    per_share = bool(re.search(r"\bper\s+(?:(?:common|ordinary|basic|diluted|stock)\s+){0,2}shares?\b|\bper\s+common\s+stock\b", label, re.I))
    is_quantity = bool(re.search(r"\bshares\b|\bshare\s+count\b", label, re.I)) or (
        re.search(r"\b(?:Denominator|dilutive)\b", section, re.I) is not None)
    monetary_measure = bool(re.search(r"\b(?:expense|compensation|cost|gains?|losses?|price)\b", label, re.I))
    if not unit and any("$" in raw for raw in raws):
        unit = "USD"
    if per_share:
        return "USD per share" if unit.startswith("USD") else ""
    if is_quantity and not monetary_measure:
        scale = next((scale for scale in ("thousands", "millions", "billions") if scale in unit), "")
        return "shares " + scale if scale else "count"
    if not unit and re.search(r"\b(?:RSAs?|RSUs?|stock options)\b", label, re.I):
        return "count"
    return unit


def _plan(question, evidence):
    if not question or not evidence:
        return None
    from verifiqa.agent.source_ppe_basis import resolve_ppe_basis
    ppe_applies, ppe_plan = resolve_ppe_basis(question, evidence)
    if ppe_applies:
        return ppe_plan
    # Population tables need their own typed-column parser. A name followed by
    # its age/salary must never become a financial statement's monetary row.
    people = _people_plan(question, evidence)
    if people is not None:
        return people
    from verifiqa.agent.source_rank_planner import rank_plan
    ranked = rank_plan(question, evidence)
    if ranked is not None:
        return ranked
    operation = _operation(question)
    if operation == "quantity_times_price":
        from verifiqa.agent.source_eligible_awards_planner import eligible_awards_plan
        return eligible_awards_plan(question, evidence, operation)
    if operation == "exclusion_or_remaining_amount":
        from verifiqa.agent.source_exclusion_planner import exclusion_plan
        return exclusion_plan(question, evidence, operation)
    from verifiqa.agent.source_percent_planner import reported_percent_tables, disclosed_change_plan
    disclosed = disclosed_change_plan(question, evidence, operation)
    if operation not in {"average", "sum", "percent_change", "period_change",
                         "line_item_difference", "proportion", "generic_ratio", "sum_then_ratio", "change_of_growth_rates"}:
        return disclosed
    tables = _annual_tables(evidence)
    from verifiqa.agent.source_table_layouts import supplemental_tables
    tables.extend(supplemental_tables(evidence))
    tables.extend(reported_percent_tables(evidence))
    from verifiqa.agent.source_typed_tables import typed_source_tables, typed_source_plans, compound_table_plan
    tables.extend(typed_source_tables(evidence, question=question))
    candidates = _narrative_comparisons(question, evidence, operation)
    candidates.extend(typed_source_plans(question, evidence, operation))
    if operation == "proportion":
        from verifiqa.agent.source_exclusion_planner import loan_proportion_plan
        loan = loan_proportion_plan(question, evidence, operation)
        if loan is not None:
            candidates.append(loan)
    from verifiqa.agent.source_column_planner import column_layout_plans
    candidates.extend(column_layout_plans(question, evidence, operation))
    if disclosed is not None:
        candidates.append(disclosed)
    maturity = _maturity_plan(question, evidence, operation)
    if maturity is not None:
        candidates.append(maturity)
    for table in tables:
        compound = compound_table_plan(question, operation, table)
        if compound is not None:
            candidates.append(compound)
        boundary = _opening_closing_plan(question, operation, table)
        if boundary is not None:
            candidates.append(boundary)
        candidate = _table_plan(question, operation, table)
        if candidate is not None:
            candidates.append(candidate)
    # Repeated retrieval chunks are equivalent only when their entire source
    # table and chosen cells are identical. Two distinct tables stay ambiguous.
    unique = {}
    for candidate in candidates:
        key = (_signature(candidate.spec), tuple(
            (name, fact.value, fact.unit, fact.source_column, fact.source_table_id)
            for name, fact in candidate.facts.items()))
        unique[key] = candidate
    return next(iter(unique.values())) if len(unique) == 1 else None


def _opening_closing_plan(question, operation, table):
    """Select the same measure's explicitly named opening and closing rows."""
    if operation not in {"period_change", "percent_change", "line_item_difference"}:
        return None
    if not (re.search(r"\b(?:beginning|opening|start)\b", question, re.I)
            and re.search(r"\b(?:end|ending|closing)\b", question, re.I)):
        return None
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    if len(years) != 1:
        return None
    columns = [i for i, column in enumerate(table.columns)
               if re.findall(r"\b(?:19|20)\d{2}\b", column) == list(years)]
    if len(columns) != 1:
        return None
    opening, closing = [], []
    for row in table.rows:
        begin = bool(re.search(r"\b(?:beginning|opening|start)\b", row.label, re.I))
        end = bool(re.search(r"\b(?:end|ending|closing)\b", row.label, re.I))
        if begin == end or not row.unit or not _scope_supported(question, table, row):
            continue
        measure = re.sub(r"\b(?:beginning|opening|start|end|ending|closing)\s+(?:of\s+)?(?:the\s+)?(?:period|year)?\b",
                         " ", row.label, flags=re.I)
        if not _row_match(_Row(measure, [], [], ""), question):
            continue
        (opening if begin else closing).append(row)
    if len(opening) != 1 or len(closing) != 1 or opening[0].unit != closing[0].unit:
        return None
    selected_operation = "percent_change" if operation == "percent_change" else "period_change"
    return _build(question, selected_operation, table,
                  [(opening[0], columns[0]), (closing[0], columns[0])])


def _narrative_comparisons(question, evidence, operation):
    """Read an explicit dated comparative percentage from a single sentence.

    Parenthetical ownership comparisons are common in filings. Embedded general
    partner interests, dates and distribution rights are separate measures and
    cannot enter the arithmetic merely because they are nearby numbers.
    """
    if operation not in {"period_change", "percent_change"}:
        return []
    month = (r"January|February|March|April|May|June|July|August|September|October|November|December")
    pattern = re.compile(
        r"\bAs\s+of\s+(?P<date>(?:" + month + r")\s+\d{1,2},?\s+(?P<year>(?:19|20)\d{2}))"
        r",\s+(?:we|the\s+company)\s+held\s+(?:a\s+)?"
        r"(?P<current>\d+(?:\.\d+)?)\s*%\s*"
        r"\(\s*(?P<prior_year>(?:19|20)\d{2})\s*:\s*(?P<prior>\d+(?:\.\d+)?)\s*%\s*\)\s*"
        r"(?P<label>ownership\s+interest\s+in\s+[^().;\n]+)", re.I)
    result = []
    for match in pattern.finditer(evidence):
        year, prior_year = match.group("year"), match.group("prior_year")
        if int(prior_year) >= int(year):
            continue
        label = match.group("label").strip().rstrip(",")
        row = _Row(label, [float(match.group("prior")), float(match.group("current"))],
                   [match.group("prior") + "%", match.group("current") + "%"],
                   match.group().rstrip(), unit="percent")
        if len(label.split()) > 12 or not _row_match(row, question):
            continue
        asked = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
        if asked and asked != {prior_year, year}:
            continue
        table = _Table([prior_year, match.group("date")], [row], label, "percent",
                       _identity(match.group(), label))
        if not _scope_supported(question, table, row):
            continue
        plan = _build(question, operation, table, [(row, 0), (row, 1)])
        if plan is not None:
            result.append(plan)
    return result


def _maturity_plan(question, evidence, operation):
    """Sum complete question-selected maturity buckets, never split a bucket."""
    if operation != "sum":
        return None
    from verifiqa.agent.table_fact_extractor import _annual_row_groups, _caption_money_unit
    from verifiqa.verification.generic_ops import _year_span_from_question
    years = _year_span_from_question(question)
    horizon = re.search(r"\bwithin\s+(\d{1,2})\s+years?\b", question, re.I)
    if not years and not horizon:
        return None
    if years is None:
        return None
    lines = evidence.splitlines()
    candidates = []
    bucket = r"(?:19|20)\d{2}(?:\s*[-–—]\s*(?:19|20)\d{2}|\s+and\s+thereafter)?|Other|Total"
    for index, line in enumerate(lines):
        context_lines = " ".join(lines[max(0, index - 3):index + 1])
        if not re.search(r"\b(?:Payments?\s+Due|Contractual\s+Obligations)\b", context_lines, re.I):
            continue
        unit = _caption_money_unit(line)
        for prior in reversed(lines[max(0, index - 3):index]):
            if unit:
                break
            unit = _caption_money_unit(prior)
        if not unit:
            continue
        columns = []
        if re.search(r"\bLess\s+than\s+1\s+Year\b", line, re.I):
            header = re.fullmatch(r"\s*(?:Contractual\s+Obligations\s+)?(?P<cols>Total\s+Less\s+than\s+1\s+Year\s+"
                                  r"1\s*[-–—]\s*3\s+Years\s+3\s*[-–—]\s*5\s+Years\s+More\s+than\s+5\s+Years)\s*", line, re.I)
            if not header:
                continue
            columns = [m.group().strip() for m in re.finditer(
                r"Total|Less\s+than\s+1\s+Year|1\s*[-–—]\s*3\s+Years|3\s*[-–—]\s*5\s+Years|More\s+than\s+5\s+Years",
                header.group("cols"), re.I)]
        else:
            captionless = re.sub(r"^\s*\(?(?:in\s+)?(?:thousands|millions|billions)\)?\s*", "", line, flags=re.I)
            matches = list(re.finditer(bucket, captionless, re.I))
            if len(matches) < 2 or "".join(m.group() for m in matches).replace(" ", "") != captionless.replace(" ", ""):
                continue
            columns = [m.group().strip() for m in matches]
        if len(set(column.lower() for column in columns)) != len(columns):
            continue
        selected = _maturity_columns(columns, years, int(horizon.group(1)) if horizon else None)
        if selected is None or len(selected) < 2:
            continue
        reporting_date = ""
        if horizon:
            month = (r"January|February|March|April|May|June|July|August|September|October|November|December")
            dates = []
            for nearby in reversed(lines[max(0, index - 4):index]):
                dates = re.findall(r"\b(?:as\s+of|at)\s+((?:" + month + r")\s+\d{1,2},?\s+(?:19|20)\d{2})", nearby, re.I)
                if dates:
                    break
            if len(set(dates)) != 1:
                continue
            reporting_date = dates[0]
        section = " / ".join(lines[max(0, index - 2):index]).strip()
        table = _Table(columns, section=section, unit=unit, identity=_identity(line, section))
        for line_body in lines[index + 1:]:
            if not line_body.strip():
                continue
            parsed = _annual_row_groups(line_body, len(columns))
            if parsed is None:
                break
            for row_section, label, raws in parsed:
                row = _Row(label, [_number(raw) for raw in raws], raws, line_body, row_section, unit)
                table.rows.append(row)
        rows = [row for row in table.rows if _row_match(row, question)
                and _scope_supported(question, table, row)]
        if len(rows) != 1 or any(rows[0].values[i] is None for i in selected):
            continue
        row = rows[0]
        roles, facts = [], {}
        for column_index in selected:
            column = columns[column_index]
            name = _slug(row.label + " " + column)
            period = reporting_date or column
            roles.append(RoleSpec(name, [row.label], period))
            facts[name] = GroundedFact(name, row.values[column_index], unit, row.quote,
                                       period=period, row_label=row.label, fact_type="source_table_numeric",
                                       source_value=row.values[column_index], source_unit=unit,
                                       source_section=section, source_column=column, source_table_id=table.identity)
        spec = ClaimSpec(_slug(row.label) + "_sum", " + ".join(facts), roles, unit,
                         answer_spec_from_question(question).tolerance, "generic_operation",
                         period=roles[-1].period, operation="sum", planning_source="deterministic_source_table")
        candidates.append(_Plan(spec, facts))
    return candidates[0] if len(candidates) == 1 else None


def _maturity_columns(columns, years, horizon):
    if years:
        wanted, coverage, selected = set(years), set(), []
        for i, column in enumerate(columns):
            match = re.fullmatch(r"((?:19|20)\d{2})(?:\s*[-–—]\s*((?:19|20)\d{2}))?", column)
            if not match:
                continue
            first, last = int(match.group(1)), int(match.group(2) or match.group(1))
            if last < first or last - first > 20:
                return None
            group = set(range(first, last + 1))
            if group & wanted and not group <= wanted:
                return None
            if group <= wanted:
                if coverage & group:
                    return None
                coverage |= group
                selected.append(i)
        return selected if coverage == wanted else None
    if horizon:
        selected, end = [], 0
        for i, column in enumerate(columns):
            if re.fullmatch(r"Less\s+than\s+1\s+Year", column, re.I):
                interval = (0, 1)
            else:
                match = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)\s+Years?", column, re.I)
                if not match:
                    continue
                interval = tuple(int(value) for value in match.groups())
            start, stop = interval
            if start >= horizon:
                continue
            if start != end or stop > horizon:
                return None
            selected.append(i)
            end = stop
        return selected if end == horizon else None
    return None


def _operation(question):
    from verifiqa.verification.generic_ops import classify_operation
    # A bare "total assets" is a reported measure, not permission to add all
    # visible annual balances. Only explicit registered arithmetic grammar can
    # authorize a sum; typed officer populations are handled separately.
    return classify_operation(question)


def _annual_tables(evidence):
    """Read complete, explicitly annual rows without inferring missing cells."""
    from verifiqa.agent.table_fact_extractor import (
        _annual_pdf_header, _annual_row_groups, _caption_money_unit,
        _inline_captioned_grid, _join_split_annual_headers, _pipe_tables_with_context, _printed_heading,
        _subsection,
    )
    tables = []
    # Structured tables retain precise dates/quarters for downstream selection.
    for rows, title, raw_lines in _pipe_tables_with_context(evidence):
        header = rows[0]
        # Standard Markdown's outer pipes are permitted only when every row
        # contains the same empty outer cells.
        if header and not header[0] and not header[-1] and all(not r[0] and not r[-1] for r in rows):
            rows = [r[1:-1] for r in rows]
            header = rows[0]
        if len(header) < 2 or not all(re.search(r"\b(?:19|20)\d{2}\b", h) for h in header[1:]):
            continue
        if len(set(header[1:])) != len(header) - 1:
            continue
        table_unit = (_caption_money_unit(header[0]) or _caption_money_unit(" ".join(header))
                      or _caption_money_unit(title))
        table = _Table(header[1:], section=title,
                       unit=table_unit, unit_declared=bool(table_unit),
                       identity=_identity("\n".join(raw_lines), title))
        section = ""
        for row, line in zip(rows[1:], raw_lines[1:]):
            if all(re.fullmatch(r"\s*:?-+:?\s*", c or "") for c in row):
                continue
            if len(row) != len(header):
                table = None
                break
            if not any(row[1:]):
                section = row[0]
                continue
            numbers = [_number(raw) for raw in row[1:]]
            if any(n is None for n in numbers):
                table = None
                break
            percents = ["%" in raw for raw in row[1:]]
            if any(percents) and not all(percents):
                table = None
                break
            table.rows.append(_Row(row[0], numbers, row[1:], line, section,
                                   source_row_unit(row[0], section,
                                                   "percent" if all(percents) else table.unit, row[1:])))
        if table and table.rows:
            tables.append(table)
    lines = _join_split_annual_headers(evidence.splitlines())
    footer_units = {_caption_money_unit(_pagewide_currency_caption(line)) for line in lines
                    if _pagewide_currency_caption(line)}
    footer_units.discard("")
    notes_units = {_caption_money_unit(_notes_header_currency_caption(line)) for line in lines
                   if _notes_header_currency_caption(line)}
    if len(footer_units | notes_units) > 1:
        return []  # Conflicting page-wide declarations need explicit page boundaries.
    table, title, unit, section, page_unit = None, "", "", "", next(iter(footer_units), "")
    percent_grid = False
    lines = _join_date_headers(lines)
    for index, line in enumerate(lines):
        if "|" in line:
            table = None
            continue
        declared = _caption_money_unit(line)
        if declared:
            unit = declared
            if _notes_header_currency_caption(line):
                page_unit = declared
            if re.fullmatch(r"\s*\(?\s*(?:Expressed|Tabular\s+amounts)\s+in\s+"
                            r"(?:thousands|millions|billions)\b.{0,100}\)?\s*", line, re.I):
                page_unit = declared
            if re.fullmatch(r"\s*\(?\s*(?:U\.?\s*S\.?\s*dollars?|USD|dollars?)\s+in\s+"
                            r"(?:thousands|millions|billions)\b.{0,100}\)?\s*", line, re.I):
                page_unit = declared
            if declared == "USD" and re.fullmatch(r"\s*\(?\s*(?:all\s+)?amounts?\s+in\s+"
                                                  r"(?:U\.?\s*S\.?\s*dollars?|USD|dollars?)\s*\)?\s*", line, re.I):
                page_unit = declared
        if declared and re.fullmatch(
                r"\s*\(?\s*(?:(?:amounts|figures|dollars)\s+)?in\s+(?:USD\s+)?"
                r"(?:thousands|millions|billions)(?:\s+of\s+\$)?"
                r"(?:\s*,\s*except\s+(?:for\s+)?per\s+share\s+(?:amounts|data))?\s*\)?\s*", line, re.I):
            if table:
                table.unit = declared
                table.unit_declared = True
            continue
        if declared and (re.fullmatch(r"\s*\(?\s*(?:U\.?\s*S\.?\s*dollars?|USD|dollars?)\s+in\s+"
                                      r"(?:thousands|millions|billions)\b.{0,100}\)?\s*", line, re.I)
                         or (declared == "USD" and re.fullmatch(
                             r"\s*\(?\s*(?:(?:all\s+)?amounts?\s+)?in\s+(?:U\.?\s*S\.?\s*dollars?|USD|dollars?)\s*\)?\s*", line, re.I))):
            if table:
                table.unit = declared
                table.unit_declared = True
            continue
        inline = _inline_captioned_grid(line)
        if inline is not None:
            years, inline_unit, inline_rows = inline
            table = _Table(years, section=title, unit=inline_unit,
                           identity=_identity(line, title + "|" + str(index)), unit_declared=True)
            for label, raws in inline_rows:
                numbers = [_number(raw) for raw in raws]
                table.rows.append(_Row(label, numbers, raws, line,
                                       unit=source_row_unit(label, "", inline_unit, raws)))
            tables.append(table)
            unit, section = inline_unit, ""
            continue
        header = _annual_pdf_header(line)
        body = ""
        if header is None:
            header = _flattened_header(line)
        if header is not None:
            years, prefix, body = header
            # A table with several subcolumns per year cannot be flattened to
            # one annual value per row (e.g. carrying vs fair value).
            if len(set(years)) != len(years):
                table = None
                continue
            if isinstance(prefix, dict):
                columns = [prefix[y] for y in years]
            else:
                columns = [(str(prefix) + " " + y).strip() for y in years]
            # Currency subcolumns are an independent axis. Never average a
            # translated USD value together with its source-currency value.
            if re.search(r"NT\$|US\$|HK\$|RMB|Carrying Value.*Fair Value", body, re.I):
                table = None
                continue
            percent_grid = bool(re.match(r"(?:\s*%){" + str(len(years)) + r"}(?:\s|$)", body))
            if percent_grid:
                body = re.sub(r"^(?:\s*%){" + str(len(years)) + r"}", "", body).strip()
            nearby = " ".join(lines[max(0, index - 3):index])
            source_context = title
            # Preserve an explicit prose table definition as context. Without
            # it, an impairment charge or subset of pension plans could be
            # silently presented as an unrestricted balance.
            if re.search(r"\b(?:following\s+table|shown\s+below|as\s+follows)\b", nearby, re.I):
                source_context = nearby
            definitions = [_table_definition(prior) for prior in lines[max(0, index - 10):index]]
            definitions = [definition for definition in definitions if definition]
            if definitions:
                # A new explicit table definition supersedes a preceding
                # grid's geography or accounting population. Keep the printed
                # section heading, but do not import neighboring table rows or
                # an older definition through the arbitrary lookback window.
                source_context = " / ".join(dict.fromkeys(part for part in (title, definitions[-1]) if part))
            table = _Table(columns, section=source_context, unit=unit or page_unit,
                           identity=_identity(line, title + "|" + str(index)), unit_declared=bool(unit or page_unit))
            tables.append(table)
            section = ""
            if not body:
                continue
        else:
            body = line
        if not body.strip():
            continue
        # A standalone scale/date caption can separate header from body.
        if table and not table.rows and (re.fullmatch(r"\(?\s*(?:in\s+)?(?:thousands|millions|billions)(?:\s*,\s*except\s+(?:for\s+)?per\s+share\s+(?:amounts|data))?\s*\)?", body.strip(), re.I)
                or re.fullmatch(r"(?:Years?\s+Ended\s+)?(?:December|June|March|September)\s+\d{1,2},?", body.strip(), re.I)):
            table.unit = unit
            continue
        subsection = _subsection(body)
        if table and ((subsection or _group_heading(body)) and (body.strip().endswith(":") or body.strip() in {
                "Numerator", "Denominator", "Deferred tax assets", "Deferred tax liabilities"})):
            section = subsection or _group_heading(body)
            continue
        if table:
            parsed = _annual_row_groups(body, len(table.columns))
            if parsed is None and ":" in body:
                group, rest = body.split(":", 1)
                if _group_heading(group + ":"):
                    parsed = _annual_row_groups(rest.strip(), len(table.columns))
                    if parsed:
                        section = group.strip()
            if parsed:
                for row_section, label, raws in parsed:
                    section = row_section or section
                    numbers = [_number(raw) for raw in raws]
                    row_unit = "percent" if percent_grid or all("%" in raw for raw in raws) else (table.unit or unit)
                    if not row_unit and any("$" in raw for raw in raws):
                        row_unit = "USD"
                        table.unit = row_unit
                    # A printed '%' on every first-row value can establish a
                    # percent table only under an explicit percentage caption.
                    if not table.rows and all("%" in raw for raw in raws) and re.search(
                            r"%|\bpercent(?:age)?s?\b", table.section, re.I):
                        table.unit = "percent"
                    row_unit = source_row_unit(label, section, row_unit, raws)
                    table.rows.append(_Row(label, numbers, raws, line, section, row_unit))
                continue
            # A malformed row ends this grid. It cannot silently become an
            # additional member of a requested average or sum.
            table = None
        heading_text = re.sub(r"\b(?:[A-Za-z]\.){2,}", lambda m: m.group().replace(".", ""), body)
        if _printed_heading(heading_text):
            title = body.strip()
        # A narrative scale declaration applies only to the immediately
        # following grid; a different prose paragraph resets it.
        elif not declared and re.search(r"[.!?]", body):
            unit = ""
    return [table for table in tables if table.rows]


def _table_definition(text):
    """Return the printed definition sentence, not its surrounding paragraph.

    PDF text can place a long narrative and the next table's definition on
    one line. Geography or accounting restrictions from those earlier
    sentences are not restrictions on the newly introduced table.
    """
    def protect_initialism(match):
        value = match.group()
        # An initialism may itself end a sentence. Clear sentence openings
        # after it remain boundaries; periods inside U.S. stay protected.
        if re.match(r"\s+(?:The|This|These|For|During|Our|We|It)\b", text[match.end():]):
            return value[:-1].replace(".", "\0") + "."
        return value.replace(".", "\0")

    protected = re.sub(r"\b(?:[A-Za-z]\.){2,}", protect_initialism, text)
    protected = re.sub(r"\b(?:Mr|Mrs|Ms|Dr)\.", lambda match: match.group().replace(".", "\0"), protected)
    definitions = []
    for sentence in re.split(r"(?<=[.!?])\s+", protected):
        sentence = sentence.replace("\0", ".").strip()
        if re.search(r"\b(?:following\s+table|shown\s+below|as\s+follows)\b|\bfollowing\b[^.]{0,120}%", sentence, re.I):
            definitions.append(sentence)
        elif sentence.endswith(":") and re.search(
                r"\bfollowing\s+(?:items|components|balances|categories|amounts)\b|"
                r"\b(?:consists?|consisted|comprised)\s+of\s+the\s+following\b|"
                r"\breconciliation\b.*\b(?:presented|provided)\s+below\b", sentence, re.I):
            definitions.append(sentence)
    return definitions[-1] if definitions else ""


def _pagewide_currency_caption(text):
    """A literal note/page declaration may be printed in the footer."""
    match = re.search(r"\(United\s+States\s+Dollars\s+in\s+(?:thousands|millions|billions),\s*"
                      r"except\s+per\s+share\s+data,\s*unless\s+otherwise\s+stated\)\s*$", text, re.I)
    if match is None:
        return ""
    prefix = text[:match.start()].strip()
    if prefix and (len(prefix) > 200 or not re.search(
            r"\bNOTES\s+TO\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS\b", prefix, re.I)):
        return ""
    return match.group().strip()


def _notes_header_currency_caption(text):
    """A literal notes-header scale applies from that heading forward."""
    match = re.search(r"\(in\s+(?:thousands|millions|billions),\s*except\s+"
                      r"(?:for\s+)?per\s+share\s+(?:amounts|data)\)\s*$", text, re.I)
    if match is None:
        return ""
    prefix = text[:match.start()].strip()
    if len(prefix) > 200 or not re.search(
            r"\bNOTES\s+TO\s+(?:THE\s+)?(?:CONSOLIDATED\s+)?FINANCIAL\s+STATEMENTS\b",
            prefix, re.I):
        return ""
    from verifiqa.verification.money_units import explicit_money_currencies
    if explicit_money_currencies(prefix) - {"USD"}:
        return ""
    return match.group().strip()


def _group_heading(text):
    if (text.strip().endswith(":") and len(text.split()) <= 10
            and not re.search(r"\d|[.!?]", text)):
        return text.strip().rstrip(":")
    return ""


def _join_date_headers(lines):
    """Join complete dated columns separated by repeated 'Year Ended' labels."""
    from verifiqa.agent.table_fact_extractor import _MONTH
    date = re.compile(r"\s*(?:" + _MONTH + r")\s+\d{1,2},?\s+(?:19|20)\d{2}\s*", re.I)
    result, index = [], 0
    while index < len(lines):
        if not date.fullmatch(lines[index]):
            result.append(lines[index])
            index += 1
            continue
        dates, end = [lines[index].strip()], index + 1
        while end < len(lines):
            next_date = end + 1 if re.fullmatch(r"\s*Year\s+Ended\s*", lines[end], re.I) else end
            if next_date >= len(lines) or not date.fullmatch(lines[next_date]):
                break
            dates.append(lines[next_date].strip())
            end = next_date + 1
        if len(dates) > 1:
            result.append(" ".join(dates))
            index = end
        else:
            result.append(lines[index])
            index += 1
    return result


def _flattened_header(line):
    # PDF extraction commonly inserts the label-column heading before the year
    # columns: "Years Ended December 31, Region 2017 2018 2019 % % % ...".
    matched = re.match(
        r"^\s*(?P<prefix>(?:Years?\s+Ended|As\s+of|Fiscal\s+Years?).{0,55}?)"
        r"(?P<years>(?:19|20)\d{2}(?:\s+(?:19|20)\d{2})+)"
        r"(?P<body>\s+.*)?$", line, re.I)
    if not matched:
        return None
    prefix = matched.group("prefix").strip()
    if re.search(r"\bquarter\b|\bQ[1-4]\b|\b201\d\b|\b202\d\b", prefix, re.I):
        return None
    years = re.findall(r"(?:19|20)\d{2}", matched.group("years"))
    # The row-axis title isn't part of a temporal column.
    prefix = re.sub(r"\b(?:Region|Application|Customer\s+Type)\s*$", "", prefix, flags=re.I).strip()
    return years, prefix, (matched.group("body") or "").strip()


def _identity(text, title):
    return "source_plan:" + hashlib.sha256((title + "\n" + text).encode()).hexdigest()[:20]


def _selected_periods(question, table, operation):
    from verifiqa.agent.table_fact_extractor import _MONTH
    from verifiqa.verification.periods import period_key
    if re.search(r"\bquarter\b|\bq[1-4]\b", question, re.I):
        return None
    if operation in {"period_change", "percent_change"}:
        dates = list(re.finditer(r"\b(?:" + _MONTH + r")\s+(?:\d{1,2},?\s+)?(?:19|20)\d{2}\b", question, re.I))
        if len(dates) == 2:
            wanted = [period_key(match.group()) for match in dates]
            source = [period_key(column) for column in table.columns]
            if None in wanted or wanted[0] == wanted[1]:
                return None
            # A written month/year may name one exact quarterly/as-of date in
            # that month. An explicitly written day remains exact, and two
            # source observations in the same month stay ambiguous.
            choices = [[index for index, key in enumerate(source)
                        if key is not None and key[:2] == target[:2]
                        and (not target[2] or key[2] == target[2])]
                       for target in wanted]
            if any(len(options) != 1 for options in choices):
                return None
            if (wanted[0] > wanted[1] and re.search(r"\bfrom\s*$", question[:dates[0].start()], re.I)
                    and re.fullmatch(r"\s+to\s+", question[dates[0].end():dates[1].start()], re.I)):
                return None
            return sorted((options[0] for options in choices), key=lambda i: source[i])
    from verifiqa.verification.generic_ops import _year_span_from_question
    years = _year_span_from_question(question)
    if years is None:
        return None
    explicit_years = sorted(set(int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", question)))
    if len(explicit_years) == 1 and not years:
        years = explicit_years
    reverse = re.search(r"\bfrom\s+((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})\b", question, re.I)
    if operation in {"period_change", "percent_change"} and reverse and int(reverse.group(1)) > int(reverse.group(2)):
        return None
    source_years = [re.findall(r"\b(?:19|20)\d{2}\b", column) for column in table.columns]
    if any(len(ys) != 1 for ys in source_years):
        return None
    source_years = [int(ys[0]) for ys in source_years]
    if len(set(source_years)) != len(source_years):
        return None
    if re.search(r"\bquarter\b|\bq[1-4]\b", question, re.I):
        return None  # Quarter ordinal and frequency are not annual years.
    if years:
        wanted = years if operation in {"average", "sum"} else sorted(set(years))
        if operation in {"percent_change", "period_change"} and len(wanted) != 2:
            return None
        if any(year not in source_years for year in wanted):
            return None
        return [source_years.index(year) for year in wanted]
    if operation in {"average", "sum"}:
        count = re.search(r"\b(?:the\s+)?([2-9])\s+years?\b", question, re.I)
        if count and len(table.columns) != int(count.group(1)):
            return None
        if re.search(r"\bboth\b", question, re.I) and len(table.columns) != 2:
            return None
        return sorted(range(len(source_years)), key=source_years.__getitem__)
    # An undated change is ambiguous even if exactly two values are present.
    return None


def _row_match(row, question):
    """Require the complete source label in the question, including scope."""
    have = set(_tokens(row.label))
    wanted = set(_tokens(question))
    # Parenthetical abbreviation is optional if the complete expanded name was
    # requested, but the alias itself can also be the explicit question target.
    plain = re.sub(r"\([^)]*\)", " ", row.label)
    options = [have, set(_tokens(plain))]
    # A printed Add:/Less: introduces an already signed component row. It is
    # not part of the measure's name; retain the actual label and cell sign.
    # An internal phrase such as "debt, less current maturities" is untouched.
    if re.match(r"\s*(?:Add|Less):\s*", row.label, re.I):
        options.append(set(_tokens(re.sub(r"^\s*(?:Add|Less):\s*", "", row.label, flags=re.I))))
    for abbreviation in re.findall(r"\([‘’\"“”']*([A-Z][A-Z0-9]+s?)[‘’\"“”']*\)", row.label):
        options.append(set(_tokens(abbreviation)))
    # Numeric geometry labels (e.g. 40 nanometers) are source measure names,
    # not observed values; their number must be explicitly question-selected.
    return any(option and option <= wanted for option in options)


def _scope_supported(question, table, row, *, compound_entities=False):
    from verifiqa.types import VerificationFact
    from verifiqa.verification.xbrl_scope import requested_scopes, _words
    context = (table.section + " " + row.section).lower()
    q = question.lower()
    from verifiqa.verification.question_measure_types import source_measure_supported
    if not source_measure_supported(question, row.label, context, row.unit):
        return False
    # Output-unit language does not change the target measure. A request for a
    # percentage change in counts still needs counts, and customer quantities
    # cannot be replaced by a geographic share of revenue.
    if re.search(r"\b(?:number|count)\s+of\b|\b(?:number|count)\b.*\b(?:shares|connections|customers|employees)\b", q):
        if not (row.unit == "count" or row.unit.startswith("shares ")):
            return False
    if re.search(r"\bcustomers?\b", q) and row.unit == "percent" and not re.search(r"\b(?:revenues?|sales)\b", q):
        return False
    if row.unit == "percent" and _operation(question) in {
            "average", "sum", "period_change", "percent_change", "percentage_point_change",
            "change_of_growth_rates", "difference_of_changes"}:
        from verifiqa.verification.question_measure_types import percentage_observation_supported
        if not percentage_observation_supported(question):
            return False
    if re.search(r"%\s+of\b|\bpercent(?:age)?\s+of\b", row.label, re.I):
        target_text = re.sub(r"\bpercent(?:age)?\s+(?:change|increase|decrease|difference)\b", " ", question, flags=re.I)
        if not re.search(r"%|\bpercent(?:age)?\b", target_text, re.I):
            return False
    # Unknown acronyms are not disposable words. In particular APRU (a typo of
    # ARPU) cannot reduce to an 'on-net' label and select connection counts.
    acronyms = re.findall(r"\b[A-Z][A-Z0-9]{1,}(?:s)?\b", question)
    observed = set(_tokens(row.label + " " + row.section + " " + table.section))
    for acronym in acronyms:
        if acronym in {"USD", "US", "USA"} or re.fullmatch(r"FY\d{2,4}|Q[1-4]", acronym):
            continue
        from verifiqa.verification.money_units import money_currency
        if money_currency(acronym) is not None and money_currency(acronym) == money_currency(row.unit):
            continue
        if not set(_tokens(acronym)) <= observed:
            return False
    wanted_scopes = set(requested_scopes(VerificationFact(question, 0, "")))
    source_scopes = set(requested_scopes(VerificationFact(
        "", 0, "", row_label=row.label, source_section=context)))
    if compound_entities:
        named = {scope for scope in wanted_scopes if re.fullmatch(r"(?:customer|supplier) [a-z]", scope[0])}
        if len(named) > 1:
            families = {scope[0].split()[0] for scope in named}
            # Only this explicit two-item operation can bind each named entity
            # to its own literal row. All geography and other scope qualifiers
            # remain global requirements for each operand.
            literal = re.fullmatch(r"\s*(customer|supplier)\s+([a-z])\s*", row.label, re.I)
            identity = (" ".join(literal.groups()).lower(),) if literal else None
            if len(families) != 1 or identity not in named:
                return False
            wanted_scopes = (wanted_scopes - named) | {identity}
    if wanted_scopes and not wanted_scopes <= source_scopes:
        return False
    geographic = {"united states", "united kingdom", "south africa", "north america",
                  "south america", "asia pacific", "rest of world", "canada", "taiwan",
                  "singapore", "europe", "foreign", "domestic"}
    wanted_geographies = {scope for scope in wanted_scopes if scope[0] in geographic}
    source_geographies = {scope for scope in source_scopes if scope[0] in geographic}
    if len(wanted_geographies) == 1 and source_geographies != wanted_geographies:
        return False
    us = ("united states", "us", "usa")
    if us in wanted_scopes:
        negative_us = r"\b(?:non|outside|excluding|other than)\s+(?:the\s+)?(?:u s|us|usa|united states)\b"
        if bool(re.search(negative_us, _words(question))) != bool(re.search(
                negative_us, _words(row.label + " " + context))):
            return False
    # A deferred-tax component isn't an underlying asset/liability, and an
    # impairment charge isn't a balance-sheet carrying amount. These contexts
    # must be requested explicitly rather than silently erased.
    if re.search(r"\bdeferred\s+tax\b", context) and not re.search(r"\b(?:deferred\s+tax|tax\s+(?:asset|liabilit))", q):
        return False
    if "impairment" in context and "impairment" not in q:
        return False
    if re.search(r"\b(?:pro\s*forma|adjusted)\b", context) and not re.search(r"\b(?:pro\s*forma|adjusted)\b", q):
        return False
    if re.search(r"\b(?:supplier|customer)\s+[a-z]\b", row.label, re.I) and row.section:
        if not set(_tokens(row.section)) <= set(_tokens(q)):
            return False
    if "in excess of" in context and "in excess of" not in q:
        # A threshold can define which named customer rows are disclosed; it
        # does not change an individually identified customer's revenue amount.
        # An aggregate/subset total or unnamed customer remains restricted.
        named_customer_revenue = bool(
            re.fullmatch(r"\s*Customer\s+[A-Z]\s*", row.label, re.I)
            and re.search(r"\b" + r"\s+".join(re.escape(part) for part in row.label.split()) + r"\b", question, re.I)
            and row.unit in {"USD", "USD thousands", "USD millions", "USD billions"}
            and re.search(r"\b(?:revenues?|sales)\b", q))
        if not named_customer_revenue:
            return False
    if set(_tokens(row.label)) <= {"total", "other", "net"}:
        # A bare total has no measure identity of its own. Its actual printed
        # table caption must supply every requested measure word; "total" in a
        # question cannot authorize summing an unrelated table's total row.
        filler = {"what", "wa", "was", "is", "are", "were", "average", "mean", "sum", "total",
                  "annual", "fiscal", "year", "both", "period", "value", "amount", "change",
                  "percentage", "percent", "company", "s", "between", "during", "all"}
        wanted = {token for token in _tokens(question)
                  if token not in filler and not token.isdigit()}
        if not wanted or not wanted <= set(_tokens(context)):
            return False
        # A heading can cover several disjoint measures. Matching one phrase
        # in "Amortization ... and Acquisition-Related Costs" does not make its
        # combined Total the requested amortization amount. Use the component
        # labels to detect a partial request without inventing a new sum.
        component_matches = []
        for component in table.rows:
            words = set(_tokens(component.label))
            if not words or words <= {"total", "subtotal", "other", "net"}:
                continue
            if "intangible" in words:
                words.discard("asset")  # intangibles == intangible assets
            component_matches.append(words <= wanted)
        if any(component_matches) and not all(component_matches):
            return False
    return True


def _table_plan(question, operation, table):
    if operation in {"line_item_difference", "proportion", "generic_ratio"}:
        return _two_row_plan(question, operation, table)
    selected = [row for row in table.rows if _row_match(row, question)
                and _scope_supported(question, table, row)]
    # A question specifying a total cannot select a similarly named component.
    # Prefer the most complete matching label, but never break equal ties.
    if not selected:
        return None
    best_size = max(len(_tokens(row.label)) for row in selected)
    selected = [row for row in selected if len(_tokens(row.label)) == best_size]
    if len(selected) != 1:
        return None
    row = selected[0]
    if re.search(r"\bless\b", question, re.I) and not re.search(r"\bless\b", row.label, re.I):
        return None
    if operation in {"average", "sum", "period_change"}:
        if re.search(r"%|\bpercent(?:age)?\b", question, re.I) and row.unit != "percent":
            return None
        if re.search(r"\b(?:USD|dollars?|in\s+(?:thousands|millions|billions))\b", question, re.I) and not row.unit.startswith("USD"):
            return None
    # An unscaled generic annual USD observation must not preempt the filing's
    # known unit. Explicitly typed dated-stock tables separately support a
    # relative change whose common undisclosed scale cancels.
    if (row.unit == "USD" and not table.unit_declared
            and not (operation == "percent_change" and table.relative_scale_only)
            and not re.search(r"\b(?:in\s+(?:(?:U\.?S\.?\s*)?dollars?|USD)|amounts?\s+in\s+USD)\b", table.section, re.I)):
        return None
    indices = _selected_periods(question, table, operation)
    if indices is None or len(indices) < 2:
        return None
    if not row.unit:
        # Units omitted from every cell cannot be recovered by reading a
        # claimed answer. A complete explicit percent grid already has a unit.
        return None
    pairs = [(row, i) for i in indices]
    return _build(question, operation, table, pairs)


def _two_row_plan(question, operation, table):
    from verifiqa.verification.generic_ops import _ratio_phrases
    text = question.lower().strip().rstrip("?")
    phrases = []
    if operation == "generic_ratio":
        pair = _ratio_phrases(question)
        if pair:
            phrases = [pair]
    elif operation == "proportion":
        match = re.search(r"(?:what\s+(?:is|was|are)\s+)?(?:the\s+)?percentage\s+of\s+(.+?)\s+(?:as\s+(?:a\s+)?(?:percentage|total)\s+of|of)\s+(.+)$", text)
        if match:
            phrases = [(match.group(1), match.group(2))]
        else:
            match = re.search(r"(?:what\s+(?:is|was|are)\s+)?(.+?)\s+as\s+a\s+percentage\s+of\s+(.+)$", text)
            if match:
                phrases = [(match.group(1), match.group(2))]
        if not phrases:
            # Reuse the authority parser's explicit numerator/denominator
            # grammar, including 'value expressed as a percentage to'. Whole
            # source labels retain net/gross qualifiers and their direction.
            from verifiqa.verification.operation_bindings import compound_ratio_phrases
            pair = compound_ratio_phrases(question)
            if pair:
                phrases = [pair]
    else:
        match = re.search(r"\bdifference\s+(?:between|in)\s+(.+)$", text)
        if match:
            for boundary in re.finditer(r"\s+(?:and|versus|vs\.?|against)\s+", match.group(1)):
                phrases.append((match.group(1)[:boundary.start()], match.group(1)[boundary.end():]))
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", question)))
    if len(years) != 1:
        return None
    columns = [i for i, header in enumerate(table.columns) if re.search(r"\b" + years[0] + r"\b", header)]
    if len(columns) != 1:
        return None
    possible = []
    for first, second in phrases:
        from verifiqa.agent.source_typed_tables import table_phrase_rows
        left = _named_customer_phrase_rows(question, first, operation, table)
        right = _named_customer_phrase_rows(question, second, operation, table)
        if left is None:
            left = table_phrase_rows(table, first, columns[0]) or []
        if right is None:
            right = table_phrase_rows(table, second, columns[0]) or []
        left = [row for row in left if _scope_supported(
            question, table, row, compound_entities=operation == "line_item_difference")]
        right = [row for row in right if _scope_supported(
            question, table, row, compound_entities=operation == "line_item_difference")]
        if len(left) == len(right) == 1 and left[0] is not right[0] and left[0].unit and left[0].unit == right[0].unit:
            possible.append((left[0], right[0]))
    if len(possible) != 1:
        return None
    return _build(question, operation, table, [(row, columns[0]) for row in possible[0]])


def _named_customer_phrase_rows(question, phrase, operation, table):
    """Bind an explicitly lettered customer within a monetary revenue clause."""
    if operation != "line_item_difference" or not re.search(r"\b(?:revenues?|sales)\b", question, re.I):
        return None
    entities = re.findall(r"\bcustomer\s+([a-z])\b", phrase, re.I)
    if len(entities) != 1:
        return None
    # Unlike an article, the A in Customer A is an identity. Qualifying revenue
    # words must be supplied by the printed table definition, not assumed from
    # an otherwise unrelated receivables/percentage table.
    if not re.search(r"\b(?:revenues?|sales)\s+(?:earned\s+)?from\s+customers?\b|"
                     r"\bcustomers?\b[^.]{0,100}\bderived\s+(?:revenues?|sales)\b", table.section, re.I):
        return None
    remainder = re.sub(r"\bcustomer\s+[a-z]\b|\b(?:19|20)\d{2}\b", " ", phrase, flags=re.I)
    from verifiqa.verification.generic_ops import _content_tokens
    if _content_tokens(remainder) - {"revenue", "sale", "sales", "earned"}:
        return []
    return [row for row in table.rows
            if re.fullmatch(r"\s*Customer\s+" + entities[0] + r"\s*", row.label, re.I)
            and row.unit in {"USD", "USD thousands", "USD millions", "USD billions"}]


def _most_complete_rows(rows):
    if not rows:
        return []
    size = max(len(_tokens(row.label)) for row in rows)
    return [row for row in rows if len(_tokens(row.label)) == size]


def _build(question, operation, table, pairs):
    if operation in {"average", "sum", "period_change", "percent_change"}:
        from verifiqa.verification.question_measure_types import percentage_observation_supported
        if any(row.unit == "percent" for row, _ in pairs) and not percentage_observation_supported(question):
            return None
    roles, facts = [], {}
    base_names = []
    for row, index in pairs:
        years = re.findall(r"\b(?:19|20)\d{2}\b", table.columns[index])
        base_names.append(_slug(row.label) + "_" + (years[0] if years else _slug(table.columns[index])))
    for row, index in pairs:
        if row.values[index] is None:
            return None
        column = table.columns[index]
        years = re.findall(r"\b(?:19|20)\d{2}\b", column)
        name = _slug(row.label) + "_" + (years[0] if years else _slug(column))
        if base_names.count(name) > 1:
            name = _slug(row.label) + "_" + _slug(column)
        if name in facts:
            return None
        roles.append(RoleSpec(name, [row.label], column))
        facts[name] = GroundedFact(name, row.values[index], row.unit, row.quote,
                                   period=column, row_label=row.label, fact_type="source_table_numeric",
                                   source_value=row.values[index], source_unit=row.unit,
                                   source_section=" / ".join(p for p in [table.section, row.section] if p),
                                   source_column=column, source_table_id=table.identity)
    names = list(facts)
    if operation == "average":
        formula = "(" + " + ".join(names) + ") / " + str(len(names))
    elif operation == "sum":
        formula = " + ".join(names)
    elif operation == "percent_change":
        if facts[names[0]].value == 0:
            return None
        formula = f"({names[-1]} - {names[0]}) / {names[0]} * 100"
    elif operation == "period_change":
        from verifiqa.verification.generic_ops import decline_magnitude_requested
        formula = (f"{names[0]} - {names[-1]}" if decline_magnitude_requested(question)
                   else f"{names[-1]} - {names[0]}")
    elif operation == "change_of_growth_rates":
        if len(names) != 2 or any(fact.unit != "percent" for fact in facts.values()) or not all(
                re.search(r"\bgrowth\b", row.label, re.I) for row, _ in pairs):
            return None
        formula = f"{names[-1]} - {names[0]}"
    elif operation in {"proportion", "generic_ratio"}:
        if facts[names[1]].value == 0:
            return None
        formula = f"{names[0]} / {names[1]}" + (" * 100" if operation == "proportion" else "")
    elif operation == "line_item_difference":
        formula = f"{names[0]} - {names[1]}"
    else:
        return None
    unit = "percent" if operation in {"percent_change", "proportion"} else (
        "ratio" if operation == "generic_ratio" else next(iter(facts.values())).unit)
    if operation not in {"proportion", "generic_ratio"} and len({f.unit for f in facts.values()}) != 1:
        return None
    spec = ClaimSpec(metric=_slug(pairs[0][0].label) + "_" + operation,
                     formula=formula, roles=roles, claim_unit=unit,
                     tolerance=answer_spec_from_question(question).tolerance,
                     formula_source="generic_operation", period=roles[-1].period,
                     operation=operation, planning_source="deterministic_source_table")
    return _Plan(spec, facts)


def _people_plan(question, evidence):
    q = question.lower()
    if re.search(r"\baverage\s+age\b", q):
        return _officer_ages(question, evidence)
    if re.search(r"\b(?:neos?|named\s+executive\s+officers?)\b", q) and re.search(r"\bsalar(?:y|ies)\b", q):
        return _executive_salaries(question, evidence)
    return None


def _officer_ages(question, evidence):
    if not re.search(r"\b(?:executive\s+officers?|vice\s+presidents?)\b", question, re.I):
        return None
    lines = evidence.splitlines()
    headers = [i for i, line in enumerate(lines) if re.fullmatch(r"\s*Name\s+Age\s+Position\s*", line, re.I)]
    if len(headers) != 1:
        return None
    start = headers[0]
    intro = " ".join(lines[max(0, start - 4):start])
    if not re.search(r"\bexecutive\s+officers\b", intro, re.I):
        return None
    people = []
    complete = False
    for line in lines[start + 1:]:
        if re.match(r"(?:The\s+remaining\s+information|Item\s+\d|PART\s+[IVX]+|The\s+following)", line.strip(), re.I):
            complete = True
            break
        match = re.fullmatch(r"(?P<name>[A-Z][A-Za-z.‘’' -]+?)\s+(?P<age>\d{1,3})\s+(?P<position>[A-Za-z].+)", line.strip())
        if not match:
            # A known table boundary establishes the complete population. A
            # wrapped/truncated row or an ellipsis cannot close it safely.
            if not line.strip():
                continue
            complete = bool(re.match(r"(?:The\s+remaining\s+information|Item\s+\d|PART\s+[IVX]+|The\s+following)", line.strip(), re.I))
            break
        age = int(match.group("age"))
        if not 18 <= age <= 110:
            return None
        position = match.group("position")
        # Current office is before the first 'since/from'. Historical offices
        # later in a biography must not qualify a former vice president.
        current = re.split(r"\b(?:since|from)\b", position, maxsplit=1, flags=re.I)[0]
        people.append((match.group("name").strip(), age, current, line))
    if not complete or not people or len({name for name, *_ in people}) != len(people):
        return None
    population_source = "\n".join(person[3] for person in people)
    if re.search(r"\bvice\s+presidents?\b", question, re.I):
        people = [person for person in people if re.search(r"\bvice\s+president\b", person[2], re.I)]
    if not people:
        return None
    # Requested years must agree with the stated roster date, if present.
    asked = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    dated = re.search(r"\bas\s+of\s+(.+?\b(?:19|20)\d{2})", intro, re.I)
    if asked and (not dated or asked != set(re.findall(r"\b(?:19|20)\d{2}\b", dated.group(1)))):
        return None
    period = dated.group(1) if dated else ""
    identity = _identity(population_source, intro)
    roles, facts = [], {}
    for name, age, position, quote in people:
        role_name = _slug(name) + "_age"
        roles.append(RoleSpec(role_name, [name + " age"], period))
        facts[role_name] = GroundedFact(role_name, float(age), "years", quote, period=period,
                                       row_label=name + " age", fact_type="source_table_numeric",
                                       source_value=float(age), source_unit="years",
                                       source_section="Executive officers / " + position.strip(),
                                       source_column="Age", source_table_id=identity)
    names = list(facts)
    spec = ClaimSpec("officer_age_average", "(" + " + ".join(names) + ") / " + str(len(names)),
                     roles, "years", answer_spec_from_question(question).tolerance,
                     "generic_operation", period=period, operation="average",
                     planning_source="deterministic_source_table")
    return _Plan(spec, facts)


def _executive_salaries(question, evidence):
    operation = "average" if re.search(r"\b(?:average|mean)\b", question, re.I) else (
        "sum" if re.search(r"\b(?:total|sum)\b", question, re.I) else "")
    if not operation:
        return None
    lines = evidence.splitlines()
    headers = []
    for i, line in enumerate(lines):
        if re.search(r"\bNEO\s+Annual\s+Salary\b", line, re.I) and re.search(r"\bChange\s+in\b", line, re.I):
            headers.append((i, "comparison"))
        elif re.search(r"\bBase\s+Target\s+Performance\s+Performance\s+Payout\s+NEO\s+Salary\b", line, re.I):
            headers.append((i, "incentive"))
    if len(headers) != 1:
        return None
    index, kind = headers[0]
    context = " ".join(lines[max(0, index - 3):index + 1])
    # An explicit annual salary column or base salary column is needed; payout
    # amounts and total compensation must not be mistaken for salary.
    fiscal = re.findall(r"\bFY(\d{2}|\d{4})\b", context, re.I)
    question_fiscal = re.findall(r"\bFY(\d{2}|\d{4})\b", question, re.I)
    if question_fiscal and (not fiscal or question_fiscal[-1][-2:] != fiscal[-1][-2:]):
        return None
    period = "20" + fiscal[-1] if fiscal and len(fiscal[-1]) == 2 else (fiscal[-1] if fiscal else "")
    # PDF extractors often flatten all rows into one line. A printed '.' after
    # each name separates its optional note marker from the typed numeric cells.
    data_lines = []
    complete = False
    for line in lines[index + 1:]:
        if re.match(r"\s*\(\d+\)|\s*(?:As\s+presented|II?I?\.\s|The\s+following)", line, re.I):
            complete = True
            break
        if line.strip():
            data_lines.append(line)
    if not complete or not data_lines:
        return None
    body = " ".join(data_lines)
    name_pattern = r"(?P<name>[A-Z][A-Za-z.‘’' -]+?)(?:\s*\(\d+\))?\s+\.\s+"
    cell = r"(?:\$?\s*(?:\d[\d,]*(?:\.\d+)?|[—–-]|n/a))(?:\s*\(\d+\))?"
    count = 3 if kind == "comparison" else 5
    row_pattern = re.compile(name_pattern + r"(?P<values>" + cell + r"(?:\s+" + cell + r"){"
                             + str(count - 1) + r"})(?=\s+[A-Z]|\s*$)", re.I)
    people = []
    remaining = body
    while remaining.strip():
        match = row_pattern.match(remaining)
        if not match:
            return None
        name = match.group("name").strip()
        clean = re.sub(r"\(\d+\)", "", match.group("values"))
        raws = re.findall(r"\$?\s*\d[\d,]*(?:\.\d+)?|[—–-]|n/a", clean, re.I)
        if len(raws) != count:
            return None
        raw = raws[-1] if kind == "comparison" else raws[0]
        salary = _number(raw)
        if salary is None or salary < 0:
            return None
        people.append((name, salary, match.group(0).strip()))
        remaining = remaining[match.end():].lstrip()
    if not people or len({name for name, *_ in people}) != len(people):
        return None
    identity = _identity(body, context)
    roles, facts = [], {}
    for name, salary, quote in people:
        role_name = _slug(name) + "_salary"
        roles.append(RoleSpec(role_name, [name + " salary"], period))
        # Keep the quote as a literal substring, including original source
        # spacing, when rows were printed one-per-line or flattened together.
        if quote not in evidence:
            return None
        facts[role_name] = GroundedFact(role_name, salary, "USD", quote, period=period,
                                       row_label=name + " salary", fact_type="source_table_numeric",
                                       source_value=salary, source_unit="USD",
                                       source_section="Named executive officers / base salary",
                                       source_column=("FY" + period[-2:] + " Annual Salary") if period else "Base Salary",
                                       source_table_id=identity)
    names = list(facts)
    formula = " + ".join(names)
    if operation == "average":
        formula = "(" + formula + ") / " + str(len(names))
    spec = ClaimSpec("executive_salary_" + operation, formula, roles, "USD",
                     answer_spec_from_question(question).tolerance,
                     "generic_operation", period=period, operation=operation,
                     planning_source="deterministic_source_table")
    return _Plan(spec, facts)
