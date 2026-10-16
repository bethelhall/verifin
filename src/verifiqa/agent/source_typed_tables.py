"""Bounded source layouts whose non-numeric axes determine the measure.

No answer, benchmark identifier or evaluator annotation is accepted. Commodity
units, observation dates, comparative narrative scope and complete compound
ratio phrases are selected before any arithmetic is constructed.
"""
from __future__ import annotations

import re


def typed_source_plans(question, evidence, operation):
    plans = _commodity_price_plans(question, evidence, operation)
    plans.extend(_narrative_amount_plans(question, evidence, operation))
    plans.extend(promotion_salary_plans(question, evidence, operation))
    plans.extend(_quarterly_market_price_plans(question, evidence, operation))
    return plans


def typed_source_tables(evidence, *, question=""):
    from verifiqa.agent.table_fact_extractor import _MONTH
    dated = _dated_month_tables(evidence) if re.search(r"\b(?:" + _MONTH + r")\b", question, re.I) else []
    return (dated + _complete_prefix_tables(evidence) + _share_rollforward_tables(evidence, question)
            + _quarterly_per_share_tables(evidence, question))


def _foreign_currency(text):
    return bool(re.search(r"\b(?:EUR|euros?|CAD|Canadian\s+dollars?|AUD|Australian\s+dollars?|HKD|"
                          r"Hong\s+Kong\s+dollars?|CNY|RMB|GBP|sterling|JPY|yen|NTD)\b|NT\$|HK\$|C\$|A\$|€|£", text, re.I))


def _quarterly_market_price_plans(question, evidence, operation):
    """Bind one named price column in a complete, explicitly dated FY block."""
    from verifiqa.agent.source_question_planner import _Plan, _identity, _number
    from verifiqa.agent.types import ClaimSpec, GroundedFact, RoleSpec
    from verifiqa.answer_spec import answer_spec_from_question
    from verifiqa.verification.money_units import explicit_money_currencies, unrecognized_currency_unit
    if operation not in {"percent_change", "period_change"}:
        return []
    ordinal = r"(?:first|second|third|fourth)"
    request = re.fullmatch(r"\s*What (?:is|was) the (?:(?:percent|percentage) )?(?:difference|change) "
                           r"between (?:the )?(?P<first>" + ordinal + r")(?: quarter)? and (?:the )?"
                           r"(?P<last>" + ordinal + r") quarter (?P<kind>high|low) "
                           r"in (?:fiscal year|fiscal|FY)\s*(?P<year>(?:19|20)\d{2})\s*\??\s*", question, re.I)
    if request is None or request.group("first").lower() == request.group("last").lower():
        return []
    quarter_numbers = {name: index for index, name in enumerate(("first", "second", "third", "fourth"), 1)}
    first, last = sorted((quarter_numbers[request.group("first").lower()], quarter_numbers[request.group("last").lower()]))
    year, kind = request.group("year"), request.group("kind").lower()
    cell = r"\$?\s*\d[\d,]*(?:\.\d+)?"
    row_pattern = re.compile(r"\s*(?P<quarter>" + ordinal + r") Quarter\s+(?P<high>" + cell
                             + r")\s+(?P<low>" + cell + r")\s*", re.I)
    year_pattern = re.compile(r"\s*(?:(?P<before>(?:19|20)\d{2}) Fiscal Year|Fiscal Year (?P<after>(?:19|20)\d{2}))\s*", re.I)
    lines, plans = evidence.splitlines(), []
    for index, line in enumerate(lines):
        if (not re.fullmatch(r"\s*High\s+Low\s*", line, re.I) or index < 1
                or not re.fullmatch(r"\s*Period\s+Sales\s+Prices\s*", lines[index - 1], re.I)):
            continue
        caption = " ".join(lines[max(0, index - 3):index - 1])
        declared_codes = re.findall(r"\b(?:quoted|denominated|expressed|reported|amounts?|prices?(?:\s+are)?)\s+in\s+([A-Za-z]{3})\b"
                                    r"|\bcurrency\s*:\s*([A-Za-z]{3})\b", caption, re.I)
        for prior in lines[max(0, index - 3):index - 1]:
            declaration = re.fullmatch(r"\s*\(?\s*In\s+([A-Za-z]{3})(?:\s+(?:dollars?|thousands?|millions?))?\s*\)?\s*", prior, re.I)
            if declaration:
                declared_codes.append((declaration.group(1), ""))
        if (not re.search(r"\bhigh and low sales prices for (?:our|the Company[’']s) common stock\b", caption, re.I)
                or explicit_money_currencies(caption) - {"USD"}
                or any(unrecognized_currency_unit(code.upper()) for pair in declared_codes for code in pair
                       if code and code.lower() not in {"the", "our"})
                or _foreign_currency(caption)):
            continue
        blocks, current, current_quote, rows, invalid = [], "", "", {}, False

        def finish():
            if current == year:
                blocks.append((current_quote, dict(rows), invalid))

        for quote in lines[index + 1:]:
            block = year_pattern.fullmatch(quote)
            if block is not None:
                finish()
                current, current_quote = block.group("before") or block.group("after"), quote
                rows, invalid = {}, False
                continue
            if not quote.strip():
                continue
            if current != year:
                # Other fiscal-year blocks can contain explicitly partial
                # quarters. Their observations never supply this year's cell.
                if current and re.match(r"\s*" + ordinal + r" Quarter\b", quote, re.I):
                    continue
                break
            row = row_pattern.fullmatch(quote)
            if row is None:
                invalid = bool(re.match(r"\s*" + ordinal + r" Quarter\b", quote, re.I))
                break
            quarter = quarter_numbers[row.group("quarter").lower()]
            if quarter in rows:
                invalid = True
                break
            rows[quarter] = (quote, row.group("quarter").title() + " Quarter", row.group("high"), row.group("low"))
        finish()
        if len(blocks) != 1:
            continue
        year_quote, rows, invalid = blocks[0]
        if invalid or set(rows) != {1, 2, 3, 4} or not any("$" in cell for row in rows.values() for cell in row[2:]):
            continue
        section = caption + " / " + year_quote
        identity = _identity("\n".join(row[0] for row in rows.values()), section + line)
        roles, facts = [], {}
        for quarter in (first, last):
            quote, label, high, low = rows[quarter]
            raw = high if kind == "high" else low
            value = _number(raw)
            if value is None:
                break
            name, period = f"{kind}_q{quarter}_fy{year}", f"Q{quarter} FY{year}"
            roles.append(RoleSpec(name, [label + " " + kind.title()], period))
            facts[name] = GroundedFact(name, value, "USD per share", quote, period=period, row_label=label,
                fact_type="source_table_numeric", source_value=value, source_unit="USD per share",
                source_section=section, source_column=kind.title(), source_table_id=identity)
        if len(roles) != 2 or (operation == "percent_change" and facts[roles[0].name].value == 0):
            continue
        before, after = (role.name for role in roles)
        formula = f"({after} - {before}) / {before} * 100" if operation == "percent_change" else f"{after} - {before}"
        spec = ClaimSpec("quarterly_" + kind + "_price_" + operation, formula, roles,
            "percent" if operation == "percent_change" else "USD per share", answer_spec_from_question(question).tolerance,
            "generic_operation", period=f"FY{year}", operation=operation, planning_source="deterministic_source_table")
        plans.append(_Plan(spec, facts))
    return plans


def promotion_salary_plans(question, evidence, operation):
    """Bind an explicitly dated promotion's before/after salary, not FY pay."""
    from datetime import date
    from verifiqa.agent.source_question_planner import _Plan, _identity, _number, _slug
    from verifiqa.agent.types import ClaimSpec, GroundedFact, RoleSpec
    from verifiqa.answer_spec import answer_spec_from_question
    from verifiqa.agent.table_fact_extractor import _MONTH
    from verifiqa.verification.periods import period_key
    if (operation != "percent_change" or not re.search(r"\bpromotion\b", question, re.I)
            or re.search(r"\b(?:19|20)\d{2}\b|\bFY\d{2,4}\b", question, re.I)):
        return []
    named = re.findall(r"\bMr\.?\s+([A-Za-z][A-Za-z'’-]+)(?:['’]s)?\s+salary\b", question, re.I)
    if len(named) != 1:
        return []
    surname = re.sub(r"['’]s$", "", named[0], flags=re.I)
    lines = evidence.splitlines()
    headers = [index for index, line in enumerate(lines)
               if re.search(r"\bNEO\s+Annual\s+Salary\s*\(\$\)", line, re.I)]
    if len(headers) != 1:
        return []
    index = headers[0]
    heading = " ".join(lines[max(0, index - 2):index + 1])
    if not re.search(r"\b(?:NEO|named\s+executive\s+officers?).*\bbase\s+salary\b", heading, re.I) or _foreign_currency(heading):
        return []
    body = "\n".join(lines[index + 1:])
    name_pattern = re.compile(r"(?P<name>[A-Z][A-Za-z.'’-]*(?:\s+[A-Z][A-Za-z.'’-]*){1,5})\s*"
                              r"(?P<note>\(\d+\))?\s+\.\s+\$?\s*\d")
    people = [(match.group("name").strip(), match.group("note")) for match in name_pattern.finditer(body)
              if match.group("name").split()[-1].lower() == surname.lower()]
    if len(people) != 1 or not people[0][1]:
        return []
    person, marker = people[0]
    amount = r"\$\s*\d[\d,]*(?:\.\d+)?"
    footnote_pattern = re.compile(
        r"^\s*" + re.escape(marker) + r"\s+Mr\.?\s+" + re.escape(surname)
        + r"\s+(?P<context>[^.\n]{1,300})\.\s+His\s+salary\s+increased\s+from\s+"
        + r"(?P<before>" + amount + r")\s+to\s+(?P<after>" + amount + r")\s+effective\s+"
        + r"(?P<date>(?:" + _MONTH + r")\s+\d{1,2},?\s+(?:19|20)\d{2})\.", re.I | re.M)
    plans = []
    for match in footnote_pattern.finditer(body):
        if (not re.search(r"\bpromotion\b", match.group("context"), re.I)
                or not re.search(r"\bsalary\s+increase\b", match.group("context"), re.I)
                or re.search(r"\b(?:Mr|Ms|Mrs|Dr)\.?\s+[A-Za-z]|\b[A-Z][a-z]+(?:[ '-][A-Z][a-z]+)+\b", match.group("context"))
                or _foreign_currency(match.group())):
            continue
        event_date = match.group("date")
        key = period_key(event_date)
        try:
            date(*key)
        except (TypeError, ValueError):
            continue
        values = [_number(match.group(state)) for state in ("before", "after")]
        if any(value is None or value <= 0 for value in values) or values[1] <= values[0]:
            continue
        quote = match.group().strip()
        identity = _identity(heading + "\n" + person + "\n" + quote, "Dated promotion salary")
        roles, facts = [], {}
        for state, value in zip(("before", "after"), values):
            name = _slug(person + " salary " + state + " promotion " + event_date)
            label = person + " salary " + state + " promotion"
            roles.append(RoleSpec(name, [label], event_date))
            facts[name] = GroundedFact(name, value, "USD", quote, period=event_date,
                                       row_label=label, fact_type="source_table_numeric", source_value=value,
                                       source_unit="USD", source_section=heading + " / promotion / " + event_date,
                                       source_column="Salary " + state + " promotion effective " + event_date,
                                       source_table_id=identity)
        before, after = list(facts)
        spec = ClaimSpec(_slug(person) + "_promotion_salary_percent_change",
                         f"({after} - {before}) / {before} * 100", roles, "percent",
                         answer_spec_from_question(question).tolerance, "generic_operation",
                         period=event_date, operation=operation, planning_source="deterministic_source_table")
        plans.append(_Plan(spec, facts))
    return plans


def _quarterly_per_share_tables(evidence, question):
    """Retain the EPS group, share class, four exact dates and unscaled rate."""
    from datetime import date
    from verifiqa.agent.source_question_planner import _Table, _Row, _identity, _number
    from verifiqa.agent.table_fact_extractor import _annual_row_groups, _MONTH
    from verifiqa.verification.periods import period_key
    if (not re.search(r"\bper\s+share\b", question, re.I)
            or not re.search(r"\b(?:basic|diluted)\b", question, re.I)):
        return []
    lines, tables = evidence.splitlines(), []
    for index, line in enumerate(lines[:-2]):
        if not re.fullmatch(r"\s*Three\s+Months\s+Ended\s*", line, re.I) or index == 0:
            continue
        fiscal = lines[index - 1]
        if not re.fullmatch(r"\s*Fiscal\s+Year\s+(?:19|20)\d{2}\s*", fiscal, re.I):
            continue
        header = lines[index + 1]
        caption = re.match(r"\s*\(in\s+(?:thousands|millions|billions),\s*except\s+(?:for\s+)?per\s+share\s+data\)\s*", header, re.I)
        if not caption or _foreign_currency(" ".join(lines[max(0, index - 2):index + 2])):
            continue
        date_text = header[caption.end():]
        dates = [match.group().strip() for match in re.finditer(
            r"\b(?:" + _MONTH + r")\s+\d{1,2},?\s+(?:19|20)\d{2}\b", date_text, re.I)]
        if len(dates) != 4 or re.sub(r"\s+", "", "".join(dates)) != re.sub(r"\s+", "", date_text):
            continue
        keys = [period_key(value) for value in dates]
        if None in keys or len(set(keys)) != 4:
            continue
        month_indices = [year * 12 + month for year, month, _day in keys]
        gaps = [left - right for left, right in zip(month_indices, month_indices[1:])]
        if gaps not in ([3, 3, 3], [-3, -3, -3]):
            continue
        try:
            for key in keys:
                date(*key)
        except ValueError:
            continue
        body_lines = []
        for body in lines[index + 2:]:
            if re.match(r"\s*(?:Fiscal\s+Year\s+(?:19|20)\d{2}\b|\(\d+\)(?:\s|$))", body, re.I):
                break
            if not body.strip():
                if body_lines:
                    break
                continue
            body_lines.append(body)
        body = "\n".join(body_lines)
        groups = list(re.finditer(r"(?P<group>(?:\(Loss\)\s*)?Earnings\s+per\s+share\b[A-Za-z .,’'&()-]{0,180}?)\s+(?=Basic\s*\.)", body, re.I))
        if len(groups) != 1:
            continue
        group = groups[0].group("group").strip()
        suffix = body[groups[0].end():]
        parsed = _annual_row_groups(suffix, 4)
        if not parsed or len(parsed) != 2 or [label.lower() for _, label, _ in parsed] != ["basic", "diluted"]:
            continue
        if not any("$" in raw for _, _, raws in parsed for raw in raws):
            continue
        quote = body[groups[0].start():]
        table = _Table(["Three months ended " + value for value in dates],
                       section=fiscal.strip() + " / " + group, unit="USD per share",
                       identity=_identity(header + "\n" + quote, fiscal), unit_declared=True)
        for row_section, label, raws in parsed:
            values = [_number(raw) for raw in raws]
            if any(value is None for value in values) or any("%" in raw for raw in raws):
                table = None
                break
            table.rows.append(_Row(label + " " + group, values, raws, quote, row_section, "USD per share"))
        if table is not None:
            tables.append(table)
    return tables


def _commodity_price_plans(question, evidence, operation):
    from verifiqa.agent.source_question_planner import _Table, _Row, _identity, _number, _table_plan
    if operation not in {"average", "period_change", "percent_change"} or not re.search(r"\bprices?\b", question, re.I):
        return []
    lines, result = evidence.splitlines(), []
    unit_pattern = r"Troy\s+(?:oz\.?|ounces?)|Lb\.?|pounds?|kg|kilograms?|grams?|metric\s+tons?|oz\.?"
    numeric = r"\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?"
    for index, line in enumerate(lines):
        matched = re.fullmatch(r"\s*Measure\s+((?:19|20)\d{2}(?:\s+(?:19|20)\d{2})+)\s*", line, re.I)
        if not matched:
            continue
        years = re.findall(r"(?:19|20)\d{2}", matched.group(1))
        if len(set(years)) != len(years):
            continue
        context = " ".join(lines[max(0, index - 3):index])
        if (_foreign_currency(context)
                or not re.search(r"\b(?:average\s+)?prices?\s+(?:incurred|paid)\b", context, re.I)):
            continue
        row_pattern = re.compile(r"\s*(?P<label>[A-Za-z][A-Za-z -]*?)\s+(?P<measure>" + unit_pattern
                                 + r")\s+(?P<values>" + numeric + r"(?:\s+" + numeric + r"){"
                                 + str(len(years) - 1) + r"})\s*", re.I)
        table = _Table(years, section="Commodity prices / " + context, unit_declared=True)
        raw_lines, has_currency = [line], False
        for row_line in lines[index + 1:]:
            parsed = row_pattern.fullmatch(row_line)
            if not parsed:
                if any(re.match(r"\s*" + re.escape(row.label) + r"\b", row_line, re.I) for row in table.rows):
                    table = None
                break
            raws = [match.group().strip() for match in re.finditer(numeric, parsed.group("values"))]
            values = [_number(raw) for raw in raws]
            if len(raws) != len(years) or any(value is None for value in values):
                table = None
                break
            physical_unit = _physical_unit(parsed.group("measure"))
            if not physical_unit:
                table = None
                break
            has_currency |= any("$" in raw for raw in raws)
            requested_unit = re.search(r"\bper\s+(" + unit_pattern + r")\b", question, re.I)
            if requested_unit and _physical_unit(requested_unit.group(1)) != physical_unit:
                continue
            table.rows.append(_Row(parsed.group("label").strip(), values, raws, row_line,
                                   "Price per " + parsed.group("measure"), "USD per " + physical_unit))
            raw_lines.append(row_line)
        if not table or not table.rows or not has_currency:
            continue
        table.identity = _identity("\n".join(raw_lines), context)
        plan = _table_plan(question, operation, table)
        if plan is not None:
            result.append(plan)
    return result


def _physical_unit(text):
    text = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return {"troy oz": "troy ounce", "troy ounce": "troy ounce", "troy ounces": "troy ounce",
            "lb": "pound", "pound": "pound", "pounds": "pound",
            "kg": "kilogram", "kilogram": "kilogram", "kilograms": "kilogram",
            "gram": "gram", "grams": "gram", "metric ton": "metric ton", "metric tons": "metric ton",
            "oz": "ounce"}.get(text)


def _narrative_amount_plans(question, evidence, operation):
    from verifiqa.agent.source_question_planner import _Table, _Row, _identity, _number, _table_plan
    if operation not in {"average", "period_change", "percent_change"}:
        return []
    from verifiqa.agent.table_fact_extractor import _MONTH
    number = r"\$\s*\(?-?\d[\d,]*(?:\.\d+)?\)?"
    pattern = re.compile(
        r"(?:^|(?<=[.!?])\s+)(?:The\s+)?(?P<label>[A-Za-z][A-Za-z ,’'/-]{2,180}?)\s+(?:was|were)\s+"
        r"(?P<a>" + number + r")\s+(?P<scale_a>thousand|million|billion)s?\s+and\s+"
        r"(?P<b>" + number + r")\s+(?P<scale_b>thousand|million|billion)s?\s+at\s+"
        r"(?P<day>(?:" + _MONTH + r")\s+\d{1,2}),?\s+(?P<year_a>(?:19|20)\d{2})\s+and\s+"
        r"(?P<year_b>(?:19|20)\d{2}),?\s+respectively\.", re.I | re.M)
    plans = []
    for match in pattern.finditer(evidence):
        if _foreign_currency(evidence[max(0, match.start() - 250):match.end()]):
            continue
        if match.group("scale_a").lower() != match.group("scale_b").lower() or match.group("year_a") == match.group("year_b"):
            continue
        label = match.group("label").strip()
        unit = "USD " + match.group("scale_a").lower() + "s"
        raws = [match.group("a"), match.group("b")]
        values = [_number(raw) for raw in raws]
        if any(value is None for value in values):
            continue
        quote = match.group().lstrip()
        section = _narrative_section(evidence, match.start(), label)
        table = _Table([match.group("day") + " " + year for year in (match.group("year_a"), match.group("year_b"))],
                       [_Row(label, values, raws, quote, unit=unit)], section, unit,
                       _identity(quote, section), unit_declared=True)
        plan = _table_plan(question, operation, table)
        if plan is not None:
            plans.append(plan)
    return plans


def _narrative_section(evidence, start, label):
    """Keep nearby printed headings without importing a preceding sentence.

    A comparative sentence may describe a component under an impairment or
    tax heading. Its otherwise ordinary row label cannot erase that context.
    A prior narrative sentence, however, is a different observation and must
    not lend its geography to the next sentence.
    """
    line_start = evidence.rfind("\n", 0, start) + 1
    if evidence[line_start:start].strip():
        return label
    headings, blanks = [], 0
    for line in reversed(evidence[:line_start].splitlines()[-4:]):
        text = line.strip()
        if not text:
            if headings or blanks:
                break
            blanks += 1
            continue
        if (len(text) > 160 or len(text.split()) > 18
                or re.search(r"[.!?]\s*$|\$\s*\d|\b(?:was|were|is|are|reported|included|amounted)\b", text, re.I)):
            break
        headings.append(text)
    return " / ".join(list(reversed(headings)) + [label])


def _share_rollforward_tables(evidence, question):
    """Type a bounded, explicitly unscaled common-share rollforward as counts."""
    from verifiqa.agent.source_question_planner import _Table, _Row, _identity, _number
    from verifiqa.agent.table_fact_extractor import _annual_row_groups, _MONTH
    if re.search(r"\b(?:treasury|preferred|compensation|expense|cash|dollars?|USD)\b", question, re.I):
        return []
    lines, tables = evidence.splitlines(), []
    caption = re.compile(r"\s*(?:A\s+)?roll\s*forward\s+of\s+common\s+shares\s+outstanding\s+is\s+as\s+follows:\s*", re.I)
    for index, line in enumerate(lines[:-3]):
        if not caption.fullmatch(line):
            continue
        if not re.fullmatch(r"\s*As\s+of\s+(?:" + _MONTH + r")\s+\d{1,2},?\s*", lines[index + 1], re.I):
            continue
        if not re.fullmatch(r"\s*(?:19|20)\d{2}(?:\s+(?:19|20)\d{2})+\s*", lines[index + 2]):
            continue
        years = re.findall(r"(?:19|20)\d{2}", lines[index + 2])
        if len(set(years)) != len(years):
            continue
        section = line.strip()
        table = _Table([lines[index + 1].strip() + " " + year for year in years],
                       section=section, unit="count", unit_declared=True)
        raw_lines = [line, lines[index + 1], lines[index + 2]]
        for body in lines[index + 3:]:
            if not body.strip() or re.match(r"\s*(?:NOTE\s+\d+|The\s+following\s+table)\b", body, re.I):
                break
            parsed = _annual_row_groups(body, len(years))
            if not parsed or "$" in body or "%" in body:
                table = None
                break
            for row_section, label, raws in parsed:
                if re.search(r"\b(?:expense|compensation|cash|price|value|cost)\b", label, re.I):
                    table = None
                    break
                values = [_number(raw) for raw in raws]
                if any(value is None for value in values):
                    table = None
                    break
                table.rows.append(_Row(label, values, raws, body, row_section, "count"))
            if table is None:
                break
            raw_lines.append(body)
        if table is not None and len(table.rows) >= 2:
            table.identity = _identity("\n".join(raw_lines), section)
            tables.append(table)
    return tables


def _dated_month_tables(evidence):
    from verifiqa.agent.source_question_planner import _Table, _Row, _identity, _number, source_row_unit
    from verifiqa.agent.table_fact_extractor import _MONTH, _annual_row_groups, _caption_money_unit
    lines, tables = evidence.splitlines(), []
    pattern = re.compile(r"\s*(?P<label>[A-Za-z][A-Za-z ,&'’/-]+?)\s+at\s+"
                         r"(?P<date_a>(?:" + _MONTH + r")\s+(?:19|20)\d{2})\s+and\s+"
                         r"(?P<date_b>(?:" + _MONTH + r")\s+(?:19|20)\d{2})\s+"
                         r"(?:consisted|consists)\s+of\s+the\s+following:\s*", re.I)
    for index, line in enumerate(lines[:-1]):
        caption = pattern.fullmatch(line)
        if not caption:
            continue
        dates = [caption.group("date_a"), caption.group("date_b")]
        expected = [re.search(r"(?:19|20)\d{2}", value).group() for value in dates]
        if expected[0] == expected[1] or re.findall(r"(?:19|20)\d{2}", lines[index + 1]) != expected:
            continue
        if re.sub(r"(?:19|20)\d{2}|\s", "", lines[index + 1]):
            continue
        unit = _caption_money_unit(line)
        table = _Table(dates, section=caption.group("label"), unit=unit,
                       identity=_identity(line + "\n" + lines[index + 1], "dated month table"), unit_declared=bool(unit),
                       relative_scale_only=True)
        for row_line in lines[index + 2:]:
            parsed = _annual_row_groups(row_line, 2)
            if parsed is None:
                break
            for section, label, raws in parsed:
                row_unit = source_row_unit(label, section, table.unit, raws)
                if not table.unit and any("$" in raw for raw in raws):
                    table.unit = "USD"
                table.rows.append(_Row(label, [_number(raw) for raw in raws], raws, row_line, section, row_unit))
        if table.rows:
            tables.append(table)
    return tables


def _complete_prefix_tables(evidence):
    from verifiqa.agent.source_question_planner import _Table, _Row, _identity, _number, source_row_unit
    from verifiqa.agent.table_fact_extractor import _PDF_CELL, _annual_row_groups, _caption_money_unit, _MONTH
    lines, tables = evidence.splitlines(), []
    for index, line in enumerate(lines):
        if not re.fullmatch(r"\s*(?:19|20)\d{2}(?:\s+(?:19|20)\d{2})+\s*", line):
            continue
        years = re.findall(r"(?:19|20)\d{2}", line)
        if len(set(years)) != len(years) or index == 0 or index + 1 >= len(lines):
            continue
        prefix = lines[index - 1]
        if not re.fullmatch(r"\s*(?:Fiscal\s+)?Years?\s+Ended\s+(?:" + _MONTH + r")\s+\d{1,2},?\s*", prefix, re.I):
            continue
        unit = next((_caption_money_unit(text) for text in reversed(lines[max(0, index - 3):index])
                     if _caption_money_unit(text)), "")
        if not unit:
            continue
        body = lines[index + 1]
        if _annual_row_groups(body, len(years)) is not None:
            continue  # The ordinary parser already supports this complete row.
        number = _PDF_CELL + r"(?:\s*%)?"
        values = number + r"(?:\s+" + number + r"){" + str(len(years) - 1) + r"}"
        label = r"[A-Za-z](?:[A-Za-z\s.,/&'’“”\-—–]|\(\d{1,2}\))*?"
        row_pattern = re.compile(r"\s*(?P<label>" + label + r")\s+(?P<values>" + values + r")(?=\s*$|\s+[A-Za-z])")
        table = _Table([(prefix.strip() + " " + year).replace(", ", " ") for year in years],
                       section=" ".join(lines[max(0, index - 2):index]), unit=unit,
                       identity=_identity(body, prefix + " " + line), unit_declared=True)
        remaining = body
        while remaining.strip():
            parsed = row_pattern.match(remaining)
            if not parsed:
                if re.match(r"\s*[$(\d—–-]", remaining):
                    table = None  # Extra numeric cells cannot become labels.
                break
            row_label = re.sub(r"\s*\(\d{1,2}\)\s*[. ]*$", "", parsed.group("label")).strip(" .")
            raws = [m.group().strip() for m in re.finditer(number, parsed.group("values"))]
            if len(raws) != len(years):
                table = None
                break
            table.rows.append(_Row(row_label, [_number(raw) for raw in raws], raws,
                                   parsed.group().strip(), unit=source_row_unit(row_label, "", unit, raws)))
            remaining = remaining[parsed.end():]
        if table is not None and table.rows and remaining.strip():
            tail = re.sub(r"[^a-z0-9]+", " ", remaining.lower())
            if any(" " + re.sub(r"[^a-z0-9]+", " ", row.label.lower()).strip() + " " in " " + tail + " "
                   for row in table.rows):
                continue
            tables.append(table)
    return tables


def table_phrase_rows(table, phrase, column_index):
    """Return the exact complete source rows covering one ratio side."""
    from verifiqa.types import VerificationIR, VerificationFact
    from verifiqa.verification.operation_bindings import _complete_phrase_parts
    facts, rows = {}, {}
    for index, row in enumerate(table.rows):
        if not row.unit or row.values[column_index] is None:
            continue
        name = "source_row_" + str(index)
        rows[name] = row
        facts[name] = VerificationFact(name, row.values[column_index], row.unit,
                                       row_label=row.label, period=table.columns[column_index],
                                       source_column=table.columns[column_index])
    ir = VerificationIR("source_phrase", "", facts, 0, "", 0)
    selected = _complete_phrase_parts(ir, phrase)
    return [rows[name] for name in selected] if selected else None


def compound_table_plan(question, operation, table):
    if operation not in {"proportion", "generic_ratio", "sum_then_ratio"}:
        return None
    from verifiqa.agent.source_question_planner import _Plan, _scope_supported, _slug
    from verifiqa.agent.types import ClaimSpec, GroundedFact, RoleSpec
    from verifiqa.answer_spec import answer_spec_from_question
    from verifiqa.verification.operation_bindings import compound_ratio_phrases
    phrases = compound_ratio_phrases(question)
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    if not phrases or len(years) != 1:
        return None
    wanted = next(iter(years))
    columns = [i for i, column in enumerate(table.columns)
               if re.findall(r"\b(?:19|20)\d{2}\b", column) == [wanted]]
    if len(columns) != 1:
        return None
    index = columns[0]
    numerator, denominator = [table_phrase_rows(table, phrase, index) for phrase in phrases]
    if not numerator or not denominator or len(numerator) + len(denominator) < 3:
        return None
    if len({id(row) for row in numerator + denominator}) != len(numerator) + len(denominator):
        return None
    if any(not _scope_supported(question, table, row) for row in numerator + denominator):
        return None
    units = {row.unit for row in numerator + denominator}
    if len(units) != 1 or sum(row.values[index] for row in denominator) == 0:
        return None
    roles, facts, groups = [], {}, []
    for rows in (numerator, denominator):
        names = []
        for row in rows:
            name = _slug(row.label) + "_" + wanted
            if name in facts:
                return None
            roles.append(RoleSpec(name, [row.label], table.columns[index]))
            facts[name] = GroundedFact(name, row.values[index], row.unit, row.quote,
                                       period=table.columns[index], row_label=row.label, fact_type="source_table_numeric",
                                       source_value=row.values[index], source_unit=row.unit,
                                       source_section=" / ".join(part for part in (table.section, row.section) if part),
                                       source_column=table.columns[index], source_table_id=table.identity)
            names.append(name)
        groups.append(" + ".join(names))
    is_percent = operation == "proportion" or bool(re.search(r"\bpercent(?:age)?\b", question, re.I))
    formula = f"({groups[0]}) / ({groups[1]})" + (" * 100" if is_percent else "")
    spec = ClaimSpec(_slug(phrases[0]) + "_ratio", formula, roles, "percent" if is_percent else "ratio",
                     answer_spec_from_question(question).tolerance, "generic_operation", period=table.columns[index],
                     operation=operation, planning_source="deterministic_source_table")
    return _Plan(spec, facts)
