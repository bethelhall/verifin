"""Question-bound plans for explicitly named financial table subcolumns.

Column types belong to the selected observation. A fair value is not a
carrying value, and a target allocation is not an observed allocation. These
readers require the complete printed header before reading any numeric cell.
"""
from __future__ import annotations

import re


_YEAR = r"(?:19|20)\d{2}"
_MONTH = (r"January|February|March|April|May|June|July|August|September|October|"
          r"November|December")
_DATE = r"(?:" + _MONTH + r")\s+\d{1,2},?\s+" + _YEAR


def _foreign_currency_caption(section):
    # The shared legacy caption reader assumes USD for an unqualified scale.
    # That default cannot overwrite an explicitly different currency here.
    return bool(re.search(r"\b(?:EUR|GBP|JPY|CHF|CAD|CNY|AUD|HKD|RMB|SEK|SGD|KRW|TWD|ZAR)\b"
                          r"|\b(?:euros?|yen|pounds\s+sterling)\b|NT\$|HK\$|C\$|A\$"
                          r"|\b(?:Canadian|Australian|New\s+Zealand|Hong\s+Kong|Singapore|(?:New\s+)?Taiwan)\s+dollars?\b", section, re.I))


def _starts_row(text, label):
    return bool(re.match(r"\s*" + r"\s+".join(re.escape(word) for word in label.split()) + r"\b", text, re.I))


def column_layout_plans(question: str, evidence: str, operation: str | None):
    """Return independently supported plans; the caller rejects competing ones."""
    plans = _denominated_currency_plans(question, evidence, operation)
    plans.extend(_customer_amount_plans(question, evidence, operation))
    plans.extend(_repurchase_price_plans(question, evidence, operation))
    plans.extend(_revenue_column_plans(question, evidence, operation))
    plans.extend(_undated_total_debt_value_plans(question, evidence, operation))
    if operation != "line_item_difference":
        return plans
    if re.search(r"\bcarrying\s+value\b", question, re.I) and re.search(
            r"\bfair\s+value\b", question, re.I):
        plans.extend(_carrying_fair_plans(question, evidence))
    if re.search(r"\btarget\s+allocations?\b", question, re.I):
        plans.extend(_target_allocation_plans(question, evidence))
    return plans


def _undated_total_debt_value_plans(question, evidence, operation):
    """Use the unique complete two-date debt comparison, including its footer.

    An undated change can use these source dates only when the specifically
    named debt table has exactly one comparison and a reconciled total. A
    different period population or a second such table remains ambiguous.
    """
    from datetime import date
    from decimal import Decimal
    from verifiqa.agent.source_question_planner import _Plan, _identity, _number, _table_definition
    from verifiqa.agent.source_table_layouts import _CELL, _row_groups
    from verifiqa.agent.table_fact_extractor import _caption_money_unit, _printed_heading
    from verifiqa.agent.types import ClaimSpec, GroundedFact, RoleSpec
    from verifiqa.answer_spec import answer_spec_from_question
    from verifiqa.verification.periods import period_key
    if operation != "period_change":
        return []
    request = re.fullmatch(r"\s*What (?:is|was) the (?:increase\s*/\s*\(decrease\)|change) in "
                           r"the total debt (?P<kind>carrying|fair) value\s*\??\s*", question, re.I)
    if request is None:
        return []
    kind = request.group("kind").title() + " Value"
    lines, plans, comparable = evidence.splitlines(), [], 0
    cells = _CELL + r"(?:\s+" + _CELL + r"){3}"
    footer_pattern = re.compile(r"\s*(?P<values>" + cells + r")\s*", re.I)
    for index, line in enumerate(lines[:-1]):
        if not re.fullmatch(r"\s*(?:" + _DATE + r"\s*){2,}\s*", line, re.I):
            continue
        prior = [line for line in lines[max(0, index - 3):index] if line.strip() and line.strip() != "?"]
        caption = _table_definition(prior[-1]) if prior else ""
        if not re.fullmatch(r"The (?:fair value and carrying value|carrying value and fair value) of "
                            r"the Company[’']s long[- ]term debt were as follows(?:\s*\([^)]*\))?\s*:", caption, re.I):
            continue
        comparable += 1
        dates = [match.group().strip() for match in re.finditer(_DATE, line, re.I)]
        keys = [period_key(value) for value in dates]
        if (len(dates) != 2 or any(key is None or not key[2] for key in keys)
                or len({key[0] for key in keys}) != 2):
            continue
        try:
            for key in keys:
                date(*key)
        except (TypeError, ValueError):
            continue
        if not re.fullmatch(r"\s*Carrying Value Fair Value Carrying Value Fair Value\s*", lines[index + 1], re.I):
            continue
        unit = _caption_money_unit(caption)
        if not unit or _foreign_currency_caption(caption):
            continue
        components, total, invalid = [], None, False
        for quote in lines[index + 2:]:
            if not quote.strip():
                continue
            footer = footer_pattern.fullmatch(quote)
            parsed = _row_groups(quote, 4) if footer is None else None
            if footer is not None:
                if total is not None:
                    invalid = True
                    break
                total = ([match.group().strip() for match in re.finditer(_CELL, footer.group("values"))], quote)
                continue
            if parsed is None:
                if (total is None or any(_starts_row(quote, label) for label, _, _ in components)
                        or (not _printed_heading(quote) and not re.search(r"[.!?]", quote))):
                    invalid = True
                break
            if total is not None:
                invalid = True  # A total cannot hide further members of this grid.
                break
            components.extend((label, raws, quote) for label, raws in parsed)
        if (invalid or total is None or len(components) < 2
                or len({label.lower() for label, _, _ in components}) != len(components)):
            continue
        raws, quote = total
        all_raws = [row_raws for _, row_raws, _ in components] + [raws]
        if any(len(row) != 4 or any("%" in raw or _number(raw) is None for raw in row) for row in all_raws):
            continue
        def precise(raw):
            value = re.sub(r"[$,\s]", "", raw)
            return -Decimal(value[1:-1]) if value.startswith("(") else Decimal(value)
        if any(sum(precise(row[column]) for _, row, _ in components) != precise(raws[column]) for column in range(4)):
            continue
        identity = _identity("\n".join([line, lines[index + 1]] + [row_quote for _, _, row_quote in components] + [quote]), caption)
        roles, facts = [], {}
        for position in sorted(range(2), key=lambda position: keys[position]):
            period = dates[position]
            name = "total_debt_" + kind.lower().replace(" ", "_") + "_" + str(keys[position][0])
            value = _number(raws[position * 2 + (0 if kind == "Carrying Value" else 1)])
            roles.append(RoleSpec(name, ["Total debt " + kind.lower()], period))
            facts[name] = GroundedFact(name, value, unit, quote, period=period, row_label="Total debt " + kind.lower(),
                fact_type="source_table_numeric", source_value=value, source_unit=unit, source_section=caption,
                source_column=period + " | " + kind, source_table_id=identity)
        before, after = (role.name for role in roles)
        spec = ClaimSpec("total_debt_" + kind.lower().replace(" ", "_") + "_period_change", f"{after} - {before}",
            roles, unit, answer_spec_from_question(question).tolerance, "generic_operation", period=roles[-1].period,
            operation="period_change", planning_source="deterministic_source_table")
        plans.append(_Plan(spec, facts))
    return plans if comparable == 1 else []


def _repurchase_price_plans(question, evidence, operation):
    """Select reported fiscal-year per-share prices, never monthly means."""
    from verifiqa.agent.source_question_planner import _Row, _Table, _identity, _number, _table_plan
    from verifiqa.agent.source_table_layouts import _CELL
    if operation not in {"average", "period_change", "percent_change"} or not re.search(
            r"\baverage\s+price\s+paid\s+per\s+share\b", question, re.I):
        return []
    lines, plans = evidence.splitlines(), []
    header_pattern = re.compile(
        r"Period\s+Total\s+number\s+of\s+shares\s+purchased\s+Average\s+price\s+paid\s+per\s+share\s+"
        r"Total\s+number\s+of\s+shares\s+purchased\s+as\s+part\s+of\s+a\s+publicly\s+announced\s+program"
        r"(?:\s*\([a-z]\))?\s+Approximate\s+Dollar\s+Value\s+of\s+Shares\s+that\s+may\s+yet\s+be\s+"
        r"Purchased\s+Under\s+the\s+Plans\s+or\s+Programs\s*\(\$\s+in\s+thousands\)", re.I)
    annual = re.compile(r"\s*Fiscal\s+Year\s+(?P<year>" + _YEAR + r")\s+(?P<values>" + _CELL +
                        r"(?:\s+" + _CELL + r"){3})\s*", re.I)
    for index, line in enumerate(lines):
        if not re.fullmatch(r"\s*Period\s*", line, re.I):
            continue
        header_end = index
        while header_end < len(lines) and not re.match(r"\s*(?:\d{1,2}/\d{1,2}/|Fiscal\s+Year\s+" + _YEAR + r")", lines[header_end], re.I):
            header_end += 1
        header = " ".join(lines[index:header_end])
        title = " ".join(lines[max(0, index - 2):index])
        if (not header_pattern.fullmatch(header.strip())
                or not re.search(r"Issuer\s+Purchases\s+of\s+Equity\s+Securities", title, re.I)
                or _foreign_currency_caption(header + " " + title)):
            continue
        observations, invalid, end = [], False, len(lines)
        for row_index in range(header_end, len(lines)):
            quote = lines[row_index]
            if re.match(r"\s*\([a-z]\)\s|\s*ITEM\s+\d", quote, re.I):
                end = row_index
                break
            if not re.match(r"\s*Fiscal\s+Year\b", quote, re.I):
                continue
            match = annual.fullmatch(quote)
            if match is None:
                invalid = True
                break
            raws = [m.group().strip() for m in re.finditer(_CELL, match.group("values"))]
            if len(raws) != 4 or any("%" in raw for raw in raws) or "$" not in raws[1]:
                invalid = True
                break
            observations.append((match.group("year"), raws[1]))
        if invalid or len(observations) < 2 or len({year for year, _ in observations}) != len(observations):
            continue
        table_quote = "\n".join(lines[index:end])
        row = _Row("Average price paid per share", [_number(raw) for _, raw in observations],
                   [raw for _, raw in observations], table_quote, unit="USD per share")
        table = _Table(["Fiscal Year " + year for year, _ in observations], [row], title,
                       "USD per share", _identity(table_quote, title), True)
        plan = _table_plan(question, operation, table)
        if plan is not None:
            for fact in plan.facts.values():
                fact.source_column += " | Average price paid per share"
            plans.append(plan)
    return plans


def _revenue_column_plans(question, evidence, operation):
    """Use a closed, reconciled revenue grid's named column and total column."""
    from verifiqa.agent.source_question_planner import _Row, _Table, _build, _identity, _number, _scope_supported, _tokens
    from verifiqa.agent.source_table_layouts import _row_groups
    from verifiqa.agent.table_fact_extractor import _caption_money_unit
    if operation != "proportion":
        return []
    request = re.search(r"\bwhat\s+percentage\s+of\s+(?P<base>.+?)\s+consists\s+of\s+(?P<part>.+?)\??\s*$", question, re.I)
    if request is None:
        return []
    base_words = set(_tokens(request.group("base")))
    if not {"total", "revenue"} <= base_words or not base_words <= {"total", "revenue", "all", "source"}:
        return []
    lines, plans = evidence.splitlines(), []
    for index, line in enumerate(lines):
        caption = re.fullmatch(r"\s*The\s+following\s+table\s+disaggregates\s+(?:our|the\s+Company[’']s)\s+"
                               r"revenue\s+by\s+major\s+source\s+for\s+the\s+year\s+ended\s+(?P<date>" + _DATE + r"):\s*", line, re.I)
        if caption is None or index + 5 >= len(lines):
            continue
        year = re.findall(_YEAR, caption.group("date"))[0]
        requested_years = set(re.findall(_YEAR, question))
        if requested_years and requested_years != {year}:
            continue
        unit = _caption_money_unit(lines[index + 1])
        if not unit.startswith("USD") or _foreign_currency_caption(lines[index + 1]):
            continue
        first = re.fullmatch(r"\s*([A-Za-z][A-Za-z '&’-]+)\s*", lines[index + 2])
        second_a = re.fullmatch(r"\s*([A-Za-z][A-Za-z '-]*&)\s*", lines[index + 3])
        second_b = re.fullmatch(r"\s*([A-Za-z][A-Za-z '-]*)\s+Total\s*", lines[index + 4])
        if not first or not second_a or not second_b:
            continue
        headers = [first.group(1).strip(), second_a.group(1).strip() + " " + second_b.group(1).strip(), "Total"]
        part_words = {word for word in _tokens(request.group("part")) if not word.isdigit()}
        selected = [i for i, header in enumerate(headers[:2])
                    if set(_tokens(header)) <= part_words and part_words <= set(_tokens(header)) | {"total", "revenue"}]
        if len(selected) != 1:
            continue
        source_rows, closed = [], False
        close_index = None
        for source_index in range(index + 5, len(lines)):
            quote = lines[source_index]
            parsed = _row_groups(quote, 3)
            if parsed is None:
                break
            for label, raws in parsed:
                values = [_number(raw) for raw in raws]
                if any(value is None for value in values) or any("%" in raw for raw in raws):
                    source_rows = []
                    break
                if abs(values[0] + values[1] - values[2]) > 1e-8:
                    source_rows = []
                    break
                source_rows.append((label, values, raws, quote))
                if label.lower() == "total":
                    closed = True
                    close_index = source_index
                    break
            if closed or not source_rows:
                break
        if not closed or len(source_rows) < 2 or len({label.lower() for label, _, _, _ in source_rows}) != len(source_rows):
            continue
        if close_index + 1 < len(lines):
            following = _row_groups(lines[close_index + 1], 3)
            if following and any(label.lower() == "total" for label, _ in following):
                continue
        total = source_rows[-1]
        if any(abs(sum(row[1][i] for row in source_rows[:-1]) - total[1][i]) > 1e-8 for i in range(3)):
            continue
        col = selected[0]
        quote = total[3]
        rows = [_Row("Total revenue " + headers[col], [total[1][col]], [total[2][col]], quote, unit=unit),
                _Row("Total revenue", [total[1][2]], [total[2][2]], quote, unit=unit)]
        table = _Table([caption.group("date")], rows, line, unit,
                       _identity("\n".join(row[3] for row in source_rows), " ".join(headers) + line), True)
        if not all(_scope_supported(question, table, row) for row in rows):
            continue
        plan = _build(question, "proportion", table, [(row, 0) for row in rows])
        if plan is not None:
            for fact, header in zip(plan.facts.values(), (headers[col], "Total")):
                fact.source_column += " | " + header
            plans.append(plan)
    return plans


def _customer_amount_plans(question, evidence, operation):
    """Keep named-customer monetary revenue separate from revenue percentages."""
    from verifiqa.agent.source_question_planner import _Row, _Table, _identity, _number, _table_plan
    from verifiqa.agent.source_table_layouts import _row_groups
    from verifiqa.agent.table_fact_extractor import _caption_money_unit

    if operation not in {"line_item_difference", "period_change", "percent_change", "average"}:
        return []
    if not re.search(r"\b(?:revenues?|sales)\b", question, re.I):
        return []
    lines, plans = evidence.splitlines(), []
    for index in range(len(lines) - 3):
        line = lines[index]
        if not re.fullmatch(r"\s*(?:" + _DATE + r"\s*){2}\s*", line, re.I):
            continue
        dates = [m.group().strip() for m in re.finditer(_DATE, line, re.I)]
        if len({re.findall(_YEAR, date)[0] for date in dates}) != 2:
            continue
        if not re.fullmatch(r"\s*%\s+of\s+%\s+of\s*", lines[index + 1], re.I):
            continue
        if not re.fullmatch(r"\s*Amount\s+Revenues\s+Amount\s+Revenues\s*", lines[index + 2], re.I):
            continue
        unit = _caption_money_unit(lines[index + 3])
        if not unit.startswith("USD") or _foreign_currency_caption(lines[index + 3]):
            continue
        section = " / ".join(lines[max(0, index - 4):index])
        if not re.search(r"\bcustomers?\b.*\b(?:derived\s+revenues?|sales)\b", section, re.I):
            continue
        rows, invalid = [], False
        for quote in lines[index + 4:]:
            parsed = _row_groups(quote, 4)
            if parsed is None:
                invalid = any(_starts_row(quote, row.label) for row in rows)
                break
            for label, raws in parsed:
                if (not re.fullmatch(r"Customer\s+[A-Z]", label, re.I)
                        or any("%" in raws[i] for i in (0, 2))
                        or any("%" not in raws[i] for i in (1, 3))):
                    invalid = True
                    break
                selected = [raws[0], raws[2]]
                rows.append(_Row(label, [_number(raw) for raw in selected], selected, quote, unit=unit))
            if invalid:
                break
        if invalid or not rows:
            continue
        table = _Table(dates, rows, section, unit,
                       _identity("\n".join(row.quote for row in rows), section + line), True)
        plan = _table_plan(question, operation, table)
        if plan is not None:
            for fact in plan.facts.values():
                fact.source_column += " | Revenue amount"
            plans.append(plan)
    return plans


def _denominated_currency_plans(question, evidence, operation):
    """Read USD observations separately from the denominated-currency block."""
    from verifiqa.agent.source_question_planner import _Row, _Table, _identity, _number, _table_plan
    from verifiqa.agent.source_table_layouts import _CELL, _context
    if operation not in {"period_change", "percent_change", "average"} or not re.search(
            r"\bcarrying\s+value\s*\(\s*USD\s*\)", question, re.I):
        return []
    lines, plans = evidence.splitlines(), []
    note = r"(?:\s*\(\d{1,2}\))?"
    cells = _CELL + r"(?:\s+" + _CELL + r"){3}"
    rate = r"(?:\d+(?:\.\d+)?%|[—–-]%?|Various)"
    maturity = r"(?:" + _DATE + r"|Various)"
    row_pattern = re.compile(r"\s*(?P<label>[A-Za-z][A-Za-z .'’/&-]*?)" + note +
                             r"\s+(?!\(\d{1,2}\)\s)(?P<values>" + cells + r")\s+" + rate + r"\s+" + maturity + r"\s*", re.I)
    for index in range(2, len(lines)):
        if not re.fullmatch(r"\s*Carrying\s+Value\s*\(Denominated\s+Currency\)" + note + r"\s*", lines[index - 2], re.I):
            continue
        if not re.fullmatch(r"\s*Carrying\s+Value\s*\(USD\)" + note +
                            r"\s+Interest\s+Rate\s+Maturity\s+Date\s*", lines[index - 1], re.I):
            continue
        years = re.findall(_YEAR, lines[index])
        if not re.fullmatch(r"\s*" + _YEAR + r"(?:\s+" + _YEAR + r"){3}\s*", lines[index]):
            continue
        if years[:2] != years[2:] or len(set(years[:2])) != 2:
            continue
        section, unit = _context(lines, index)
        if not unit.startswith("USD") or _foreign_currency_caption(section):
            continue
        rows, unparsed = [], []
        for quote in lines[index + 1:]:
            if not quote.strip():
                continue
            if re.match(r"\s*(?:_{3,}|\(\d{1,2}\)\s)", quote):
                break  # Printed footnotes close the financial grid.
            parsed = row_pattern.fullmatch(quote)
            if parsed is None:
                # Preserve unsupported rows to check for a malformed duplicate
                # of any otherwise selected instrument before the grid closes.
                unparsed.append(quote)
                continue
            raws = [m.group().strip() for m in re.finditer(_CELL, parsed.group("values"))]
            if len(raws) != 4 or any("%" in raw for raw in raws):
                rows = []
                break
            rows.append(_Row(parsed.group("label").strip(), [_number(raw) for raw in raws[2:]],
                             raws[2:], quote, "Carrying Value (USD)", unit))
        if not rows or any(_starts_row(quote, row.label) for quote in unparsed for row in rows):
            continue
        table = _Table(years[2:], rows, section, unit,
                       _identity("\n".join(row.quote for row in rows), section + lines[index]), True)
        plan = _table_plan(question, operation, table)
        if plan is not None:
            for fact in plan.facts.values():
                fact.source_column += " | Carrying Value (USD)"
            plans.append(plan)
    return plans


def _carrying_fair_plans(question, evidence):
    from verifiqa.agent.source_question_planner import _Row, _Table, _build, _identity, _number, _scope_supported, _tokens
    from verifiqa.agent.source_table_layouts import _context, _row_groups
    from verifiqa.verification.operation_bindings import _difference_pairs

    years = set(re.findall(_YEAR, question))
    if len(years) != 1:
        return []
    orders = set()
    for first, second in _difference_pairs(question):
        kinds = [{kind for kind in ("carrying value", "fair value") if kind in phrase.lower()}
                 for phrase in (first, second)]
        if len(kinds[0]) == len(kinds[1]) == 1 and kinds[0] != kinds[1]:
            orders.add((next(iter(kinds[0])), next(iter(kinds[1]))))
    if len(orders) != 1:
        return []
    order = next(iter(orders))
    lines, plans = evidence.splitlines(), []
    for index, line in enumerate(lines[:-1]):
        if not re.fullmatch(r"\s*(?:" + _DATE + r"\s*){2,}\s*", line, re.I):
            continue
        dates = [m.group().strip() for m in re.finditer(_DATE, line, re.I)]
        date_years = [re.findall(_YEAR, date)[0] for date in dates]
        if len(date_years) != len(set(date_years)) or next(iter(years)) not in date_years:
            continue
        # The repetition establishes the type nesting independently of values.
        types = re.findall(r"Carrying\s+Value|Fair\s+Value", lines[index + 1], re.I)
        if not re.fullmatch(r"\s*(?:Carrying\s+Value\s+Fair\s+Value\s*){" + str(len(dates)) + r"}\s*",
                            lines[index + 1], re.I) or len(types) != 2 * len(dates):
            continue
        # Some PDF text exports retain a standalone '?' layout marker between
        # the caption and header. It has no numeric or measurement semantics.
        prior = [prior for prior in lines[:index] if prior.strip() != "?"]
        section, unit = _context(prior, len(prior))
        if not unit or _foreign_currency_caption(section):
            continue
        width, source_rows, invalid = 2 * len(dates), [], False
        for quote in lines[index + 2:]:
            parsed = _row_groups(quote, width)
            if parsed is None:
                invalid = any(_starts_row(quote, label) for label, _, _ in source_rows)
                break
            for label, raws in parsed:
                if any("%" in raw for raw in raws):
                    invalid = True
                    break
                source_rows.append((label, raws, quote))
            if invalid:
                break
        if invalid:
            continue
        wanted = set(_tokens(question))
        matching = [(label, raws, quote) for label, raws, quote in source_rows
                    if set(_tokens(label)) and set(_tokens(label)) <= wanted]
        if not matching:
            continue
        best = max(len(_tokens(label)) for label, _, _ in matching)
        matching = [row for row in matching if len(_tokens(row[0])) == best]
        if len(matching) != 1:
            continue
        label, raws, quote = matching[0]
        rows = [_Row(label + " " + kind, [_number(raws[2 * i + offset]) for i in range(len(dates))],
                     [raws[2 * i + offset] for i in range(len(dates))], quote, unit=unit)
                for offset, kind in enumerate(("carrying value", "fair value"))]
        table = _Table(dates, rows, section, unit,
                       _identity("\n".join((line, lines[index + 1], quote)), section), True)
        if not all(_scope_supported(question, table, row) for row in rows):
            continue
        # Match question order. A difference in A and B is A minus B.
        if order[0] == "fair value":
            rows.reverse()
        col = date_years.index(next(iter(years)))
        plan = _build(question, "line_item_difference", table, [(row, col) for row in rows])
        if plan is not None:
            for fact, row in zip(plan.facts.values(), rows):
                kind = "Carrying Value" if row.label.endswith("carrying value") else "Fair Value"
                fact.source_column = dates[col] + " | " + kind
            plans.append(plan)
    return plans


def _target_allocation_plans(question, evidence):
    from verifiqa.agent.source_question_planner import _Row, _Table, _build, _identity, _number, _scope_supported, _tokens
    from verifiqa.agent.source_table_layouts import _row_groups
    from verifiqa.verification.operation_bindings import _difference_pairs

    years = set(re.findall(_YEAR, question))
    if len(years) != 1:
        return []
    lines, plans = evidence.splitlines(), []
    for index, line in enumerate(lines):
        header = re.fullmatch(r"\s*Asset\s+Category\s+(" + _YEAR + r"(?:\s+" + _YEAR + r")+)\s*", line, re.I)
        if not header:
            continue
        columns = re.findall(_YEAR, header.group(1))
        if len(columns) != len(set(columns)) or next(iter(years)) != columns[0]:
            continue
        context = " ".join(lines[max(0, index - 6):index])
        # The introductory sentence independently binds the target year. A
        # generic heading alone cannot assign target/actual column semantics.
        if not re.search(r"target\s+allocation\s+for\s+" + columns[0] + r"\b", context, re.I):
            continue
        if not re.search(r"Target\s+Allocations\s+Percentage\s+of\s+Plan\s+Assets", context, re.I):
            continue
        rows = []
        for quote in lines[index + 1:]:
            parsed = _row_groups(quote, len(columns))
            if parsed is None:
                break
            for label, raws in parsed:
                if not all("%" in raw for raw in raws):
                    return []
                # 'Other' is explicitly an Asset Category, not a company-wide
                # total or a different table's similarly named row.
                if label.lower() == "other":
                    label = "Other assets"
                rows.append(_Row(label, [_number(raw) for raw in raws], raws, quote, unit="percent"))
        if not rows or len({row.label.lower() for row in rows}) != len(rows):
            continue
        pairs = []
        for first, second in _difference_pairs(question):
            selected = [[row for row in rows if set(_tokens(row.label)) <= set(_tokens(phrase))]
                        for phrase in (first, second)]
            if len(selected[0]) == len(selected[1]) == 1 and selected[0][0] is not selected[1][0]:
                pairs.append((selected[0][0], selected[1][0]))
        if len(pairs) != 1:
            continue
        table = _Table([columns[0] + " Target allocation"] +
                       [year + " Observed allocation" for year in columns[1:]], rows,
                       context, "percent", _identity("\n".join(row.quote for row in rows), line + context), True)
        if not all(_scope_supported(question, table, row) for row in pairs[0]):
            continue
        plan = _build(question, "line_item_difference", table, [(row, 0) for row in pairs[0]])
        if plan is not None:
            plans.append(plan)
    return plans
