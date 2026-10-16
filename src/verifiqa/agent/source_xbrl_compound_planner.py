"""Question/source arithmetic populations with explicit filing-schema scope.

Selection uses printed measures, dates, award/acquisition membership and units.
Fact amounts and proposed answers are never inputs to selecting an operand.
The ordinary XBRL binder still supplies and validates every selected value.
"""
from __future__ import annotations

from datetime import date
import re

from verifiqa.agent.types import ClaimSpec, RoleSpec
from verifiqa.answer_spec import answer_spec_from_question
from verifiqa.formulas.evaluator import formula_variables, normalize_formula


_MARKER = "deterministic_xbrl_schema"
_RECOGNIZED = "us-gaap:AllocatedShareBasedCompensationExpense"
_OPTIONS = "us-gaap:EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedStockOptions"
_RSUS = "us-gaap:EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions"
_GOODWILL = "us-gaap:GoodwillAcquiredDuringPeriod"
_FAIR_VALUE = "us-gaap:AssetsHeldForSaleLongLivedFairValueDisclosure"
_AWARD_AXIS = "us-gaap:AwardTypeAxis"
_OPTION_MEMBER = "us-gaap:EmployeeStockOptionMember"
_RSU_MEMBER = "us-gaap:RestrictedStockUnitsRSUMember"
_ACQUISITION_AXIS = "us-gaap:BusinessAcquisitionAxis"
_NONRECURRING = "us-gaap:FairValueByMeasurementFrequencyAxis=us-gaap:FairValueMeasurementsNonrecurringMember"
_AMOUNT = r"\$\s*\d[\d,]*(?:\.\d+)?"


def compound_plan_from_source_xbrl(question, evidence, bundle):
    """Return one fully selected source/schema plan, without reading amounts."""
    if not evidence or getattr(bundle, "status", "") != "ok":
        return None
    from verifiqa.verification.money_units import explicit_money_currencies
    if explicit_money_currencies(evidence) - {"USD"}:
        return None  # This bounded schema reader does not infer FX or column currency.
    plans = [plan for plan in (
        _recognized_and_unrecognized(question, evidence, bundle),
        _goodwill_excluding_acquisition(question, evidence, bundle),
        _nonrecurring_fair_value_operation(question, evidence, bundle),
        _defined_time_charter_revenues(question, evidence, bundle),
    ) if plan is not None]
    return plans[0] if len(plans) == 1 else None


def compound_source_xbrl_plan_matches(spec, question, evidence, bundle):
    """A marker, chosen concept or context ID alone never authorizes a plan."""
    rebuilt = compound_plan_from_source_xbrl(question, evidence, bundle)
    return rebuilt is not None and _signature(rebuilt) == _signature(spec)


def compound_source_xbrl_ir_matches(ir, question, evidence, bundle):
    """Rebuild the exact operand membership before authorizing arithmetic."""
    rebuilt = compound_plan_from_source_xbrl(question, evidence, bundle)
    if rebuilt is None or normalize_formula(ir.formula) != normalize_formula(rebuilt.formula):
        return False
    expected = {role.name: role for role in rebuilt.roles}
    if set(ir.facts) != set(expected) or formula_variables(ir.formula) != set(expected):
        return False
    from verifiqa.verification.money_units import money_currency
    from verifiqa.verification.xbrl_linkbase import _unit_kind_from_unit_text
    for name, role in expected.items():
        fact = ir.facts[name]
        if (fact.concept != role.concept or getattr(fact, "xbrl_context_id", "") != role.xbrl_context_id
                or str(fact.period) != role.period or money_currency(fact.unit) != "USD"
                or _unit_kind_from_unit_text(fact.unit) != "money"):
            return False
    return True


def _signature(spec):
    return (spec.metric, normalize_formula(spec.formula), spec.formula_source,
            spec.operation, spec.claim_unit, spec.period, spec.tolerance,
            spec.planning_source, getattr(spec, "source_schema_sha256", ""),
            getattr(spec, "source_report_unit", ""),
            tuple((role.name, tuple(role.aliases), role.period, role.concept,
                   getattr(role, "xbrl_context_id", "")) for role in spec.roles))


def _words(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _question(text):
    return re.sub(r"\s+", " ", text).strip().rstrip("?").strip()


def _date(value):
    from verifiqa.verification.periods import period_key
    key = period_key(value)
    try:
        return date(*key).isoformat()
    except (TypeError, ValueError):
        return ""


def _money_fact(fact, bundle):
    # Explicit filing currency is authoritative. A printed '$' alone does not
    # establish its scale or permit another currency to stand in for USD.
    return fact.value is not None and bundle.units.get(fact.unit_ref) == ("iso4217:USD",)


def _select(bundle, concept, predicate):
    matches = []
    for fact in bundle.facts_by_local_context.values():
        context = bundle.contexts.get(fact.context_id)
        if fact.concept == concept and context is not None and _money_fact(fact, bundle) and predicate(context):
            matches.append(fact)
    return matches[0] if len(matches) == 1 else None


def _annual(context, end):
    if context.instant or context.end_date != end or not context.start_date:
        return False
    try:
        return 350 <= (date.fromisoformat(end) - date.fromisoformat(context.start_date)).days <= 380
    except ValueError:
        return False


def _instant(context, when):
    return context.instant == when and not context.start_date and not context.end_date


def _printed_unit(evidence):
    from verifiqa.agent.table_fact_extractor import _caption_money_unit
    declarations = set()
    for line in evidence.splitlines():
        # Restrict this to literal standalone report-unit declarations. A
        # narrative '$100 million' is not the scale of the neighboring grid.
        if re.fullmatch(r"\s*\(?\s*(?:(?:all\s+)?amounts?\s+)?(?:expressed\s+)?in\s+"
                        r"(?:(?:U\.?S\.?\s+dollars?|USD)\s+)?(?:thousands|millions|billions)"
                        r"(?:\s+of\s+U\.?S\.?\s+dollars?)?"
                        r"(?:\s*,\s*except\s+(?:for\s+)?per\s+share\s+(?:amounts|data))?\s*\)?\s*", line, re.I) or re.fullmatch(
                            r"\s*\(?\s*(?:(?:all\s+)?amounts?\s+)?(?:in\s+)?(?:USD|U\.?S\.?\s+dollars?)\s*\)?\s*", line, re.I):
            unit = _caption_money_unit(line)
            if unit:
                declarations.add(unit)
    return next(iter(declarations)) if len(declarations) == 1 else ""


def _schema_hash(evidence, bundle):
    # Pin raw retrieved source and filing schema, never XML amount fields.
    from verifiqa.agent.source_xbrl_schema_planner import source_xbrl_schema_sha256
    return source_xbrl_schema_sha256(evidence, bundle)


def _spec(question, evidence, bundle, metric, operation, roles, formula, year):
    return ClaimSpec(metric, formula, roles, "USD millions", answer_spec_from_question(question).tolerance,
                     "generic_operation", period=year, operation=operation, planning_source=_MARKER,
                     source_schema_sha256=_schema_hash(evidence, bundle),
                     source_report_unit=_printed_unit(evidence))


def _recognized_and_unrecognized(question, evidence, bundle):
    match = re.fullmatch(r"What (?:is|was) the (?:total|sum of) stock[- ]based compensation expense and "
                         r"unrecognized stock[- ]based compensation expense in ((?:19|20)\d{2})", _question(question), re.I)
    if match is None:
        return None
    year = match.group(1)
    from verifiqa.agent.table_fact_extractor import _MONTH
    narrative = re.compile(r"As of (?P<date>(?:" + _MONTH + r")\s+\d{1,2},?\s+" + year + r"),\s*"
                           r"there was " + _AMOUNT + r" of unrecognized stock[- ]based compensation expense "
                           r"related to unvested employee stock options and " + _AMOUNT
                           + r" of unrecognized stock[- ]based compensation expense related to unvested RSUs\.", re.I)
    observations = list(narrative.finditer(evidence))
    if len(observations) != 1 or len(re.findall(r"unrecognized stock[- ]based compensation expense", evidence, re.I)) != 2:
        return None
    when = _date(observations[0].group("date"))
    if not when:
        return None
    from verifiqa.agent.source_question_planner import _annual_tables
    tables = [table for table in _annual_tables(evidence)
              if sum(bool(re.fullmatch(r"Total stock[- ]based compensation expense", row.label, re.I)) for row in table.rows) == 1
              and sum(bool(re.fullmatch(r"Total stock[- ]based compensation expense", row.label, re.I)
                          and row.unit in {"USD", "USD thousands", "USD millions", "USD billions"})
                     for row in table.rows) == 1
              and sum(_date(column) == when for column in table.columns) == 1
              and re.search(r"\bYears?\s+Ended\b", table.section + " " + " ".join(table.columns), re.I)]
    if len(tables) != 1:
        return None
    total = _select(bundle, _RECOGNIZED, lambda context: _annual(context, when) and not context.dimensions)
    options = _select(bundle, _OPTIONS, lambda context: _instant(context, when)
                      and context.dimensions == (_AWARD_AXIS + "=" + _OPTION_MEMBER,))
    rsus = _select(bundle, _RSUS, lambda context: _instant(context, when)
                   and context.dimensions == (_AWARD_AXIS + "=" + _RSU_MEMBER,))
    if any(fact is None for fact in (total, options, rsus)):
        return None
    chosen = {options.context_id, rsus.context_id}
    for fact in bundle.facts_by_local_context.values():
        context = bundle.contexts.get(fact.context_id)
        if ("NonvestedAwardsTotalCompensationCostNotYetRecognized" in fact.local_name
                and context is not None and _instant(context, when) and _money_fact(fact, bundle)
                and (fact.context_id not in chosen or fact.concept not in {_OPTIONS, _RSUS})):
            return None
    roles = [RoleSpec("recognized_stock_based_compensation_" + year,
                      ["Total stock-based compensation expense"], year, _RECOGNIZED, total.context_id),
             RoleSpec("unrecognized_stock_option_compensation_" + year,
                      ["unrecognized stock-based compensation expense related to unvested employee stock options"],
                      observations[0].group("date"), _OPTIONS, options.context_id),
             RoleSpec("unrecognized_rsu_compensation_" + year,
                      ["unrecognized stock-based compensation expense related to unvested RSUs"],
                      observations[0].group("date"), _RSUS, rsus.context_id)]
    return _spec(question, evidence, bundle, "recognized_and_unrecognized_stock_compensation_sum", "sum", roles,
                 " + ".join(role.name for role in roles), year)


def _goodwill_excluding_acquisition(question, evidence, bundle):
    match = re.fullmatch(r"How much goodwill was acquired during fiscal ((?:19|20)\d{2}), excluding "
                         r"the goodwill acquired from the (?:purchase|acquisition) of (.+)", _question(question), re.I)
    if match is None:
        return None
    year, entity = match.groups()
    from verifiqa.agent.table_fact_extractor import _MONTH
    dates = re.findall(r"fiscal years? ended ((?:" + _MONTH + r")\s+\d{1,2},?\s+" + year + r")\b", evidence, re.I)
    ends = {_date(value) for value in dates}
    if len(ends) != 1 or "" in ends:
        return None
    end = next(iter(ends))
    paragraphs = re.findall(r"Goodwill acquired during fiscal " + year + r" totaled " + _AMOUNT
                            + r"\s*,\s*with[^\n]+", evidence, re.I)
    entity_words = _words(entity)
    entity_pattern = r"[\W_]+".join(re.escape(word) for word in entity_words.split())
    if (not entity_words or len(paragraphs) != 1 or not re.search(
            _AMOUNT + r"\s+of that resulting from the purchase of " + entity_pattern + r"\b", paragraphs[0], re.I)):
        return None
    total = _select(bundle, _GOODWILL, lambda context: _annual(context, end) and not context.dimensions)

    def member_context(context):
        if not _annual(context, end) or len(context.dimensions) != 1:
            return False
        axis, separator, member = context.dimensions[0].partition("=")
        return (separator and axis == _ACQUISITION_AXIS
                and _words(re.sub(r"Member$", "", member.split(":")[-1])) == entity_words)

    excluded = _select(bundle, _GOODWILL, member_context)
    if total is None or excluded is None:
        return None
    total_context, excluded_context = bundle.contexts[total.context_id], bundle.contexts[excluded.context_id]
    if total_context.start_date != excluded_context.start_date:
        return None
    roles = [RoleSpec("total_goodwill_acquired_" + year, ["Goodwill acquired during fiscal " + year], year,
                      _GOODWILL, total.context_id),
             RoleSpec("excluded_acquisition_goodwill_" + year, ["Goodwill acquired from the purchase of " + entity], year,
                      _GOODWILL, excluded.context_id)]
    return _spec(question, evidence, bundle, "goodwill_acquired_excluding_named_acquisition", "exclusion_or_remaining_amount",
                 roles, roles[0].name + " - " + roles[1].name, year)


def _nonrecurring_fair_value_operation(question, evidence, bundle):
    # Select the valuation basis before choosing the temporal arithmetic. An
    # average uses the same fair-value disclosures as a change, never nearby
    # carrying amounts or a hierarchy-level subtotal with a similar row label.
    match = re.fullmatch(
        r"What (?:is|was) the (?P<request>change in|average(?: of)?|sum of) "
        r"(?:the )?total fair value of long[- ]lived assets held for sale "
        r"(?P<connector>between|for|from) (?P<first>(?:19|20)\d{2}) "
        r"(?P<join>and|to) (?P<last>(?:19|20)\d{2})", _question(question), re.I)
    if match is None or match.group("first") == match.group("last"):
        return None
    first, last = match.group("first", "last")
    connector, join = match.group("connector").lower(), match.group("join").lower()
    if ((connector == "from" and (join != "to" or first > last))
            or (connector != "from" and join != "and")):
        return None
    request = match.group("request").lower()
    operation = "period_change" if request == "change in" else "average" if request.startswith("average") else "sum"
    # 'Between'/'from' denotes a range, so nonadjacent years would require
    # intermediate observations for an average or sum. This layout reader
    # supports two observations, not an inferred incomplete range.
    if operation != "period_change" and connector != "for" and abs(int(last) - int(first)) != 1:
        return None
    years = sorted((first, last))
    if not re.search(r"Total\s+Fair\s+Value", evidence, re.I) or not re.search(r"Level\s+1\s+Level\s+2\s+Level\s+3", evidence, re.I):
        return None
    sections = re.split(r"Non[- ]Recurring Fair Value Measurements", evidence, flags=re.I)
    if len(sections) != 2:
        return None
    section = re.split(r"\bNOTE\s+\d+\.", sections[1], maxsplit=1, flags=re.I)[0]
    from verifiqa.agent.table_fact_extractor import _MONTH, _annual_row_groups
    lines, dates, current = section.splitlines(), {}, ""
    for line in lines:
        if re.fullmatch(r"\s*(?:" + _MONTH + r")\s+\d{1,2},?\s+(?:19|20)\d{2}\s*", line, re.I):
            current = _date(line)
        elif re.match(r"\s*Long[- ]lived assets held for sale\b", line, re.I):
            parsed = _annual_row_groups(line, 4)
            if (not current or parsed is None or len(parsed) != 1
                    or not re.fullmatch(r"Long[- ]lived assets held for sale(?:\s*\([a-z]\))?", parsed[0][1], re.I)
                    or any("%" in raw for raw in parsed[0][2])
                    or current[:4] in dates):
                return None
            dates[current[:4]] = current
    if set(dates) != set(years):
        return None
    roles = []
    for year in years:
        fact = _select(bundle, _FAIR_VALUE, lambda context: _instant(context, dates[year])
                       and context.dimensions == (_NONRECURRING,))
        if fact is None:
            return None
        roles.append(RoleSpec("total_nonrecurring_fair_value_" + year, ["Long-lived assets held for sale"],
                              dates[year], _FAIR_VALUE, fact.context_id))
    if operation == "period_change":
        formula = roles[1].name + " - " + roles[0].name
        suffix = "change"
    else:
        formula = " + ".join(role.name for role in roles)
        if operation == "average":
            formula = "(" + formula + ") / " + str(len(roles))
        suffix = operation
    return _spec(question, evidence, bundle, "total_nonrecurring_held_for_sale_fair_value_" + suffix,
                 operation, roles, formula, years[-1])


def _defined_time_charter_revenues(question, evidence, bundle):
    """Expand an explicit disclosed reconciliation before temporal arithmetic."""
    from verifiqa.agent.table_fact_extractor import _MONTH
    prefix = re.fullmatch(r"What (?:is|was) the (?P<request>change in|average(?: of)?|sum of|total) "
                          r"(?:the )?(?:time charter equivalent|TCE) revenues? "
                          r"(?P<periods>(?:(?:for|between|from)\s+)?(?:(?:fiscal\s+)?Years?\s+Ended\s+)?"
                          r"(?:(?:" + _MONTH + r")\s+\d{1,2},?\s+)?(?:19|20)\d{2}\s+(?:to|and)\s+(?:19|20)\d{2})",
                          _question(question), re.I)
    if prefix is None:
        return None
    written_years = re.findall(r"\b(?:19|20)\d{2}\b", prefix.group("periods"))
    if (re.match(r"from\b", prefix.group("periods"), re.I)
            and len(written_years) == 2 and written_years[0] > written_years[1]):
        return None
    request = prefix.group("request").lower()
    operation = "period_change" if request == "change in" else "average" if request.startswith("average") else "sum"
    definitions = re.findall(r"\b(?:TCE|time charter equivalent) revenues,?\s+"
                             r"(?:which represents?|represent|are defined as) shipping revenues less voyage expenses\b", evidence, re.I)
    if len(definitions) != 1:
        return None
    from verifiqa.agent.source_question_planner import _annual_tables, _selected_periods
    from verifiqa.agent.source_xbrl_schema_planner import _annual_header, _question_date_matches
    plans = []
    for table in _annual_tables(evidence):
        if not all(re.fullmatch(r"(?:19|20)\d{2}", column) for column in table.columns):
            continue
        labels = [_words(re.sub(r"^\s*(?:Add|Less):\s*", "", row.label, flags=re.I)) for row in table.rows]
        required = {"time charter equivalent revenues", "shipping revenues", "voyage expenses"}
        if set(labels) != required or len(labels) != 3:
            continue
        if any(row.unit not in {"USD", "USD thousands", "USD millions", "USD billions"} for row in table.rows):
            continue
        quote = table.rows[0].quote
        if not quote or evidence.count(quote) != 1:
            continue
        nearby = evidence[:evidence.index(quote)].splitlines()[-4:]
        if sum(bool(re.fullmatch(r"\s*Reconciliations? of TCE revenues,?\s+"
                                r"(?:a non[- ]GAAP measure,?\s+)?to shipping revenues "
                                r"as reported in the consolidated statements of operations follows:\s*", line, re.I))
               for line in nearby) != 1:
            continue
        selected = _selected_periods(question, table, operation)
        header = _annual_header(table, evidence)
        if (selected is None or len(selected) < 2 or header is None
                or not _question_date_matches(question, header)):
            continue
        numeric = {fact.concept for fact in bundle.facts_by_local_context.values() if _money_fact(fact, bundle)}
        concepts = {}
        for measure in ("shipping revenues", "voyage expenses"):
            matches = {concept for concept, names in bundle.concept_labels.items() if concept in numeric
                       and any(re.sub(r"^total\s+", "", _words(name)) == measure for name in names)}
            if measure == "shipping revenues":
                # The caption explicitly reconciles to consolidated revenue.
                # An investee's summarized revenue may share its terse label;
                # it is not the standard consolidated revenue observation.
                matches &= {"us-gaap:Revenues"}
            if len(matches) != 1:
                break
            concepts[measure] = next(iter(matches))
        if len(concepts) != 2 or len(set(concepts.values())) != 2:
            continue
        roles, derived = [], []
        month, day = header
        for index in selected:
            year = table.columns[index]
            try:
                end = date(int(year), month, day).isoformat()
            except ValueError:
                break
            pair = []
            for measure in ("shipping revenues", "voyage expenses"):
                concept = concepts[measure]
                fact = _select(bundle, concept, lambda context: _annual(context, end) and not context.dimensions)
                if fact is None:
                    break
                role = RoleSpec(measure.replace(" ", "_") + "_" + year,
                                ["Shipping revenues" if measure == "shipping revenues" else "Voyage expenses"],
                                year, concept, fact.context_id)
                roles.append(role)
                pair.append(role.name)
            if len(pair) != 2:
                break
            derived.append("(" + pair[0] + " - " + pair[1] + ")")
        if len(derived) != len(selected) or len(roles) != len(selected) * 2:
            continue
        if operation == "period_change":
            if len(derived) != 2:
                continue
            formula = derived[-1] + " - " + derived[0]
        elif operation == "average":
            formula = "(" + " + ".join(derived) + ") / " + str(len(derived))
        else:
            formula = " + ".join(derived)
        plans.append(_spec(question, evidence, bundle, "defined_time_charter_revenues_" + operation,
                           operation, roles, formula, table.columns[selected[-1]]))
    return plans[0] if len(plans) == 1 else None
