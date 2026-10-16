"""Original principal less the explicitly linked, disclosed outstanding balance.

The question and literal source establish the instrument before either amount
is read. An allowance or a net carrying value cannot stand in for principal.
"""
from __future__ import annotations

from decimal import Decimal
import re


_YEAR = r"(?:19|20)\d{2}"
_QUESTION = re.compile(
    r"\s*How\s+much\s+of\s+the\s+note\s+receivable\s+related\s+to\s+"
    r"(?P<project>.+?)\s+in\s+(?P<origin>" + _YEAR + r")\s+has\s+been\s+"
    r"received\s+by\s+(?:the\s+)?company\s+so\s+far\s*\??\s*", re.I)
_ORIGINAL = re.compile(
    r"\bThe\s+note\s+receivable\s+represents\s+the\s+remaining\s+outstanding\s+"
    r"balance\s+of\s+an?\s+original\s+note\s+related\s+to\s+"
    r"(?P<project>[^.?!;]+?)\s+in\s+(?P<origin>" + _YEAR + r")\s+in\s+the\s+"
    r"amount\s+of\s+\$\s*(?P<amount>\d[\d,]*(?:\.\d+)?)\s+"
    r"(?P<scale>thousand|million|billion)s?\s*\.", re.I)
_SCALES = {"thousand": Decimal(1000), "million": Decimal(1000000),
           "billion": Decimal(1000000000)}
_SCALE = r"(?:thousand|million|billion)s?"
_AMOUNT = r"\d[\d,]*(?:\.\d+)?"
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE = _MONTH + r"\s+\d{1,2},?\s+" + _YEAR
_COST_QUESTION = re.compile(
    r"\s*For\s+Fiscal\s+year\s+(?P<year>" + _YEAR + r"),\s*excluding\s+the\s+increase\s+in\s+"
    r"(?P<component>[A-Za-z][A-Za-z -]{0,100}?)\s+of\s+\$\s*(?P<amount>" + _AMOUNT + r")\s+"
    r"(?P<scale>" + _SCALE + r"),\s*what\s+is\s+the\s+(?P<measure>Cost\s+of\s+[A-Za-z][A-Za-z '&-]{0,100}?)\s+"
    r"for\s+fiscal\s+year\s+(?P<target>" + _YEAR + r")\s+in\s+(?P<unit>" + _SCALE + r")"
    r"(?:\s*\([\d,]+s?\))?\s*\??\s*", re.I)
_COST_NARRATIVE = re.compile(
    r"\b(?P<measure>Cost\s+of\s+[A-Za-z][A-Za-z '&-]{0,100}?)\s+(?:decreased|increased)\s+by\s+"
    r"\$\s*" + _AMOUNT + r"\s+" + _SCALE + r"\s+during\s+the\s+year\s+ended\s+"
    r"(?P<date>" + _DATE + r")\s+as\s+compared\s+to\s+the\s+prior\s+fiscal\s+year,\s*"
    r"due\s+to\s+a[n]?\s+(?:decrease|increase)\s+in\s+[A-Za-z][A-Za-z -]{0,100}?\s+of\s+"
    r"approximately\s+\$\s*" + _AMOUNT + r"\s+" + _SCALE + r",\s*partially\s+offset\s+by\s+"
    r"an?\s+increase\s+in\s+(?P<component>[A-Za-z][A-Za-z -]{0,100}?)\s+of\s+"
    r"\$\s*(?P<amount>" + _AMOUNT + r")\s+(?P<scale>" + _SCALE + r")\s*\.", re.I)
_COST_GRID = re.compile(
    r"\bYear\s+Ended\s+(?P<date>" + _MONTH + r"\s+\d{1,2},?)\s+"
    r"\(In\s+(?P<scale>" + _SCALE + r")\)\s+(?P<current>" + _YEAR + r")\s+"
    r"Change\s+increase\s*\(decrease\)\s+(?P<prior>" + _YEAR + r")\s+"
    r"Change\s+increase\s*\(decrease\)\s+(?P<oldest>" + _YEAR + r")\b", re.I)
_LOAN_QUESTION = re.compile(
    r"\s*What\s+is\s+the\s+percentage\s+of\s+(?:the\s+)?loan\s+for\s+the\s+"
    r"acquisition\s+of\s+(?P<equipment>.+?)\s+that\s+is\s+outstanding\s*\??\s*", re.I)
_LOAN_INTRO = (
    r"\bOn\s+(?P<origin>" + _DATE + r"),?\s+the\s+Company\s+entered\s+into\s+a\s+"
    r"(?:\d+(?:\.\d+)?%\s+interest\s+)?loan\s+agreement\s+for\s+the\s+acquisition\s+of\s+"
    r"(?P<equipment>[^.?!;]+?)\s+in\s+the\s+principal\s+amount\s+of\s+"
    r"\$\s*(?P<principal>\d[\d,]*(?:\.\d+)?)")
_LOAN = re.compile(
    _LOAN_INTRO + r"(?:\s+bearing\s+interest\s+at\s+\d+(?:\.\d+)?%)?\s*\.\s*"
    r"The\s+loan\s+agreement\s+calls\s+for\s+monthly\s+payments\s+of\s+"
    r"\$\s*\d[\d,]*(?:\.\d+)?\s+and\s+is\s+scheduled\s+to\s+mature\s+on\s+" + _DATE + r"\s*\.\s*"
    r"The\s+loan\s+is\s+secured\s+by\s+the\s+equipment\s*\.\s*"
    r"The\s+outstanding\s+balance\s+at\s+(?P<asof>" + _DATE + r"),?\s+is\s+"
    r"\$\s*(?P<outstanding>\d[\d,]*(?:\.\d+)?)\s*\.", re.I)


def exclusion_plan(question, evidence, operation):
    """Return one fully linked principal-repayment plan, or decline to infer it.

    This narrow reader requires an explicit original-note sentence, an exact
    annual ``Note receivable`` row in the same source section, and a declared
    monetary scale. It does not infer principal from a reserve or net balance.
    """
    if operation != "exclusion_or_remaining_amount":
        return None
    if _COST_QUESTION.fullmatch(question or ""):
        return _cost_exclusion_plan(question, evidence)
    requested = _QUESTION.fullmatch(question or "")
    if not requested:
        return None
    from verifiqa.agent.source_question_planner import (
        _Plan, _annual_tables, _identity, _tokens,
    )
    from verifiqa.agent.source_column_planner import _foreign_currency_caption
    from verifiqa.agent.table_fact_extractor import _caption_money_unit
    from verifiqa.agent.types import ClaimSpec, GroundedFact, RoleSpec
    from verifiqa.answer_spec import answer_spec_from_question

    originals = list(_ORIGINAL.finditer(evidence))
    # Another original note cannot be silently ignored because its prose uses
    # a different layout. Ambiguous multiple-instrument disclosures need their
    # own explicit source relation rather than a nearby numeric match.
    if len(originals) != 1 or len(re.findall(r"\boriginal\s+notes?\b", evidence, re.I)) != 1:
        return None
    original = originals[0]
    if (original.group("origin") != requested.group("origin")
            or _tokens(original.group("project")) != _tokens(requested.group("project"))):
        return None
    if not _tokens(requested.group("project")):
        return None

    candidates = []
    tables = _annual_tables(evidence)
    for table in tables:
        related = [row for row in table.rows if re.search(r"\bnotes?\s+receivables?\b", row.label, re.I)]
        principal = [row for row in related if re.fullmatch(r"\s*Note\s+receivable\s*", row.label, re.I)]
        if related and not principal:
            return None
        if len(principal) != 1:
            continue
        if any(not re.fullmatch(r"\s*(?:Note\s+receivable|Allowance\s+for\s+note\s+receivable)\s*", row.label, re.I)
               for row in related):
            return None
        row = principal[0]
        if (not table.unit_declared or not re.fullmatch(r"USD(?: thousands| millions| billions)?", table.unit)
                or row.unit != table.unit or evidence.count(row.quote) != 1):
            return None
        position = evidence.find(row.quote)
        if position >= original.start():
            return None
        between = evidence[position:original.start()]
        if re.search(r"(?m)^\s*\d{1,2}\.\s+[A-Za-z]", between):
            return None
        # A scale inherited from an earlier note is not this table's scale.
        headings = list(re.finditer(r"(?m)^\s*\d{1,2}\.\s+[A-Za-z][^\n]*", evidence[:position]))
        section_start = headings[-1].start() if headings else 0
        local_prefix = evidence[section_start:position]
        captions = [line for line in local_prefix.splitlines() if _caption_money_unit(line)]
        if (not captions or _caption_money_unit(captions[-1]) != table.unit
                or _foreign_currency_caption(local_prefix + original.group())
                or re.search(r"\bq[1-4]\b|\bquarter\b|\bmonths?\s+ended\b", local_prefix, re.I)):
            return None
        years = []
        for column in table.columns:
            found = re.findall(r"\b" + _YEAR + r"\b", column)
            if (len(found) != 1 or re.search(r"\bq[1-4]\b|\bquarter\b|\bmonths?\b|\byears?\s+ended\b", column, re.I)):
                return None
            years.append(int(found[0]))
        if (not years or len(set(years)) != len(years) or any(value is None for value in row.values)
                or int(original.group("origin")) > max(years)):
            return None
        latest = years.index(max(years))
        candidates.append((table, row, latest, position, str(max(years))))
    if len(candidates) != 1:
        return None
    table, row, latest, position, year = candidates[0]
    # The sentence must explain this table, not a different intervening grid.
    for other in tables:
        if other.identity == table.identity:
            continue
        if any(position < evidence.find(r.quote) < original.start() for r in other.rows):
            return None
    if re.search(r"\b(?:another|second|additional|separate|other)\s+note(?:\s+receivable)?\b",
                 evidence[position:original.end()], re.I):
        return None

    raw_original = Decimal(original.group("amount").replace(",", ""))
    source_scale = original.group("scale").lower()
    target_scale = next((scale for word, scale in _SCALES.items() if word in table.unit), Decimal(1))
    original_value = float(raw_original * _SCALES[source_scale] / target_scale)
    remaining_value = row.values[latest]
    # Negative or increased principal can reflect non-cash changes or another
    # instrument; the source relation alone does not authorize a receipt amount.
    if remaining_value < 0 or remaining_value > original_value:
        return None
    origin = original.group("origin")
    whole, remainder = "original_note_principal", "remaining_note_receivable"
    identity = _identity(table.identity + "\n" + original.group(), "note principal and outstanding balance")
    facts = {
        whole: GroundedFact(
            whole, original_value, table.unit, original.group(), period=origin,
            row_label="Original note principal", fact_type="source_table_numeric",
            source_value=float(raw_original), source_unit="USD " + source_scale + "s",
            source_section=original.group(), source_column=origin + " original principal",
            source_table_id=identity),
        remainder: GroundedFact(
            remainder, remaining_value, table.unit, row.quote, period=year,
            row_label=row.label, fact_type="source_table_numeric", source_value=remaining_value,
            source_unit=table.unit, source_section=original.group(),
            source_column=table.columns[latest], source_table_id=identity),
    }
    spec = ClaimSpec(
        "note_principal_received", whole + " - " + remainder,
        [RoleSpec(whole, ["Original note principal"], origin), RoleSpec(remainder, [row.label], year)],
        table.unit, answer_spec_from_question(question).tolerance, "generic_operation",
        period=year, operation=operation, planning_source="deterministic_source_table")
    return _Plan(spec, facts)


def _cost_exclusion_plan(question, evidence):
    """Read a cost row and its separately disclosed, same-year cost component."""
    from datetime import date
    from verifiqa.agent.source_question_planner import _Plan, _identity, _number, _slug, _tokens
    from verifiqa.agent.source_column_planner import _foreign_currency_caption
    from verifiqa.agent.table_fact_extractor import _annual_row_groups
    from verifiqa.agent.types import ClaimSpec, GroundedFact, RoleSpec
    from verifiqa.answer_spec import answer_spec_from_question
    from verifiqa.verification.periods import period_key

    requested = _COST_QUESTION.fullmatch(question)
    if requested is None or requested.group("year") != requested.group("target"):
        return None
    year = requested.group("year")
    narratives = [match for match in _COST_NARRATIVE.finditer(evidence)
                  if _tokens(match.group("measure")) == _tokens(requested.group("measure"))
                  and _tokens(match.group("component")) == _tokens(requested.group("component"))
                  and re.findall(_YEAR, match.group("date")) == [year]]
    if len(narratives) != 1:
        return None
    narrative = narratives[0]
    try:
        date(*period_key(narrative.group("date")))
    except (TypeError, ValueError):
        return None
    if _foreign_currency_caption(narrative.group()):
        return None
    # The source determines the operand. A number repeated only in the question
    # never supplies evidence; a conflicting question premise also stays unresolved.
    source_scale = narrative.group("scale").lower().rstrip("s")
    question_scale = requested.group("scale").lower().rstrip("s")
    raw_component = Decimal(narrative.group("amount").replace(",", ""))
    component_base = raw_component * _SCALES[source_scale]
    stated_base = Decimal(requested.group("amount").replace(",", "")) * _SCALES[question_scale]
    if component_base != stated_base:
        return None
    candidates = []
    for header in _COST_GRID.finditer(evidence):
        if header.end() >= narrative.start():
            continue
        years = [header.group(name) for name in ("current", "prior", "oldest")]
        if len(set(years)) != 3 or year not in years or sorted(years, reverse=True) != years:
            continue
        target_scale = header.group("scale").lower().rstrip("s")
        if target_scale != requested.group("unit").lower().rstrip("s"):
            continue
        region = evidence[header.end():narrative.start()]
        target_rows, invalid = [], False
        prefix = r"\s*" + r"\s+".join(re.escape(word) for word in narrative.group("measure").split()) + r"\b"
        for line in region.splitlines():
            if not re.match(prefix, line, re.I):
                continue
            rows = _annual_row_groups(line, 5)
            matching = [(label, raws) for _section, label, raws in (rows or [])
                        if _tokens(label) == _tokens(requested.group("measure"))]
            if len(matching) != 1 or any("%" in raw for raw in matching[0][1]):
                invalid = True
                break
            label, raws = matching[0]
            values = [_number(raw) for raw in raws]
            if any(value is None for value in values):
                invalid = True
                break
            target_rows.append((label, values[2 * years.index(year)], line))
        if invalid or len(target_rows) != 1 or _foreign_currency_caption(header.group()):
            continue
        label, value, quote = target_rows[0]
        if value < 0 or component_base / _SCALES[target_scale] > Decimal(str(value)):
            continue
        candidates.append((header, label, value, quote, target_scale))
    if len(candidates) != 1:
        return None
    header, label, value, quote, target_scale = candidates[0]
    unit = "USD " + target_scale + "s"
    whole, component = _slug(label) + "_" + year, _slug("increase in " + narrative.group("component"))
    identity = _identity(header.group() + "\n" + quote + "\n" + narrative.group(), label)
    facts = {
        whole: GroundedFact(
            whole, value, unit, quote, period=year, row_label=label, fact_type="source_table_numeric",
            source_value=value, source_unit=unit, source_section=label,
            source_column="Year ended " + header.group("date") + " " + year,
            source_table_id=identity),
        component: GroundedFact(
            component, float(component_base / _SCALES[target_scale]), unit, narrative.group(),
            period=year, row_label="Increase in " + narrative.group("component"),
            fact_type="source_table_numeric", source_value=float(raw_component),
            source_unit="USD " + source_scale + "s", source_section=label,
            source_column="Year ended " + narrative.group("date") + " | Increase in " + narrative.group("component"),
            source_table_id=identity),
    }
    spec = ClaimSpec(
        _slug(label) + "_excluding_cost_increase", whole + " - " + component,
        [RoleSpec(name, [fact.row_label], year) for name, fact in facts.items()], unit,
        answer_spec_from_question(question).tolerance, "generic_operation", period=year,
        operation="exclusion_or_remaining_amount", planning_source="deterministic_source_table")
    return _Plan(spec, facts)


def loan_proportion_plan(question, evidence, operation):
    """Bind original and outstanding principal from one complete loan paragraph.

    Dollar amounts use the same disclosed unit, so any common page scale
    cancels in this dimensionless proportion. No absolute scale is inferred.
    An adjacent loan's balance or a second loan for the same purpose makes the
    requested instrument unresolved instead of supplying a convenient number.
    """
    if operation != "proportion":
        return None
    requested = _LOAN_QUESTION.fullmatch(question or "")
    if not requested:
        return None
    from verifiqa.agent.source_question_planner import _Plan, _identity, _slug, _tokens
    from verifiqa.agent.source_column_planner import _foreign_currency_caption
    from verifiqa.agent.types import ClaimSpec, GroundedFact, RoleSpec
    from verifiqa.answer_spec import answer_spec_from_question

    wanted = _tokens(requested.group("equipment"))
    if not wanted:
        return None
    intros = [match for match in re.finditer(_LOAN_INTRO, evidence, re.I)
              if _tokens(match.group("equipment")) == wanted]
    if len(intros) != 1:
        return None
    source = _LOAN.match(evidence, intros[0].start())
    if source is None or _foreign_currency_caption(source.group()):
        return None
    following = evidence[source.end():]
    boundary = re.search(r"\bOn\s+" + _DATE + r",|^\s*\d{1,2}\.\s+[A-Za-z]", following, re.I | re.M)
    tail = following[:boundary.start()] if boundary else following
    if re.search(r"\boutstanding\s+balances?\b|\bprincipal\s+amount\b", tail, re.I):
        return None
    principal = float(Decimal(source.group("principal").replace(",", "")))
    outstanding = float(Decimal(source.group("outstanding").replace(",", "")))
    if not 0 <= outstanding <= principal or principal <= 0:
        return None
    from datetime import date
    from verifiqa.verification.periods import period_key
    origin, asof = source.group("origin"), source.group("asof")
    if not period_key(origin) or not period_key(asof) or period_key(origin) > period_key(asof):
        return None
    try:
        date(*period_key(origin))
        date(*period_key(asof))
    except ValueError:
        return None
    purpose = "Loan for the acquisition of " + source.group("equipment").strip()
    whole, part = _slug(purpose) + "_original_principal", "outstanding_loan_balance"
    identity = _identity(source.group(), purpose)
    facts = {}
    for name, value, label, period, column in (
        (whole, principal, purpose + " original principal", origin, "Original principal"),
        (part, outstanding, "Outstanding balance", asof, "Outstanding balance"),
    ):
        facts[name] = GroundedFact(
            name, value, "USD", source.group(), period=period, row_label=label,
            fact_type="source_table_numeric", source_value=value, source_unit="USD",
            source_section=purpose, source_column=period + " | " + column,
            source_table_id=identity)
    spec = ClaimSpec(
        _slug(purpose) + "_outstanding_percentage", part + " / " + whole + " * 100",
        [RoleSpec(name, [fact.row_label], fact.period) for name, fact in facts.items()],
        "percent", answer_spec_from_question(question).tolerance, "generic_operation",
        period=asof, operation=operation, planning_source="deterministic_source_table")
    return _Plan(spec, facts)


def loan_proportion_matches(ir, question, evidence_text):
    """Independently check the requested loan for source and frozen LLM plans."""
    if not re.search(r"\bpercentage\b.*\bloan\b.*\bacquisition\b.*\boutstanding\b", question, re.I):
        return True
    plan = loan_proportion_plan(question, evidence_text, "proportion")
    if plan is None or len(ir.facts) != len(plan.facts):
        return False
    from verifiqa.verification.operation_bindings import _source_population_member_matches
    unmatched = list(ir.facts.values())
    for source in plan.facts.values():
        matches = [fact for fact in unmatched if _source_population_member_matches(fact, source)]
        if len(matches) != 1:
            return False
        unmatched.remove(matches[0])
    return not unmatched


def exclusion_source_matches(ir, question, evidence_text):
    """Require independent source reconstruction for the supported exclusions."""
    if not (_QUESTION.fullmatch(question or "") or _COST_QUESTION.fullmatch(question or "")):
        return True
    plan = exclusion_plan(question, evidence_text, "exclusion_or_remaining_amount")
    if plan is None or len(ir.facts) != len(plan.facts):
        return False
    from verifiqa.verification.operation_bindings import _source_population_member_matches
    unmatched = list(ir.facts.values())
    for source in plan.facts.values():
        matches = [fact for fact in unmatched if _source_population_member_matches(fact, source)
                   and fact.source_column == source.source_column
                   and fact.source_table_id == source.source_table_id]
        if len(matches) != 1:
            return False
        unmatched.remove(matches[0])
    return not unmatched
