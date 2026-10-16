"""Bind exercise-eligible options to their plan's disclosed exercise price.

A plan's inclusive rights/RSU count is not an exercise-eligible population.
Both the count and the exclusion of non-exercisable awards must be printed.
"""
from __future__ import annotations

import hashlib
import math
import re

from verifiqa.agent.types import ClaimSpec, GroundedFact, RoleSpec


_MONTH = ("January|February|March|April|May|June|July|August|September|October|November|December")
_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_INTEGER = r"\d[\d,]*"


def _named_plan(question):
    match = re.fullmatch(r"\s*What\s+(?:is|was)\s+the\s+total\s+value\s+of\s+outstanding\s+"
                         r"options\s+and\s+warrants\s+under\s+(?:the\s+)?(?P<plan>[^?\n]+)\??\s*",
                         question, re.I)
    if match is None:
        return ""
    name = match.group("plan").strip()
    if (len(name.split()) < 3 or not re.search(r"\b(?:19|20)\d{2}\b", name)
            or not re.search(r"\bPlan$", name, re.I)):
        return ""
    return name


def eligible_awards_plan(question, evidence, operation):
    """Return a complete question/source-only quantity-times-price plan."""
    name = _named_plan(question)
    if operation != "quantity_times_price" or not name or not evidence:
        return None
    from verifiqa.agent.source_question_planner import _Plan
    introduction = re.compile(
        r"The\s+following\s+table\s+sets\s+forth,?\s+as\s+of\s+"
        r"(?P<date>(?:" + _MONTH + r")\s+\d{1,2},?\s+(?:19|20)\d{2}),?\s+"
        r"certain\s+information\s+related\s+to\s+our\s+compensation\s+plans", re.I)
    intros = list(introduction.finditer(evidence))
    if len(intros) != 1:
        return None
    intro = intros[0]
    header = re.compile(
        r"Number\s+of\s+Shares\s+to\s+be\s+Issued\s+Upon\s+Exercise\s+of\s+"
        r"Outstanding\s+Options,?\s+Warrants\s+and\s+Rights\s+"
        r"(?P<price>Weighted[-\s]+Average\s+Exercise\s+Price\s+of\s+Outstanding\s+Options,?\s+"
        r"Warrants\s+and\s+Rights\s*\((?P<price_note>\d{1,2})\))\s+"
        r"Number\s+of\s+Shares\s+Remaining\s+Available\s+for\s+Future\s+Issuance\s+Under\s+"
        r"Equity\s+Compensation\s+Plans\s*\(Excluding\s+Securities\s+Reflected\s+in\s+1st\s+Column\)", re.I)
    headers = list(header.finditer(evidence, intro.end()))
    if len(headers) != 1:
        return None
    matched_header = headers[0]
    words = re.findall(r"[A-Za-z0-9]+", name)
    plan_pattern = r"[\W_]+".join(re.escape(word) for word in words)
    row_pattern = re.compile(
        r"(?<!\w)" + plan_pattern + r"\s+(?P<inclusive>" + _INTEGER + r")\s*"
        r"\((?P<note>\d{1,2})\)\s+(?P<price>\$?\s*" + _NUMBER + r")\s+"
        r"(?P<future>" + _INTEGER + r"|[—–-])(?=\s|$)", re.I)
    rows = list(row_pattern.finditer(evidence, matched_header.end()))
    if len(rows) != 1:
        return None
    row = rows[0]
    if re.search(r"\b(?:thousand|million|billion)s?\b", evidence[intro.start():row.end()], re.I):
        return None
    # The exact note must exhaust the population, rather than merely mention
    # an option count alongside another instrument or a second plan.
    note = row.group("note")
    footnote = re.compile(
        r"(?:^|\n)\s*\(" + re.escape(note) + r"\)\s*Consists\s+of\s+"
        r"(?P<rsus>" + _INTEGER + r")\s+restricted\s+share\s+units"
        r"(?:,\s*with\s+performance[-\s]+based\s+awards\s+assuming\s+maximum\s+performance,)?"
        r"\s+and\s+(?P<options>" + _INTEGER + r")\s+stock\s+options\s*\.(?=\s|$)", re.I)
    notes = list(footnote.finditer(evidence, row.end()))
    if len(notes) != 1:
        return None
    note_match = notes[0]
    # Duplicate note numbers, even with a different definition, are ambiguous.
    if len(re.findall(r"(?:^|\n)\s*\(" + re.escape(note) + r"\)", evidence)) != 1:
        return None
    price_note = matched_header.group("price_note")
    exclusion = re.compile(
        r"(?:^|\n)\s*\(" + re.escape(price_note) + r"\)\s*Does\s+not\s+reflect\s+"
        r"restricted\s+stock\s+units\s+because\s+these\s+awards\s+have\s+no\s+exercise\s+price\s*\.", re.I)
    exclusions = list(exclusion.finditer(evidence, row.end()))
    if len(exclusions) != 1 or len(re.findall(
            r"(?:^|\n)\s*\(" + re.escape(price_note) + r"\)", evidence)) != 1:
        return None
    # A literal dollar sign in this typed price column supplies currency; it
    # never supplies a scale for a monetary total from another table.
    table_prefix = evidence[matched_header.end():row.end()]
    table_context = evidence[intro.start():row.end()]
    from verifiqa.verification.money_units import explicit_money_currencies, unrecognized_currency_unit
    if ("$" not in table_prefix or explicit_money_currencies(table_context) - {"USD"}
            or re.search(r"(?:NT|HK|CA|AU|SG|[A-Za-z]{3})\s*\$|[€£]", table_context, re.I)):
        return None
    captions = re.findall(r"(?:\bcurrency\s*[:=]\s*|\b(?:prices?|amounts?|figures?)\s+"
                          r"(?:are\s+)?(?:in|expressed\s+in)\s+)([A-Za-z]{3})\b", table_context, re.I)
    captions.extend(re.findall(r"^\s*\(?\s*([A-Za-z]{3})\s*\)?\s*$", table_context, re.M))
    if any(unrecognized_currency_unit(code.upper()) for code in captions):
        return None
    integer = lambda text: int(text.replace(",", ""))
    inclusive, rsus, quantity = (integer(row.group("inclusive")), integer(note_match.group("rsus")),
                                 integer(note_match.group("options")))
    if quantity <= 0 or rsus < 0 or inclusive != rsus + quantity:
        return None
    # This check validates the independently selected complete population; it
    # cannot choose a subset or a count by agreement with a proposed answer.
    price = float(row.group("price").replace("$", "").replace(",", "").strip())
    if not math.isfinite(price) or price <= 0:
        return None
    when = re.sub(r"\s+", " ", intro.group("date"))
    section = "Securities Authorized for Issuance under Equity Compensation Plans / " + name + " / " + when
    identity = "source_eligible_awards:" + hashlib.sha256((intro.group() + matched_header.group()
                + row.group() + note_match.group() + exclusions[0].group()).encode()).hexdigest()[:20]
    labels = {"eligible_option_count": "Eligible option count under " + name,
              "applicable_exercise_price": "Weighted-average exercise price under " + name}
    columns = {"eligible_option_count": "Exercise-eligible stock options in footnote (" + note + ")",
               "applicable_exercise_price": re.sub(r"\s+", " ", matched_header.group("price"))}
    facts = {
        "eligible_option_count": GroundedFact("eligible_option_count", quantity, "shares", note_match.group(),
            period=when, row_label=labels["eligible_option_count"], fact_type="source_table_numeric",
            source_value=quantity, source_unit="shares", source_section=section,
            source_column=columns["eligible_option_count"], source_table_id=identity),
        "applicable_exercise_price": GroundedFact("applicable_exercise_price", price, "USD per share", row.group(),
            period=when, row_label=labels["applicable_exercise_price"], fact_type="source_table_numeric",
            source_value=price, source_unit="USD per share", source_section=section,
            source_column=columns["applicable_exercise_price"], source_table_id=identity),
    }
    spec = ClaimSpec("quantity_times_price", "eligible_option_count * applicable_exercise_price",
                     [RoleSpec(key, [labels[key]], when) for key in facts], "USD", .01, "generic_operation",
                     period=when, operation=operation, planning_source="deterministic_source_table")
    return _Plan(spec, facts)


def eligible_awards_source_matches(ir, question, evidence):
    """Guard only recognized named-plan questions; legacy other products pass."""
    if not _named_plan(question):
        return True
    plan = eligible_awards_plan(question, evidence, "quantity_times_price")
    if plan is None or len(ir.facts) != len(plan.facts):
        return False
    used = set()
    for expected in plan.facts.values():
        candidates = [(name, fact) for name, fact in ir.facts.items()
                      if (fact.row_label, fact.period, fact.unit, fact.source_quote,
                          fact.source_section, fact.source_column, fact.source_table_id) ==
                         (expected.row_label, expected.period, expected.unit, expected.source_quote,
                          expected.source_section, expected.source_column, expected.source_table_id)]
        if len(candidates) != 1:
            return False
        name, fact = candidates[0]
        if name in used or not math.isclose(fact.value, expected.value, rel_tol=1e-12, abs_tol=1e-9):
            return False
        used.add(name)
    return len(used) == len(ir.facts)
