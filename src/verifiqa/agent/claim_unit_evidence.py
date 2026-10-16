"""Display units from the claimed answer, without a reference answer or facts."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from verifiqa.verification.money_units import explicit_money_currencies


class AmbiguousClaimUnit(ValueError):
    pass


_CURRENCY = r"(?:U\.?S\.?\s+dollars?|US\$|NT\$|HK\$|[A-Za-z]{3}\s*\$?|[$€£]|dollars?|euros?)"
_GRAMMAR = {"the", "and", "for", "net", "per", "tax", "was", "are", "our", "its", "has", "had", "not", "all"}


@dataclass(frozen=True)
class ClaimMoneyUnitEvidence:
    unit: str = ""
    display_kind: str = "absent"  # scaled | base | currency_only | absent


def explicit_claim_money_unit(answer: str, claimed_value: float, source_currency: str) -> str:
    """Return the displayed currency/unit, without claiming its scale is known."""
    return claim_money_unit_evidence(answer, claimed_value, source_currency).unit


def explicit_claim_day_unit(answer: str, claimed_value: float) -> str:
    """Validate a day-valued claim's literal units; bare numbers stay unlabeled."""
    from verifiqa.agent.table_fact_extractor import _parse_number
    answer = (answer or "").replace("\N{MINUS SIGN}", "-")
    number = r"(?:\(\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\)|[-+]?\d[\d,]*(?:\.\d+)?)"
    other = r"(?:%|percent(?:age)?|years?|months?|weeks?|hours?|thousands?|millions?|billions?|" + _CURRENCY + r")"
    pattern = (r"(?<![\w.])(?P<currency_prefix>[-+]?\s*(?:" + _CURRENCY + r")\s*)?"
               r"(?P<number>" + number + r")(?=days?(?![A-Za-z])|" + other + r"(?![A-Za-z])|[^\w.]|$)")
    seen_days = conflicting = last_has_days = False
    for match in re.finditer(pattern, answer, re.I):
        value = _parse_number(match.group("number"))
        if value is not None and (match.group("currency_prefix") or "").lstrip().startswith("-"):
            value = -value
        if value is None or not math.isclose(value, claimed_value, rel_tol=1e-12, abs_tol=1e-12):
            continue
        start, end = match.span("number")
        prefix, suffix = answer[max(0, start - 30):start], answer[end:end + 55]
        day = re.match(r"\s*days?(?![A-Za-z])", suffix, re.I)
        before = re.search(r"(?<!\w)(?P<unit>" + _CURRENCY + r")\s*$", prefix, re.I)
        after = re.match(r"\s*(?P<unit>" + other + r")(?![A-Za-z])", suffix[day.end():] if day else suffix, re.I)
        # This is prose, rather than a structured currency-unit field. A
        # lowercase word such as "now" is not an explicit currency code.
        # Known currencies and unknown uppercase codes still conflict.
        def prose_word(match):
            token = match.group("unit").strip()
            return (token.lower() in _GRAMMAR or
                    (len(token) == 3 and token.isalpha() and token.islower()
                     and not explicit_money_currencies(token)))
        if before and prose_word(before):
            before = None
        if after and prose_word(after):
            after = None
        seen_days = seen_days or bool(day)
        conflicting = conflicting or bool(before or after)
        last_has_days = bool(day)
    if conflicting or (seen_days and not last_has_days):
        raise AmbiguousClaimUnit("conflicting_or_missing_final_unit")
    return "days" if seen_days else ""


def claim_money_unit_evidence(answer: str, claimed_value: float,
                              source_currency: str) -> ClaimMoneyUnitEvidence:
    """Read currency/scale next to this displayed number; never infer magnitude.

    Multiple occurrences must agree. A bare final number cannot borrow the
    display scale of a different operand in an explanation.
    """
    from verifiqa.agent.table_fact_extractor import _parse_number
    answer = (answer or "").replace("\N{MINUS SIGN}", "-")
    number = r"(?:\(\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\)|[-+]?\d[\d,]*(?:\.\d+)?)"
    # Include an attached currency prefix in the match, while retaining the
    # number's own span for adjacent-unit reading. EUR325 must not become a
    # unitless 325 that can borrow a USD report format.
    pattern = (r"(?<![\w.])(?P<currency_prefix>[-+]?\s*(?:" + _CURRENCY + r")\s*)?"
               r"(?P<number>" + number + r")"
               r"(?=(?:" + _CURRENCY + r"|thousands?|millions?|billions?)(?![A-Za-z])|[^\w.]|$)")
    units = set()
    last_has_unit = False
    last_kind = "absent"
    for match in re.finditer(pattern, answer or ""):
        value = _parse_number(match.group("number"))
        currency_prefix = match.group("currency_prefix") or ""
        if value is not None and currency_prefix.lstrip().startswith("-"):
            value = -value
        if value is None or not math.isclose(value, claimed_value, rel_tol=1e-12, abs_tol=1e-12):
            continue
        last_has_unit = False
        last_kind = "absent"
        start, end = match.span("number")
        prefix, suffix = answer[max(0, start - 30):start], answer[end:end + 55]
        # A currency prefix is a complete token, not the tail of preceding
        # prose such as "were", "generated", or "Approximately".
        before = re.search(r"(?<!\w)(?P<unit>" + _CURRENCY + r")\s*$", prefix, re.I)
        after = re.match(r"\s*(?P<unit>(?:" + _CURRENCY + r"\s+)?"
                         r"(?:thousands?|millions?|billions?)(?:\s+(?:of\s+)?" + _CURRENCY + r")?"
                         r"|" + _CURRENCY + r")(?![A-Za-z])", suffix, re.I)
        if before and before.group("unit").strip().lower() in _GRAMMAR:
            before = None
        if after and after.group("unit").strip().lower() in _GRAMMAR:
            after = None
        if not before and not after:
            continue
        text = (before.group("unit") if before else "") + " " + (after.group("unit") if after else "")
        unknown = {code.upper() for code in re.findall(r"\b[A-Za-z]{3}\b", text)
                   if code.lower() not in _GRAMMAR and not explicit_money_currencies(code)}
        currencies = explicit_money_currencies(text)
        if len(currencies) > 1:
            raise AmbiguousClaimUnit("conflicting_currencies")
        currency = " ".join(sorted(unknown)) if unknown else next(iter(currencies), source_currency)
        scale = next((name + "s" for name in ("thousand", "million", "billion") if name in text.lower()), "")
        units.add(currency + (" " + scale if scale else ""))
        # A currency symbol/code does not state the display scale of a filing
        # excerpt. Physical unit words or an explicit scale do state it.
        last_kind = ("scaled" if scale else "base" if re.search(r"\b(?:dollars?|euros?)\b", text, re.I)
                     else "currency_only")
        last_has_unit = True
    if len(units) > 1 or (units and not last_has_unit):
        raise AmbiguousClaimUnit("conflicting_or_missing_final_unit")
    return ClaimMoneyUnitEvidence(next(iter(units)), last_kind) if units else ClaimMoneyUnitEvidence()
