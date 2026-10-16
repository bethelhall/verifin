"""Currency identity for monetary display scales; no exchange rates are inferred."""
from __future__ import annotations

import re


def explicit_money_currencies(text: str) -> set[str]:
    currencies = set()
    patterns = {
        "TWD": r"\bTWD\b|\bNT\s*\$|\b(?:New\s+)?Taiwan\s+dollars?\b",
        "EUR": r"\bEUR\b|€|\beuros?\b",
        "GBP": r"\bGBP\b|£|\bpounds?\s+sterling\b",
        "CAD": r"\bCAD\b|\bC\$|\bCanadian\s+dollars?\b",
        "AUD": r"\bAUD\b|\bA\$|\bAustralian\s+dollars?\b",
        "HKD": r"\bHKD\b|\bHK\$|\bHong\s+Kong\s+dollars?\b",
        "SGD": r"\bSGD\b|\bSingapore\s+dollars?\b",
        "JPY": r"\bJPY\b|\byen\b",
        "CHF": r"\bCHF\b|\bSwiss\s+francs?\b",
        "USD": r"\bUSD\b|\bUS\s*\$|\bU\.?\s*S\.?\s+dollars?\b",
    }
    for currency, pattern in patterns.items():
        if re.search(pattern, text or "", re.I):
            currencies.add(currency)
    # A bare dollar sign/dollar word retains the existing USD convention. A
    # qualified symbol such as NT$ must never add a second, spurious currency.
    if not currencies and ("$" in (text or "") or re.search(r"\bdollars?\b", text or "", re.I)):
        currencies.add("USD")
    return currencies


def money_currency(unit: str) -> str | None:
    text = unit or ""
    if re.search(r"\bshares?\b|\bper\b|/|%|\bpercent", text, re.I):
        return None
    currencies = explicit_money_currencies(text)
    if len(currencies) == 1:
        return next(iter(currencies))
    if not currencies and re.fullmatch(r"\s*(?:thousand|million|billion)s?\s*", text, re.I):
        return "USD"  # Legacy unqualified monetary-scale aliases.
    return None


def unrecognized_currency_unit(unit: str) -> bool:
    """Reject an explicit currency code whose identity we cannot preserve.

    This inspects structured units only, rather than treating three-letter
    acronyms in questions or evidence as currencies. Unsupported currencies
    must not silently become dimensionless numbers or acquire a USD label.
    """
    text = (unit or "").strip()
    codes = re.findall(r"\biso4217:([A-Za-z]{3})\b", text, re.I)
    # Parentheses, currency signs and display qualifiers do not erase a code.
    # Unit fields are structured metadata, so an unknown uppercase token must
    # fail closed even when a bare dollar sign would otherwise imply USD.
    codes.extend(code for code in re.findall(r"(?<![A-Za-z])([A-Z]{3})(?![A-Za-z])", text)
                 if code not in {"PER", "AND"})
    prefix = re.match(r"([A-Za-z]{3})(?=[\s$(]|$)", text)
    if prefix and re.search(r"\b(?:thousand|million|billion)s?\b", text, re.I):
        codes.append(prefix.group(1))
    return any(not explicit_money_currencies(code) for code in codes)
