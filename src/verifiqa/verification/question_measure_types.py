"""Keep a reported percentage distinct from the underlying amount or count."""
from __future__ import annotations

import re


def _valuation_bases(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    return {basis for basis, pattern in {
        "fair": r"\bfair[-\s]+value\b",
        "intrinsic": r"\bintrinsic[-\s]+value\b",
        "carrying": r"\bcarrying[-\s]+(?:value|amount)\b",
    }.items() if re.search(pattern, text, re.I)}


def _temporal_valuation_basis_supported(ir, question: str, operation: str | None) -> bool:
    """Keep an explicitly requested fair/intrinsic basis across observations.

    This does not constrain ratios, cross-measure formulas or comparisons of
    two valuation bases. A general fair-value note heading cannot supply the
    missing type of a source row: the row, selected column or QName must do so.
    """
    if operation not in {"average", "sum", "period_change", "percent_change"}:
        return True
    # Context before the actual question is not the requested measure, e.g.
    # "Regarding fair value reporting, what was the change in total assets?"
    target = re.split(r"\b(?:what|how)\b", question or "", maxsplit=1, flags=re.I)[-1]
    requested = _valuation_bases(target)
    if requested not in ({"fair"}, {"intrinsic"}):
        return True
    from .operation_bindings import same_measure
    from .periods import fact_period_key
    names = list(ir.facts)
    periods = {fact_period_key(name, ir.facts[name]) for name in names}
    if len(periods) < 2 or None in periods or not same_measure(ir, names):
        return True
    for fact in ir.facts.values():
        typed_source = " ".join((fact.row_label, getattr(fact, "source_column", ""),
                                 getattr(fact, "column", ""), getattr(fact, "concept", "")))
        if _valuation_bases(typed_source) != requested:
            return False
    return True


def source_measure_supported(question: str, label: str, section: str = "", unit: str = "", concept: str = "") -> bool:
    """Keep source-qualified amounts distinct from underlying balances."""
    concept_words = re.sub(r"([a-z])([A-Z])", r"\1 \2", concept or "")
    context = label + " " + section + " " + concept_words
    valuation_allowance = bool(re.search(r"\bvaluation\s+allowances?\b", question, re.I)
                               and re.search(r"\bvaluation\s+allowances?\b", label + " " + concept_words, re.I))
    if (re.search(r"\bdeferred\s+tax\b", context, re.I)
            and not valuation_allowance
            and not re.search(r"\b(?:deferred\s+tax|tax\s+(?:assets?|liabilit\w*|effects?))\b", question, re.I)):
        # A tax-effect reserve component is not the associated reserve itself.
        # The QName preserves this identity even when an LLM drops the caption.
        return False
    award = (r"\b(?:RSUs?|RSAs?|restricted\s+(?:stock|share)\s+(?:units?|awards?)|"
             r"stock\s+options?|SARs?|stock\s+appreciation\s+rights?)\b")
    expense = r"\b(?:expenses?|compensation|costs?)\b"
    if not re.search(award, question, re.I):
        return True
    if (re.search(r"\banti[-\s]?dilutive\b", label + " " + section, re.I)
            and not re.search(r"\banti[-\s]?dilutive\b|\bexcluded\b", question, re.I)):
        # Excluded awards are a subset, rather than the complete award balance.
        return False
    # A typed value/count row remains its own measure inside a broader
    # compensation note. Only an actual expense measure is barred here.
    value_kind = re.search(r"\b(intrinsic|fair)\s+value\b", label, re.I)
    count = (re.search(r"\b(?:number|count)\s+of\b|\bshares?\s+outstanding\b", label, re.I)
             or re.fullmatch(r"\s*(?:(?:thousand|million|billion)s?\s+)?(?:counts?|shares?)\s*", unit, re.I))
    if re.search(expense, question, re.I):
        return not (value_kind or count) or bool(re.search(r"\bexpenses?\b", label, re.I))
    if value_kind:
        return bool(re.search(r"\b" + value_kind.group(1) + r"\s+value\b", question, re.I))
    if count:
        return not bool(re.search(r"\bvalue\b", question, re.I))
    from .money_units import money_currency
    if money_currency(unit):
        # A legacy extraction can lose its expense caption. A naked award
        # label in dollars still cannot establish a requested award quantity
        # or an unspecified valuation method.
        return False
    return not bool(re.search(expense, label + " " + section, re.I))


def percentage_observation_supported(question: str) -> bool:
    """Whether this question can use a reported percentage as an operand.

    Asking for percentage *change* in revenue does not make a revenue-share
    table sufficient: monetary revenue and its share of an unknown total can
    change differently. Explicit share/percentage or reported-figure language
    can identify the percentage measure. Unqualified categories retain their
    source unit (e.g. a mean of the Consumer row in a percentage table).
    """
    text = question.lower()
    if re.search(r"\bcustomers?\b", text) and not re.search(r"\b(?:revenues?|sales)\b", text):
        return False
    monetary = re.search(
        r"\b(?:revenues?|sales|expenses?|costs?|purchases?|assets?|liabilit(?:y|ies)|"
        r"payables?|receivables?|debts?|goodwill|income|cash)\b", text)
    if monetary is None:
        return True
    target = re.sub(r"(?:\bpercent(?:age)?\s+|%\s*)(?:change|increase|decrease|growth|difference)\b",
                    " ", text)
    return bool(re.search(r"%|\b(?:percent(?:age)?s?|shares?|proportions?|fractions?|figures?|"
                          r"rates?|growth|margins?|yields?|returns?)\b", target))


def question_input_types_match(ir, question: str, operation: str | None) -> bool:
    """Apply the same operand-type guard to source plans and LLM fallbacks."""
    if not all(source_measure_supported(question, fact.row_label, getattr(fact, "source_section", ""), fact.unit,
                                       getattr(fact, "concept", ""))
               for fact in ir.facts.values()):
        return False
    if not _temporal_valuation_basis_supported(ir, question, operation):
        return False
    from .money_units import explicit_money_currencies, unrecognized_currency_unit
    if any(unrecognized_currency_unit(fact.unit) for fact in ir.facts.values()):
        return False
    currencies = set().union(*(explicit_money_currencies(fact.unit) for fact in ir.facts.values()))
    if len(currencies) > 1:
        return False
    if operation not in {"average", "sum", "period_change", "percent_change", "percentage_point_change",
                         "change_of_growth_rates", "difference_of_changes"}:
        return True
    percentages = [fact for fact in ir.facts.values()
                   if re.search(r"%|\bpercent(?:age)?", fact.unit, re.I)]
    return not percentages or percentage_observation_supported(question)
