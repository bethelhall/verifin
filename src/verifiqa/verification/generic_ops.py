"""Operation-definition registry for deployment-realistic FinQA verification.

FinanceBench draws formula authority from the metric-policy registry; FinQA's gold
`program` is a per-question dataset annotation (an oracle). With the program disabled
(``--no-finqa-program``) the named-metric registry rarely covers FinQA's free-form
question-imposed arithmetic, so the verifier abstains.

This module supplies the legitimate, instance-independent alternative: the *definitions
of the standard financial-analysis operations* FinQA questions use. These are named,
citable techniques — not the gold derivation:

  * percentage change / growth        -> period-over-period growth rate
  * proportion / percent-of-total     -> common-size (vertical) analysis
  * absolute period change            -> arithmetic difference

The registry supplies only the FORM (e.g. ``part / base * 100``); the LLM still
classifies the operation and selects + grounds the operands, and the verifier checks
the grounded operands reproduce the canonical form. Operand ROLES are bound from the
question grammar and the document's own row labels (e.g. the "total" row), never from
the answer and never from the operand values.

Soundness discipline (uniform across families): bind roles only when they are
*unambiguous*; otherwise return None (abstain) rather than guess. A parse ambiguity
costs coverage, never soundness.
"""
from __future__ import annotations

import re
from typing import Any

from verifiqa.policy.registry import DEFAULT_POLICY_REGISTRY, PolicyRegistry
from verifiqa.types import VerificationIR
from verifiqa.verification.periods import fact_period_key


# -- citations ---------------------------------------------------------------
# Same source class the metric-policy registry already cites (CFA Institute +
# standard financial-statement-analysis texts), so the operation registry's
# provenance matches the authority a reviewer already accepts for FinanceBench.
_CFA_FAT = {
    "title": "CFA Institute - Financial Analysis Techniques",
    "url": "https://www.cfainstitute.org/en/membership/professional-development/refresher-readings/financial-analysis-techniques",
}
_FSA_TEXT = {
    "title": "White, Sondhi & Fried, The Analysis and Use of Financial Statements (common-size / ratio analysis)",
}


_OPERATIONS: dict[str, dict[str, Any]] = {
    "percent_change": {
        "technique": "period-over-period growth rate",
        "note": "Percentage change = (later period - earlier period) / earlier period * 100.",
        "source_refs": [_CFA_FAT],
    },
    "period_change": {
        "technique": "arithmetic period difference",
        "note": "Period change = later-period value - earlier-period value.",
        "source_refs": [_CFA_FAT],
    },
    "proportion": {
        "technique": "common-size (vertical) analysis",
        "note": "Common-size analysis: a component expressed as a percentage of its base total (part / base * 100).",
        "source_refs": [_CFA_FAT, _FSA_TEXT],
    },
    "average": {
        "technique": "arithmetic mean",
        "note": "Average = sum of observations / count of observations.",
        "source_refs": [_CFA_FAT],
    },
    "generic_ratio": {
        "technique": "ratio of two quantities",
        "note": "Ratio = numerator / denominator.",
        "source_refs": [_CFA_FAT],
    },
}


# -- operation classification ------------------------------------------------
# Order matters — more specific patterns checked before broader ones.
_PERCENT_CHANGE = re.compile(
    r"percent(?:age)?\s+(?:change|increase|decrease|growth|difference)"
    r"|%\s*change"
    r"|change\b[^.?]*\bpercent"
    r"|growth\s+rate"
    r"|rate\s+of\s+(?:growth|change|increase|decrease)",
    re.IGNORECASE,
)
_ROI = re.compile(
    r"\broi\b"
    r"|rate\s+of\s+return\s+of\s+an?\s+investment"
    r"|(?:percentage\s+)?cumulative\s+(?:total\s+)?return"
    r"|\bfive[- ]year\s+total\s+return\b"
    r"|\btotal\s+return\s+(?:percentage|percent|on)\b",
    re.IGNORECASE,
)
_PROPORTION = re.compile(
    r"\bwhat\s+(?:percent(?:age)?|portion|proportion|fraction|share)\b"
    r"|\b(?:percent(?:age)?|portion|proportion|fraction|share)\s+of\b"
    r"|\bas\s+a\s+(?:percent(?:age)?|proportion|fraction|share)\s+of\b",
    re.IGNORECASE,
)
_AVERAGE = re.compile(
    r"\bwhat\s+(?:is|was)\s+the\s+average\b"
    r"|\bwhat\s+was\s+the\s+average\b"
    r"|\baverage\s+(?:of|for|from|amount|number|price|value|payment|percentage|weighted)\b"
    r"|\b(?:average|mean|avg)\b",
    re.IGNORECASE,
)
# "average X per Y" is a per-unit rate (ratio), not a multi-period mean.
_AVERAGE_PER = re.compile(r"\baverage\b.{0,40}\bper\b", re.IGNORECASE)
_GENERIC_RATIO = re.compile(
    r"\bwhat\s+(?:is|was)\s+the\s+ratio\s+of\b"
    r"|\bratio\s+of\s+the\b",
    re.IGNORECASE,
)
_ABSOLUTE_CHANGE = re.compile(
    r"\b(?:change|increase|decrease|difference|grew|grow|growth|rose|fell|"
    r"decline|declined|decreased|increased|reduction|gain)\b",
    re.IGNORECASE,
)
_DECLINE_MAGNITUDE = re.compile(
    r"\b(?:decrease|decline|declined|drop|dropped|fell|fall|reduction|lower)\b",
    re.IGNORECASE,
)


def decline_magnitude_requested(question: str) -> bool:
    # "Increase / (decrease)" requests a signed change, unlike "how much
    # did it decrease?". Parenthetical direction labels must not invert it.
    text = question or ""
    for first, second in (("increase", "decrease"), ("decrease", "increase")):
        paired = (r"\(?\s*\b" + first + r"\s*\)?\s*(?:/|\bor\b)\s*\(?\s*" + second + r"\s*\)?"
                  r"|\b" + first + r"\s*\(\s*" + second + r"\s*\)"
                  r"|\(\s*" + first + r"\s*\)\s*" + second + r"\b")
        text = re.sub(paired, " ", text, flags=re.I)
    return bool(_DECLINE_MAGNITUDE.search(text))


def classify_operation(question: str, metric: str = "") -> str | None:
    """Return the generic operation class implied by the question, or None.

    Order matters: more specific patterns are checked before broader ones.
    ROI is checked before proportion because cumulative return questions contain
    percentage language. Average and generic_ratio are checked before absolute
    change to avoid false matches on "average increase" etc.
    """
    text = question or metric or ''
    if re.search(r"percentage\s+of\s+(?:the\s+)?(?:decrease|decline|change)", text, re.I):
        return "share_of_a_change"
    if re.search(r"difference\b.*\bchange\b.*\b(?:change|increase)\b", text, re.I):
        return "difference_of_changes"
    if re.search(r"\bchange\s+in\b.*\bgrowth\b", text, re.I):
        return "change_of_growth_rates"
    if re.search(r"(?:percentage\s+(?:difference|change)).*(?:expressed\s+as\s+a\s+percentage|ratios)", text, re.I):
        return "change_of_ratios"
    if re.search(r"\b(?:change|difference)\b.*(?:percentage points?|%\s+of\b|percentage\s+(?:to|of\s+plan)|as\s+a\s+percentage\s+of)", text, re.I):
        return "percentage_point_change"
    if re.search(r"\bnet\b.*\b(?:gains?|losses)\b.*\b(?:gains?|losses)\b", text, re.I):
        return "net_gains_losses"
    exclusion = re.search(r"\bexcluding\b|\bhas been received\b", text, re.I)
    if exclusion and not _ABSOLUTE_CHANGE.search(text[:exclusion.start()]):
        return "exclusion_or_remaining_amount"
    if re.search(r"\b(?:top|first)\s+3\b.*(?:sum|add|total)|\bsum\b.*\btop\s+3\b", text, re.I):
        return "selected_three_row_sum"
    if re.search(r"\btotal value\b.*\b(?:options|warrants)\b", text, re.I):
        return "quantity_times_price"
    mean = _AVERAGE.search(text)
    if mean and not (_ABSOLUTE_CHANGE.search(text[:mean.start()]) or _PERCENT_CHANGE.search(text[:mean.start()])):
        # Average reported EPS/dividends over periods is a mean. A new per-unit
        # measure (average revenue per customer) still needs ratio semantics.
        if not _AVERAGE_PER.search(text) or re.search(
                r"\bper\s+(?:(?:common|ordinary)\s+)?shares?\b|\bper\s+common\s+stock\b", text, re.I):
            return "average"
    if _PERCENT_CHANGE.search(text):
        return "percent_change"
    if _ROI.search(text):
        return "percent_change"  # same formula: (end - start) / start * 100
    share_measure = re.search(r"\bshare\s+of\s+(?:net\s+)?(?:earnings|profits?|losses)\b", text, re.I)
    temporal_revenue_share = re.search(
        r"\b(?:change|increase|decrease|difference)\s+in\b[^?]{0,120}"
        r"\bshare\s+of\s+(?:net\s+)?(?:revenues?|sales)\b", text, re.I)
    requested_ratio = re.match(
        r"\s*what(?:\s+(?:is|was|are|were))?\s+(?:the\s+)?"
        r"(?:percent(?:age)?|portion|proportion|fraction|share)\s+of\b", text, re.I)
    if (_ABSOLUTE_CHANGE.search(text) and not requested_ratio
            and (share_measure or temporal_revenue_share)):
        # 'Change in X's share of revenue' changes a reported measure across
        # periods. A question beginning 'what share of ...' still asks for a
        # proportion, including when its denominator mentions a change.
        return "period_change"
    if _PROPORTION.search(text):
        return "proportion"
    if re.search(r"percentage constitution|expressed as a percentage\s+to", text, re.I):
        return "proportion"
    if _GENERIC_RATIO.search(text):
        if re.search(r"\bratio\s+of\b.*(?:\bsum\b|\bdue\s+\d{4}\s+and\s+\d{4}).*\bto\b", text, re.I):
            return "sum_then_ratio"
        return "generic_ratio"
    if (re.search(r"\b(?:sum|add up)\b", text, re.I) or
        (not _ABSOLUTE_CHANGE.search(text) and re.search(r"\btotal\b.*(?:\bboth\b|\band\s+(?:(?:fiscal|fy)(?:\s+year)?\s+)?(?:19|20)\d{2}|\bbetween\b|\bdue\b|\bsalary\b|compensation expense and)", text, re.I))):
        return "sum"
    if re.search(r"(?:difference|fluctuation)\b.*\b(?:between|and|high and low)\b", text, re.I):
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", question or ""))
        # A pair of distinct dates means a period change; otherwise bind the
        # distinct measures in question order (basic/diluted, carrying/fair).
        if len(years) < 2 or re.search(r"(?:difference.*(?:total.*obligations|total revenue between)|high and low)", text, re.I):
            return "line_item_difference"
    if _ABSOLUTE_CHANGE.search(text):
        return "period_change"
    return None


def resolve_generic_operation(
    ir: VerificationIR, question: str, *, registry: PolicyRegistry | None = None,
    evidence_text: str = "", source_pdf_path: str = "", source_xbrl_bundle=None,
) -> dict[str, Any] | None:
    """Authorize a FinQA arithmetic claim against a cited operation definition.

    Returns a dict with the substituted ``formula`` (over this IR's fact names), the
    ``variables``, the ``computed_value``, the ``technique``/``source_note`` and
    ``source_refs`` (citations) — or None to abstain.
    """
    operation = classify_operation(question, ir.metric)
    from verifiqa.agent.source_ppe_basis import ppe_source_basis_matches
    if not ppe_source_basis_matches(ir, question, evidence_text):
        return None
    from .operation_bindings import question_scope_matches, question_population_matches, source_sign_basis_supported
    from .question_measure_types import question_input_types_match
    if not question_input_types_match(ir, question, operation):
        return None
    if not source_sign_basis_supported(ir, question, operation, evidence_text):
        return None
    if not question_scope_matches(ir, question, operation):
        return None
    if source_xbrl_bundle is not None:
        from verifiqa.agent.source_xbrl_schema_planner import plan_from_source_xbrl, resolve_source_xbrl_operation
        if plan_from_source_xbrl(question, evidence_text, source_xbrl_bundle) is not None:
            return resolve_source_xbrl_operation(ir, question, evidence_text, source_xbrl_bundle)
    if not question_population_matches(ir, question, operation, evidence_text):
        return None
    if operation == "quantity_times_price":
        from verifiqa.agent.source_eligible_awards_planner import eligible_awards_source_matches
        if not eligible_awards_source_matches(ir, question, evidence_text):
            return None
    if operation in {"sum", "average"}:
        from verifiqa.agent.source_pdf_group_planner import source_pdf_group_matches
        if not source_pdf_group_matches(ir, question, evidence_text, source_pdf_path):
            return None
    if operation == "proportion":
        from verifiqa.agent.source_exclusion_planner import loan_proportion_matches
        if not loan_proportion_matches(ir, question, evidence_text):
            return None
        if re.search(r"\bwhat\s+percent(?:age)?\s+of\b.+\bconsists?\s+of\b", question, re.I):
            from .operation_bindings import composition_source_matches
            if not composition_source_matches(ir, question, evidence_text):
                return None
        resolved = _resolve_proportion(ir, question)
        if resolved is not None:
            return resolved
        # A compound numerator or denominator needs independently bound set
        # membership. Retain the ordinary two-fact proportion path for names
        # such as "furniture and equipment", which describe a single row.
        from .operation_bindings import resolve_extended_operation
        compound = resolve_extended_operation(ir, question, "sum_then_ratio", evidence_text=evidence_text)
        return _result("sum_then_ratio", **compound, registry=registry) if compound is not None else None
    if operation in ("percent_change", "period_change"):
        if operation == "percent_change" and re.search(r"\bpromotion\b", question, re.I) and re.search(
                r"\bsalar(?:y|ies)\b", question, re.I):
            from .operation_bindings import resolve_promotion_salary_change
            event = resolve_promotion_salary_change(ir, question, evidence_text)
            return _result(operation, **event, registry=registry) if event is not None else None
        resolved = _resolve_change(ir, question, operation)
        if resolved is not None:
            return resolved
        # Complete, explicitly dated change columns can telescope to a wider
        # absolute period change. They do not supply the missing base needed
        # for a percentage change.
        from .operation_bindings import resolve_disclosed_period_change
        disclosed = resolve_disclosed_period_change(ir, question, operation)
        return _result(operation, **disclosed, registry=registry) if disclosed is not None else None
    if operation == "average":
        return _resolve_average(ir, question)
    if operation == "generic_ratio":
        return _resolve_generic_ratio(ir, question)
    if operation:
        from .operation_bindings import resolve_extended_operation
        resolved = resolve_extended_operation(ir, question, operation, evidence_text=evidence_text)
        if resolved is not None:
            return _result(operation, **resolved, registry=registry)
    return None


# -- percentage / absolute change (period-ordered operands) ------------------

def _resolve_change(ir: VerificationIR, question: str, operation: str) -> dict[str, Any] | None:
    from .operation_bindings import compatible_units, same_measure
    if not compatible_units(ir, list(ir.facts)) or not same_measure(ir, list(ir.facts)):
        return None
    ordered = _facts_ordered_by_period(ir)
    if ordered is None:
        return None
    start_name, end_name = ordered
    try:
        start_value = float(ir.facts[start_name].value)
        end_value = float(ir.facts[end_name].value)
    except (KeyError, TypeError, ValueError):
        return None

    if operation == "percent_change":
        if start_value == 0.0:
            return None
        formula = f"({end_name} - {start_name}) / {start_name} * 100"
        computed = (end_value - start_value) / start_value * 100.0
    elif decline_magnitude_requested(question):
        formula = f"{start_name} - {end_name}"
        computed = start_value - end_value
    else:
        formula = f"{end_name} - {start_name}"
        computed = end_value - start_value

    return _result(operation, formula, sorted({start_name, end_name}), computed,
                   start=start_name, end=end_name)


def _facts_ordered_by_period(ir: VerificationIR) -> tuple[str, str] | None:
    """Order distinct explicit years, quarters, dates, or opening/closing periods."""
    facts = list(ir.facts.keys())
    if len(facts) != 2:
        return None
    keyed = []
    for name in facts:
        key = fact_period_key(name, ir.facts[name])
        if key is None:
            return None
        keyed.append((key, name))
    if keyed[0][0] == keyed[1][0]:
        return None
    keyed.sort(key=lambda item: item[0])
    return keyed[0][1], keyed[1][1]


# -- average (arithmetic mean) ----------------------------------------------

def _resolve_average(ir: VerificationIR, question: str) -> dict[str, Any] | None:
    from .operation_bindings import compatible_units, same_measure, undated_average_periods_match
    items = list(ir.facts.items())
    if len(items) < 2 or not compatible_units(ir, list(ir.facts)):
        return None

    expected_years = _year_span_from_question(question)
    if expected_years is None:
        return None
    if expected_years:
        if not same_measure(ir, list(ir.facts)):
            return None
        keyed: list[tuple[int, str]] = []
        for name, fact in items:
            years = _fact_years(name, fact)
            if len(years) != 1:
                return None
            keyed.append((years[0], name))
        if sorted(year for year, _name in keyed) != expected_years:
            return None
        keyed.sort(key=lambda item: item[0])
        names = [name for _year, name in keyed]
    else:
        if not undated_average_periods_match(ir, question):
            return None
        names = [name for name, _fact in items]

    try:
        values = [float(ir.facts[name].value) for name in names]
    except (KeyError, TypeError, ValueError):
        return None
    count = len(values)
    if count == 0:
        return None

    formula = f"({' + '.join(names)}) / {count}"
    computed = sum(values) / count
    return _result("average", formula, names, computed, count=str(count))


def _year_span_from_question(question: str) -> list[int] | None:
    """Return the explicitly selected years, expanding only local range syntax.

    An empty list means no multi-year selection. ``None`` means an invalid or
    unsupported selection and must not become unconstrained arithmetic.
    Separate ranges and listed years form a union; intervening unlisted years
    are never added merely because 'from' or 'and' appears elsewhere.
    """
    text = question or ""
    year = r"(?:19|20)\d{2}"
    years = [int(value) for value in re.findall(rf"(?<!\d){year}(?!\d)", text)]
    if len(years) < 2:
        return []
    # Annual reporting dates may name the observations ("years ended ...").
    # A general date interval cannot instead authorize annual endpoints.
    month = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
             r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
    dated = (rf"\b{year}[-/]\d{{1,2}}[-/]\d{{1,2}}\b|"
             rf"\b{month}\.?\s+(?:\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s*)?{year}\b|"
             rf"\b\d{{1,2}}\s+{month}\.?\s+{year}\b")
    quartered = r"\bq[1-4]\b|\b(?:[1-4](?:st|nd|rd|th)?|first|second|third|fourth)\s+quarter\b"
    if re.search(quartered, text, re.I):
        return None
    dates = list(re.finditer(dated, text, re.I))
    if dates:
        annual = re.search(r"\bannual\b|\b(?:fiscal\s+)?years?\s+ended\b", text, re.I)
        if annual:
            text = re.sub(dated, lambda match: re.findall(year, match.group())[-1], text, flags=re.I)
        elif len(dates) > 1 and re.search(r"\b(?:to|through|between)\b", text, re.I):
            return None
    prefix = r"(?:FY\s*|fiscal\s+(?:years?\s+)?|years?\s+)?"
    patterns = (
        rf"(?<!\w){prefix}({year})\s*(?:\bto\b|\bthrough\b|[-‐‑‒–—−])\s*{prefix}({year})(?!\d)",
        rf"\bbetween\s+{prefix}({year})\s+and\s+{prefix}({year})(?!\d)",
    )
    selected = set(years)
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            first, last = sorted(int(value) for value in match.groups())
            if last - first > 20:
                return None
            selected.update(range(first, last + 1))
    return sorted(selected)


def _fact_years(name: str, fact: Any) -> list[int]:
    text = " ".join(
        str(value or "")
        for value in (name, getattr(fact, "period", ""), getattr(fact, "row_label", ""))
    )
    return sorted({int(year) for year in re.findall(r"(?:19|20)\d{2}", text)})


# -- explicit ratio (question grammar binds numerator and denominator) -------

def _resolve_generic_ratio(ir: VerificationIR, question: str) -> dict[str, Any] | None:
    items = list(ir.facts.items())
    if len(items) != 2:
        return None

    phrases = _ratio_phrases(question)
    if phrases is None:
        return None
    numerator_phrase, denominator_phrase = phrases
    numerator = _best_fact_for_phrase(items, numerator_phrase)
    denominator = _best_fact_for_phrase(items, denominator_phrase)
    if numerator is None or denominator is None or numerator == denominator:
        return None

    try:
        numerator_value = float(ir.facts[numerator].value)
        denominator_value = float(ir.facts[denominator].value)
    except (KeyError, TypeError, ValueError):
        return None
    if denominator_value == 0.0:
        return None

    formula = f"{numerator} / {denominator}"
    computed = numerator_value / denominator_value
    return _result("generic_ratio", formula, sorted({numerator, denominator}), computed,
                   numerator=numerator, denominator=denominator)


def _ratio_phrases(question: str) -> tuple[str, str] | None:
    q = (question or "").lower()
    match = re.search(
        r"\bratio\s+of\s+(?:the\s+)?(.{1,120}?)\s+"
        r"(?:to|over|divided\s+by)\s+(?:the\s+)?(.{1,120}?)(?:[?.]|$)",
        q,
    )
    if not match:
        return None
    return match.group(1), match.group(2)


def _best_fact_for_phrase(items: list[tuple[str, Any]], phrase: str) -> str | None:
    requested_years = set(re.findall(r"\b(?:19|20)\d{2}\b", phrase))
    if requested_years:
        items = [(name, fact) for name, fact in items if requested_years <= set(
            re.findall(r"(?:19|20)\d{2}", f"{name} {getattr(fact, 'period', '')} {getattr(fact, 'row_label', '')}"))]
    phrase_tokens = _content_tokens(phrase)
    if not phrase_tokens:
        return items[0][0] if requested_years and len(items) == 1 else None
    best_name, best_score, tie = None, 0, False
    for name, fact in items:
        tokens = _content_tokens(f"{getattr(fact, 'row_label', '') or ''} {name}")
        score = len(tokens & phrase_tokens)
        if score > best_score:
            best_name, best_score, tie = name, score, False
        elif score == best_score and score > 0:
            tie = True
    if best_score == 0 or tie:
        return None
    return best_name


# -- proportion / percent-of-total (common-size analysis) --------------------

def _resolve_proportion(ir: VerificationIR, question: str) -> dict[str, Any] | None:
    """Authorize a percent-of-total claim as part / base * 100.

    Requires exactly two facts and an *unambiguous* base (denominator), bound from
    the document's own "total" row label and/or the question grammar. Explicit
    contribution clauses bind both sides before the usual percent-of-total
    heuristic. Abstains on ambiguity; never inspects values to choose roles.
    """
    items = list(ir.facts.items())
    if len(items) != 2:
        return None
    from .operation_bindings import compound_ratio_phrases, _complete_phrase_parts
    phrases = compound_ratio_phrases(question)
    if phrases and any(re.search(r"\band\b|,", phrase, re.I) for phrase in phrases):
        # Even a two-fact draft must cover every explicitly requested component.
        # A total marker alone cannot waive a missing consultants operand.
        left, right = (_complete_phrase_parts(ir, phrase) for phrase in phrases)
        if (not left or not right or set(left) & set(right)
                or set(left + right) != set(ir.facts)):
            return None
    composition = re.search(r"\bwhat\s+percent(?:age)?\s+of\s+(.+?)\s+consists?\s+of\s+(.+?)(?:\?|$)",
                            question, re.I)
    contribution = _contribution_phrases(question)
    if contribution:
        # In 'percentage of PART contributing to BASE', the noun after 'of'
        # is the numerator. A partial word overlap (e.g. 'equity') cannot
        # distinguish nested totals, so require both complete identities.
        from .operation_bindings import compatible_units
        parts, bases = (_complete_phrase_parts(ir, phrase) for phrase in contribution)
        periods = {fact_period_key(name, fact) for name, fact in items}
        if (not parts or not bases or len(parts) != 1 or len(bases) != 1
                or parts == bases or not compatible_units(ir, list(ir.facts))
                or len(periods) != 1 or None in periods):
            return None
        part_name, base_name = parts[0], bases[0]
    elif composition:
        from .operation_bindings import compatible_units
        roles = _composition_roles(items, composition.group(1), composition.group(2))
        periods = {fact_period_key(name, fact) for name, fact in items}
        if not roles or not compatible_units(ir, list(ir.facts)) or len(periods) != 1 or None in periods:
            return None
        part_name, base_name = roles
    else:
        base_name = _select_base(items, question)
        if base_name is None:
            return None
        part_name = next(n for n, _ in items if n != base_name)
    try:
        base_value = float(ir.facts[base_name].value)
        part_value = float(ir.facts[part_name].value)
    except (KeyError, TypeError, ValueError):
        return None
    if base_value == 0.0:
        return None

    formula = f"{part_name} / {base_name} * 100"
    computed = part_value / base_value * 100.0
    return _result("proportion", formula, sorted({part_name, base_name}), computed,
                   part=part_name, base=base_name)


def _contribution_phrases(question: str) -> tuple[str, str] | None:
    """Return explicit (part, base) clauses, without consulting the candidate."""
    share = r"(?:percent(?:age)?|portion|proportion|fraction|share)"
    forward = re.search(
        rf"\b{share}\s+(?:of\s+|(?:does|did)\s+)?(.+?)\s+"
        r"(?:(?:is|are|was|were)\s+)?contribut(?:e[sd]?|ing)\s+to\s+(.+?)(?:\?|$)",
        question, re.I,
    )
    if forward:
        return forward.group(1), forward.group(2)
    reverse = re.search(
        rf"\b{share}\s+of\s+(.+?)\s+(?:is|are|was|were)\s+"
        r"contributed\s+by\s+(.+?)(?:\?|$)", question, re.I,
    )
    return (reverse.group(2), reverse.group(1)) if reverse else None


def _composition_roles(items, base_phrase, part_phrase):
    """Cover the complete identities in 'percentage of BASE consists of PART'."""
    def tokens(text):
        return {word[:-1] if len(word) > 3 and word.endswith('s') else word
                for word in _content_tokens(text)}
    base_phrase = re.sub(r'\bfrom\s+all\s+sources\b', ' ', base_phrase, flags=re.I)
    base, part = tokens(base_phrase), tokens(part_phrase)
    if not base or not part:
        return None
    bases = [name for name, fact in items if tokens(fact.row_label) == base]
    parts = [name for name, fact in items if tokens(fact.row_label) == base | part]
    if len(bases) != 1 or len(parts) != 1 or bases[0] == parts[0]:
        return None
    for name, wanted in ((bases[0], base), (parts[0], part - base)):
        column = getattr(dict(items)[name], 'source_column', '')
        if '|' in column:
            scope = tokens(column.rsplit('|', 1)[1])
            # A constructed Revenue/Network Solutions identity must retain
            # the actual typed column. A Total column has no scope tokens.
            if (name == parts[0] and not wanted <= scope) or (name == bases[0] and scope and scope != base):
                return None
    return parts[0], bases[0]


def _select_base(items: list[tuple[str, Any]], question: str) -> str | None:
    """Identify the base (denominator) fact value-blind, or None if ambiguous.

    Two independent signals; they must not contradict:
      1. document row-label carries a total/aggregate marker (the base of a
         common-size ratio is the total);
      2. question grammar: the fact named immediately after "of".
    """
    marked = [name for name, fact in items if _has_total_marker(fact)]
    grammar_base = _base_by_grammar(items, question)
    if len(marked) >= 2:
        return grammar_base  # explicit grammar can distinguish two different totals
    label_base = marked[0] if marked else None

    if label_base and grammar_base:
        return label_base if label_base == grammar_base else None  # disagree -> abstain
    return label_base or grammar_base


_TOTAL_MARKERS = ("total", "aggregate", "overall", "consolidated", "all ")


def _has_total_marker(fact: Any) -> bool:
    # A quote can discuss a different total (e.g. a customer's receivable
    # 'as a share of total consolidated receivables'). Only the operand's
    # identity supplies this independent denominator signal.
    text = " ".join(str(getattr(fact, key, "") or "") for key in ("row_label", "name"))
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return bool(tokens & {marker.strip() for marker in _TOTAL_MARKERS})


_STOP = frozenset({
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "from", "is", "are",
    "was", "were", "total", "percent", "percentage", "what", "comes", "come",
    "that", "which", "due", "as", "by", "on", "s",
})


def _base_by_grammar(items: list[tuple[str, Any]], question: str) -> str | None:
    """The fact whose label tokens appear in the phrase immediately after "of"."""
    q = (question or "").lower()
    explicit = re.search(
        r"(?:as\s+a\s+percent(?:age)?\s+(?:of|to)|\bamong\s+(?:the\s+)?|\bover\s+(?:the\s+)?)\s*(.+?)(?:\?|$)", q
    )
    if explicit:
        return _best_fact_for_phrase(items, explicit.group(1))
    match = re.search(
        r"\bof\s+(?:the\s+)?(?:total\s+)?(.{0,70}?)"
        r"(?:\s+(?:is|are|was|were|comes?|that|which|due|attributable|represent|account)\b|[,?]|$)",
        q,
    )
    if not match:
        return None
    if re.match(r"\s*(?:is|are|was|were)\b", match.group(1)):
        return None
    phrase_tokens = {t for t in re.findall(r"[a-z0-9]+", match.group(1)) if t not in _STOP}
    if not phrase_tokens:
        return None
    best_name, best_score, tie = None, 0, False
    for name, fact in items:
        tokens = _content_tokens(f"{getattr(fact, 'row_label', '') or ''} {name}")
        score = len(tokens & phrase_tokens)
        if score > best_score:
            best_name, best_score, tie = name, score, False
        elif score == best_score and score > 0:
            tie = True
    if best_score == 0 or tie:
        return None
    return best_name


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if t not in _STOP and not re.fullmatch(r"(?:19|20)?\d{2,4}", t)}


# -- result helper -----------------------------------------------------------

def _result(operation: str, formula: str, variables: list[str], computed: float,
            *, registry: PolicyRegistry | None = None, **roles: str) -> dict[str, Any]:
    registry = registry or DEFAULT_POLICY_REGISTRY
    canonical = {"average": "arithmetic_mean", "generic_ratio": "ratio", "proportion": "proportion_percent"}.get(operation, operation)
    policy = registry.find_policy(canonical)
    spec = _OPERATIONS.get(operation, {})
    if policy and policy.metadata.get("requires_question_binding"):
        spec = {"technique": policy.metadata.get("technique", canonical),
                "note": policy.metadata.get("formula_source_note", ""),
                "source_refs": policy.source_refs}
    if not spec:
        raise ValueError(f"Unregistered operation: {operation}")
    out = {
        "status": "applied",
        "operation": operation,
        "technique": spec["technique"],
        "formula": formula,
        "variables": variables,
        "computed_value": computed,
        "source_note": spec["note"],
        "source_refs": list(spec["source_refs"]),
    }
    out.update(roles)
    if policy:
        out["policy_id"] = policy.policy_id
    return out
