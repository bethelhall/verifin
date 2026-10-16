"""Strict, gold-independent quantities for corrected shared-candidate baselines.

Historical paper scoring lives in ``candidate_scoring`` and ``paper_scoring``.
This module deliberately does not reproduce its magnitude-only, any-number,
one-percent rule. Ambiguous prose needs a reviewed quantity, not a first-number
guess. Normalization never examines a gold answer or a verifier's verdict.
"""
from __future__ import annotations

import math
import ast
import hashlib
import json
import re
from decimal import Decimal
from typing import Any, Mapping

from .candidate_scoring import unit_info


_NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SCALE = r"(?:thousands?|millions?|billions?|mn|bn|k|m|b)"
_CURRENCY = r"(?:USD|US\$|NT\$|TWD|EUR|GBP|dollars?|euros?|pounds?|[$€£])"
_UNIT = re.compile(
    rf"^(?:{_CURRENCY}(?:\s*{_SCALE})?(?:\s*(?:per|/)\s*shares?)?"
    rf"|{_SCALE}(?:\s+(?:of\s+)?(?:{_CURRENCY}|shares?))?"
    rf"|(?:shares?|count)(?:\s+{_SCALE})?"
    r"|%|percents?|percentage|percentage\s+points?|pp"
    r"|ratio|proportion|fraction|dimensionless|unitless|pure|times|x"
    r"|days?|years?)$", re.IGNORECASE,
)
_SCALAR = re.compile(
    rf"^(?P<sign>[-+]?)\s*(?P<prefix>{_CURRENCY})?\s*"
    rf"(?P<number>{_NUMBER})\s*(?P<unit>.*?)$", re.IGNORECASE,
)
_PLACES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
           "five": 5, "six": 6}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _unit(text: Any, default_unit: str = "") -> dict[str, Any] | None:
    raw = str(text or "").strip().rstrip(".")
    if not raw:
        return None
    if not _UNIT.fullmatch(raw):
        return None
    if re.fullmatch(r"days?", raw, re.I):
        return {"unit": "days", "family": "days", "scale": 1., "currency": ""}
    if raw.lower() in {"times", "x"}:
        raw = "ratio"
    if raw.lower() == "percents":
        raw = "percent"
    info = unit_info(raw, default_unit=default_unit)
    return None if info["family"] in {None, "unknown"} else info


def parse_quantity(text: Any, default_unit: str = "") -> dict[str, Any] | None:
    """Parse a whole scalar answer; never extract a number from prose.

    Explicit units take precedence over defaults. Scale-only labels can inherit
    a currency/count dimension from an independent question or metadata unit.
    Bare numbers keep the empty unit when no such unit is available.
    """
    if isinstance(text, bool) or text is None:
        return None
    raw = str(text).strip()
    if raw.startswith("(") and raw.endswith(")"):
        inner = parse_quantity(raw[1:-1], default_unit)
        return {**inner, "value": -abs(inner["value"])} if inner else None
    match = _SCALAR.fullmatch(raw)
    if not match:
        return None
    try:
        value = float((match["sign"] or "") + match["number"].replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    unit_text = " ".join(part for part in (match["prefix"], match["unit"]) if part)
    if unit_text:
        info = _unit(unit_text, default_unit)
        if info is None:
            return None
        unit_source = "scalar_text"
    elif default_unit:
        info = _unit(default_unit)
        if info is None:
            return None
        unit_source = "default_unit"
    else:
        info = {"unit": "", "family": None, "scale": 1., "currency": ""}
        unit_source = "missing"
    return {"value": value, **info, "unit_source": unit_source}


def requested_unit(question: Any) -> str:
    """Read explicit output units, leaving ambiguous requests unresolved."""
    text = str(question or "")
    matches = []
    for match in re.finditer(
        rf"(?:\bin\s+(?:units\s+of\s+)?|\banswer\s+in\s+)"
        rf"(?P<unit>{_CURRENCY}(?:\s+{_SCALE})?(?:\s+per\s+share)?"
        rf"|{_SCALE}\s+(?:of\s+)?shares?|percentage\s+points?|percents?|%)",
        text, re.I,
    ):
        info = _unit(match["unit"])
        if info:
            matches.append(info["unit"])
    if len(set(matches)) == 1:
        return matches[0]
    if matches:
        return ""
    if re.search(r"\b(?:how many\s+)?percentage points?\b", text, re.I):
        return "percentage points"
    if re.search(r"\bdays payable outstanding\b|\bcash conversion cycle\b", text, re.I):
        return "days"
    if re.search(r"\bratio\b", text, re.I):
        return "ratio"
    return ""


def convert_value(value: Any, source_unit: str, target_unit: str) -> float | None:
    """Convert compatible explicit units. Unknown units are never guessed."""
    value = _finite(value)
    source, target = _unit(source_unit), _unit(target_unit)
    if value is None or source is None or target is None:
        return None
    if (source["family"], source["currency"]) != (target["family"], target["currency"]):
        return None
    converted = (Decimal(str(value)) * Decimal(str(source["scale"]))
                 / Decimal(str(target["scale"])))
    return _finite(float(converted))


def quantities_match(value: Any, unit: str, target_value: Any, target_unit: str,
                     absolute_tolerance: float = 0.) -> bool | None:
    """Compare signed quantities, with tolerance expressed in ``unit``.

    Missing/unknown units yield None. Explicit different dimensions/currencies
    yield False. A tiny floating-point guard is not an answer tolerance.
    """
    left, right = _finite(value), _finite(target_value)
    tolerance = _finite(absolute_tolerance)
    if tolerance is None or tolerance < 0:
        raise ValueError("absolute_tolerance must be finite and nonnegative")
    lhs, rhs = _unit(unit), _unit(target_unit)
    if left is None or right is None or lhs is None or rhs is None:
        return None
    if (lhs["family"], lhs["currency"]) != (rhs["family"], rhs["currency"]):
        return False
    converted = convert_value(right, target_unit, unit)
    if converted is None:
        return None
    guard = 4 * max(math.ulp(left), math.ulp(converted), math.ulp(tolerance))
    return abs(left - converted) <= tolerance + guard


def parse_candidate_quantity(text: Any, default_unit: str = "") -> dict[str, Any] | None:
    """Read a scalar, a scalar with explanation, or an equation's stated RHS.

    This candidate-only grammar preserves the declared leading answer. It does
    not evaluate the explanation, repair its arithmetic, or search prose for a
    number. Corrections and unresolved alternatives still require review. The
    reference-answer parser ``parse_quantity`` remains whole-scalar only.
    """
    # Normalize only the mathematical minus sign, retaining the original text
    # in the source record and judge prompt. An en/em dash is not a minus sign.
    raw = text.strip().replace("\N{MINUS SIGN}", "-") if isinstance(text, str) else text
    scalar = parse_quantity(raw, default_unit)
    if scalar is not None or not isinstance(text, str):
        return scalar
    if raw.count("=") == 1:
        expression, answer = raw.split("=")
        head = parse_quantity(answer.strip(), default_unit)
        if head is not None and _is_arithmetic_expression(expression.strip()):
            return {**head, "answer_source": "equation_answer",
                    "unit_source": "equation_answer_text" if head["unit_source"] == "scalar_text"
                                   else head["unit_source"]}
    # Try each opening parenthesis so an accounting-negative head also works.
    for opening in re.finditer(r"\s+\(", raw):
        head = parse_quantity(raw[:opening.start()], default_unit)
        if head is None:
            continue
        suffix = raw[opening.end() - 1:].rstrip(".").rstrip()
        depth, closing = 0, None
        for index, char in enumerate(suffix):
            depth += (char == "(") - (char == ")")
            if depth == 0:
                closing = index
                break
        if closing != len(suffix) - 1 or depth != 0:
            return None
        body = suffix[1:-1].strip()
        if not body or re.search(
            r"\b(?:actually|instead|alternatively|correction|corrected|correct|"
            r"not|should|rather|maybe|perhaps|either|uncertain|answer|"
            r"revised|revision|replaced)\b", body, re.I,
        ):
            return None
        # A parenthesized alternate scalar must express the same quantity.
        alternate = parse_quantity(body, head["unit"])
        if alternate is not None and quantities_match(
            head["value"], head["unit"], alternate["value"], alternate["unit"]
        ) is not True:
            return None
        # Permit explicit equivalent restatements, e.g. "or $302,578
        # thousand as reported ..." or "= 0.397 or 39.7%". Other uses of
        # "or" may offer competing answers and remain unresolved.
        alternatives = re.split(r"\bor\b", body, flags=re.I)
        if len(alternatives) > 1:
            if len(alternatives) != 2:
                return None
            before, after = alternatives
            after = re.split(r"\bas reported\b", after, maxsplit=1, flags=re.I)[0]
            restatement = parse_quantity(after.strip(), head["unit"])
            if restatement is None or quantities_match(
                head["value"], head["unit"], restatement["value"], restatement["unit"]
            ) is not True:
                return None
            if before.strip():
                # An unlabelled value in a percent conversion is a ratio.
                prior_unit = "ratio" if head["unit"] == "percent" else head["unit"]
                prior = parse_quantity(before.rsplit("=", 1)[-1].strip(), prior_unit)
                if prior is None or quantities_match(
                    head["value"], head["unit"], prior["value"], prior["unit"]
                ) is not True:
                    return None
        # A list of quantities without explanatory prose is ambiguous too.
        # A comma is a grouping separator only before exactly three digits
        # followed by a non-digit/end; leave those separators for parse_quantity.
        parts = re.split(r"\band\b|;|(?<!\d),|,(?!\d{3}(?:\D|$))", body, flags=re.I)
        if len(parts) > 1 and all(parse_quantity(p.strip(), head["unit"]) for p in parts):
            return None
        return {**head, "answer_source": "leading_answer",
                "unit_source": "leading_answer_text" if head["unit_source"] == "scalar_text"
                               else head["unit_source"]}
    return None


def _is_arithmetic_expression(text: str) -> bool:
    """Validate a bounded arithmetic LHS without calculating its result."""
    if not text or len(text) > 10000 or not re.fullmatch(r"[\d,eE.+*/()\s-]+", text):
        return False
    # Remove thousands separators only within well-formed numeric literals;
    # malformed commas remain and are rejected as tuples or syntax errors.
    expression = re.sub(_NUMBER, lambda m: m[0].replace(",", ""), text)
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return False
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub)
    for index, node in enumerate(ast.walk(tree)):
        if index >= 1000 or not isinstance(node, allowed):
            return False
        if isinstance(node, ast.Constant) and _finite(node.value) is None:
            return False
    return True


def question_tolerance(question: Any, candidate_unit: str) -> tuple[float | None, str]:
    """Read requested rounding, otherwise compare exactly (up to FP noise).

    Never use candidate precision or the gold answer to widen the threshold.
    """
    text = str(question or "")
    decimal = re.search(
        r"\bround\b[^.!?]{0,45}?\b(?:to\s+)?(zero|one|two|three|four|five|six|\d+)"
        r"\s+decimal\s+places?", text, re.I,
    )
    if decimal:
        places = _PLACES.get(decimal[1].lower())
        if places is None:
            places = int(decimal[1])
        if places > 15:
            return None, "unsupported_decimal_precision"
        output_unit = requested_unit(text) or candidate_unit
        converted = convert_value(0.5 * 10 ** -places, output_unit, candidate_unit)
        return converted, "question_decimal_rounding"
    nearest = re.search(r"\bnearest\s+(million|thousand|billion|integer|whole number)\b", text, re.I)
    if nearest:
        scale = {"million": 1e6, "thousand": 1e3, "billion": 1e9,
                 "integer": 1., "whole number": 1.}[nearest[1].lower()]
        info = _unit(candidate_unit)
        if info is None:
            return None, "rounding_unit_unresolved"
        if nearest[1].lower() in {"integer", "whole number"}:
            output_unit = requested_unit(text) or candidate_unit
            return convert_value(.5, output_unit, candidate_unit), "question_integer_rounding"
        if info["family"] not in {"money", "count"}:
            return None, "rounding_unit_unresolved"
        return .5 * scale / info["scale"], "question_scale_rounding"
    return 0., "exact_no_rounding_requested"


def _declared_stored_unit(row: Mapping[str, Any]) -> str:
    for key in ("claimed_value_unit", "claimed_unit", "claim_unit"):
        raw = row.get(key)
        if raw:
            info = _unit(raw)
            return info["unit"] if info else ""
    return ""


def _display_metadata_unit(row: Mapping[str, Any]) -> str:
    diagnostics = row.get("diagnostics") or {}
    spec = diagnostics.get("claim_spec") or {}
    for value in (spec.get("claimed_unit"), spec.get("claim_unit"),
                  (row.get("claim") or {}).get("unit")):
        if value:
            info = _unit(value)
            if info:
                return info["unit"]
    return ""


def _unanimous_fact_unit(row: Mapping[str, Any]) -> str:
    facts = row.get("facts") or {}
    facts = facts.values() if isinstance(facts, Mapping) else facts
    units = [_unit(fact.get("unit")) for fact in facts if isinstance(fact, Mapping)]
    if units and all(info is not None and info == units[0] for info in units):
        return units[0]["unit"]
    return ""


def candidate_review_fingerprint(row: Mapping[str, Any]) -> str:
    """Bind a review to stripped candidate text and the original stored value.

    Saved baseline rows preserve ``candidate_source_claimed_value`` when a
    missing source value was recovered. The fingerprint deliberately contains
    no gold answer, verifier status, or inferred quantity.
    """
    text = row.get("claim_input_answer")
    if not isinstance(text, str):
        raise ValueError("Candidate review requires string claim_input_answer")
    value = row.get("candidate_source_claimed_value", row.get("claimed_value"))
    if value is not None and _finite(value) is None:
        raise ValueError("Candidate review requires finite original claimed_value or null")
    payload = {"claim_input_answer": text.strip(), "claimed_value": value}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _reviewed_quantity(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    review = row.get("candidate_quantity_review")
    if not isinstance(review, Mapping):
        return None, "invalid_candidate_review"
    if not all(isinstance(review.get(key), str) and review[key].strip()
               for key in ("reviewer", "reason", "review_source", "sha256", "unit")):
        return None, "incomplete_candidate_review"
    source_value = row.get("candidate_source_claimed_value", row.get("claimed_value"))
    try:
        fingerprint = candidate_review_fingerprint(row)
    except ValueError:
        return None, "invalid_candidate_review_source"
    if (review.get("claim_input_answer") != str(row.get("claim_input_answer") or "").strip()
            or "claimed_value" not in review or review["claimed_value"] != source_value
            or review["sha256"] != fingerprint):
        return None, "stale_candidate_review"
    value = _finite(review.get("value"))
    info = _unit(review.get("unit"))
    if value is None or info is None:
        return None, "invalid_reviewed_quantity"
    return {"value": value, **info, "unit_source": "explicit_candidate_review"}, ""


def normalize_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve display and stored numeric representations without gold access.

    Accept an explicit scalar answer, an equation's stated RHS, or a leading
    scalar with a parenthesized explanation. Other prose needs a quantity review
    bound to its exact text and source value; never infer a narrative's answer.
    """
    stored_raw = row.get("claimed_value")
    stored = _finite(stored_raw)
    result = {"value": None, "unit": "", "stored_value": stored,
              "stored_unit": "", "status": "unresolved", "reason": "",
              "tolerance": None, "tolerance_source": "unresolved",
              "value_source": "stored_claim" if stored is not None else "unavailable",
              "unit_source": "unresolved"}
    if stored_raw is not None and stored is None:
        return {**result, "reason": "invalid_stored_claim"}
    text = row.get("claim_input_answer")
    if not isinstance(text, str) or not text.strip():
        return {**result, "reason": "missing_candidate_text"}
    request = requested_unit(row.get("question"))
    declared = _declared_stored_unit(row)
    if any(row.get(key) for key in ("claimed_value_unit", "claimed_unit", "claim_unit")) and not declared:
        return {**result, "reason": "invalid_stored_unit"}
    display_metadata = _display_metadata_unit(row)
    default = request or display_metadata or declared
    reviewed = row.get("candidate_quantity_review") is not None
    if reviewed:
        scalar, error = _reviewed_quantity(row)
        if error:
            return {**result, "reason": error}
        ordinary = parse_candidate_quantity(text, default)
        if ordinary and ordinary["unit"] and quantities_match(
            ordinary["value"], ordinary["unit"], scalar["value"], scalar["unit"]
        ) is not True:
            return {**result, "reason": "review_disagrees_with_scalar_candidate"}
    else:
        scalar = parse_candidate_quantity(text, default)
    facts_unit = _unanimous_fact_unit(row)
    # A scale-only scalar (e.g. "2,018 million") supplies its own scale; the
    # unanimous fact currency may supply only its otherwise missing dimension.
    if scalar is None and not reviewed and not default and facts_unit:
        scalar = parse_candidate_quantity(text, facts_unit)
    if scalar is None:
        return {**result, "reason": "non_scalar_candidate_requires_review"}
    if not scalar["unit"]:
        return {**result, "reason": "missing_candidate_unit"}
    value, unit = scalar["value"], scalar["unit"]
    review = row.get("candidate_quantity_review") or {}
    if reviewed and stored is not None and value != stored and not review.get("stored_unit"):
        return {**result, "reason": "review_requires_stored_unit"}
    if reviewed and review.get("stored_unit"):
        review_unit = _unit(review["stored_unit"])
        if review_unit is None or (declared and review_unit["unit"] != declared):
            return {**result, "reason": "invalid_reviewed_stored_unit"}
        declared = review_unit["unit"]
    if stored is None:
        stored, stored_unit, source = value, unit, "scalar_reparse"
        if scalar.get("answer_source"):
            source = scalar["answer_source"] + "_reparse"
    else:
        source = "stored_claim"
        # Without explicit persisted units, equality establishes the candidate's
        # representation. A different numeric magnitude requires a documented
        # compatible runtime fact-unit conversion, never a power-of-ten guess.
        stored_unit = declared or unit
        equal = quantities_match(value, unit, stored, stored_unit)
        if equal is not True and not declared and facts_unit:
            if quantities_match(value, unit, stored, facts_unit) is True:
                stored_unit, equal = facts_unit, True
        if equal is not True:
            return {**result, "reason": "candidate_text_value_mismatch",
                    "stored_unit": declared, "unit": unit}
    tolerance, tolerance_source = question_tolerance(row.get("question"), unit)
    if tolerance is None:
        return {**result, "reason": "unresolved_question_tolerance"}
    return {"value": value, "unit": unit, "stored_value": stored,
            "stored_unit": stored_unit, "status": "resolved", "reason": "consistent_quantity",
            "tolerance": tolerance, "tolerance_source": tolerance_source,
            "value_source": "reviewed_candidate" if reviewed else source,
            "unit_source": scalar["unit_source"] if scalar["unit_source"] != "default_unit"
                           else "question_or_explicit_metadata"}


# Kept as a descriptive alias for consumers that use resolution terminology.
resolve_candidate = normalize_candidate
