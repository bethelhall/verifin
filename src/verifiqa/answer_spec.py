from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Tuple

from verifiqa.formulas.evaluator import (
    FormulaError,
    evaluate_formula,
    formula_variables,
    normalize_formula,
)
from verifiqa.types import AnswerSpec, Claim, VerificationCertificate, VerificationSchema


_ROUND_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
}

# When the question does not state a rounding rule, precision is inferred from the
# model's printed answer. The printed digit count can exceed the model's actual
# rounding accuracy, so a pure half-ULP band false-violates answers that are
# faithful to the evidence but rounded slightly coarser or off by ~1 ULP. A small
# relative floor admits those while keeping genuinely wrong answers (off by more
# than a percent) violated. Only applied when precision is inferred, never when
# the question explicitly specifies the rounding.
_ANSWER_PRECISION_REL_TOL = 0.001


def answer_spec_from_question(question: str, answer: str = "") -> AnswerSpec:
    text = " ".join(question.lower().split())
    precision = _precision_digits(text)
    percent_expected = _expects_percent(text)
    percentage_points_expected = "percentage point" in text or "percentage-point" in text
    expected_unit = _expected_unit(text, percent_expected, percentage_points_expected)
    scale = _value_scale(text, expected_unit)
    tolerance = _tolerance(text, precision, expected_unit)
    question_precision = precision is not None or bool(re.search(
        r"\bnearest\s+(?:million|billion|thousand)s?\b", text,
    ))
    tolerance_source = "question_precision" if question_precision else "default"

    return AnswerSpec(
        expected_unit=expected_unit,
        value_scale=scale,
        tolerance=tolerance,
        precision_digits=precision,
        precision_kind="question" if question_precision else "default",
        percent_expected=percent_expected,
        percentage_points_expected=percentage_points_expected,
        tolerance_source=tolerance_source,
    )


def effective_tolerance_from_answer(
    spec: AnswerSpec,
    claimed_value: float,
    answer: str = "",
    reported_value: float | None = None,
) -> tuple[float, int | None, str]:
    """Return the absolute tolerance to enforce for a displayed answer claim.

    ``AnswerSpec.tolerance`` is already in the output unit expected by SMT. When
    the question does not state a rounding rule, infer precision from the printed
    answer token that matches the claimed value, mirroring ``apply_answer_spec``.
    """
    precision_digits = spec.precision_digits
    tolerance_source = spec.tolerance_source
    tolerance = spec.tolerance
    if precision_digits is None and tolerance_source != "question_precision":
        inferred_precision = _precision_digits_from_answer(
            answer,
            claimed_value,
            reported_value,
        )
        if inferred_precision is not None:
            precision_digits = inferred_precision
            half_ulp = 0.5 * (10 ** (-precision_digits))
            # Anchor the relative floor to the authoritative reported value when we
            # have it, never the claimed value: a model must not be able to widen its
            # own acceptance band by claiming a larger number.
            rel_anchor = abs(reported_value) if reported_value is not None else abs(claimed_value)
            rel_floor = _ANSWER_PRECISION_REL_TOL * rel_anchor
            tolerance = max(half_ulp, rel_floor)
            tolerance_source = "answer_precision"
    return tolerance, precision_digits, tolerance_source


def apply_answer_spec(
    certificate: VerificationCertificate,
    claim: Claim,
    schema: VerificationSchema,
    spec: AnswerSpec,
    answer: str = "",
) -> Tuple[VerificationCertificate, Claim, VerificationSchema]:
    claimed_value, formula_scale = _normalize_claimed_value(
        certificate.claim.claimed_value,
        certificate.claim.unit,
        spec,
    )
    reported_value = certificate.claim.reported_value
    if reported_value is not None:
        reported_value, _ = _normalize_claimed_value(reported_value, certificate.claim.unit, spec)

    formula = certificate.formula
    output_unit = spec.expected_unit if spec.expected_unit != "unspecified" else certificate.claim.unit
    if spec.expected_unit == "ratio":
        formula = _strip_percent_multiplier(formula)
    elif formula_scale != 1.0:
        formula = f"({formula}) * {_format_scale(formula_scale)}"
    formula = normalize_formula(formula, output_unit)

    tol, precision_digits, tolerance_source = effective_tolerance_from_answer(
        spec,
        claimed_value,
        answer,
        reported_value,
    )

    normalized_certificate = replace(
        certificate,
        formula=formula,
        claim=replace(
            certificate.claim,
            claimed_value=claimed_value,
            reported_value=reported_value,
            unit=spec.expected_unit if spec.expected_unit != "unspecified" else certificate.claim.unit,
        ),
        tolerance=tol,
    )
    normalized_claim = replace(claim, claimed_value=claimed_value)
    normalized_schema = replace(
        schema,
        formula=formula,
        tolerance=tol,
        unit_policy=f"answer_spec:{spec.expected_unit}",
        claim_unit=normalized_certificate.claim.unit,
        computed_unit=normalized_certificate.claim.unit,
        precision_digits=precision_digits,
        tolerance_source=tolerance_source,
    )
    return normalized_certificate, normalized_claim, normalized_schema


def apply_output_contract(
    certificate: VerificationCertificate,
    claim: Claim,
    schema: VerificationSchema,
    spec: AnswerSpec,
    answer: str = "",
) -> Tuple[VerificationCertificate, Claim, VerificationSchema, dict[str, Any]]:
    """Bind the verifier claim to the numeric value actually displayed by the answer.

    ``apply_answer_spec`` normalizes units and tolerance. This step checks that the
    answer text itself contains a compatible numeric claim, and when it does, uses
    that displayed value for SMT. That prevents hidden verifier-only rounding from
    turning an answer like ``-0.0142`` into a verified ``-0.01``.
    """

    check = output_contract_check(certificate, claim, schema, spec, answer)
    displayed_value = check.get("normalized_claimed_value")
    if check.get("valid") is True and displayed_value is not None:
        displayed_value = float(displayed_value)
        effective_tolerance = check.get("tolerance")
        if effective_tolerance is not None:
            effective_tolerance = float(effective_tolerance)
        certificate = replace(
            certificate,
            claim=replace(certificate.claim, claimed_value=displayed_value, reported_value=displayed_value),
            tolerance=effective_tolerance if effective_tolerance is not None else certificate.tolerance,
        )
        claim = replace(claim, claimed_value=displayed_value)
        schema = replace(
            schema,
            tolerance=effective_tolerance if effective_tolerance is not None else schema.tolerance,
            tolerance_source=check.get("tolerance_source") or schema.tolerance_source,
        )
    return certificate, claim, schema, check


def output_contract_check(
    certificate: VerificationCertificate,
    claim: Claim,
    schema: VerificationSchema,
    spec: AnswerSpec,
    answer: str = "",
) -> dict[str, Any]:
    precision_digits = schema.precision_digits
    tolerance = float(schema.tolerance if schema.tolerance is not None else spec.tolerance)
    target = float(claim.claimed_value)
    tokens = _numeric_tokens(answer)
    base = {
        "enabled": True,
        "expected_unit": spec.expected_unit,
        "precision_digits": precision_digits,
        "tolerance": tolerance,
        "tolerance_source": schema.tolerance_source or spec.tolerance_source,
        "claimed_value": target,
    }
    if not tokens:
        return {
            **base,
            "status": "failed",
            "valid": False,
            "reason": "no_numeric_claim_in_answer",
            "solver_enforced": True,
        }

    candidates = []
    for token in tokens:
        normalized, _ = _normalize_claimed_value(
            token["value"],
            "percent" if token["has_percent"] else certificate.claim.unit,
            spec,
        )
        distance = abs(float(normalized) - target)
        if distance <= max(tolerance, 1e-8) + 1e-12:
            candidates.append({**token, "normalized_value": float(normalized), "distance": distance})

    if not candidates:
        # A derived/relational claim (e.g. a change, difference, or growth) is
        # often presented through its components rather than as a single literal:
        # "the rate dropped from 24.6% to 21.6%" supports a -3.0 pp claim without
        # printing -3.0. Accept the answer when it states the formula's input
        # facts and those inputs compute to the claimed value.
        component = _answer_states_claim_components(
            certificate, target, tokens, spec, max(tolerance, 1e-8) + 1e-12
        )
        if component is not None:
            return {
                **base,
                "status": "passed",
                "valid": True,
                "reason": "answer_states_claim_components",
                "claim_components": component,
                "solver_enforced": False,
            }
        return {
            **base,
            "status": "failed",
            "valid": False,
            "reason": "claimed_value_not_in_answer",
            "answer_numbers": [token["raw"] for token in tokens],
            "solver_enforced": True,
        }

    if precision_digits is not None:
        valid_precision = [candidate for candidate in candidates if candidate["decimals"] <= precision_digits]
        if not valid_precision:
            candidate = min(candidates, key=lambda item: (item["decimals"], item["distance"]))
            return {
                **base,
                "status": "failed",
                "valid": False,
                "reason": f"answer_precision_exceeds_question:{candidate['decimals']}>{precision_digits}",
                "claim_token": candidate["raw"],
                "claim_token_decimals": candidate["decimals"],
                "normalized_claimed_value": candidate["normalized_value"],
                "solver_enforced": True,
            }
        candidates = valid_precision

    chosen = min(candidates, key=lambda item: (item["distance"], item["index"]))
    effective_tolerance, effective_tolerance_source = _display_token_tolerance(
        chosen,
        spec,
        schema.tolerance_source or spec.tolerance_source,
        tolerance,
    )
    return {
        **base,
        "status": "passed",
        "valid": True,
        "reason": "answer_output_matches_claim",
        "claim_token": chosen["raw"],
        "claim_token_decimals": chosen["decimals"],
        "normalized_claimed_value": chosen["normalized_value"],
        "tolerance": effective_tolerance,
        "tolerance_source": effective_tolerance_source,
        "solver_enforced": False,
    }


def _precision_digits(text: str) -> int | None:
    match = re.search(r"round(?:ed)?(?: your answer)? to (\d+) decimal places?", text)
    if match:
        return int(match.group(1))
    match = re.search(r"round(?:ed)?(?: your answer)? to ([a-z]+) decimal places?", text)
    if match:
        return _ROUND_WORDS.get(match.group(1))
    if "nearest integer" in text or "whole number" in text:
        return 0
    match = re.search(r"round(?:ed)?(?: your answer)? to (\d+) decimal", text)
    if match:
        return int(match.group(1))
    return None


def _expects_percent(text: str) -> bool:
    return "percent" in text or "percents" in text or "percentage" in text or "%" in text


def _expected_unit(text: str, percent_expected: bool, percentage_points_expected: bool) -> str:
    if percentage_points_expected:
        return "percentage_points"
    if percent_expected:
        return "percent"
    target = text.split("?", 1)[0]
    day_measure = re.search(
        r"\bdays?\s+(?:payable|sales|inventory|receivable)\s+outstanding\b"
        r"|\b(?:dpo|dso|dio|ccc)\b|\bcash\s+conversion\s+cycle\b",
        target,
    )
    if (day_measure and not re.search(r"\bratio\s+of\b", target)) or re.search(
            r"\b(?:in|units?\s+of)\s+days\b|\bhow\s+many\s+days\b", text):
        return "days"
    if "usd billions" in text or "usd billion" in text or "$ billions" in text or "$ billion" in text:
        return "USD billions"
    if "usd millions" in text or "usd million" in text or "$ millions" in text or "$ million" in text:
        return "USD millions"
    if "usd thousands" in text or "usd thousand" in text or "$ thousands" in text or "$ thousand" in text:
        return "USD thousands"
    # Bare "in billions"/"in millions"/"in thousands" without a $/usd prefix.
    if "in billions" in text or "in billion" in text:
        return "USD billions"
    if "in millions" in text or "in million" in text:
        return "USD millions"
    if "in thousands" in text or "in thousand" in text:
        return "USD thousands"
    if re.search(r"\bin\s+(?:usd|(?:u\.?s\.?\s+)?dollars?)\b", text):
        return "USD"
    if re.search(r"\bratio\b|\breturn\s+on\s+assets\b|\broa\b", text):
        return "ratio"
    return "unspecified"


def _value_scale(text: str, expected_unit: str) -> float:
    if expected_unit == "USD billions":
        return 1_000.0
    if expected_unit == "USD thousands":
        return 0.001
    if expected_unit == "USD":
        return 0.000001
    return 1.0


def _tolerance(text: str, precision: int | None, expected_unit: str) -> float:
    if precision is not None:
        return 0.5 * (10 ** (-precision))
    nearest = re.search(r"\bnearest\s+(million|billion|thousand)s?\b", text)
    if nearest:
        quantum = {"thousand": 1_000., "million": 1_000_000., "billion": 1_000_000_000.}[nearest[1]]
        output_scale = {"USD": 1., "USD thousands": 1_000., "USD millions": 1_000_000.,
                        "USD billions": 1_000_000_000.}.get(expected_unit)
        if output_scale is not None:
            return 0.5 * quantum / output_scale
        return 0.5
    if expected_unit in {"percent", "percentage_points"}:
        return 0.05
    if expected_unit == "USD millions":
        return 0.5 if ("amount" in text or "how much" in text) else 0.01
    if expected_unit == "USD billions":
        return 0.005
    return 0.01


def _precision_digits_from_answer(
    answer: str,
    claimed_value: float,
    reported_value: float | None,
) -> int | None:
    candidates = [claimed_value]
    if reported_value is not None:
        candidates.append(reported_value)
    best: int | None = None
    for raw in re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?", answer):
        token = raw.replace("$", "").replace(",", "")
        percent = token.endswith("%")
        if percent:
            token = token[:-1]
        try:
            value = float(token)
        except ValueError:
            continue
        decimals = len(token.split(".", 1)[1]) if "." in token else 0
        values = [value, value / 100.0] if percent else [value]
        if any(_close(v, candidate) for v in values for candidate in candidates):
            best = decimals if best is None else max(best, decimals)
    return best


def _numeric_tokens(answer: str) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for index, match in enumerate(re.finditer(r"\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?", answer or "")):
        raw = match.group(0).strip()
        token = raw
        negative = token.startswith("(") and token.endswith(")")
        token = token.strip("()").replace("$", "").replace(",", "")
        after = (answer or "")[match.end(): match.end() + 24].lower()
        has_percent = token.endswith("%") or bool(re.match(r"\s*(?:percent|percentage)\b", after))
        if token.endswith("%"):
            token = token[:-1]
        try:
            value = float(token)
        except ValueError:
            continue
        if negative:
            value = -value
        decimals = len(token.split(".", 1)[1]) if "." in token else 0
        tokens.append(
            {
                "raw": raw,
                "value": value,
                "decimals": decimals,
                "has_percent": has_percent,
                "index": index,
            }
        )
    return tokens


def _display_token_tolerance(
    token: dict[str, Any],
    spec: AnswerSpec,
    source: str,
    fallback: float,
) -> tuple[float, str]:
    tolerance = float(fallback)
    tolerance_source = source or "answer_precision"
    # A percent-written answer for a ratio-valued metric: the token's own decimals
    # are in percent space, so converting that display band to ratio space means
    # /100. Otherwise "-1.47%" gets the same 0.005 ratio tolerance as "-0.01",
    # which admits a full half percentage point rather than half of the printed
    # percent unit.
    if spec.expected_unit == "ratio" and token.get("has_percent"):
        tolerance /= 100.0
        tolerance_source = f"{tolerance_source}:percent_token_to_ratio"
    return tolerance, tolerance_source


def _normalize_claimed_value(value: float, unit: str, spec: AnswerSpec) -> tuple[float, float]:
    out = float(value)
    unit_text = unit.lower()
    if spec.expected_unit == "ratio" and ("%" in unit_text or "percent" in unit_text):
        return (out / 100.0 if abs(out) > 1.0 else out), 1.0
    if spec.expected_unit in {"percent", "percentage_points"}:
        if "ratio" in unit_text and abs(out) <= 1.0:
            return out * 100.0, 100.0
        return out, 1.0
    return out, 1.0


def _strip_percent_multiplier(formula: str) -> str:
    text = str(formula or "").strip()
    if not text:
        return text
    compact = re.sub(r"\s+", "", text)
    compact = _strip_outer_parens(compact)
    for suffix in ("*100.0", "*100"):
        if compact.endswith(suffix):
            return _strip_outer_parens(compact[:-len(suffix)])
    for prefix in ("100.0*", "100*"):
        if compact.startswith(prefix):
            return _strip_outer_parens(compact[len(prefix):])
    return text


def _strip_outer_parens(value: str) -> str:
    text = value
    while text.startswith("(") and text.endswith(")") and _outer_parens_wrap(text):
        text = text[1:-1]
    return text


def _outer_parens_wrap(text: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
    return depth == 0


def _format_scale(scale: float) -> str:
    return f"{scale:.12g}"


def _close(left: float, right: float) -> bool:
    denom = max(1.0, abs(left), abs(right))
    return abs(left - right) <= max(1e-8, denom * 1e-8)


def _answer_states_claim_components(
    certificate: VerificationCertificate,
    target: float,
    tokens: list[dict[str, Any]],
    spec: AnswerSpec,
    tolerance: float,
) -> dict[str, float] | None:
    """Accept a derived claim when the answer states the formula's inputs.

    Returns the input fact values when (a) the formula combines more than one
    input, (b) every formula-input fact value appears as a number in the answer,
    and (c) evaluating the formula over those values reproduces the claim. Single
    -variable formulas are excluded: their claim equals the fact and must appear
    directly.
    """
    variables = formula_variables(certificate.formula)
    if len(variables) < 2:
        return None
    values: dict[str, float] = {}
    for fact in certificate.facts:
        if fact.name in variables:
            values[fact.name] = float(fact.value)
    if set(values) != set(variables):
        return None

    # Every input must be visible in the answer (allowing percent display).
    for value in values.values():
        if not any(
            _close_within(token["value"], value, tolerance)
            or _close_within(token["value"], value * 100.0, tolerance)
            or _close_within(token["value"], value / 100.0, tolerance)
            for token in tokens
        ):
            return None

    try:
        computed = evaluate_formula(certificate.formula, values)
    except (FormulaError, ZeroDivisionError, ValueError):
        return None
    if abs(float(computed) - target) > tolerance:
        return None
    return values


def _close_within(left: float, right: float, tolerance: float) -> bool:
    denom = max(1.0, abs(left), abs(right))
    return abs(left - right) <= max(tolerance, denom * 1e-3)
