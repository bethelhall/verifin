"""Screen references without consulting generated answers or verifier decisions.

An XBRLFiling calculation question is eligible only when its supplied weighted
operands and reported reference agree at the precision requested by the question.
This is an evaluation exclusion, not a replacement gold or a corrected verdict.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import re
from typing import Any, Mapping


POLICY = {
    "id": "xbrl_reference_consistency_v1",
    "dataset": "xbrlfiling",
    "scope": "All XBRLFiling questions, independently of candidate answers and decisions",
    "unit": "USD millions",
    "precision": "nearest 1 million",
    "rounding": "Decimal ROUND_HALF_UP",
    "eligibility": "round(sum(operand.weight * operand.value_millions)) == round(ground_truth.value)",
    "treatment": "Exclude mismatches from correctness metrics for every method; preserve saved status counts",
    "reference_replacement": False,
}


def _number(value: Any, *, context: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{context}: expected a finite numeric value")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{context}: expected a finite numeric value") from error
    if not number.is_finite():
        raise ValueError(f"{context}: expected a finite numeric value")
    return number


def screen_reference_consistency(
    datasets: Mapping[str, Mapping[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return mismatches from canonical references; other datasets are unchanged.

    Missing or unsupported proof schemas fail closed instead of quietly passing
    the row. Decimal strings in the result make the screen independently auditable.
    """
    exclusions: dict[str, dict[str, dict[str, Any]]] = {dataset: {} for dataset in datasets}
    for rid, row in datasets.get("xbrlfiling", {}).items():
        context = f"xbrlfiling/{rid}"
        question = row.get("question")
        if not isinstance(question, str) or not re.search(r"\bin\s+USD\s+millions\b", question, re.I):
            raise ValueError(f"{context}: reference screen requires a question in USD millions")
        if not re.search(r"\bround\s+to\s+the\s+nearest\s+million\b", question, re.I):
            raise ValueError(f"{context}: reference screen requires nearest-million question precision")
        proof = row.get("ground_truth")
        if not isinstance(proof, dict) or proof.get("unit") != POLICY["unit"]:
            raise ValueError(f"{context}: reference screen requires a ground_truth proof in USD millions")
        operands = proof.get("operands")
        if not isinstance(operands, list) or not operands:
            raise ValueError(f"{context}: reference screen requires nonempty ground_truth.operands")
        reference = _number(proof.get("value"), context=f"{context}/ground_truth.value")
        parsed = []
        for position, operand in enumerate(operands):
            if not isinstance(operand, dict) or not isinstance(operand.get("concept"), str) or not operand["concept"]:
                raise ValueError(f"{context}/operands[{position}]: expected a named operand")
            weight = _number(operand.get("weight"), context=f"{context}/operands[{position}].weight")
            value = _number(operand.get("value_millions"), context=f"{context}/operands[{position}].value_millions")
            parsed.append((operand["concept"], weight, value))
        # Exact decimal operations, including cancellation, must not depend on
        # binary floating point or a caller's decimal context.
        digit_budget = sum(len(number.as_tuple().digits) + abs(number.as_tuple().exponent)
                           for number in [reference, *(v for _, w, x in parsed for v in (w, x))])
        with localcontext() as decimal_context:
            decimal_context.prec = max(50, digit_budget + len(parsed) + 10)
            calculated = sum((weight * value for _, weight, value in parsed), Decimal(0))
            rounded_calculated = calculated.quantize(Decimal(1), rounding=ROUND_HALF_UP)
            rounded_reference = reference.quantize(Decimal(1), rounding=ROUND_HALF_UP)
            difference = calculated - reference
        if rounded_calculated != rounded_reference:
            exclusions["xbrlfiling"][rid] = {
                "id": rid, "policy": POLICY["id"], "unit": POLICY["unit"],
                "reason": "supplied_operand_calculation_disagrees_with_reported_reference_at_question_precision",
                "calculated_value": str(calculated), "reference_value": str(reference),
                "rounded_calculated_value": str(rounded_calculated), "rounded_reference_value": str(rounded_reference),
                "calculated_minus_reference": str(difference),
                "operands": [dict(concept=concept, weight=str(weight), value_millions=str(value))
                             for concept, weight, value in parsed],
            }
    return exclusions
