"""Evaluation-only quantities for the candidate that the verifier decided on.

The stored numeric claim is authoritative. Numbers mentioned in its explanation
are never alternative answers. Unit conversions use explicit gold units or a
reviewed conversion, and leave the candidate value and scoring tolerance intact.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Any, Mapping


_NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TOKEN = re.compile(rf"(?<![\w.])\(?\s*[-+]?\s*[$€£]?\s*{_NUMBER}\s*\)?")
_SCALAR = re.compile(
    rf"^\s*(?P<open>\()?\s*(?P<sign>[-+]?)\s*"
    rf"(?P<prefix>USD|US\$|NT\$|TWD|EUR|GBP|[$€£])?\s*"
    rf"(?P<number>{_NUMBER})\s*(?P<close>\))?\s*(?P<unit>[^\d]*)\s*$",
    re.IGNORECASE,
)
_SCALES = {
    "thousand": 1_000.0, "thousands": 1_000.0, "k": 1_000.0,
    "million": 1_000_000.0, "millions": 1_000_000.0,
    "mn": 1_000_000.0, "m": 1_000_000.0,
    "billion": 1_000_000_000.0, "billions": 1_000_000_000.0,
    "bn": 1_000_000_000.0, "b": 1_000_000_000.0,
    "ones": 1.0,
}
_SCALE_LABELS = {1.0: "", 1_000.0: "thousands", 1_000_000.0: "millions",
                 1_000_000_000.0: "billions"}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def unit_info(text: Any, *, default_unit: str = "") -> dict[str, Any]:
    """Describe an explicit unit, optionally resolving a scale-only label.

    ``count`` and ``shares`` denote the same count dimension. Currency amounts,
    per-share amounts, and fractions remain distinct dimensions. This parser
    never examines numeric magnitudes.
    """
    raw = str(text or "").strip()
    norm = " ".join(raw.lower().replace("_", " ").split())
    norm = norm.rstrip(".")
    if not norm:
        return {"unit": "", "family": None, "scale": 1.0, "currency": ""}
    scale_matches = [value for word, value in _SCALES.items()
                     if re.search(rf"\b{re.escape(word)}\b", norm)]
    scale = scale_matches[0] if scale_matches else 1.0
    if len(set(scale_matches)) > 1:
        return {"unit": raw, "family": "unknown", "scale": 1.0, "currency": ""}

    currency = ""
    if re.search(r"nt\$|\btwd\b|taiwan", norm):
        currency = "TWD"
    elif re.search(r"€|\beur\b|\beuros?\b", norm):
        currency = "EUR"
    elif re.search(r"£|\bgbp\b|\bpounds?\b", norm):
        currency = "GBP"
    elif re.search(r"\$|\busd\b|\bdollars?\b", norm):
        currency = "USD"
    if currency:
        denominator = ""
        if re.search(r"(?:per|/)\s*shares?\b", norm):
            denominator = "share"
        elif re.search(r"(?:per|/)\s*(?:troy\s+)?ounces?\b", norm):
            denominator = "troy ounce"
        label = " ".join(part for part in (currency, _SCALE_LABELS[scale]) if part)
        if denominator:
            label += f" per {denominator}"
        return {"unit": label, "family": f"money_per_{denominator}" if denominator else "money",
                "scale": scale, "currency": currency}
    if re.search(r"percentage\s+points?|\bpp\b", norm):
        return {"unit": "percentage points", "family": "percentage_points", "scale": 1.0, "currency": ""}
    if "%" in norm or re.search(r"\bpercent(?:age)?\b", norm):
        return {"unit": "percent", "family": "fraction", "scale": 0.01, "currency": ""}
    if norm in {"ratio", "proportion", "fraction", "dimensionless", "unitless", "pure"}:
        return {"unit": "ratio", "family": "fraction", "scale": 1.0, "currency": ""}
    if re.search(r"\b(?:shares?|count|counts|number)\b", norm):
        label = " ".join(part for part in ("count", _SCALE_LABELS[scale]) if part)
        return {"unit": label, "family": "count", "scale": scale, "currency": ""}
    if norm in _SCALES and default_unit:
        default = unit_info(default_unit)
        if default["family"] in {"money", "count"}:
            label = " ".join(part for part in
                             (default["currency"] or "count", _SCALE_LABELS[scale]) if part)
            return {**default, "unit": label, "scale": scale}
    if re.search(r"\byears?\b", norm):
        return {"unit": "years", "family": "years", "scale": 1.0, "currency": ""}
    return {"unit": raw, "family": "unknown", "scale": 1.0, "currency": ""}


def _scalar_quantity(text: Any, *, default_unit: str = "") -> dict[str, Any] | None:
    if isinstance(text, bool):
        return None
    raw = str("" if text is None else text).strip()
    if raw.startswith("(") and raw.endswith(")"):
        inner = _scalar_quantity(raw[1:-1], default_unit=default_unit)
        return {**inner, "value": -abs(inner["value"])} if inner else None
    match = _SCALAR.fullmatch(raw)
    if not match or bool(match["open"]) != bool(match["close"]):
        return None
    unit_text = " ".join(part for part in (match["prefix"], match["unit"].strip()) if part)
    info = unit_info(unit_text, default_unit=default_unit)
    if unit_text and info["family"] == "unknown":
        return None
    try:
        value = float((match["sign"] or "") + match["number"].replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if match["open"]:
        value = -abs(value)
    return {"value": value, **info}


def _claim_unit(row: Mapping[str, Any], scalar: Mapping[str, Any] | None,
                *, scalar_value_fallback: bool = False) -> dict[str, Any]:
    diagnostics = row.get("diagnostics") or {}
    spec = diagnostics.get("claim_spec") or {}
    claim = row.get("claim") or {}
    ir = row.get("verification_ir") or {}
    # These fields state the numeric representation of the stored candidate,
    # rather than the display/requested units in the original decomposition.
    for field in ("claimed_value_unit", "claimed_unit", "claim_unit"):
        raw = row.get(field)
        if raw:
            return {**unit_info(raw), "unit_source": field, "declared_unit": str(raw)}
    if scalar_value_fallback and scalar and scalar["unit"]:
        return {**{key: scalar[key] for key in ("unit", "family", "scale", "currency")},
                "unit_source": "legacy_scalar_answer", "declared_unit": scalar["unit"]}
    declared = None
    for raw, source in ((spec.get("claimed_unit"), "diagnostics.claim_spec.claimed_unit"),
                        (spec.get("claim_unit"), "diagnostics.claim_spec.claim_unit"),
                        (claim.get("unit"), "claim.unit"),
                        (ir.get("claim_unit"), "verification_ir.claim_unit")):
        if raw:
            declared = {**unit_info(raw), "unit_source": source, "declared_unit": str(raw)}
            break
    # VerificationAgent._prepare_inputs rescales monetary and count claims into
    # homogeneous fact units before saving claimed_value. The original ClaimSpec
    # unit can remain the PDF's display scale. Fractions and per-share amounts do
    # not inherit the units of their monetary operands.
    facts = row.get("facts") or {}
    fact_rows = list(facts.values()) if isinstance(facts, Mapping) else facts
    infos = [unit_info(fact.get("unit")) for fact in fact_rows if isinstance(fact, Mapping)]
    if infos and infos[0]["family"] in {"money", "count"}:
        first = infos[0]
        compatible_declared = declared is None or (
            (first["family"], first["currency"]) == (declared["family"], declared["currency"])
        )
        if all(info == first for info in infos) and compatible_declared:
            return {**first, "unit_source": "normalized_fact_unit",
                    "declared_unit": declared["declared_unit"] if declared else ""}
    if declared:
        return declared
    if scalar and scalar["unit"]:
        return {**{key: scalar[key] for key in ("unit", "family", "scale", "currency")},
                "unit_source": "legacy_scalar_answer", "declared_unit": scalar["unit"]}
    # Explicit requested units are usable for old scalar-answer rows. This does
    # not infer an answer from the question or rescale a number to make it match.
    question = str(row.get("question") or "")
    for pattern in (
        r"(?:USD|US\$|NT\$|TWD|EUR|GBP|dollars?)\s+(?:thousands?|millions?|billions?)(?:\s+per\s+share)?",
        r"(?:thousands?|millions?|billions?)\s+of\s+shares",
        r"(?:thousands?|millions?|billions?)\s+shares",
        r"percentage\s+points?|\bpercent(?:age)?\b|\bratio\b",
    ):
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return {**unit_info(match[0]), "unit_source": "explicit_question_unit",
                    "declared_unit": match[0]}
    return {**unit_info(""), "unit_source": "missing", "declared_unit": ""}


def claim_quantity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the stored final numeric claim or an old scalar-only answer.

    A present malformed, null, nonfinite, or boolean ``claimed_value`` is invalid
    and cannot be rescued by a number elsewhere in an answer. Older pipeline
    rows can carry the value in ``claim`` or ``verification_ir``.
    """
    scalar = None
    source = "missing"
    value = None
    containers = ((row, "claimed_value", "claimed_value"),
                  (row.get("claim") or {}, "claimed_value", "claim.claimed_value"),
                  (row.get("verification_ir") or {}, "claimed_value", "verification_ir.claimed_value"))
    for container, key, label in containers:
        if key in container:
            source = label
            value = _finite_number(container[key])
            break
    else:
        answer = row.get("claim_input_answer")
        if answer is None:
            answer = row.get("answer")
        scalar = _scalar_quantity(answer)
        if scalar:
            source = "legacy_scalar_answer"
            value = scalar["value"]
    # Scalar suffixes can supply a unit for the stored claim, but never replace
    # its numeric value. A multi-number explanation cannot supply this fallback.
    if scalar is None:
        scalar = _scalar_quantity(row.get("claim_input_answer", row.get("answer")))
    return {"value": value,
            **_claim_unit(row, scalar, scalar_value_fallback=source == "legacy_scalar_answer"),
            "source": source,
            "valid": value is not None}


def _gold_values(text: Any) -> list[float]:
    scalar = _scalar_quantity(text)
    if scalar:
        return [scalar["value"]]
    values = []
    for match in _TOKEN.finditer(str("" if text is None else text)):
        raw = match[0].strip()
        negative = raw.startswith("(") and raw.endswith(")")
        token = re.sub(r"[$€£(),\s]", "", raw)
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(-abs(value) if negative else value)
    return values


def normalize_gold_targets(
    row: Mapping[str, Any],
    gold_text: Any,
    *,
    gold_scale: str = "",
    reviewed_conversion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Express explicit gold quantities in the stored candidate's units.

    A reviewed record has ``factor`` and ``comparison_unit`` (or ``claim_unit``)
    and may bind itself to an exact ``raw_answer``. Bare benchmark numbers are
    assumed to use the candidate unit for compatibility, with that assumption
    reported. An optional source-unit ``absolute_tolerance`` and
    ``tolerance_unit`` convert with the target, preserving the same physical
    tolerance across saved candidate scales. Incompatible explicit dimensions
    have no allowed numeric target.
    """
    claim = claim_quantity(row)
    # A reviewed source can repair a missing denominator in an old saved unit
    # label. This changes only the quantity type, never its amount or scale.
    # The correction applies equally to every model/candidate for that source.
    if reviewed_conversion and reviewed_conversion.get("claim_unit_qualifier"):
        qualifier = reviewed_conversion["claim_unit_qualifier"]
        if qualifier != "per share":
            raise ValueError("Unsupported reviewed claim unit qualifier")
        target_info = unit_info(reviewed_conversion.get("comparison_unit"))
        if target_info["family"] != "money_per_share":
            raise ValueError("Reviewed per-share correction requires a per-share target")
        if claim["family"] == "money" and claim["currency"] == target_info["currency"]:
            claim = {**claim, **unit_info(f"{claim['unit']} per share"),
                     "unit_source": f"{claim['unit_source']};reviewed_missing_denominator"}
    targets = _gold_values(gold_text)
    raw_targets = list(targets)
    scalar = _scalar_quantity(gold_text, default_unit=claim["unit"])
    metadata_unit = row.get("gold_unit") or ""
    explicit_unit = gold_scale or metadata_unit
    # A scale-only metadata label inherits an explicit gold currency first. It
    # must never reinterpret a euro amount as USD because the candidate is USD.
    default_gold_unit = metadata_unit or (scalar["unit"] if scalar else "") or claim["unit"]
    gold = unit_info(explicit_unit, default_unit=default_gold_unit)
    if not explicit_unit and scalar:
        gold = {key: scalar[key] for key in ("unit", "family", "scale", "currency")}
    reason = "same_unit"
    if reviewed_conversion is not None:
        expected = reviewed_conversion.get("raw_answer")
        if expected is not None and str(expected).strip() != str(gold_text).strip():
            raise ValueError("Reviewed conversion does not match the raw gold answer")
        factor = _finite_number(reviewed_conversion.get("factor"))
        raw_unit = (reviewed_conversion.get("comparison_unit")
                    or reviewed_conversion.get("claim_unit") or reviewed_conversion.get("unit"))
        if factor is None or factor <= 0 or not raw_unit:
            raise ValueError("Reviewed conversion requires a positive finite factor and comparison unit")
        targets = [float(Decimal(str(value)) * Decimal(str(factor))) for value in targets]
        gold = unit_info(raw_unit, default_unit=claim["unit"])
        reason = "reviewed_conversion"
    compatible = True
    if not gold["unit"]:
        gold = {key: claim[key] for key in ("unit", "family", "scale", "currency")}
        reason = "bare_gold_assumed_claim_unit"
    elif not claim["unit"]:
        reason = "missing_claim_unit_assumed_gold_unit"
    elif (gold["family"], gold["currency"]) != (claim["family"], claim["currency"]):
        compatible = False
        reason = "incompatible_units"
    elif gold["family"] == "unknown" and gold["unit"] != claim["unit"]:
        compatible = False
        reason = "incompatible_units"
    elif gold["scale"] != claim["scale"]:
        targets = [float(Decimal(str(value)) * Decimal(str(gold["scale"]))
                         / Decimal(str(claim["scale"]))) for value in targets]
        reason = f"{reason};explicit_unit_conversion" if reviewed_conversion else "explicit_unit_conversion"
    absolute_tolerance = None
    if reviewed_conversion and ("absolute_tolerance" in reviewed_conversion
                                or "tolerance_unit" in reviewed_conversion):
        raw_tolerance = _finite_number(reviewed_conversion.get("absolute_tolerance"))
        tolerance_info = unit_info(reviewed_conversion.get("tolerance_unit"))
        if raw_tolerance is None or raw_tolerance < 0 or not tolerance_info["unit"]:
            raise ValueError("Reviewed tolerance requires a finite nonnegative amount and explicit unit")
        if ((tolerance_info["family"], tolerance_info["currency"])
                != (gold["family"], gold["currency"])):
            raise ValueError("Reviewed tolerance must have the target quantity type and currency")
        if compatible:
            absolute_tolerance = float(Decimal(str(raw_tolerance))
                                       * Decimal(str(tolerance_info["scale"]))
                                       / Decimal(str(claim["scale"])))
    return {"candidate_value": claim["value"], "claim_unit": claim["unit"],
            "claim_unit_source": claim["unit_source"], "declared_claim_unit": claim["declared_unit"],
            "claim_source": claim["source"], "targets": targets if compatible else [],
            "raw_targets": raw_targets, "gold_unit": gold["unit"],
            "units_compatible": compatible, "conversion_reason": reason,
            "absolute_tolerance_in_claim_units": absolute_tolerance}


def numeric_claim_comparison(
    row: Mapping[str, Any], gold_text: Any, *, gold_scale: str = "",
    reviewed_conversion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Public comparison input shared by CLI and notebook evaluation."""
    return normalize_gold_targets(row, gold_text, gold_scale=gold_scale,
                                  reviewed_conversion=reviewed_conversion)


def candidate_matches_gold(
    row: Mapping[str, Any], gold_text: Any, *, gold_scale: str = "",
    reviewed_conversion: Mapping[str, Any] | None = None,
    absolute_floor: float = 1.0, relative_tolerance: float = 0.01,
    compare_magnitudes: bool = True,
) -> bool | None:
    """Match the decided candidate with the published numeric policy.

    Missing numeric candidates or gold targets are ungradeable. Incompatible
    explicit quantities are incorrect, even when their numeric magnitudes match.
    Runtime SMT tolerances are independent of this evaluation-only comparison.
    """
    absolute = _finite_number(absolute_floor)
    relative = _finite_number(relative_tolerance)
    if absolute is None or relative is None or absolute < 0 or relative < 0:
        raise ValueError("Scoring tolerances must be finite nonnegative numbers")
    comparison = numeric_claim_comparison(
        row, gold_text, gold_scale=gold_scale, reviewed_conversion=reviewed_conversion,
    )
    candidate = comparison["candidate_value"]
    if candidate is None or not comparison["raw_targets"]:
        return None
    if not comparison["units_compatible"]:
        return False
    if comparison["absolute_tolerance_in_claim_units"] is not None:
        absolute = comparison["absolute_tolerance_in_claim_units"]
    for target in comparison["targets"]:
        value = candidate
        if compare_magnitudes:
            value, target = abs(value), abs(target)
        if abs(value - target) <= max(absolute, relative * abs(target)):
            return True
    return False
