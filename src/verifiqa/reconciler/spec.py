from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


class ReconcilerSpecError(ValueError):
    pass


@dataclass(frozen=True)
class ReconcilerSpec:
    fact_bindings: dict[str, str]
    formula: str
    claim_scale: float = 1.0
    claim_unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ALLOWED_KEYS = {"fact_bindings", "formula", "claim_scale", "claim_unit"}


def parse_reconciler_spec(text: str) -> ReconcilerSpec:
    payload = _extract_json_object(text)
    extra = sorted(set(payload) - _ALLOWED_KEYS)
    if extra:
        raise ReconcilerSpecError(f"unexpected_spec_keys:{','.join(extra)}")

    bindings = payload.get("fact_bindings")
    if not isinstance(bindings, dict):
        raise ReconcilerSpecError("fact_bindings_must_be_object")
    clean_bindings: dict[str, str] = {}
    for key, value in bindings.items():
        if not isinstance(key, str) or not key.strip():
            raise ReconcilerSpecError("fact_binding_key_must_be_nonempty_string")
        if not isinstance(value, str) or not value.strip():
            raise ReconcilerSpecError(f"fact_binding_value_must_be_nonempty_string:{key}")
        clean_bindings[key.strip()] = value.strip()

    formula = payload.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        raise ReconcilerSpecError("formula_must_be_nonempty_string")

    claim_scale = payload.get("claim_scale", 1)
    try:
        claim_scale_float = float(claim_scale)
    except (TypeError, ValueError) as exc:
        raise ReconcilerSpecError("claim_scale_must_be_numeric_multiplier") from exc
    if not claim_scale_float:
        raise ReconcilerSpecError("claim_scale_must_be_nonzero")

    claim_unit = payload.get("claim_unit", "")
    if claim_unit is None:
        claim_unit = ""
    if not isinstance(claim_unit, str):
        raise ReconcilerSpecError("claim_unit_must_be_string")

    return ReconcilerSpec(
        fact_bindings=clean_bindings,
        formula=formula.strip(),
        claim_scale=claim_scale_float,
        claim_unit=claim_unit.strip(),
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ReconcilerSpecError("no_json_object")
