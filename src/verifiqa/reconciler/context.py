from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from verifiqa.policy.registry import PolicyRegistry
from verifiqa.types import FinanceBenchExample, VerificationIR


@dataclass(frozen=True)
class GroundedSourceFact:
    source_id: str
    fact_name: str
    concept: str
    period: str
    unit: str
    source: str
    value: float
    grounded: bool = True

    def prompt_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "fact_name": self.fact_name,
            "concept": self.concept,
            "period": self.period,
            "unit": self.unit,
            "source": self.source,
            "grounded": self.grounded,
        }


@dataclass(frozen=True)
class ReconcilerContext:
    prompt_payload: dict[str, Any]
    source_facts: list[GroundedSourceFact]
    input_hash: str


def build_reconciler_context(
    example: FinanceBenchExample,
    ir: VerificationIR,
    registry: PolicyRegistry,
) -> ReconcilerContext:
    source_facts = _source_facts_from_ir(ir)
    filing_concepts = _filing_concepts(ir)
    prompt_payload = {
        "task": "value_blind_reconciler_spec",
        "question": example.question,
        "company": example.company,
        "doc_name": example.doc_name,
        "metric_from_current_ir": ir.metric,
        "claim": {
            "value": "MASKED",
            "unit": ir.claim_unit,
            "period": ir.period,
            "tolerance": "MASKED",
        },
        "source_facts_no_values": [fact.prompt_record() for fact in source_facts],
        "filing_concepts_no_values": filing_concepts,
        "formula_registry": _registry_payload(registry),
        "output_schema": {
            "fact_bindings": {
                "type": "object",
                "description": "Map each selected registry formula variable to one source_id, concept, or fact_name from source_facts_no_values.",
            },
            "formula": "policy_id from formula_registry, or identity, or none",
            "claim_scale": "numeric multiplier applied to the masked claim value; use 1 when already normalized",
            "claim_unit": "claimed answer unit after scaling, such as ratio, percent, USD, USD millions",
        },
    }
    canonical = json.dumps(prompt_payload, sort_keys=True, separators=(",", ":"))
    return ReconcilerContext(
        prompt_payload=prompt_payload,
        source_facts=source_facts,
        input_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _source_facts_from_ir(ir: VerificationIR) -> list[GroundedSourceFact]:
    xbrl = ir.xbrl_calculations or {}
    bindings = xbrl.get("bindings") or []
    source = str(xbrl.get("source") or xbrl.get("grounding_status") or "grounding")
    facts: list[GroundedSourceFact] = []
    for binding in bindings:
        value = binding.get("xbrl_value")
        if value is None:
            continue
        try:
            multiplier = float(binding.get("binding_multiplier", 1.0))
            grounded_value = float(value) * multiplier
        except (TypeError, ValueError):
            continue
        concept = str(binding.get("concept") or "")
        context_id = str(binding.get("context_id") or "")
        fact_name = str(binding.get("fact_name") or "")
        source_id = str(binding.get("canonical_key") or "")
        if not source_id:
            source_id = f"{concept}|{context_id}" if concept or context_id else fact_name
        facts.append(
            GroundedSourceFact(
                source_id=source_id,
                fact_name=fact_name,
                concept=concept,
                period=_period_label(binding.get("context_period")) or str(binding.get("period") or ""),
                unit=_unit_label(binding),
                source=source,
                value=grounded_value,
                grounded=True,
            )
        )

    if facts:
        return _dedupe_source_facts(facts)

    # Last-resort catalog for shadow diagnostics only. These values came from the
    # current IR, not an independent source binding, so the compiler refuses to
    # verify with them. They still help explain why the reconciler could not run.
    return [
        GroundedSourceFact(
            source_id=f"ir_fact:{name}",
            fact_name=name,
            concept=fact.row_label or name,
            period=fact.period,
            unit=fact.unit,
            source="current_ir_unverified",
            value=float(fact.value),
            grounded=False,
        )
        for name, fact in sorted(ir.facts.items())
    ]


def _filing_concepts(ir: VerificationIR) -> list[dict[str, Any]]:
    xbrl = ir.xbrl_calculations or {}
    concepts: dict[str, dict[str, Any]] = {}
    for binding in xbrl.get("bindings") or []:
        concept = str(binding.get("concept") or "")
        if concept:
            concepts.setdefault(concept, {"concept": concept, "source": "binding"})
    for constraint in xbrl.get("constraints") or []:
        parent = str(constraint.get("parent_concept") or "")
        if parent:
            concepts.setdefault(parent, {"concept": parent, "source": "calculation_linkbase"})
        for child in constraint.get("children") or []:
            child_concept = str(child.get("concept") or "")
            if child_concept:
                concepts.setdefault(child_concept, {"concept": child_concept, "source": "calculation_linkbase"})
    return sorted(concepts.values(), key=lambda item: item["concept"])


def _registry_payload(registry: PolicyRegistry) -> list[dict[str, Any]]:
    rows = []
    for policy in registry.policies:
        rows.append(
            {
                "policy_id": policy.policy_id,
                "metric": policy.metric,
                "metric_aliases": list(policy.metric_aliases),
                "formula_templates": list(policy.formula_templates),
                "roles": policy.roles,
                "source_type": policy.source_type,
                "source_refs": policy.source_refs,
            }
        )
    return rows


def _dedupe_source_facts(facts: list[GroundedSourceFact]) -> list[GroundedSourceFact]:
    seen: set[str] = set()
    result: list[GroundedSourceFact] = []
    for fact in facts:
        key = fact.source_id
        if key in seen:
            suffix = 2
            while f"{key}#{suffix}" in seen:
                suffix += 1
            fact = GroundedSourceFact(
                source_id=f"{key}#{suffix}",
                fact_name=fact.fact_name,
                concept=fact.concept,
                period=fact.period,
                unit=fact.unit,
                source=fact.source,
                value=fact.value,
                grounded=fact.grounded,
            )
            key = fact.source_id
        seen.add(key)
        result.append(fact)
    return result


def _unit_label(binding: dict[str, Any]) -> str:
    measures = binding.get("unit_measures")
    if isinstance(measures, list) and measures:
        return " ".join(str(measure) for measure in measures if str(measure).strip())
    return str(binding.get("unit_ref") or binding.get("unit") or "")


def _period_label(period: Any) -> str:
    if not isinstance(period, dict):
        return ""
    instant = str(period.get("instant") or "")
    if instant:
        return instant
    start = str(period.get("start_date") or "")
    end = str(period.get("end_date") or "")
    if start or end:
        return f"{start}/{end}".strip("/")
    return ""
