"""Carry independently selected source identities through numeric verification.

References are evidence, not authority: every use reconstructs the question-only
plan and checks the selected records. Candidate values never select a record.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from verifiqa.agent.types import ClaimSpec


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _plan_payload(spec):
    payload = asdict(spec)
    for key in ("claimed_value", "claimed_unit", "source_selection_id"):
        payload.pop(key, None)
    for role in payload["roles"]:
        role.pop("source_ref", None)
    return payload


def _reconstruct(question, evidence_text, source_pdf_path, source_xbrl_bundle):
    from verifiqa.agent.source_question_planner import plan_from_source
    planned = plan_from_source(question, evidence_text, source_pdf_path=source_pdf_path)
    if planned is None and source_xbrl_bundle is not None:
        from verifiqa.agent.source_xbrl_schema_planner import plan_from_source_xbrl
        planned = plan_from_source_xbrl(question, evidence_text, source_xbrl_bundle)
    return planned


def attach_source_selection(spec: ClaimSpec, question: str, evidence_text: str, *,
                            source_pdf_path: str = "", source_xbrl_bundle=None) -> ClaimSpec:
    """Return a copied plan with exact source references, or reject it.

    This accepts only a complete independently reconstructed deterministic plan;
    neither a supplied plan nor a cached reference can authorize itself.
    """
    planned = _reconstruct(question, evidence_text, source_pdf_path, source_xbrl_bundle)
    if planned is None or _plan_payload(spec) != _plan_payload(planned):
        raise ValueError("source_selection_plan_mismatch")
    selected = deepcopy(spec)
    provenance = {
        "evidence_sha256": hashlib.sha256(evidence_text.encode("utf-8")).hexdigest(),
        "document_sha256": planned.source_document_sha256,
    }
    if planned.planning_source == "deterministic_source_table":
        from verifiqa.agent.source_question_planner import source_plan_facts
        facts = source_plan_facts(planned, question, evidence_text, source_pdf_path=source_pdf_path)
        if facts is None or set(facts) != {role.name for role in planned.roles}:
            raise ValueError("source_selection_incomplete")
        for role in selected.roles:
            fact = facts[role.name]
            ref = {**provenance, "kind": "source_cell", "table_id": fact.source_table_id,
                   "row_label": fact.row_label, "column": fact.source_column,
                   "section": fact.source_section, "period": fact.period, "unit": fact.unit,
                   "quote_sha256": _digest(fact.source_quote)}
            role.source_ref = {**ref, "record_id": _digest(ref)}
    elif planned.planning_source == "deterministic_xbrl_schema" and source_xbrl_bundle is not None:
        instance_path = source_xbrl_bundle.instance_path
        filing = {"doc_name": source_xbrl_bundle.doc_name, "cik": source_xbrl_bundle.cik,
                  "instance_sha256": hashlib.sha256(Path(instance_path).read_bytes()).hexdigest()
                  if instance_path is not None else ""}
        for role in selected.roles:
            records = [fact for fact in source_xbrl_bundle.facts_by_local_context.values()
                       if fact.concept == role.concept and fact.context_id == role.xbrl_context_id
                       and fact.value is not None]
            context = source_xbrl_bundle.contexts.get(role.xbrl_context_id)
            if len(records) != 1 or context is None:
                raise ValueError("source_selection_ambiguous_xbrl_record")
            record = records[0]
            ref = {**provenance, "filing": filing, "kind": "xbrl_fact", "concept": role.concept,
                   "context_id": role.xbrl_context_id, "context": asdict(context),
                   "unit_ref": record.unit_ref,
                   "unit": list(source_xbrl_bundle.units.get(record.unit_ref, ())),
                   "schema_sha256": planned.source_schema_sha256,
                   "fact_sha256": _digest(asdict(record))}
            role.source_ref = {**ref, "record_id": _digest(ref)}
    else:
        raise ValueError("source_selection_unsupported_plan")
    selected.source_selection_id = _digest({
        "question": question, "plan": _plan_payload(selected),
        "references": {role.name: role.source_ref for role in selected.roles},
    })
    return selected


def validate_source_selection(spec, question, evidence_text, *,
                              source_pdf_path="", source_xbrl_bundle=None):
    try:
        expected = attach_source_selection(spec, question, evidence_text,
            source_pdf_path=source_pdf_path, source_xbrl_bundle=source_xbrl_bundle)
        return (bool(spec.source_selection_id)
                and spec.source_selection_id == expected.source_selection_id
                and [role.source_ref for role in spec.roles] == [role.source_ref for role in expected.roles])
    except (ValueError, TypeError, OSError, KeyError):
        return False


def bind_selected_facts(spec, facts, question, evidence_text, *,
                        source_pdf_path="", source_xbrl_bundle=None):
    """Check selected identities and amounts before unit normalization.

    A label re-resolution or extraction cannot silently replace a selected cell.
    Existing formula/type/scope/XBRL checks still run after this additional check.
    """
    if not validate_source_selection(spec, question, evidence_text,
            source_pdf_path=source_pdf_path, source_xbrl_bundle=source_xbrl_bundle):
        return None
    if set(facts) != {role.name for role in spec.roles} or any(f is None for f in facts.values()):
        return None
    if spec.planning_source == "deterministic_source_table":
        from verifiqa.agent.source_question_planner import source_plan_facts
        expected = source_plan_facts(spec, question, evidence_text, source_pdf_path=source_pdf_path)
        fields = ("unit", "period", "row_label", "source_quote", "source_section",
                  "source_column", "source_table_id", "source_value", "source_unit", "fact_type")
        if expected is None:
            return None
        for name, fact in facts.items():
            source = expected[name]
            if (any(getattr(fact, key) != getattr(source, key) for key in fields)
                    or not Decimal(str(fact.value)).is_finite()
                    or Decimal(str(fact.value)) != Decimal(str(source.value))
                    or not _table_concept_matches(fact, source, spec, source_xbrl_bundle)):
                return None
    else:
        from verifiqa.agent.agent import _build_ir
        from verifiqa.agent.source_xbrl_schema_planner import source_xbrl_facts_match
        if not source_xbrl_facts_match(_build_ir(spec, facts, 0.0, 0.0),
                                       question, evidence_text, source_xbrl_bundle):
            return None
    bound = {}
    for role in spec.roles:
        fact = facts[role.name]
        if fact.source_ref and fact.source_ref != role.source_ref:
            return None
        bound[role.name] = replace(fact, source_ref=deepcopy(role.source_ref))
    return bound


def _table_concept_matches(fact, source, spec, bundle):
    """Permit only independently witnessed XBRL enrichment of a source cell."""
    if (fact.concept, fact.xbrl_context_id) == (source.concept, source.xbrl_context_id):
        return True
    if bundle is None or not fact.concept:
        return False
    from verifiqa.agent.agent import _source_binding_column
    from verifiqa.types import VerificationFact, VerificationIR
    from verifiqa.verification.xbrl_linkbase import (
        _bind_fact_local, _candidate_concept_locals, _exact_numeric_label_concepts,
    )
    probe = VerificationFact(source.name, source.value, source.unit,
        fact_type="source_table_numeric", column=_source_binding_column(source.source_column),
        period=source.period, row_label=source.row_label, source_quote=source.source_quote,
        source_section=source.source_section, source_column=source.source_column,
        source_table_id=source.source_table_id)
    labels = _exact_numeric_label_concepts([source.row_label], bundle)
    allowed = labels or {record.concept for record in bundle.facts_by_local_context.values()
                         if record.local_name in _candidate_concept_locals(probe)}
    ir = VerificationIR(spec.metric, spec.formula, {source.name: probe},
                        0.0, spec.claim_unit, spec.tolerance, period=spec.period)
    matches = [binding for concept in sorted(allowed)
               if (binding := _bind_fact_local(replace(probe, concept=concept), ir, bundle,
                                                allow_label_fallback=False)) is not None]
    return (len(matches) == 1 and matches[0].get("concept") == fact.concept
            and (not fact.xbrl_context_id or fact.xbrl_context_id == matches[0].get("context_id")))
