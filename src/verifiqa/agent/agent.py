"""Agentic verification pipeline.

Flow (no repair loops):
  1. Decompose question → ClaimSpec  (LLM-1, no proposed answer)
  2. Parse claimed value from answer (LLM-2, fixed ClaimSpec)
  3. Extract facts in parallel       (LLM-3 per role, full oracle evidence text)
  4. Grounding check                 (deterministic: source_quote in evidence_text)
  5. Build VerificationIR            (deterministic)
  6. Generate SMT                    (LLM, existing SmtGenerator)
  7. Z3 single pass
  8. If UNSAT: re-extract facts with computed-mismatch feedback (once by default)
"""
from __future__ import annotations

import json
import re
from decimal import Decimal
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

from verifiqa.agent.claim_classifier import (
    decompose_question,
    decompose_question_consensus,
    parse_claimed_answer,
)

from verifiqa.agent.fact_extractor import (
    extract_fact,
    extract_facts_parallel,
    extract_fact_with_feedback,
)
from verifiqa.agent.types import AgentResult, ClaimSpec, GroundedFact, RoleSpec
from verifiqa.agent.table_fact_extractor import AmbiguousTableLookup, lookup_fact
from verifiqa.agent.utils import extract_json
from verifiqa.answer_spec import answer_spec_from_question, effective_tolerance_from_answer
from verifiqa.formulas.evaluator import evaluate_formula, formula_variables
from verifiqa.policy.registry import DEFAULT_POLICY_REGISTRY
from verifiqa.types import SolverResult, VerificationFact, VerificationIR
from verifiqa.verification.generic_ops import resolve_generic_operation
from verifiqa.verification.smt_generator import SmtGenerator, render_ir_smt
from verifiqa.verification.xbrl_linkbase import (
    FilingXbrl,
    SourceBindingConflict,
    _actual_unit_kind,
    _bind_fact_local,
    _bind_ir_facts,
    _candidate_concept_locals,
    _exact_numeric_label_concepts,
    _source_label_identity,
    _source_period_candidates,
    _unit_kind_compatible_for_fact,
    attach_xbrl_calculations,
    augment_smt_with_xbrl,
    load_filing_xbrl,
)
from verifiqa.verification.z3_runner import Z3Runner

@dataclass
class AgentConfig:
    max_workers: int = 4
    # A single fact-repair pass is the experimental default. Tests and targeted
    # diagnostics may opt into additional passes explicitly.
    max_unsat_fact_retries: int = 1
    xbrl_artifacts_dir: Optional[Path] = None
    source_plan_cache_dir: Optional[Path] = None
    require_claimspec_consensus: bool = False
    require_smt_consensus: bool = False
    # On a Z3-INVALID (unparseable LLM SMT), regenerate the SMT once with the parser
    # error fed back, and re-run. Recovers solver_INVALID abstentions without a
    # deterministic renderer. Soundness is unaffected: only SAT/UNSAT decide a verdict.
    repair_invalid_smt: bool = False
    # Render the SMT deterministically from the IR (render_ir_smt) instead of using the
    # LLM autoformalizer. Always well-formed and faithful, so it removes solver_INVALID
    # abstentions entirely; no LLM in the SMT step.
    deterministic_smt: bool = False

@dataclass
class _PreparedInputs:
    claimed_value: float
    tolerance: float
    precision_digits: int | None
    tolerance_source: str
    unit: str


class VerificationAgent:
    """Classify → extract (parallel) → SMT (LLM) → Z3."""

    def __init__(
        self,
        llm_client,
        smt_generator: SmtGenerator,
        z3_runner: Optional[Z3Runner] = None,
        config: Optional[AgentConfig] = None,
        progress_callback: Optional[Callable[..., None]] = None,
    ):
        self.llm = llm_client
        self.smt_gen = smt_generator
        self.z3 = z3_runner or Z3Runner()
        self.config = config or AgentConfig()
        self._cb = progress_callback or (lambda *a, **kw: None)

    def run(self, question: str, answer: str, evidence_text: str, doc_name: str = "", *, source_pdf_path: str = "") -> AgentResult:
        # 1. Resolve the verification target from question+evidence only. The
        # generated answer must not be allowed to choose or reinterpret formula.
        self._cb("decompose_question", question=question)
        decomposition_diagnostics: dict = {}
        source_xbrl_bundle = None
        if doc_name and self.config.xbrl_artifacts_dir:
            try:
                source_xbrl_bundle = load_filing_xbrl(doc_name, self.config.xbrl_artifacts_dir)
            except Exception as exc:
                return self._abstain(question, answer, f"source_xbrl_load_failed:{type(exc).__name__}")
            if source_xbrl_bundle.status != "ok":
                source_xbrl_bundle = None
        decomposition_kwargs = {"evidence_text": evidence_text, "source_pdf_path": source_pdf_path}
        if self.config.source_plan_cache_dir is not None:
            decomposition_kwargs["source_plan_cache_dir"] = self.config.source_plan_cache_dir
        if source_xbrl_bundle is not None:
            decomposition_kwargs["source_xbrl_bundle"] = source_xbrl_bundle
        try:
            if self.config.require_claimspec_consensus:
                claim_spec, decomposition_diagnostics, failure = decompose_question_consensus(
                    question,
                    self.llm,
                    **decomposition_kwargs,
                )
                if failure:
                    return AgentResult(
                        question=question,
                        answer=answer,
                        status="ABSTAIN",
                        metric=claim_spec.metric if claim_spec else "",
                        formula=claim_spec.formula if claim_spec else "",
                        formula_source=claim_spec.formula_source if claim_spec else "",
                        failure_reason=failure,
                        diagnostics=decomposition_diagnostics,
                    )
                if claim_spec is None:
                    return AgentResult(
                        question=question,
                        answer=answer,
                        status="ABSTAIN",
                        failure_reason="claimspec_consensus_failed",
                        diagnostics=decomposition_diagnostics,
                    )
            else:
                claim_spec = decompose_question(question, self.llm, **decomposition_kwargs)
        except Exception as exc:
            return self._abstain(question, answer, f"decomposition_failed:{type(exc).__name__}")
        claim_spec, xbrl_claimspec_diagnostics = _maybe_authorize_xbrl_linkbase_claim(
            claim_spec,
            question,
            evidence_text,
            doc_name,
            self.config.xbrl_artifacts_dir,
        )
        decomposition_diagnostics = _merge_diagnostics(
            decomposition_diagnostics,
            xbrl_claimspec_diagnostics,
            {"claim_spec": {
                "metric": claim_spec.metric,
                "formula": claim_spec.formula,
                "formula_source": claim_spec.formula_source,
                "claim_unit": claim_spec.claim_unit,
                "roles": [vars(role).copy() for role in claim_spec.roles],
                "planning_source": claim_spec.planning_source,
                "source_document_sha256": claim_spec.source_document_sha256,
                "source_schema_sha256": claim_spec.source_schema_sha256,
                "source_report_unit": claim_spec.source_report_unit,
                "source_selection_id": claim_spec.source_selection_id,
            }},
        )
        if claim_spec.source_report_unit and self.config.xbrl_artifacts_dir and doc_name:
            from verifiqa.agent.source_report_units import recover_report_unit
            unit_witness = recover_report_unit(evidence_text, doc_name, self.config.xbrl_artifacts_dir)
            if unit_witness is not None and unit_witness["unit"] == claim_spec.source_report_unit:
                decomposition_diagnostics["source_report_unit_provenance"] = unit_witness
        self._cb("classified", metric=claim_spec.metric, formula_source=claim_spec.formula_source)

        if claim_spec.source_selection_id or any(role.source_ref for role in claim_spec.roles):
            from verifiqa.agent.source_selection import validate_source_selection
            if not validate_source_selection(claim_spec, question, evidence_text,
                    source_pdf_path=source_pdf_path, source_xbrl_bundle=source_xbrl_bundle):
                return AgentResult(question=question, answer=answer, status="ABSTAIN",
                    failure_reason="source_selection_invalid", diagnostics=decomposition_diagnostics)

        if claim_spec.planning_source == "deterministic_xbrl_schema":
            from verifiqa.agent.source_xbrl_schema_planner import source_xbrl_plan_matches
            if source_xbrl_bundle is None or not source_xbrl_plan_matches(
                    claim_spec, question, evidence_text, source_xbrl_bundle):
                return self._abstain(question, answer, "source_xbrl_schema_plan_invalid")

        if claim_spec.operation == "ambiguous_operation":
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=claim_spec.metric, formula_source=claim_spec.formula_source,
                failure_reason="ambiguous_operation",
                diagnostics=decomposition_diagnostics,
            )
        if claim_spec.formula_source == "no_formula" or not claim_spec.formula.strip():
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=claim_spec.metric, formula_source="no_formula",
                failure_reason="no_formula",
                diagnostics={**decomposition_diagnostics, "abstention_category": "unverified_formula"},
            )

        _TRUSTED_SOURCES = {"policy_registry", "generic_operation", "document_derived", "xbrl_linkbase"}
        if claim_spec.formula_source not in _TRUSTED_SOURCES:
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=claim_spec.metric, formula=claim_spec.formula,
                formula_source=claim_spec.formula_source,
                failure_reason=f"untrusted_formula_source:{claim_spec.formula_source}",
                diagnostics=decomposition_diagnostics,
            )

        if not claim_spec.roles:
            return AgentResult(
                question=question,
                answer=answer,
                status="ABSTAIN",
                failure_reason="no_roles_identified",
                diagnostics=decomposition_diagnostics,
            )

        # The classifier can copy a formula template and its role names inconsistently
        # (e.g. formula uses "revenue_start"/"revenue_end" while a role is instead
        # named "revenue_2015"), especially on multi-period metrics. If any variable in
        # the formula has no matching role, the SMT built from it would reference an
        # undeclared constant and Z3 would reject it as invalid -- catch that here, with
        # a clear reason, rather than reaching the solver with an unusable formula.
        try:
            formula_vars = formula_variables(claim_spec.formula)
        except (SyntaxError, ValueError):
            formula_vars = None
        role_names = {role.name for role in claim_spec.roles}
        if formula_vars is not None and not formula_vars <= role_names and claim_spec.formula_source == "generic_operation":
            # Generic formulas are rebuilt from the question and grounded operands
            # below. Template placeholders must not block that independent check.
            decomposition_diagnostics["generic_formula_placeholders"] = sorted(formula_vars - role_names)
        elif formula_vars is not None and not formula_vars <= role_names:
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=claim_spec.metric, formula=claim_spec.formula,
                formula_source=claim_spec.formula_source,
                failure_reason="formula_role_mismatch:"
                    f"{','.join(sorted(formula_vars - role_names))}",
                diagnostics=decomposition_diagnostics,
            )

        # 2. Extract only the claimed final value from the generated answer. The
        # parser sees the fixed ClaimSpec but cannot modify it.
        self._cb("parse_claim", metric=claim_spec.metric)
        try:
            claimed = parse_claimed_answer(question, answer, claim_spec, self.llm)
        except Exception as exc:
            return self._abstain(question, answer, f"claim_parse_failed:{type(exc).__name__}")
        claim_spec.claimed_value = claimed.value
        claim_spec.claimed_unit = claimed.unit
        raw_claimed_value = claimed.value
        if raw_claimed_value is None:
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=claim_spec.metric, formula=claim_spec.formula,
                formula_source=claim_spec.formula_source, failure_reason="no_numeric_answer",
                diagnostics=decomposition_diagnostics,
            )

        # Re-read deterministic plans against this evidence. A marker alone is
        # not authority, and a changed/incomplete source population cannot fall
        # through to model extraction under an already selected formula.
        planned_facts = None
        if claim_spec.planning_source == "deterministic_source_table":
            from verifiqa.agent.source_question_planner import source_plan_facts
            planned_facts = source_plan_facts(claim_spec, question, evidence_text, source_pdf_path=source_pdf_path)
            if planned_facts is None or set(planned_facts) != role_names:
                return AgentResult(
                    question=question, answer=answer, status="ABSTAIN",
                    metric=claim_spec.metric, formula=claim_spec.formula,
                    formula_source=claim_spec.formula_source,
                    claimed_value=raw_claimed_value,
                    failure_reason="source_plan_grounding_failed",
                    diagnostics={**decomposition_diagnostics,
                                 "abstention_category": "grounding_failed"},
                )

        # 3. Bind registry-backed and exact filing-label roles directly from the
        # packaged XBRL before asking the LLM to read rendered text.
        binding_context = {"question": question, "evidence_text": evidence_text}
        if planned_facts is not None:
            binding_context["source_facts"] = planned_facts
        facts, xbrl_fact_diagnostics = _bind_xbrl_role_facts(
            claim_spec,
            doc_name,
            self.config.xbrl_artifacts_dir,
            **binding_context,
        )
        decomposition_diagnostics = _merge_diagnostics(
            decomposition_diagnostics,
            xbrl_fact_diagnostics,
        )
        semantic_failure = xbrl_fact_diagnostics.get("xbrl_role_fact_binding", {}).get("semantic_failure")
        if semantic_failure:
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=claim_spec.metric, formula=claim_spec.formula,
                formula_source=claim_spec.formula_source, facts=facts,
                claimed_value=raw_claimed_value, failure_reason=semantic_failure,
                diagnostics=decomposition_diagnostics,
            )

        if planned_facts is not None:
            facts.update({name: fact for name, fact in planned_facts.items() if name not in facts})
            decomposition_diagnostics["source_plan_grounding"] = {
                "status": "passed", "roles": sorted(planned_facts),
                "source_tables": sorted({fact.source_table_id for fact in planned_facts.values()}),
            }

        remaining_roles = [r for r in claim_spec.roles if r.name not in facts]
        self._cb(
            "extract_facts",
            roles=[r.name for r in remaining_roles],
            xbrl_bound=sorted(facts),
        )
        if remaining_roles:
            extraction_spec = replace(claim_spec, roles=remaining_roles)
            facts.update(
                extract_facts_parallel(
                    extraction_spec, evidence_text, self.llm, self.config.max_workers,
                    question=question,
                )
            )

        # Retry missing facts once (same evidence, sequential)
        missing = [name for name, f in facts.items() if f is None]
        if missing:
            self._cb("retry_missing", facts=missing)
            for name in missing:
                role = next((r for r in claim_spec.roles if r.name == name), None)
                if role:
                    try:
                        facts[name] = extract_fact(role, evidence_text, claim_spec.formula, self.llm,
                                                   question=question)
                    except Exception:
                        facts[name] = None

        still_missing = [n for n, f in facts.items() if f is None]
        if still_missing:
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=claim_spec.metric, formula=claim_spec.formula,
                formula_source=claim_spec.formula_source,
                claimed_value=raw_claimed_value,
                facts={name: fact for name, fact in facts.items() if fact is not None},
                failure_reason=f"facts_not_found:{','.join(still_missing)}",
                diagnostics=_merge_diagnostics(
                    decomposition_diagnostics, {"missing_facts": still_missing}
                ),
            )

        grounded: dict[str, GroundedFact] = dict(facts)  # type: ignore[arg-type]

        if claim_spec.source_selection_id:
            from verifiqa.agent.source_selection import bind_selected_facts
            selected = bind_selected_facts(claim_spec, grounded, question, evidence_text,
                source_pdf_path=source_pdf_path, source_xbrl_bundle=source_xbrl_bundle)
            if selected is None:
                return AgentResult(question=question, answer=answer, status="ABSTAIN",
                    metric=claim_spec.metric, formula=claim_spec.formula,
                    formula_source=claim_spec.formula_source, facts=grounded,
                    claimed_value=raw_claimed_value, failure_reason="source_selection_binding_mismatch",
                    diagnostics=decomposition_diagnostics)
            grounded = selected
            decomposition_diagnostics["source_selection"] = {
                "status": "passed", "selection_id": claim_spec.source_selection_id,
                "records": {name: fact.source_ref for name, fact in grounded.items()},
            }

        # 4. Grounding check — normalized once, reused per fact
        norm_evidence = _normalize_text(evidence_text)
        for fact in grounded.values():
            if fact.source_quote and not _quote_grounded(fact.source_quote, norm_evidence):
                # Audit: preserve facts + metric in the result, don't silently discard
                return AgentResult(
                    question=question, answer=answer, status="ABSTAIN",
                    metric=claim_spec.metric, formula=claim_spec.formula,
                    formula_source=claim_spec.formula_source,
                    facts=grounded, claimed_value=raw_claimed_value,
                    failure_reason=f"grounding_failed:{fact.name}",
                    diagnostics=_merge_diagnostics(
                        decomposition_diagnostics,
                        {"ungrounded_quote": fact.source_quote},
                    ),
                )

        try:
            prepared = _prepare_inputs(question, answer, claim_spec, grounded, raw_claimed_value)
        except SourceBindingConflict as exc:
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=claim_spec.metric, formula=claim_spec.formula, formula_source=claim_spec.formula_source,
                facts=grounded, claimed_value=raw_claimed_value, failure_reason=str(exc),
                diagnostics=decomposition_diagnostics,
            )
        claimed_value = prepared.claimed_value
        if prepared.unit:
            decomposition_diagnostics["claim_normalization"] = _claim_normalization_diagnostics(
                prepared, raw_claimed_value, claimed.unit, claim_spec.claimed_unit,
            )

        # 5. Build IR
        ir = _build_ir(
            claim_spec,
            grounded,
            claimed_value,
            prepared.tolerance,
            prepared.precision_digits,
            prepared.tolerance_source,
        )
        ir, authority_diagnostics, authority_failure = _authorize_generic_operation(
            ir, question, claim_spec, evidence_text=evidence_text, source_pdf_path=source_pdf_path,
            source_xbrl_bundle=source_xbrl_bundle,
        )
        decomposition_diagnostics = _merge_diagnostics(decomposition_diagnostics, authority_diagnostics)
        if authority_failure:
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=ir.metric, formula=ir.formula, formula_source=claim_spec.formula_source,
                facts=grounded, claimed_value=claimed_value,
                failure_reason=authority_failure,
                diagnostics={**decomposition_diagnostics, "abstention_category": "unverified_formula"},
            )
        claim_spec.formula = ir.formula

        # A formula/roles can still come back empty after authorization even when no
        # authority_failure was raised (e.g. a degenerate claim spec on a weak model).
        # Catch that here rather than letting an empty obligation reach the solver,
        # where it would surface as an opaque solver_INVALID instead of a clear reason.
        if not ir.formula.strip() or not ir.facts:
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=ir.metric, formula=ir.formula, formula_source=claim_spec.formula_source,
                facts=grounded, claimed_value=claimed_value,
                failure_reason="empty_obligation_after_authorization",
                diagnostics={**decomposition_diagnostics, "abstention_category": "unverified_formula"},
            )

        # 5a. Attach XBRL calculation linkbase (if artifacts available)
        if self.config.xbrl_artifacts_dir and doc_name:
            try:
                ir = attach_xbrl_calculations(ir, doc_name, self.config.xbrl_artifacts_dir)
            except SourceBindingConflict as exc:
                return AgentResult(
                    question=question, answer=answer, status="ABSTAIN",
                    metric=ir.metric, formula=ir.formula, formula_source=claim_spec.formula_source,
                    facts=grounded, claimed_value=claimed_value, failure_reason=str(exc),
                    diagnostics=decomposition_diagnostics,
                )

        # 6. Generate SMT
        self._cb("generate_smt", metric=ir.metric)
        try:
            smtlib, result, smt_diagnostics, smt_failure = self._generate_and_run_smt(
                ir,
                doc_name=doc_name,
                label=f"agent_{ir.metric}",
            )
        except Exception as exc:
            return self._abstain(question, answer, f"smt_failed:{type(exc).__name__}")
        smt_diagnostics = _merge_diagnostics(decomposition_diagnostics, smt_diagnostics)
        if smt_failure:
            return AgentResult(
                question=question, answer=answer, status="ABSTAIN",
                metric=ir.metric, formula=ir.formula,
                formula_source=claim_spec.formula_source,
                facts=grounded, claimed_value=claimed_value,
                smtlib=smtlib,
                failure_reason=smt_failure,
                diagnostics=smt_diagnostics,
            )

        # 7. Z3 single pass
        self._cb("verify")

        # Record the first-pass verdict on every result so we can tell whether the
        # initial answer verified on its own, vs only after the UNSAT repair retry.
        smt_diagnostics = _merge_diagnostics(
            smt_diagnostics, {"first_pass_solver_status": result.solver_status}
        )

        if result.solver_status == "SAT":
            return AgentResult(
                question=question, answer=answer, status="VERIFIED",
                metric=ir.metric, formula=ir.formula, formula_source=claim_spec.formula_source,
                facts=grounded, claimed_value=claimed_value,
                smtlib=smtlib, solver_status="SAT",
                diagnostics=smt_diagnostics,
            )

        if result.solver_status == "UNSAT":
            # 8. Counterexample-aware retry: re-extract facts named in the UNSAT
            # core, or all facts when no useful core is available. The retry
            # prompt sees the computed mismatch and previous extraction so it can
            # look for a better row/period instead of repeating the same value.
            retry_diagnostics = []
            latest_ir = ir
            latest_smtlib = smtlib
            latest_result = result
            for attempt in range(1, self.config.max_unsat_fact_retries + 1):
                retry_names = _retry_fact_names(latest_result.unsat_core or [], grounded)
                retry_names = [name for name in retry_names
                               if grounded[name].fact_type not in {"xbrl_numeric", "source_table_numeric"}]
                if not retry_names:
                    break
                computed_value = _computed_formula_value(claim_spec.formula, grounded)
                self._cb(
                    "retry_unsat",
                    facts=retry_names,
                    attempt=attempt,
                    computed_value=computed_value,
                    claimed_value=claimed_value,
                )
                changed: list[str] = []
                skipped_ungrounded: list[str] = []
                for name in retry_names:
                    role = next((r for r in claim_spec.roles if r.name == name), None)
                    if role is None:
                        continue
                    previous = grounded.get(name)
                    try:
                        retried = extract_fact_with_feedback(
                            role,
                            evidence_text,
                            claim_spec.formula,
                            self.llm,
                            claimed_value=claimed_value,
                            computed_value=computed_value,
                            tolerance=latest_ir.tolerance,
                            current_facts=grounded,
                            previous_fact=previous,
                            attempt=attempt,
                            unsat_core=latest_result.unsat_core,
                            question=question,
                        )
                    except Exception:
                        retried = None
                    if retried is None or _same_fact(previous, retried):
                        continue
                    if retried.source_quote and not _quote_grounded(retried.source_quote, norm_evidence):
                        skipped_ungrounded.append(name)
                        continue
                    grounded[name] = retried
                    changed.append(name)

                retry_diagnostics.append({
                    "attempt": attempt,
                    "facts": retry_names,
                    "changed": changed,
                    "skipped_ungrounded": skipped_ungrounded,
                    "computed_value_before": computed_value,
                })
                if not changed:
                    continue

                try:
                    prepared = _prepare_inputs(question, answer, claim_spec, grounded, raw_claimed_value)
                except SourceBindingConflict as exc:
                    failed_diagnostics = _agent_diagnostics(smt_diagnostics, retry_diagnostics)
                    failed_diagnostics.pop("claim_normalization", None)
                    return AgentResult(
                        question=question, answer=answer, status="ABSTAIN",
                        metric=claim_spec.metric, formula=claim_spec.formula, formula_source=claim_spec.formula_source,
                        facts=grounded, claimed_value=raw_claimed_value, failure_reason=str(exc),
                        diagnostics=failed_diagnostics,
                    )
                claimed_value = prepared.claimed_value
                smt_diagnostics.pop("claim_normalization", None)
                if prepared.unit:
                    smt_diagnostics["claim_normalization"] = _claim_normalization_diagnostics(
                        prepared, raw_claimed_value, claimed.unit, claim_spec.claimed_unit,
                    )
                ir2 = _build_ir(
                    claim_spec,
                    grounded,
                    claimed_value,
                    prepared.tolerance,
                    prepared.precision_digits,
                    prepared.tolerance_source,
                )
                ir2, retry_authority_diagnostics, authority_failure = _authorize_generic_operation(
                    ir2,
                    question,
                    claim_spec,
                    evidence_text=evidence_text,
                    source_pdf_path=source_pdf_path,
                    source_xbrl_bundle=source_xbrl_bundle,
                )
                if retry_authority_diagnostics:
                    retry_diagnostics[-1].update(retry_authority_diagnostics)
                if authority_failure:
                    return AgentResult(
                        question=question, answer=answer, status="ABSTAIN",
                        metric=ir2.metric, formula=ir2.formula,
                        formula_source=claim_spec.formula_source,
                        facts=grounded, claimed_value=claimed_value,
                        failure_reason=authority_failure,
                        diagnostics={**_agent_diagnostics(smt_diagnostics, retry_diagnostics),
                                     "abstention_category": "unverified_formula"},
                    )
                claim_spec.formula = ir2.formula
                if self.config.xbrl_artifacts_dir and doc_name:
                    try:
                        ir2 = attach_xbrl_calculations(ir2, doc_name, self.config.xbrl_artifacts_dir)
                    except SourceBindingConflict as exc:
                        return AgentResult(
                            question=question, answer=answer, status="ABSTAIN",
                            metric=ir2.metric, formula=ir2.formula, formula_source=claim_spec.formula_source,
                            facts=grounded, claimed_value=claimed_value, failure_reason=str(exc),
                            diagnostics=_agent_diagnostics(smt_diagnostics, retry_diagnostics),
                        )
                try:
                    smtlib2, result2, retry_smt_diagnostics, smt_failure = self._generate_and_run_smt(
                        ir2,
                        doc_name=doc_name,
                        label=f"agent_{ir.metric}_retry_{attempt}",
                    )
                except Exception as exc:
                    return AgentResult(
                        question=question, answer=answer, status="ABSTAIN",
                        metric=ir2.metric, formula=ir2.formula,
                        formula_source=claim_spec.formula_source,
                        facts=grounded, claimed_value=claimed_value,
                        failure_reason=f"smt_failed:unsat_retry:{type(exc).__name__}",
                        diagnostics=_agent_diagnostics(smt_diagnostics, retry_diagnostics),
                    )
                if retry_smt_diagnostics:
                    retry_diagnostics[-1].update(retry_smt_diagnostics)
                if smt_failure:
                    return AgentResult(
                        question=question, answer=answer, status="ABSTAIN",
                        metric=ir2.metric, formula=ir2.formula,
                        formula_source=claim_spec.formula_source,
                        facts=grounded, claimed_value=claimed_value,
                        smtlib=smtlib2,
                        failure_reason=smt_failure,
                        diagnostics=_agent_diagnostics(smt_diagnostics, retry_diagnostics),
                    )
                latest_ir = ir2
                latest_smtlib = smtlib2
                latest_result = result2
                retry_diagnostics[-1]["solver_status"] = result2.solver_status
                retry_diagnostics[-1]["computed_value_after"] = _computed_formula_value(
                    claim_spec.formula, grounded
                )
                if result2.solver_status == "SAT":
                    return AgentResult(
                        question=question, answer=answer, status="VERIFIED",
                        metric=ir2.metric, formula=ir2.formula,
                        formula_source=claim_spec.formula_source,
                        facts=grounded, claimed_value=claimed_value,
                        smtlib=smtlib2, solver_status="SAT",
                        diagnostics=_agent_diagnostics(smt_diagnostics, retry_diagnostics),
                    )
                if result2.solver_status != "UNSAT":
                    return AgentResult(
                        question=question, answer=answer, status="ABSTAIN",
                        metric=ir2.metric, formula=ir2.formula,
                        formula_source=claim_spec.formula_source,
                        facts=grounded, claimed_value=claimed_value,
                        smtlib=smtlib2, solver_status=result2.solver_status,
                        failure_reason=f"solver_{result2.solver_status}",
                        diagnostics=_agent_diagnostics(smt_diagnostics, retry_diagnostics),
                    )

            return AgentResult(
                question=question, answer=answer, status="VIOLATED",
                metric=latest_ir.metric, formula=latest_ir.formula,
                formula_source=claim_spec.formula_source,
                facts=grounded, claimed_value=claimed_value,
                smtlib=latest_smtlib, solver_status="UNSAT",
                failure_reason="claim_inconsistent_with_evidence",
                diagnostics=_agent_diagnostics(smt_diagnostics, retry_diagnostics),
            )

        return self._abstain(question, answer, f"solver_{result.solver_status}")

    def _generate_and_run_smt(
        self,
        ir: VerificationIR,
        *,
        doc_name: str,
        label: str,
    ) -> tuple[str, SolverResult, dict, str]:
        xbrl = ir.xbrl_calculations or {}
        independent_evidence = bool(xbrl.get("bindings") or xbrl.get("constraints") or ir.evidence_facts)

        def augmented(smtlib):
            # Already attached IR constraints do not depend on a filesystem
            # setting to retain their semantics in either solver obligation.
            return augment_smt_with_xbrl(smtlib, ir) if independent_evidence else smtlib

        def evidence_consistency_failure(result, diagnostics):
            if str(result.solver_status).upper() != "UNSAT" or not independent_evidence:
                return ""
            # An inconsistent graph disproves every possible candidate. It
            # cannot establish that this particular generated answer is wrong.
            # Compile this query deterministically, without reading the claim,
            # rather than deleting assertions from an LLM-generated obligation.
            try:
                consistency_smt = augmented(render_ir_smt(ir, include_claim=False))
                consistency = self.z3.run(
                    consistency_smt, label=f"{label}_evidence_consistency",
                    debug_unsat_core=True,
                )
            except Exception as exc:
                diagnostics["evidence_consistency"] = {
                    "status": "failed", "solver_status": "",
                    "method": "deterministic_without_claim", "unsat_core": [],
                    "error": f"{type(exc).__name__}:{exc}",
                }
                return f"evidence_consistency_failed:{type(exc).__name__}"
            status = str(consistency.solver_status).upper()
            diagnostics["evidence_consistency"] = {
                "status": "passed" if status == "SAT" else (
                    "inconsistent" if status == "UNSAT" else "unresolved"),
                "solver_status": status, "method": "deterministic_without_claim",
                "unsat_core": list(consistency.unsat_core or []),
                "error": consistency.error,
            }
            if status == "UNSAT":
                return "inconsistent_evidence"
            if status != "SAT":
                return f"evidence_consistency_{status.lower() or 'failed'}"
            return ""

        if not self.config.require_smt_consensus:
            smtlib = (render_ir_smt(ir) if self.config.deterministic_smt
                      else self.smt_gen.generate_ir(ir))
            smtlib = augmented(smtlib)
            # No retry on an invalid SMT: an invalid obligation abstains. Regenerating
            # it with the LLM was found to break soundness (a valid-but-faithless
            # obligation can false-accept), so that retry is removed.
            result = self.z3.run(smtlib, label=label)
            diagnostics = {}
            failure = evidence_consistency_failure(result, diagnostics)
            return smtlib, result, diagnostics, failure

        runs = []
        results = []
        for index in range(3):
            smtlib = augmented(self.smt_gen.generate_ir(ir))
            result = self.z3.run(smtlib, label=f"{label}_smt_{index + 1}")
            results.append(result)
            runs.append({
                "index": index + 1,
                "solver_status": result.solver_status,
                "error": result.error,
                "smtlib": smtlib,
            })

        statuses = [str(run["solver_status"]).upper() for run in runs]
        diagnostics = {"smt_consensus": runs}
        if len(set(statuses)) == 1 and statuses[0] in {"SAT", "UNSAT"}:
            failure = evidence_consistency_failure(results[0], diagnostics)
            return runs[0]["smtlib"], results[0], diagnostics, failure
        return runs[0]["smtlib"], SolverResult(solver_status=""), diagnostics, "smt_consensus_failed"

    def _abstain(self, question: str, answer: str, reason: str) -> AgentResult:
        return AgentResult(
            question=question, answer=answer, status="ABSTAIN", failure_reason=reason
        )


_MONEY_SCALE = {"billion": 1_000_000_000.0, "million": 1_000_000.0, "thousand": 1_000.0}
_MILLION = 1_000_000.0


def _money_scale_from_unit(unit: str | None) -> float | None:
    """Dollars-per-unit for a fact's declared unit string, else None (non-monetary)."""
    text = (unit or "").lower()
    if re.search(r"\bshares?\b|\bper\b|/", text):
        return None
    scaled = next((v for k, v in _MONEY_SCALE.items() if k in text), None)
    if scaled is not None:
        return scaled
    if re.fullmatch(r"\s*(?:usd|(?:u\.?\s*s\.?\s+)?dollars?|\$|iso4217:usd)\s*", text):
        return 1.0
    from verifiqa.verification.money_units import money_currency
    if money_currency(text) and re.fullmatch(r"(?:iso4217:)?[a-z$€£\s]+", text):
        return 1.0
    return None


def _evidence_money_scale(facts) -> float | None:
    """Money scale (dollars-per-unit) shared by all monetary facts, else None.

    Returns None when facts have no monetary unit or mix scales, so callers can
    skip conversion rather than guess.
    """
    from verifiqa.verification.money_units import money_currency
    scales, currencies = set(), set()
    for f in facts:
        matched = _money_scale_from_unit(f.unit)
        currency = money_currency(f.unit)
        if matched is None or currency is None:
            return None
        scales.add(matched)
        currencies.add(currency)
    if len(scales) != 1 or len(currencies) != 1:
        return None
    return scales.pop()


def _expected_unit_money_scale(expected_unit: str, claimed_unit: str | None) -> float | None:
    """Money scale requested by the question (or the LLM-parsed claim), else None."""
    # An explicit answer unit describes the displayed number. The requested
    # format is a fallback; a dollar amount remains dollars when the question
    # asked for its equivalent in millions.
    for source in (claimed_unit or "", expected_unit or ""):
        matched = _money_scale_from_unit(source)
        if matched is not None:
            return matched
    return None


def _prepare_inputs(
    question: str,
    answer: str,
    claim_spec: ClaimSpec,
    grounded: dict[str, GroundedFact],
    raw_claimed_value: float,
) -> _PreparedInputs:
    """Normalize facts and claim/tolerance into the formula's numeric scale.

    The helper is intentionally idempotent: facts already converted to
    ``USD millions`` are left untouched, so retry paths can call it after every
    fact replacement without double-scaling old facts.
    """
    from verifiqa.verification.money_units import explicit_money_currencies, money_currency, unrecognized_currency_unit
    spec = answer_spec_from_question(question, answer)
    schema_plan = claim_spec.planning_source == "deterministic_xbrl_schema"
    if spec.expected_unit == "days":
        from verifiqa.agent.claim_unit_evidence import AmbiguousClaimUnit, explicit_claim_day_unit
        try:
            literal = explicit_claim_day_unit(answer, raw_claimed_value)
        except AmbiguousClaimUnit as exc:
            raise SourceBindingConflict("ambiguous_answer_unit") from exc
        if claim_spec.claim_unit.lower() in {"", "other", "unspecified", "day", "days"}:
            claim_spec.claim_unit = "days"
        if literal:
            claim_spec.claimed_unit = literal
        if claim_spec.claim_unit != "days" or claim_spec.claimed_unit.lower() not in {"day", "days"}:
            raise SourceBindingConflict("claim_unit_mismatch:days")
    nonmonetary = {"percent", "ratio", "percentage_points", "days"}
    monetary_output = (bool(grounded) and claim_spec.claim_unit not in nonmonetary
                       and spec.expected_unit not in nonmonetary
                       and all(_money_scale_from_unit(fact.unit) is not None
                               and money_currency(fact.unit) for fact in grounded.values()))
    if monetary_output:
        from verifiqa.agent.claim_unit_evidence import AmbiguousClaimUnit, claim_money_unit_evidence
        native = {money_currency(fact.unit) for fact in grounded.values()}
        if len(native) != 1:
            raise SourceBindingConflict("inconsistent_source_currencies")
        try:
            literal = claim_money_unit_evidence(answer, raw_claimed_value, next(iter(native)))
        except AmbiguousClaimUnit as exc:
            raise SourceBindingConflict("ambiguous_answer_unit") from exc
        requested = answer_spec_from_question(question).expected_unit
        if _money_scale_from_unit(requested) is None:
            physical_format = re.search(r"\bin\s+(?P<unit>(?:U\.?S\.?\s+)?dollars?|euros?)\b", question, re.I)
            if physical_format:
                requested_currencies = explicit_money_currencies(physical_format.group("unit"))
                if len(requested_currencies) == 1:
                    requested = next(iter(requested_currencies))
        if unrecognized_currency_unit(literal.unit):
            raise SourceBindingConflict("unsupported_currency_unit")
        literal_currencies = explicit_money_currencies(literal.unit)
        if literal_currencies and literal_currencies != native:
            raise SourceBindingConflict("currency_conversion_not_grounded")
        if literal.display_kind in {"scaled", "base"}:
            claim_spec.claimed_unit = literal.unit
        elif schema_plan and _money_scale_from_unit(requested) is not None:
            claim_spec.claimed_unit = requested
        elif schema_plan and claim_spec.source_report_unit:
            claim_spec.claimed_unit = claim_spec.source_report_unit
        elif schema_plan and raw_claimed_value != 0:
            raise SourceBindingConflict("ambiguous_answer_unit")
        elif (literal.display_kind == "currency_only" and raw_claimed_value != 0
              and _money_scale_from_unit(claim_spec.claimed_unit) is None):
            # A parser response such as "other" cannot turn an unqualified
            # currency amount into the canonical millions used by XBRL facts.
            if _money_scale_from_unit(requested) is not None:
                claim_spec.claimed_unit = requested
            elif _money_scale_from_unit(claim_spec.source_report_unit) is not None:
                claim_spec.claimed_unit = claim_spec.source_report_unit
            elif (claim_spec.planning_source == "deterministic_source_table"
                  and _money_scale_from_unit(claim_spec.claim_unit) is not None):
                claim_spec.claimed_unit = claim_spec.claim_unit
            else:
                raise SourceBindingConflict("ambiguous_answer_unit")
    if any(unrecognized_currency_unit(unit) for unit in
           [claim_spec.claimed_unit or "", claim_spec.claim_unit]
           + [fact.unit for fact in grounded.values()]):
        raise SourceBindingConflict("unsupported_currency_unit")
    _normalize_monetary_facts(grounded)
    source_currencies = set().union(*(explicit_money_currencies(fact.unit) for fact in grounded.values()))
    if len(source_currencies) > 1:
        raise SourceBindingConflict("inconsistent_source_currencies")
    if source_currencies:
        displayed = explicit_money_currencies(claim_spec.claimed_unit or "")
        requested = explicit_money_currencies(question)
        if ((displayed and displayed != source_currencies)
                or (requested and requested != source_currencies)):
            raise SourceBindingConflict("currency_conversion_not_grounded")
    tolerance, precision_digits, tolerance_source = effective_tolerance_from_answer(
        spec,
        raw_claimed_value,
        answer,
    )
    scale_factor = _claim_money_scale_factor(spec.expected_unit, claim_spec, grounded)
    # Export a unit only when this function establishes its representation.
    # Other outputs retain the scorer's existing unit interpretation.
    normalized_unit = ("days" if spec.expected_unit == "days" else
                       next(iter(grounded.values())).unit if monetary_output else "")
    if (spec.expected_unit not in {"percent", "ratio", "percentage_points"}
            and (claim_spec.claim_unit == "count" or "shares" in claim_spec.claim_unit.lower())):
        quantity_factor = _claim_quantity_scale_factor(answer, raw_claimed_value, grounded)
        if quantity_factor is not None:
            scale_factor = quantity_factor
            normalized_unit = next(iter(grounded.values())).unit
    tolerance_factor = scale_factor
    if tolerance_source == "question_precision":
        question_scale = _money_scale_from_unit(spec.expected_unit)
        evidence_scale = _evidence_money_scale(grounded.values())
        if question_scale is not None and evidence_scale is not None:
            tolerance_factor = question_scale / evidence_scale
    return _PreparedInputs(
        claimed_value=raw_claimed_value * scale_factor,
        tolerance=tolerance * abs(tolerance_factor),
        precision_digits=precision_digits,
        tolerance_source=tolerance_source,
        unit=normalized_unit,
    )


def _claim_normalization_diagnostics(
    prepared: _PreparedInputs, raw_claimed_value: float, parsed_unit: str, display_unit: str,
) -> dict:
    """Record the scale actually supplied to the solver for result export."""
    return {
        "raw_claimed_value": raw_claimed_value,
        "parsed_unit": parsed_unit,
        "display_unit": display_unit,
        "claimed_value": prepared.claimed_value,
        "unit": prepared.unit,
        "tolerance": prepared.tolerance,
        "precision_digits": prepared.precision_digits,
        "tolerance_source": prepared.tolerance_source,
    }


def _bind_xbrl_role_facts(
    claim_spec: ClaimSpec,
    doc_name: str,
    artifacts_dir: Optional[Path],
    *,
    question: str = "",
    evidence_text: str = "",
    source_facts: dict[str, GroundedFact] | None = None,
) -> tuple[dict[str, GroundedFact], dict]:
    exact_children = claim_spec.formula_source == "xbrl_linkbase"
    if not exact_children and claim_spec.formula_source not in {"policy_registry", "generic_operation"}:
        return {}, {}
    diagnostics: dict = {
        "xbrl_role_fact_binding": {
            "status": "skipped",
            "bound": [],
            "missing": [],
        }
    }
    if not artifacts_dir or not doc_name:
        diagnostics["xbrl_role_fact_binding"]["reason"] = "missing_artifacts_or_doc"
        return {}, diagnostics

    roles = [role for role in claim_spec.roles if role.concept] if exact_children else claim_spec.roles
    if not roles:
        diagnostics["xbrl_role_fact_binding"]["reason"] = "no_role_concepts"
        return {}, diagnostics

    bundle = load_filing_xbrl(doc_name, artifacts_dir)
    if bundle.status != "ok":
        diagnostics["xbrl_role_fact_binding"].update(
            {"reason": bundle.status, "bundle_status": bundle.status}
        )
        return {}, diagnostics

    schema_plan = claim_spec.planning_source == "deterministic_xbrl_schema"
    if schema_plan:
        from verifiqa.agent.source_xbrl_schema_planner import source_xbrl_plan_matches
        if not source_xbrl_plan_matches(claim_spec, question, evidence_text, bundle):
            diagnostics["xbrl_role_fact_binding"]["semantic_failure"] = "source_xbrl_schema_plan_invalid"
            return {}, diagnostics
    elif any(role.xbrl_context_id for role in roles):
        diagnostics["xbrl_role_fact_binding"]["semantic_failure"] = "untrusted_xbrl_context_selection"
        return {}, diagnostics

    from verifiqa.verification.xbrl_scope import requested_scopes

    question_scopes = requested_scopes(VerificationFact("", 0, "", row_label=question))
    placeholder_facts = {}
    skipped = {}
    anchored: dict[str, GroundedFact] = {}
    source_bindings = []
    source_witnesses = []
    failures = []
    source_concepts: dict[tuple[frozenset[str], frozenset[str], str], set[str]] = {}
    for role in roles:
        if exact_children:
            unit, concept = claim_spec.claim_unit or "USD millions", role.concept
        elif schema_plan:
            concept = role.concept
            selected = bundle.facts_by_local_context.get((concept, role.xbrl_context_id))
            measures = bundle.units.get(selected.unit_ref, ()) if selected is not None else ()
            from verifiqa.verification.money_units import explicit_money_currencies
            currencies = explicit_money_currencies(" ".join(measures))
            kind = _actual_unit_kind(measures)
            if len(currencies) != 1 or kind not in {"money", "per_share"}:
                failures.append(f"source_xbrl_schema_unit_invalid:{role.name}")
                continue
            currency = next(iter(currencies))
            unit = currency + (" millions" if kind == "money" else " per share")
        else:
            total_intent = _explicit_total_intent(role, question)
            total_concept = _requested_total_concept(role, question, bundle)
            source_role = role
            if total_intent:
                # A bare component alias supplied by the classifier cannot
                # satisfy an explicit total/sum/all request in the question.
                total_aliases = [label for label in _role_measure_labels(role)
                                 if re.search(r"\btotal\b", label, re.I)]
                source_role = replace(role, aliases=total_aliases or ["total " + _role_measure_labels(role)[0]])
            try:
                # A unique matching *numeric row* is an independent source
                # witness. Section headings and unlabeled footers are not rows.
                source_row = None
                if not total_concept:
                    source_row = (source_facts.get(role.name) if source_facts is not None else
                                  lookup_fact(source_role, evidence_text, question=question) if evidence_text else None)
            except AmbiguousTableLookup:
                failures.append(f"ambiguous_source_row:{role.name}")
                continue
            if source_row is not None and source_row.unit:
                source_probe = VerificationFact(role.name, source_row.value, source_row.unit,
                    period=source_row.period or role.period, row_label=source_row.row_label,
                    source_quote=source_row.source_quote, fact_type="source_table_numeric",
                    column=_source_binding_column(source_row.source_column), source_section=source_row.source_section,
                    source_column=source_row.source_column, source_table_id=source_row.source_table_id)
                if any(scope not in requested_scopes(source_probe) for scope in question_scopes):
                    skipped[role.name] = "question_scope_missing_from_source_row"
                    continue
                label_concepts = _exact_numeric_label_concepts([source_row.row_label], bundle)
                # A typed column is part of the source measure. "Computer
                # software" in a useful-life (years) column cannot collide with
                # the identically printed carrying-value (USD) tag elsewhere.
                # Retain conflicts whenever any same-unit label concept exists;
                # a missing date, scope, or value must not waive those checks.
                concepts = {fact.concept for fact in bundle.facts_by_local_context.values()
                            if fact.concept in label_concepts and fact.value is not None
                            and _unit_kind_compatible_for_fact(
                                source_probe, bundle.units.get(fact.unit_ref, ()), fact.unit_ref)}
                exact_source_concepts = bool(concepts)
                if not label_concepts:
                    locals_ = _candidate_concept_locals(source_probe)
                    concepts = {fact.concept for fact in bundle.facts_by_local_context.values()
                                if fact.local_name in locals_ and fact.value is not None
                                and _unit_kind_compatible_for_fact(
                                    source_probe, bundle.units.get(fact.unit_ref, ()), fact.unit_ref)}
                # A local filing can cover fewer comparative years than its
                # printed selected-financial-data table. Missing local periods
                # do not contradict an independently reconstructed source cell.
                # Keep conflicts when a same-period filing witness does exist.
                if source_row.source_table_id and source_row.source_column and source_row.source_quote:
                    from verifiqa.verification.xbrl_scope import filter_context_period
                    available = [(fact, bundle.contexts[fact.context_id])
                                 for fact in bundle.facts_by_local_context.values()
                                 if fact.concept in concepts and fact.value is not None
                                 and fact.context_id in bundle.contexts]
                    available = filter_context_period(_source_period_candidates(available, source_probe), source_probe, bundle)
                    concepts = {fact.concept for fact, _context in available}
                    exact_source_concepts = exact_source_concepts and bool(concepts)
                witness_ir = VerificationIR(claim_spec.metric, claim_spec.formula, {role.name: source_probe},
                    0.0, claim_spec.claim_unit, claim_spec.tolerance, period=claim_spec.period)
                matches = []
                for candidate in sorted(concepts):
                    binding = _bind_fact_local(replace(source_probe, concept=candidate), witness_ir, bundle,
                                               allow_label_fallback=False)
                    if binding is not None:
                        matches.append(binding)
                if (exact_source_concepts or matches) and len(matches) != 1:
                    failures.append(f"source_row_concept_ambiguous:{role.name}" if matches
                                    else f"source_row_concept_conflict:{role.name}")
                    continue
                source_row.fact_type = "source_table_numeric"
                source_row.source_value = source_row.value
                source_row.source_unit = source_row.unit
                if matches:
                    source_row.concept = str(matches[0]["concept"])
                    source_bindings.append(matches[0])
                    # Identically printed rows under distinct section headings
                    # describe distinct measures (for example pension cost and
                    # projected-obligation discount rates). Only require a
                    # stable concept across periods of the same source measure.
                    source_identity = (_source_label_identity(source_row.source_section),
                                       _source_label_identity(source_row.row_label),
                                       source_row.unit)
                    source_concepts.setdefault(source_identity, set()).add(source_row.concept)
                anchored[role.name] = source_row
                source_witnesses.append({
                    "fact_name": role.name, "row_label": source_row.row_label,
                    "source_section": source_row.source_section,
                    "source_column": source_row.source_column,
                    "source_table_id": source_row.source_table_id,
                    "period": source_row.period, "unit": source_row.unit,
                    "source_quote": source_row.source_quote,
                    "concept": source_row.concept,
                })
                continue
            exact = _exact_numeric_label_concepts(_role_measure_labels(role), bundle)
            if len(exact) > 1 and not total_concept:
                failures.append(f"ambiguous_xbrl_measure:{role.name}")
                continue
            resolved = _resolve_xbrl_role_measure(role, bundle)
            if total_concept:
                kinds = {_actual_unit_kind(bundle.units.get(f.unit_ref, ()), f.unit_ref)
                         for f in bundle.facts_by_local_context.values()
                         if f.concept == total_concept and f.value is not None}
                unit = {"money": "USD millions", "shares": "shares", "per_share": "USD/shares"}.get(
                    next(iter(kinds), "") if len(kinds) == 1 else "")
                resolved = (unit, total_concept) if unit else None
            if resolved is None:
                skipped[role.name] = "no_registry_or_exact_filing_label"
                continue
            unit, concept = resolved
        probe = VerificationFact(
            name=role.name, value=0.0, unit=unit, period=role.period,
            row_label="; ".join(role.aliases) or _role_display_label(role), concept=concept,
            xbrl_context_id=role.xbrl_context_id if schema_plan else "",
        )
        # A broad role name must not turn a country/segment question into a
        # consolidated binding. Require the requested scope on the role itself.
        if not exact_children and any(scope not in requested_scopes(probe) for scope in question_scopes):
            skipped[role.name] = "question_scope_missing_from_role"
            continue
        placeholder_facts[role.name] = probe
    temp_ir = VerificationIR(
        metric=claim_spec.metric,
        formula=claim_spec.formula,
        facts=placeholder_facts,
        claimed_value=0.0,
        claim_unit=claim_spec.claim_unit,
        tolerance=claim_spec.tolerance,
        computed_unit=claim_spec.claim_unit,
        period=claim_spec.period,
    )

    if exact_children:
        bindings = _bind_ir_facts(temp_ir, bundle)
    else:
        bindings = [binding for fact in placeholder_facts.values()
                    if (binding := _bind_fact_local(fact, temp_ir, bundle, allow_label_fallback=False)) is not None]
    role_by_name = {role.name: role for role in roles}
    grounded: dict[str, GroundedFact] = dict(anchored)
    sources: dict[str, str] = {name: "independent_source_row" for name in anchored}
    for binding in bindings:
        name = str(binding.get("fact_name") or "")
        role = role_by_name.get(name)
        value = binding.get("xbrl_value")
        if role is None or value is None:
            continue
        multiplier = float(binding.get("binding_multiplier") or 1.0)
        # Keep the filing context's actual scope when a concise display label
        # omits it. A role suffix such as ``_us`` cannot supply this proof.
        context = bundle.contexts.get(str(binding.get("context_id") or ""))
        geographic_scope = ""
        if context is not None:
            for dimension in context.dimensions:
                axis, separator, member = dimension.partition("=")
                if (separator and member == "country:US" and axis.split(":")[-1] in {
                        "StatementGeographicalAxis", "GeographyAxis", "GeographicAxis", "GeographicalAxis",
                        "GeographicAreasAxis", "GeographicalAreasAxis", "CountryAxis", "CountriesAxis"}):
                    geographic_scope = "United States"
        grounded[name] = GroundedFact(
            name=name,
            value=float(value) * multiplier,
            unit=placeholder_facts[name].unit,
            source_quote="",
            period=role.period or _binding_period_label(binding, role.period),
            row_label=_role_display_label(role),
            fact_type="xbrl_numeric",
            concept=str(binding.get("concept") or role.concept),
            source_section=geographic_scope,
            xbrl_context_id=str(binding.get("context_id") or ""),
        )
        sources[name] = str(binding.get("source") or "")

    missing = [role.name for role in roles if role.name not in grounded]
    if any(len(concepts) > 1 for concepts in source_concepts.values()):
        failures.append("source_row_concept_changed_across_periods")
    diagnostics["xbrl_role_fact_binding"].update(
        {
            "status": "ok" if not missing else "partial",
            "bound": sorted(grounded),
            "missing": missing,
            "sources": sources,
            "skipped": skipped,
            "bindings": bindings + source_bindings,
            "source_witnesses": source_witnesses,
        }
    )
    if failures:
        diagnostics["xbrl_role_fact_binding"]["semantic_failure"] = failures[0]
        diagnostics["xbrl_role_fact_binding"]["semantic_failures"] = failures
    return grounded, diagnostics


def _role_measure_labels(role: RoleSpec) -> list[str]:
    return [re.sub(r"\s+(?:fy)?(?:19|20)\d{2}$", "", text.replace("_", " "), flags=re.I)
            for text in [role.name, *role.aliases]]


def _source_binding_column(source_column: str) -> str:
    """Keep annual table headers annual while retaining their raw provenance.

    "Year ended December 31, 2019" identifies a full annual column. Treating
    its date as a point measurement would also admit a fourth-quarter flow
    and would turn a rollforward's opening balance into its closing balance.
    An explicit as-of date without the annual-period heading stays precise.
    """
    if re.match(r"^\s*(?:for\s+the\s+)?(?:fiscal\s+)?years?\s+(?:ended|ending)\b",
                source_column or "", re.I):
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", source_column))
        if len(years) == 1:
            return next(iter(years))
    return source_column


def _requested_total_concept(role: RoleSpec, question: str, bundle: FilingXbrl) -> str:
    """An explicit total request may select the parent of a label collision."""
    if not _explicit_total_intent(role, question):
        return ""
    labels = _role_measure_labels(role)
    bare_labels = [re.sub(r"\btotal\s+", "", label, flags=re.I) for label in labels]
    concepts = _exact_numeric_label_concepts(bare_labels, bundle)
    parents = {group.parent_concept for group in bundle.calc_groups
               if group.parent_concept in concepts and
               any(child.concept in concepts - {group.parent_concept} for child in group.children)}
    return next(iter(parents)) if len(parents) == 1 else ""


def _explicit_total_intent(role: RoleSpec, question: str) -> bool:
    def words(text):
        text = re.sub(r"\bnon[ -]+current\b", "noncurrent", text.lower().replace("_", " "))
        return " ".join(re.findall(r"[a-z0-9]+", text))
    requested = words(question)
    for label in _role_measure_labels(role):
        bare = re.sub(r"\btotal\s+", "", words(label))
        if bare and re.search(r"\b(?:total(?: amount| value)?(?: of)?|sum of|all(?: of)?)\s+(?:the\s+)?"
                              + re.escape(bare) + r"\b", requested):
            return True
    return False


def _resolve_xbrl_role_measure(role: RoleSpec, bundle: FilingXbrl) -> tuple[str, str] | None:
    """Resolve the operand's unit and concept without using the proposed answer.

    Exact numeric filing labels outrank a broader registry name. Otherwise only
    registered role names are eligible; fuzzy label matches still need text extraction.
    """
    inferred = DEFAULT_POLICY_REGISTRY.infer_concepts(role.name)
    semantic = DEFAULT_POLICY_REGISTRY.concepts.get(inferred.get(role.name, ""))
    if semantic is None:
        from verifiqa.verification.xbrl_scope import requested_scopes, unscoped_measure_texts
        probe = VerificationFact(role.name, 0, "", row_label="; ".join(role.aliases))
        if requested_scopes(probe):
            ids = set(DEFAULT_POLICY_REGISTRY.infer_concepts(*unscoped_measure_texts(probe)).values())
            if len(ids) == 1:
                semantic = DEFAULT_POLICY_REGISTRY.concepts[next(iter(ids))]
    exact_concepts = _exact_numeric_label_concepts(_role_measure_labels(role), bundle)
    if len(exact_concepts) > 1:
        return None
    concept = next(iter(exact_concepts), "")
    unit_by_kind = {"money": "USD millions", "shares": "shares", "per_share": "USD/shares", "percent": "percent"}
    if concept:
        kinds = {_actual_unit_kind(bundle.units.get(f.unit_ref, ()), f.unit_ref)
                 for f in bundle.facts_by_local_context.values()
                 if f.concept == concept and f.value is not None}
        if len(kinds) != 1:
            return None
        kind = next(iter(kinds))
        if kind == "ratio" and semantic and semantic.unit_kind == "percent":
            kind = "percent"
        unit = unit_by_kind.get(kind)
        return (unit, concept) if unit else None
    if semantic is None:
        return None
    qualifiers = {"miscellaneous", "diluted", "basic", "current", "noncurrent", "foreign", "domestic",
                  "gross", "restricted", "unrestricted", "accrued", "vested", "nonvested"}
    semantic_words = set(re.findall(r"[a-z]+", " ".join(semantic.aliases).lower()))
    alias_words = set(re.findall(r"[a-z]+", " ".join(role.aliases).lower()))
    if (alias_words - semantic_words) & qualifiers:
        return None
    unit = unit_by_kind.get(semantic.unit_kind)
    return (unit, "") if unit else None


def _role_display_label(role: RoleSpec) -> str:
    return role.aliases[0] if role.aliases else role.name.replace("_", " ")


def _binding_period_label(binding: dict, fallback: str) -> str:
    context = binding.get("context_period")
    if isinstance(context, dict):
        start = str(context.get("start_date") or "")
        end = str(context.get("end_date") or "")
        instant = str(context.get("instant") or "")
        if start and end:
            return f"{start} to {end}"
        if instant:
            return instant
        if end:
            return end
    return fallback


def _normalize_monetary_facts(grounded: dict[str, GroundedFact]) -> None:
    from verifiqa.verification.money_units import money_currency
    for fact in grounded.values():
        money_scale = _money_scale_from_unit(fact.unit)
        currency = money_currency(fact.unit)
        if currency and money_scale is not None and money_scale != _MILLION:
            fact.value = float(Decimal(str(fact.value)) * Decimal(str(money_scale)) / Decimal(str(_MILLION)))
            fact.unit = currency + " millions"


def _claim_money_scale_factor(
    expected_unit: str,
    claim_spec: ClaimSpec,
    grounded: dict[str, GroundedFact],
) -> float:
    if expected_unit in {"percent", "ratio", "percentage_points", "days"}:
        return 1.0
    evidence_scale = _evidence_money_scale(grounded.values())
    requested_scale = _expected_unit_money_scale(expected_unit, claim_spec.claimed_unit)
    if (requested_scale is None and not claim_spec.claimed_unit
            and claim_spec.planning_source == "deterministic_source_table"):
        requested_scale = _money_scale_from_unit(claim_spec.claim_unit)
    if evidence_scale is not None and requested_scale is not None:
        return requested_scale / evidence_scale
    if (
        "billion" in (claim_spec.claimed_unit or "").lower()
        and grounded
        and all("million" in (f.unit or "").lower() for f in grounded.values() if f.unit)
    ):
        return 1000.0
    return 1.0


def _claim_quantity_scale_factor(answer: str, raw_claimed: float, grounded: dict[str, GroundedFact]) -> float | None:
    """Align an explicitly displayed share quantity to the operand scale.

    A source caption saying millions of shares does not mean a claim saying
    '0.9 shares' is 0.9 million. Read the unit next to the displayed claim;
    unqualified counts retain the base-share scale.
    """
    if not grounded or any("shares" not in fact.unit.lower() or "usd" in fact.unit.lower()
                           for fact in grounded.values()):
        return None
    def scale(unit):
        return next((value for word, value in _MONEY_SCALE.items() if word in unit.lower()), 1.0)
    operand_scales = {scale(fact.unit) for fact in grounded.values()}
    if len(operand_scales) != 1:
        raise SourceBindingConflict("inconsistent_source_share_scale")
    displayed_scales = set()
    for match in re.finditer(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w.])", answer):
        if abs(float(match.group().replace(",", "")) - raw_claimed) > 1e-9:
            continue
        suffix = answer[match.end():match.end() + 40].lower()
        unit = re.match(r"\s*(billion|million|thousand)(?:s)?\b", suffix)
        if unit:
            displayed_scales.add(_MONEY_SCALE[unit.group(1)])
        elif re.match(r"\s*shares?\b", suffix):
            displayed_scales.add(1.0)
    # Conflicting claims cannot acquire a favorable unit by picking a token.
    if len(displayed_scales) > 1:
        raise SourceBindingConflict("ambiguous_displayed_share_scale")
    claimed_scale = next(iter(displayed_scales), 1.0)
    return claimed_scale / next(iter(operand_scales))


def _authorize_generic_operation(
    ir: VerificationIR,
    question: str,
    claim_spec: ClaimSpec,
    *, evidence_text: str = "", source_pdf_path: str = "", source_xbrl_bundle=None,
) -> tuple[VerificationIR, dict, str]:
    if claim_spec.formula_source != "generic_operation":
        return ir, {}, ""

    # The classifier gives complete text/PDF plans priority over XML-schema
    # plans. Enriching a source cell with a validated QName does not change
    # that plan into an XML-context plan or require a pinned context ID.
    schema_plan = claim_spec.planning_source == "deterministic_xbrl_schema"
    from verifiqa.verification.named_metric_authority import (
        requested_named_metric_policy, resolve_named_metric_operation,
    )
    if not schema_plan and requested_named_metric_policy(question) is not None:
        # A named metric's complete target takes priority over an inner
        # arithmetic hint. Missing roles must never turn a turnover ratio
        # into an average of its denominator observations.
        resolved = resolve_named_metric_operation(ir, question, evidence_text=evidence_text)
    else:
        resolved = (None if schema_plan and source_xbrl_bundle is None else
                    resolve_generic_operation(ir, question, evidence_text=evidence_text, source_pdf_path=source_pdf_path,
                                              source_xbrl_bundle=source_xbrl_bundle if schema_plan else None))
    if resolved is None:
        return (
            ir,
            {
                "generic_operation_authority": {
                    "status": "failed",
                    "reason": "generic_operation_authority_unresolved",
                    "operation": claim_spec.operation or "",
                    "formula": ir.formula,
                }
            },
            "generic_operation_authority_unresolved",
        )

    revised = replace(ir, formula=str(resolved["formula"]))
    diagnostics = {
        "generic_operation_authority": {
            "status": "passed",
            "reason": f"generic_operation_resolved:{resolved['operation']}",
            "operation": resolved["operation"],
            "formula": resolved["formula"],
            "computed_value": resolved.get("computed_value"),
            "variables": resolved.get("variables", []),
            "source_note": resolved.get("source_note", ""),
        }
    }
    if resolved.get("policy_id"):
        diagnostics["generic_operation_authority"].update({
            "policy_id": resolved["policy_id"],
            "role_bindings": resolved.get("role_bindings", {}),
            "source_refs": resolved.get("source_refs", []),
        })
    return revised, diagnostics, ""


def _build_ir(
    claim_spec: ClaimSpec,
    facts: dict[str, GroundedFact],
    claimed_value: float,
    tolerance: float,
    precision_digits: int | None = None,
    tolerance_source: str = "",
) -> VerificationIR:
    vfacts = {
        name: VerificationFact(
            name=name,
            value=f.value,
            unit=f.unit,
            source_quote=f.source_quote,
            chunk_id="oracle",
            period=f.period,
            row_label=f.row_label,
            fact_type=f.fact_type,
            concept=getattr(f, "concept", "") or "",
            raw_value=getattr(f, "source_value", None),
            raw_unit=getattr(f, "source_unit", ""),
            column=_source_binding_column(getattr(f, "source_column", "")),
            source_section=getattr(f, "source_section", ""),
            source_column=getattr(f, "source_column", ""),
            source_table_id=getattr(f, "source_table_id", ""),
            xbrl_context_id=getattr(f, "xbrl_context_id", ""),
            source_ref=getattr(f, "source_ref", {}),
        )
        for name, f in facts.items()
    }
    # answer_spec returns an absolute tolerance in the displayed output unit.
    # Do not rescale it here; SMT enforces |computed - claimed| <= tolerance.
    return VerificationIR(
        metric=claim_spec.metric,
        formula=claim_spec.formula,
        facts=vfacts,
        claimed_value=claimed_value,
        claim_unit=claim_spec.claim_unit,
        tolerance=tolerance,
        computed_unit=claim_spec.claim_unit,
        precision_digits=precision_digits,
        tolerance_source=tolerance_source,
        period=claim_spec.period,
    )


def _agent_diagnostics(base: dict, unsat_fact_retries: list[dict]) -> dict:
    diagnostics = dict(base or {})
    diagnostics["unsat_fact_retries"] = unsat_fact_retries
    return diagnostics


def _merge_diagnostics(*diagnostics: dict) -> dict:
    merged: dict = {}
    for diagnostic in diagnostics:
        if diagnostic:
            merged.update(diagnostic)
    return merged


_XBRL_PARENT_ALIASES = {
    "GrossProfit": ["gross profit"],
    "OperatingExpenses": ["operating expenses", "total operating expenses"],
    "OperatingIncomeLoss": ["operating income"],
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": [
        "income before income taxes",
        "income before taxes",
    ],
    "NetIncomeLoss": ["net income"],
    "Assets": ["assets", "total assets"],
    "AssetsCurrent": ["current assets", "total current assets"],
    "AssetsNoncurrent": ["non-current assets", "noncurrent assets", "total non-current assets"],
    "Liabilities": ["liabilities", "total liabilities"],
    "LiabilitiesCurrent": ["current liabilities", "total current liabilities"],
    "LiabilitiesNoncurrent": ["non-current liabilities", "noncurrent liabilities", "total non-current liabilities"],
    "StockholdersEquity": ["shareholders equity", "stockholders equity", "total shareholders equity"],
    "PropertyPlantAndEquipmentNet": ["net property plant and equipment", "property plant and equipment net"],
    "NetCashProvidedByUsedInOperatingActivities": ["net cash from operating activities"],
    "NetCashProvidedByUsedInInvestingActivities": ["net cash from investing activities"],
    "NetCashProvidedByUsedInFinancingActivities": ["net cash from financing activities"],
}


def _maybe_authorize_xbrl_linkbase_claim(
    claim_spec: ClaimSpec,
    question: str,
    evidence_text: str,
    doc_name: str,
    artifacts_dir: Optional[Path],
) -> tuple[ClaimSpec, dict]:
    """Use filing calc-linkbase authority for redacted subtotal questions.

    The classifier correctly treats "what was total X" as a lookup in normal
    FinanceBench. In the calc-required provable dataset the target subtotal is
    redacted, so the authoritative formula is the filing's XBRL calculation
    linkbase: parent subtotal = weighted sum of child concepts.
    """
    if not artifacts_dir or not doc_name:
        return claim_spec, {}
    if "[redacted]" not in (evidence_text or ""):
        return claim_spec, {}

    bundle = load_filing_xbrl(doc_name, artifacts_dir)
    if bundle.status != "ok":
        return claim_spec, {"xbrl_claimspec_status": bundle.status}

    targets = _xbrl_target_phrases(question, evidence_text)
    if not targets:
        return claim_spec, {"xbrl_claimspec_status": "no_target_phrase"}

    label_aliases = _xbrl_label_aliases(bundle)
    group, score = _best_xbrl_calc_group(bundle.calc_groups, targets, evidence_text, label_aliases)
    if group is None:
        return claim_spec, {
            "xbrl_claimspec_status": "no_matching_calc_parent",
            "xbrl_claimspec_targets": targets,
        }
    if len(group.children) < 2:
        return claim_spec, {"xbrl_claimspec_status": "matched_parent_without_children"}

    fiscal_year = _question_fiscal_year(question) or _doc_fiscal_year(doc_name) or claim_spec.period
    roles: list[RoleSpec] = []
    role_names: list[str] = []
    used_names: set[str] = set()
    for child in group.children:
        role_name = _unique_role_name(
            _xbrl_role_name(child.local_name, fiscal_year),
            used_names,
        )
        used_names.add(role_name)
        role_names.append(role_name)
        roles.append(
            RoleSpec(
                name=role_name,
                aliases=_xbrl_aliases(child.local_name, label_aliases),
                period=_xbrl_current_column_period(fiscal_year),
                concept=child.concept,  # authoritative: bind straight to this qname
            )
        )

    revised = replace(
        claim_spec,
        metric=_xbrl_role_name(group.parent_local, fiscal_year),
        formula=_xbrl_formula(role_names, [float(child.weight) for child in group.children]),
        roles=roles,
        formula_source="xbrl_linkbase",
        period=f"FY{fiscal_year}" if fiscal_year else claim_spec.period,
        operation="xbrl_linkbase",
    )
    return revised, {
        "xbrl_claimspec_status": "ok",
        "xbrl_claimspec_parent": group.parent_local,
        "xbrl_claimspec_role": group.role,
        "xbrl_claimspec_match_score": score,
        "xbrl_claimspec_targets": targets,
        "xbrl_claimspec_children": [child.local_name for child in group.children],
    }


def _xbrl_target_phrases(question: str, evidence_text: str) -> list[str]:
    targets: list[str] = []
    for line in (evidence_text or "").splitlines():
        if "[redacted]" not in line:
            continue
        label = line.split("[redacted]", 1)[0].strip()
        if label:
            targets.append(label)

    q = question or ""
    match = re.search(
        r"\bwhat\s+(?:is|was|were|are)\s+(?:.+?'s\s+)?(?:fy\s*)?(?:19|20)?\d{0,2}\s*(?P<target>.+?)(?:,\s+in\b| in usd\b| round\b|\?)",
        q,
        flags=re.I,
    )
    if match:
        targets.append(match.group("target"))
    return _dedupe_preserve_order([t for t in (_clean_xbrl_phrase(t) for t in targets) if t])


def _xbrl_current_column_period(fiscal_year: str | None = None) -> str:
    label = f" for FY{fiscal_year}" if fiscal_year else ""
    return (
        "current filing period"
        f"{label}: use the first/leftmost numeric column in this evidence table; "
        "do not choose a later column merely because its calendar-year header matches the fiscal-year label"
    )


def _best_xbrl_calc_group(
    calc_groups,
    targets: list[str],
    evidence_text: str,
    label_aliases: dict[str, list[str]],
):
    evidence_norm = _normalize_xbrl_text(evidence_text)
    scored = []
    for group in calc_groups:
        if len(group.children) < 2:
            continue
        aliases = _xbrl_aliases(group.parent_local, label_aliases) + _XBRL_PARENT_ALIASES.get(group.parent_local, [])
        score = max(
            (_xbrl_match_score(target, alias) for target in targets for alias in aliases),
            default=0,
        )
        if score <= 0:
            continue
        child_hits = sum(
            1
            for child in group.children
            if any(
                _normalize_xbrl_text(alias) in evidence_norm
                for alias in _xbrl_aliases(child.local_name, label_aliases)
            )
        )
        child_affinity = sum(
            max(
                (
                    _xbrl_match_score(target, alias)
                    for target in targets
                    for alias in _xbrl_aliases(child.local_name, label_aliases)
                ),
                default=0,
            )
            for child in group.children
        )
        scored.append((score, child_hits, child_affinity, -len(group.children), group))
    if not scored:
        return None, 0
    scored.sort(key=lambda item: item[:4], reverse=True)
    best = scored[0]
    if best[0] < 30:
        return None, best[0]
    if len(scored) > 1 and scored[1][:4] == best[:4]:
        return None, best[0]
    return best[4], best[0]


def _xbrl_match_score(target: str, alias: str) -> int:
    t = _normalize_xbrl_text(target)
    a = _normalize_xbrl_text(alias)
    if not t or not a:
        return 0
    if t == a:
        return 100
    if t in a or a in t:
        return 75 + min(len(t.split()), len(a.split()))
    t_tokens = set(t.split())
    a_tokens = set(a.split())
    if not t_tokens or not a_tokens:
        return 0
    overlap = t_tokens & a_tokens
    if len(overlap) < 2:
        return 0
    return len(overlap) * 10 - len(t_tokens ^ a_tokens)


def _xbrl_formula(role_names: list[str], weights: list[float]) -> str:
    terms = []
    for name, weight in zip(role_names, weights):
        if weight == 1.0:
            term = name
        elif weight == -1.0:
            term = f"(-1 * {name})"
        else:
            term = f"({weight:g} * {name})"
        terms.append(term)
    return " + ".join(terms)


def _xbrl_role_name(local_name: str, fiscal_year: str | None = None) -> str:
    base = _snake(_camel_words(local_name))
    if fiscal_year:
        return f"{base}_fy{fiscal_year}"
    return base


def _unique_role_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    index = 2
    while f"{name}_{index}" in used:
        index += 1
    return f"{name}_{index}"


def _xbrl_aliases(local_name: str, label_aliases: dict[str, list[str]] | None = None) -> list[str]:
    words = _camel_words(local_name)
    aliases = {words, _normalize_xbrl_text(words)}
    aliases.update(_XBRL_PARENT_ALIASES.get(local_name, []))
    aliases.update((label_aliases or {}).get(local_name, []))
    return _dedupe_preserve_order([alias for alias in aliases if alias])


def _xbrl_label_aliases(bundle) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for concept_qname, labels in (bundle.concept_labels or {}).items():
        local = concept_qname.rsplit(":", 1)[-1]
        aliases.setdefault(local, []).extend(labels)
    if not bundle.cik or bundle.edgar_dir is None:
        return aliases
    path = bundle.edgar_dir / f"CIK{int(bundle.cik):010d}.json"
    if not path.exists():
        return aliases
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return aliases
    for local, concept_data in (data.get("facts", {}).get("us-gaap", {}) or {}).items():
        label = str(concept_data.get("label") or "").strip()
        if label:
            aliases.setdefault(local, []).append(label)
    return {key: _dedupe_preserve_order(values) for key, values in aliases.items()}


def _question_fiscal_year(question: str) -> str:
    match = re.search(r"\bFY\s*((?:19|20)\d{2})\b", question or "", flags=re.I)
    return match.group(1) if match else ""


def _doc_fiscal_year(doc_name: str) -> str:
    match = re.search(r"_(\d{4})_10K$", doc_name or "")
    return match.group(1) if match else ""


def _clean_xbrl_phrase(value: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", value or "")
    text = re.sub(r"\b(?:fy|fiscal year|fiscal)\s*(?:19|20)\d{2}\b", " ", text, flags=re.I)
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = re.sub(r"\b(?:usd|dollars?|millions?|billions?|thousands?|round|nearest|million)\b", " ", text, flags=re.I)
    text = re.sub(r"\b(?:in|to|the|a|an)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_xbrl_text(value: str) -> str:
    text = _camel_words(value)
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(?:the|a|an|total)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _camel_words(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _snake(value: str) -> str:
    text = _normalize_xbrl_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "xbrl_fact"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _parse_number(text: str) -> float | None:
    """Parse the first answer-like numeric value, preserving sign.

    Financial answers often mention company names, fiscal years, quarters, or
    security tickers before the actual claim value. Skip those context tokens so
    "3M in FY2022 decreased by 1.7%" parses as 1.7, not 2022.
    """
    for match in re.finditer(r"(-?)\$?(\d[\d,]*(?:\.\d+)?|\.\d+)\s*%?", text or ""):
        sign, digits = match.group(1), match.group(2)
        if _is_context_number(text or "", match, digits):
            continue
        try:
            value = float(digits.replace(",", ""))
            return -value if sign == "-" else value
        except ValueError:
            continue
    return None


def _is_context_number(text: str, match: re.Match[str], digits: str) -> bool:
    start, end = match.span()
    before = text[start - 1:start]
    after = text[end:end + 1]
    matched = match.group(0)
    has_decimal = "." in digits
    has_percent = "%" in matched
    has_currency = "$" in matched

    # Embedded labels/tickers/dates: 3M, Q3, FY2022, Jun'23, MMM26.
    if before and (before.isalpha() or before in {"'", "_"}):
        return True
    if after and after.isupper() and not has_decimal:
        return True

    # Standalone calendar years are usually context, not the claimed metric.
    clean_digits = digits.replace(",", "")
    if (
        not has_decimal
        and not has_percent
        and not has_currency
        and clean_digits.isdigit()
        and 1900 <= int(clean_digits) <= 2099
    ):
        return True

    return False


def _facts_in_core(unsat_core: list[str], facts: dict[str, GroundedFact]) -> list[str]:
    result = []
    for assertion in unsat_core:
        for name in facts:
            if assertion == f"evidence_{name}" and name not in result:
                result.append(name)
                continue
            # Use word-boundary match to avoid 'revenue' matching 'revenue_2022'
            if re.search(r"\b" + re.escape(name) + r"\b", assertion) and name not in result:
                result.append(name)
    return result


def _retry_fact_names(unsat_core: list[str], facts: dict[str, GroundedFact]) -> list[str]:
    names = _facts_in_core(unsat_core, facts)
    return names if names else sorted(facts)


def _computed_formula_value(
    formula: str,
    facts: dict[str, GroundedFact],
) -> float | None:
    try:
        return float(evaluate_formula(formula, {name: fact.value for name, fact in facts.items()}))
    except Exception:
        return None


def _same_fact(left: GroundedFact | None, right: GroundedFact | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        abs(left.value - right.value) <= 1e-12
        and (left.unit or "") == (right.unit or "")
        and (left.period or "") == (right.period or "")
        and (left.row_label or "") == (right.row_label or "")
        and (left.source_quote or "") == (right.source_quote or "")
    )


def _normalize_text(text: str) -> str:
    """Collapse whitespace and strip formatting for fuzzy quote matching."""
    text = re.sub(r"[\$,()]", "", text)   # remove, not replace — avoids splitting numbers
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_grounded(source_quote: str, norm_evidence: str) -> bool:
    """Return True if source_quote appears in norm_evidence (pre-normalized by caller).

    Handles PDF artifact patterns:
    1. Line breaks: 'Revenue\\n40,339' vs 'Revenue 40,339'
    2. Short quotes: 'Revenue $ 40,339' → 2 tokens after normalization
    3. Word merges: 'Total net revenue' vs 'Totalnetrevenue'
    4. Multi-column tables: value not adjacent to label
    """
    if not source_quote:
        return True
    norm_quote = _normalize_text(source_quote)

    # Pass 1: full substring match
    if norm_quote in norm_evidence:
        return True

    # Pass 2: first 6 tokens (handles line-break artifacts; works for ≥2 tokens)
    tokens = norm_quote.split()
    if len(tokens) >= 2:
        fragment = " ".join(tokens[:6])
        if fragment in norm_evidence:
            return True

    # Pass 3: whitespace-stripped match (handles PDF word-merge artifacts like
    # 'Totalnetrevenue' where the PDF runs words together without spaces)
    collapsed_quote = re.sub(r"\s+", "", norm_quote)
    collapsed_evidence = re.sub(r"\s+", "", norm_evidence)
    if len(collapsed_quote) >= 8 and collapsed_quote in collapsed_evidence:
        return True

    # Pass 4: numeric value appears in evidence (multi-column tables, any remaining PDF
    # artifacts). Label adjacency not required — Z3 catches wrong values anyway.
    nums = [n for n in re.findall(r"\d+", norm_quote) if len(n) >= 3]
    if nums and all(n in norm_evidence for n in nums):
        return True

    return False
