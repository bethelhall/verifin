from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from verifiqa.answer_spec import answer_spec_from_question, apply_answer_spec, apply_output_contract
from verifiqa.formalization.certificate_validator import (
    CertificateValidationResult,
    CertificateValidator,
    certificate_provenance,
)
from verifiqa.formalization.formalizer import (
    Formalizer,
    certificate_to_claim_schema,
    prune_unused_certificate_facts,
)
from verifiqa.formulas.evaluator import FormulaError, evaluate_formula, formula_variables, result_is_money
from verifiqa.generation.answer_generator import AnswerGenerator
from verifiqa.policy import PolicySemanticChecker
from verifiqa.question_filter import is_numerical_example
from verifiqa.retrieval.evidence_retriever import EvidenceRetriever
from verifiqa.retrieval.planner import RetrievalPlanner
from verifiqa.types import (
    ABSTENTION_MESSAGE,
    VIOLATION_MESSAGE,
    CertificateClaim,
    CertificateFact,
    Claim,
    DiagnosticTrace,
    EvidenceChunk,
    FinanceBenchExample,
    RavResult,
    RetrievalFact,
    RetrievalPlan,
    SolverResult,
    VerificationCertificate,
    VerificationIR,
    VerificationSchema,
)
from verifiqa.units import normalize_certificate_quantities, _is_money_unit
from verifiqa.reconciler import ReconcilerRunner
from verifiqa.verification.finqa_program import resolve_finqa_program
from verifiqa.verification.generic_ops import resolve_generic_operation
from verifiqa.verification.smt_generator import SmtGenerator, render_ir_smt
from verifiqa.verification.smt_sanitizer import SmtSanitizer
from verifiqa.verification.ir import claim_schema_from_ir, ir_from_certificate
from verifiqa.verification.document_grounding import attach_document_calculations
from verifiqa.verification.finqa_table import attach_table_calculations
from verifiqa.verification.policy_formula import augment_smt_with_policy_formula, resolve_policy_formula
from verifiqa.verification.xbrl_linkbase import attach_xbrl_calculations, augment_smt_with_xbrl
from verifiqa.verification.z3_runner import Z3Runner
from verifiqa.verification.nongaap_reconciler import NonGaapReconciler


@dataclass
class PipelineConfig:
    evidence_top_k: int = 5
    oracle_retrieval: bool = False
    numeric_only: bool = True
    adaptive_retrieval: bool = False
    verify_original_answer: bool = False
    allow_sat_repair: bool = True
    max_repair_rounds: int = 3
    policy_semantic_check: bool = False
    certificate_grounding_check: bool = True
    # Authorize company-specific non-GAAP metrics (no registry policy) against the
    # filing's own Reg-G reconciliation total when the registry has no formula.
    document_reconciliation: bool = True
    # Use the FinQA gold reasoning program as formula authority. This is a dataset
    # ANNOTATION (the answer's derivation), not deployment-available external
    # knowledge — an oracle condition. Set False for a deployment-realistic regime
    # where FinQA is authorized only by the standard formula registry, like
    # FinanceBench. Default True preserves the program-authority (oracle) behavior.
    use_finqa_program: bool = True
    use_generic_operations: bool = True
    xbrl_calculation_dir: Optional[Path] = None
    xbrl_llm_binding: bool = False
    include_unsat_core: bool = True
    verification_mode: str = "current"
    numeric_only_verification: bool = False


@dataclass
class _FormalizationPass:
    answer: str
    certificate: VerificationCertificate
    claim: Claim
    schema: VerificationSchema
    verification_ir: VerificationIR
    verified_answer: str
    validation_answer: str
    provenance: dict
    certificate_validation: CertificateValidationResult
    output_contract: dict


class CounterexampleRavPipeline:
    """
    Minimal consistency-checking pipeline:

      1. RAG retrieves relevant filing text.
      2. Answer LLM answers from that text.
      3. Formalizer LLM emits an evidence-grounded verification certificate.
      4. Certificate validator checks fact spans, chunk ids, units, and claim support.
      5. The certificate becomes a typed VerificationIR.
      6. SMT LLM writes a counterexample query from the IR.
      7. IR semantic validation + Z3 return UNSAT for a consistent model, otherwise abstain.
    """

    def __init__(
        self,
        evidence_retriever: EvidenceRetriever,
        answer_generator: AnswerGenerator,
        formalizer: Formalizer,
        smt_generator: Optional[SmtGenerator] = None,
        retrieval_planner: Optional[RetrievalPlanner] = None,
        certificate_validator: Optional[CertificateValidator] = None,
        policy_checker: Optional[PolicySemanticChecker] = None,
        sanitizer: Optional[SmtSanitizer] = None,
        z3_runner: Optional[Z3Runner] = None,
        reconciler_runner: Optional[ReconcilerRunner] = None,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[Callable[[str, dict], None]] = None,
        evidence_callback: Optional[Callable[[FinanceBenchExample, Optional[RetrievalPlan], list[EvidenceChunk], str], None]] = None,
        **_legacy_kwargs,
    ):
        self.evidence_retriever = evidence_retriever
        self.answer_generator = answer_generator
        self.formalizer = formalizer
        self.retrieval_planner = retrieval_planner
        self.certificate_validator = certificate_validator or CertificateValidator()
        self.policy_checker = policy_checker or PolicySemanticChecker()
        if smt_generator is None:
            raise ValueError("smt_generator_required")
        self.smt_generator = smt_generator
        self.sanitizer = sanitizer or SmtSanitizer()
        self.z3_runner = z3_runner or Z3Runner()
        self.reconciler_runner = reconciler_runner
        self.config = config or PipelineConfig()
        self.reconciler = (
            NonGaapReconciler(getattr(self.formalizer, "llm_client", None))
            if self.config.document_reconciliation and getattr(self.formalizer, "llm_client", None) is not None
            else None
        )
        self.progress_callback = progress_callback
        self.evidence_callback = evidence_callback

    def run_example(self, example: FinanceBenchExample) -> RavResult:
        if (
            self.config.numeric_only
            and not _has_finqa_program(example)
            and not is_numerical_example(example)
        ):
            self._record_evidence(example, None, [], "skipped_non_numerical_question")
            return self._abstain_early(example, "non_numerical_question")

        if (
            not self.config.oracle_retrieval
            and example.doc_name
            and hasattr(self.evidence_retriever, "has_document")
            and not self.evidence_retriever.has_document(example.doc_name)
        ):
            self._record_evidence(example, None, [], f"missing_document:{example.doc_name}")
            return self._abstain_early(example, f"missing_document:{example.doc_name}")

        retrieval_plan = None
        if self.retrieval_planner is not None:
            self._progress("plan_retrieval")
            try:
                retrieval_plan = self.retrieval_planner.plan(example.question)
            except Exception as exc:
                self._record_evidence(
                    example,
                    None,
                    [],
                    f"retrieval_planning_failed:{type(exc).__name__}",
                )
                return self._abstain_early(
                    example,
                    f"retrieval_planning_failed:{type(exc).__name__}:{exc}",
                )
            self._progress("plan_retrieval", facts=len(retrieval_plan.facts), metric=retrieval_plan.metric)

        self._progress("retrieve_evidence")

        # Oracle mode: build EvidenceChunks directly from the gold evidence passages
        # embedded in the FinanceBench example, bypassing the corpus entirely.
        if self.config.oracle_retrieval and example.evidence:
            chunks = _oracle_chunks_from_example(example)
        elif retrieval_plan is not None:
            chunks = self.evidence_retriever.retrieve_with_plan(
                example.question,
                retrieval_plan,
                company=example.company,
                doc_name=example.doc_name,
                top_k=self.config.evidence_top_k,
            )
        else:
            chunks = self.evidence_retriever.retrieve(
                example.question,
                company=example.company,
                doc_name=example.doc_name,
                top_k=self.config.evidence_top_k,
            )
        retrieved_chunk_ids = [c.chunk_id for c in chunks]
        if not chunks:
            self._record_evidence(example, retrieval_plan, [], "no_retrieved_evidence")
            return self._abstain_early(
                example,
                "no_retrieved_evidence",
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieval_plan=retrieval_plan,
            )

        self._record_evidence(example, retrieval_plan, chunks, "retrieved")
        try:
            pass_result = self._formalization_pass(example, chunks, retrieval_plan)
        except Exception as exc:
            error = str(exc)
            error = error if error.startswith("answer_generation_failed:") else f"formalization_failed:{type(exc).__name__}:{exc}"
            retry = self._adaptive_retrieval_retry(
                example,
                retrieval_plan,
                chunks,
                error,
            )
            if retry is None:
                return self._abstain_early(
                    example,
                    error,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieval_plan=retrieval_plan,
                )
            chunks, pass_result = retry
            retrieved_chunk_ids = [c.chunk_id for c in chunks]

        answer = pass_result.answer
        certificate = pass_result.certificate
        claim = pass_result.claim
        schema = pass_result.schema
        verification_ir = pass_result.verification_ir
        verified_answer = pass_result.verified_answer
        provenance = pass_result.provenance
        certificate_validation = pass_result.certificate_validation
        output_contract = pass_result.output_contract
        verification_checks = _initial_verification_checks(self.config.policy_semantic_check)
        verification_checks["output_contract"] = output_contract

        if not certificate_validation.valid:
            retry = self._adaptive_retrieval_retry(
                example,
                retrieval_plan,
                chunks,
                certificate_validation.reason,
                certificate=certificate,
            )
            if retry is not None:
                chunks, pass_result = retry
                retrieved_chunk_ids = [c.chunk_id for c in chunks]
                answer = pass_result.answer
                certificate = pass_result.certificate
                claim = pass_result.claim
                schema = pass_result.schema
                verification_ir = pass_result.verification_ir
                verified_answer = pass_result.verified_answer
                provenance = pass_result.provenance
                certificate_validation = pass_result.certificate_validation
                output_contract = pass_result.output_contract
                verification_checks["output_contract"] = output_contract

        if not certificate_validation.valid:
            reason = f"certificate_validation_failed:{certificate_validation.reason}"
            self._progress("certificate_grounding_failed_continue_to_smt", reason=reason)
            verification_checks["certificate_grounding"] = {
                "enabled": True,
                "status": "failed",
                "valid": False,
                "reason": reason,
            }
        else:
            verification_checks["certificate_grounding"] = {
                "enabled": self.config.certificate_grounding_check,
                "status": "passed" if self.config.certificate_grounding_check else "skipped",
                "valid": True if self.config.certificate_grounding_check else None,
                "reason": "certificate_grounded" if self.config.certificate_grounding_check else certificate_validation.reason,
            }

        formula_authority = _formula_authority_result(
            example,
            verification_ir,
            schema,
            self.policy_checker.registry,
            certificate_validation,
            ignore_unit_matching=self.config.numeric_only_verification,
            reconciler=self.reconciler,
            chunks=chunks,
            use_finqa_program=self.config.use_finqa_program,
            use_generic_operations=self.config.use_generic_operations,
        )
        verification_checks["formula_authority"] = formula_authority["check"]
        formula_authority_valid = bool(formula_authority["valid"])
        if not formula_authority_valid:
            reason = formula_authority["check"].get("reason", "formula_authority_failed")
            # Couldn't authorize the formula -> fall back to the grounded SMT rather than
            # force a (false) violation. Forcing UNSAT here conflates "no authoritative
            # formula" with "wrong answer" and false-violates every metric without a policy.
            forces_violation = _authority_failure_forces_violation(reason)
            self._progress(
                "formula_authority_failed_continue_to_smt",
                reason=reason,
                solver_enforced=forces_violation,
            )
            verification_checks["formula_authority"]["solver_enforced"] = forces_violation
            for key in ("semantic_validity", "policy_alignment"):
                if key in verification_checks and verification_checks[key].get("status") == "pending":
                    verification_checks[key]["status"] = "skipped"
                    verification_checks[key]["valid"] = None
                    verification_checks[key]["reason"] = f"not_reached:{reason}"
        else:
            verification_ir = formula_authority.get("verification_ir") or verification_ir
            claim, schema = claim_schema_from_ir(verification_ir)

        reconciler_shadow = self._run_reconciler_shadow(example, verification_ir)
        if reconciler_shadow is not None:
            verification_checks["reconciler_shadow"] = reconciler_shadow

        policy_validation = None
        policy = None
        if self.config.policy_semantic_check and formula_authority["valid"]:
            self._progress("policy_semantic_check", metric=schema.metric)
            policy_validation = self.policy_checker.validate_ir(
                verification_ir, question=example.question,
                evidence_text="\n".join(chunk.text for chunk in chunks),
            )
            policy = self.policy_checker.registry.find_policy(verification_ir.metric)
            verification_checks.update(_checks_from_policy_validation(policy_validation, policy))
            if not policy_validation.valid:
                reason = f"policy_semantic_failed:{policy_validation.reason}"
                self._progress("policy_semantic_failed_abstain", reason=reason)
                verification_checks["mathematical_validity"] = _math_check(
                    status="skipped",
                    valid=None,
                    reason=f"not_reached:{reason}",
                    solver_status="INVALID",
                )
                verification_checks = _finalize_unreached_checks(verification_checks, reason)
                diagnostics = _diagnostics(
                    "INVALID",
                    claim,
                    schema,
                    error=reason,
                    evidence_provenance=provenance,
                )
                return self._abstain(
                    example,
                    "INVALID",
                    claim,
                    schema,
                    "",
                    diagnostics,
                    certificate,
                    retrieved_chunk_ids,
                    retrieval_plan,
                    verification_ir,
                    verification_checks=verification_checks,
                    verifier_certificate=_verifier_certificate(
                        example=example,
                        final_status="ABSTAIN",
                        answer=ABSTENTION_MESSAGE,
                        claim=claim,
                        schema=schema,
                        ir=verification_ir,
                        certificate=certificate,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        retrieval_plan=retrieval_plan,
                        validation=None,
                        solver=None,
                        verification_checks=verification_checks,
                        smtlib="",
                        failure_reason=reason,
                    ),
                )

        self._progress("generate_smt", metric=schema.metric, claimed_value=claim.claimed_value)
        try:
            base_smtlib, smtlib = _generate_smt(self.smt_generator, verification_ir, claim, schema)
        except Exception as exc:
            verification_checks["mathematical_validity"] = _math_check(
                status="failed",
                valid=False,
                reason=f"smt_generation_failed:{type(exc).__name__}:{exc}",
                smt_status="not_generated",
                solver_status="INVALID",
            )
            diagnostics = _diagnostics(
                "INVALID",
                claim,
                schema,
                error=f"smt_generation_failed:{type(exc).__name__}:{exc}",
                evidence_provenance=provenance,
            )
            return self._abstain(
                example,
                "INVALID",
                claim,
                schema,
                "",
                diagnostics,
                certificate,
                retrieved_chunk_ids,
                retrieval_plan,
                verification_ir,
                verification_checks=verification_checks,
            )

        self._progress("sanitize_smt")
        validation = self.sanitizer.validate_ir(smtlib, verification_ir)
        if validation.smt_status != "valid":
            verification_checks["mathematical_validity"] = _math_check(
                status="failed",
                valid=False,
                reason=validation.reason,
                smt_status=validation.smt_status,
                solver_status="INVALID",
            )
            diagnostics = _diagnostics(
                "INVALID",
                claim,
                schema,
                error=validation.reason,
                evidence_provenance=provenance,
            )
            return self._abstain(
                example,
                "INVALID",
                claim,
                schema,
                smtlib,
                diagnostics,
                certificate,
                retrieved_chunk_ids,
                retrieval_plan,
                verification_ir,
                verification_checks=verification_checks,
                verifier_certificate=_verifier_certificate(
                    example=example,
                    final_status="ABSTAIN",
                    answer=ABSTENTION_MESSAGE,
                    claim=claim,
                    schema=schema,
                    ir=verification_ir,
                    certificate=certificate,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieval_plan=retrieval_plan,
                    validation=validation,
                    solver=None,
                    verification_checks=verification_checks,
                    smtlib=smtlib,
                    failure_reason=validation.reason,
                ),
            )

        consistency_solver = None
        xbrl_conflict = False
        if _needs_xbrl_consistency_check(verification_ir):
            self._progress("z3_consistency_check")
            consistency_smt = _strip_claim_assertions(smtlib)
            consistency_solver = self.z3_runner.run(
                consistency_smt,
                label=f"{example.financebench_id or 'example'}_consistency",
                consistency_only=True,
            )
            if consistency_solver.solver_status == "UNSAT":
                # The filing's XBRL facts contradict the LLM-extracted evidence.
                # The filing is authoritative: drop the LLM values for bound facts and
                # run the claim check against the XBRL-grounded SMT. If the claim is
                # inconsistent with the filing, the normal UNSAT -> answer-repair path
                # recomputes the correct value from the filing instead of trusting the LLM.
                xbrl_conflict = True
                self._progress("xbrl_linkbase_conflict", resolution="recompute_from_xbrl")
                verification_checks["xbrl_consistency"] = {
                    "status": "conflict",
                    "solver_status": "UNSAT",
                    "reason": "xbrl_linkbase_conflict",
                    "resolution": "recompute_from_xbrl",
                }
            elif consistency_solver.solver_status != "SAT":
                reason = f"xbrl_consistency_{consistency_solver.solver_status.lower()}"
                self._progress("abstain", solver_status=consistency_solver.solver_status, reason=reason)
                verification_checks["mathematical_validity"] = _math_check(
                    status="failed",
                    valid=False,
                    smt_status=validation.smt_status,
                    solver_status=consistency_solver.solver_status,
                    reason=reason,
                )
                return self._abstain(
                    example,
                    consistency_solver.solver_status,
                    claim,
                    schema,
                    smtlib,
                    _diagnostics(
                        consistency_solver.solver_status,
                        claim,
                        schema,
                        error=reason,
                        solver=consistency_solver,
                        evidence_provenance=provenance,
                    ),
                    certificate=certificate,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieval_plan=retrieval_plan,
                    verification_ir=verification_ir,
                    verification_checks=verification_checks,
                    verifier_certificate=_verifier_certificate(
                        example=example,
                        final_status="ABSTAIN",
                        answer=ABSTENTION_MESSAGE,
                        claim=claim,
                        schema=schema,
                        ir=verification_ir,
                        certificate=certificate,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        retrieval_plan=retrieval_plan,
                        validation=validation,
                        solver=None,
                        consistency_solver=consistency_solver,
                        verification_checks=verification_checks,
                        smtlib=smtlib,
                        failure_reason=reason,
                    ),
                )
        else:
            self._progress("z3_consistency_check", status="skipped_no_xbrl_constraints")

        # When a fact is independently bound to an authoritative source (XBRL or
        # a FinQA table cell), strip the LLM value assertion for that fact so the
        # claim is checked against the source value. Unbound facts keep their LLM
        # evidence assertions.
        grounded_fact_recompute = _has_grounded_fact_bindings(verification_ir)
        if grounded_fact_recompute:
            smtlib = _strip_llm_evidence_for_bound_facts(smtlib, verification_ir)
        formula_authority_solver_enforced = bool(
            verification_checks.get("formula_authority", {}).get("solver_enforced")
        )
        if formula_authority_solver_enforced:
            smtlib = _augment_smt_with_formula_authority_failure(
                smtlib,
                verification_checks["formula_authority"],
            )
        output_contract_solver_enforced = bool(
            verification_checks.get("output_contract", {}).get("solver_enforced")
        )
        if output_contract_solver_enforced:
            smtlib = _augment_smt_with_output_contract_failure(
                smtlib,
                verification_checks["output_contract"],
            )
        self._progress(
            "z3_claim_check",
            xbrl_conflict_recompute=xbrl_conflict,
            grounded_fact_recompute=grounded_fact_recompute,
            formula_authority_solver_enforced=formula_authority_solver_enforced,
            output_contract_solver_enforced=output_contract_solver_enforced,
        )
        solver = self.z3_runner.run(
            smtlib,
            label=f"{example.financebench_id or 'example'}_first",
            debug_unsat_core=self.config.include_unsat_core,
        )
        if solver.solver_status == "SAT":
            if not formula_authority_valid:
                reason = _formula_authority_unverified_reason(verification_checks)
                self._progress("abstain", solver_status="SAT", reason=reason)
                verification_checks["mathematical_validity"] = _math_check(
                    status="passed_unauthorized_formula",
                    valid=None,
                    reason=reason,
                    smt_status=validation.smt_status,
                    solver_status=solver.solver_status,
                )
                diagnostics = _diagnostics(
                    "SAT",
                    claim,
                    schema,
                    error=reason,
                    solver=solver,
                    evidence_provenance=provenance,
                )
                return self._unverified_formula(
                    example,
                    "SAT",
                    claim,
                    schema,
                    smtlib,
                    diagnostics,
                    certificate,
                    retrieved_chunk_ids,
                    retrieval_plan,
                    verification_ir,
                    verification_checks=verification_checks,
                    verifier_certificate=_verifier_certificate(
                        example=example,
                        final_status="ABSTAIN",
                        answer=ABSTENTION_MESSAGE,
                        claim=claim,
                        schema=schema,
                        ir=verification_ir,
                        certificate=certificate,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        retrieval_plan=retrieval_plan,
                        validation=validation,
                        solver=solver,
                        consistency_solver=consistency_solver,
                        verification_checks=verification_checks,
                        smtlib=smtlib,
                        failure_reason=reason,
                    ),
                )

            self._progress("verified")
            verification_checks["mathematical_validity"] = _math_check(
                status="passed",
                valid=True,
                smt_status=validation.smt_status,
                solver_status=solver.solver_status,
            )
            return RavResult(
                financebench_id=example.financebench_id,
                question=example.question,
                answer=verified_answer,
                final_status="VERIFIED",
                first_pass_solver_status="SAT",
                gold_answer=example.answer,
                verified=True,
                grounding=_grounding_tier(verification_ir),
                rule_id=f"formalized_{schema.metric}",
                metric=schema.metric,
                certificate=certificate,
                claim=claim,
                smtlib=smtlib,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieval_plan=retrieval_plan,
                verification_ir=verification_ir,
                verification_checks=verification_checks,
                verifier_certificate=_verifier_certificate(
                    example=example,
                    final_status="VERIFIED",
                    answer=verified_answer,
                    claim=claim,
                    schema=schema,
                    ir=verification_ir,
                    certificate=certificate,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieval_plan=retrieval_plan,
                    validation=validation,
                    solver=solver,
                    consistency_solver=consistency_solver,
                    verification_checks=verification_checks,
                    smtlib=smtlib,
                ),
            )

        # UNSAT repair: feed the unsat-core back to the LLM so it can self-diagnose
        # formula encoding errors. Each round records what went wrong and the Z3 outcome.
        repair_history: list[dict] = []
        current_smtlib = smtlib
        current_solver = solver

        if (
            solver.solver_status == "UNSAT"
            and self.config.allow_sat_repair
            and formula_authority_valid
            and not formula_authority_solver_enforced
            and not output_contract_solver_enforced
        ):
            for repair_round in range(1, self.config.max_repair_rounds + 1):
                round_entry: dict = {
                    "round": repair_round,
                    "unsat_core": current_solver.unsat_core or [],
                }
                repaired_smtlib = None
                try:
                    self._progress(
                        "repair_smt_with_unsat_core", round=repair_round
                    )
                    repaired_smtlib = self.smt_generator.repair(
                        verification_ir, current_smtlib, current_solver.unsat_core or []
                    )
                except Exception as exc:
                    round_entry["outcome"] = "repair_error"
                    round_entry["error"] = f"{type(exc).__name__}:{exc}"
                    repair_history.append(round_entry)
                    self._progress(
                        "smt_repair_failed",
                        round=repair_round,
                        error=round_entry["error"],
                    )
                    break

                if not repaired_smtlib:
                    round_entry["outcome"] = "cannot_repair"
                    repair_history.append(round_entry)
                    self._progress("smt_repair_cannot_repair", round=repair_round)
                    break

                round_entry["repaired_smtlib"] = repaired_smtlib
                repaired_validation = self.sanitizer.validate_ir(
                    repaired_smtlib, verification_ir
                )
                if repaired_validation.smt_status != "valid":
                    round_entry["outcome"] = "invalid_repair"
                    round_entry["validation_reason"] = repaired_validation.reason
                    repair_history.append(round_entry)
                    break

                try:
                    repaired_solver = self.z3_runner.run(
                        repaired_smtlib,
                        label=f"{example.financebench_id or 'example'}_repair_{repair_round}",
                        debug_unsat_core=self.config.include_unsat_core,
                    )
                except Exception as exc:
                    round_entry["outcome"] = "z3_error"
                    round_entry["error"] = f"{type(exc).__name__}:{exc}"
                    repair_history.append(round_entry)
                    break

                round_entry["outcome"] = repaired_solver.solver_status
                repair_history.append(round_entry)

                if repaired_solver.solver_status == "SAT":
                    self._progress("verified_after_repair", round=repair_round)
                    verification_checks["mathematical_validity"] = _math_check(
                        status="passed_after_repair",
                        valid=True,
                        reason="initial_unsat_repaired_by_llm_using_unsat_core",
                        smt_status=repaired_validation.smt_status,
                        solver_status=repaired_solver.solver_status,
                        first_pass_solver_status=solver.solver_status,
                        repair_rounds=repair_round,
                    )
                    return RavResult(
                        financebench_id=example.financebench_id,
                        question=example.question,
                        answer=verified_answer,
                        final_status="REPAIRED_VERIFIED",
                        first_pass_solver_status="UNSAT",
                        gold_answer=example.answer,
                        verified=True,
                        grounding=_grounding_tier(verification_ir),
                        rule_id=f"formalized_{schema.metric}",
                        metric=schema.metric,
                        certificate=certificate,
                        claim=claim,
                        smtlib=smtlib,
                        revised_smtlib=repaired_smtlib,
                        repair_history=repair_history,
                        repair_rounds=repair_round,
                        retrieved_chunk_ids=retrieved_chunk_ids,
                        retrieval_plan=retrieval_plan,
                        verification_ir=verification_ir,
                        verification_checks=verification_checks,
                        verifier_certificate=_verifier_certificate(
                            example=example,
                            final_status="REPAIRED_VERIFIED",
                            answer=verified_answer,
                            claim=claim,
                            schema=schema,
                            ir=verification_ir,
                            certificate=certificate,
                            retrieved_chunk_ids=retrieved_chunk_ids,
                            retrieval_plan=retrieval_plan,
                            validation=validation,
                            solver=solver,
                            consistency_solver=consistency_solver,
                            verification_checks=verification_checks,
                            smtlib=smtlib,
                            revised_solver=repaired_solver,
                            revised_smtlib=repaired_smtlib,
                        ),
                    )

                # Still UNSAT — feed new core into next round
                current_smtlib = repaired_smtlib
                current_solver = repaired_solver

        # Answer repair: strip claim assertions and run Z3 to find the actual computed
        # value, then feed it back to the LLM so it can correct its answer.
        if (
            solver.solver_status == "UNSAT"
            and self.config.allow_sat_repair
            and formula_authority_valid
            and not formula_authority_solver_enforced
            and not output_contract_solver_enforced
        ):
            from verifiqa.verification.smt_generator import extract_computed_value
            no_claim_smt = _strip_claim_assertions(smtlib)
            no_claim_solver = self.z3_runner.run(
                no_claim_smt,
                label=f"{example.financebench_id or 'example'}_no_claim",
                debug_unsat_core=False,
            )
            computed_value = (
                extract_computed_value(no_claim_solver.model or "", schema.metric)
                if no_claim_solver.solver_status == "SAT" else None
            )
            if computed_value is not None:
                answer_repair_entry: dict = {
                    "round": "answer_repair",
                    "computed_value": computed_value,
                }
                try:
                    self._progress("repair_answer_with_computed_value_feedback")
                    corrected_value = self.smt_generator.repair_answer(
                        verification_ir, no_claim_solver.model or ""
                    )
                    answer_repair_entry["corrected_value"] = corrected_value
                    if (
                        corrected_value is not None
                        and abs(corrected_value - computed_value) <= verification_ir.tolerance
                    ):
                        answer_repair_entry["outcome"] = "REPAIRED_VERIFIED"
                        repair_history.append(answer_repair_entry)
                        self._progress("verified_after_answer_repair")
                        corrected_answer_str = _format_value_with_unit(
                            corrected_value,
                            verification_ir.claim_unit or schema.claim_unit,
                            schema.precision_digits,
                        )
                        verification_checks["mathematical_validity"] = _math_check(
                            status="passed_after_answer_repair",
                            valid=True,
                            reason="llm_corrected_answer_using_z3_computed_value_feedback",
                            smt_status=validation.smt_status,
                            solver_status="SAT",
                            first_pass_solver_status="UNSAT",
                        )
                        return RavResult(
                            financebench_id=example.financebench_id,
                            question=example.question,
                            answer=corrected_answer_str,
                            final_status="REPAIRED_VERIFIED",
                            first_pass_solver_status="UNSAT",
                            gold_answer=example.answer,
                            verified=True,
                            grounding=_grounding_tier(verification_ir),
                            rule_id=f"formalized_{schema.metric}",
                            metric=schema.metric,
                            certificate=certificate,
                            claim=claim,
                            smtlib=smtlib,
                            repair_history=repair_history,
                            repair_rounds=len(repair_history),
                            retrieved_chunk_ids=retrieved_chunk_ids,
                            retrieval_plan=retrieval_plan,
                            verification_ir=verification_ir,
                            verification_checks=verification_checks,
                            verifier_certificate=_verifier_certificate(
                                example=example,
                                final_status="REPAIRED_VERIFIED",
                                answer=corrected_answer_str,
                                claim=claim,
                                schema=schema,
                                ir=verification_ir,
                                certificate=certificate,
                                retrieved_chunk_ids=retrieved_chunk_ids,
                                retrieval_plan=retrieval_plan,
                                validation=validation,
                                solver=solver,
                                consistency_solver=consistency_solver,
                                verification_checks=verification_checks,
                                smtlib=smtlib,
                            ),
                        )
                    else:
                        answer_repair_entry["outcome"] = "answer_repair_failed"
                except Exception as exc:
                    answer_repair_entry["outcome"] = "answer_repair_error"
                    answer_repair_entry["error"] = f"{type(exc).__name__}:{exc}"
                repair_history.append(answer_repair_entry)

        if (
            solver.solver_status == "UNSAT"
            and not formula_authority_valid
            and not formula_authority_solver_enforced
            and not output_contract_solver_enforced
        ):
            reason = _formula_authority_unverified_reason(verification_checks)
            self._progress("abstain", solver_status="UNSAT", reason=reason)
            verification_checks["mathematical_validity"] = _math_check(
                status="failed_unauthorized_formula",
                valid=None,
                reason=reason,
                smt_status=validation.smt_status,
                solver_status="UNSAT",
            )
            diagnostics = _diagnostics(
                "UNSAT",
                claim,
                schema,
                error=reason,
                solver=solver,
                evidence_provenance=provenance,
            )
            return self._unverified_formula(
                example,
                "UNSAT",
                claim,
                schema,
                smtlib,
                diagnostics,
                certificate,
                retrieved_chunk_ids,
                retrieval_plan,
                verification_ir,
                verification_checks=verification_checks,
                verifier_certificate=_verifier_certificate(
                    example=example,
                    final_status="ABSTAIN",
                    answer=ABSTENTION_MESSAGE,
                    claim=claim,
                    schema=schema,
                    ir=verification_ir,
                    certificate=certificate,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieval_plan=retrieval_plan,
                    validation=validation,
                    solver=solver,
                    consistency_solver=consistency_solver,
                    verification_checks=verification_checks,
                    smtlib=smtlib,
                    failure_reason=reason,
                ),
            )

        if solver.solver_status == "UNSAT":
            self._progress("violated", solver_status="UNSAT")
            unsat_reason = "unsat_claim_inconsistent_with_evidence"
            if output_contract_solver_enforced:
                oc_reason = verification_checks["output_contract"].get("reason", "output_contract_failed")
                unsat_reason = f"output_contract_failed:{oc_reason}"
            elif formula_authority_solver_enforced:
                fa_reason = verification_checks["formula_authority"].get("reason", "formula_authority_failed")
                unsat_reason = f"formula_authority_failed:{fa_reason}"
            verification_checks["mathematical_validity"] = _math_check(
                status="failed",
                valid=False,
                smt_status=validation.smt_status,
                solver_status="UNSAT",
                reason=unsat_reason,
            )
            diagnostics = _diagnostics(
                "UNSAT",
                claim,
                schema,
                error=unsat_reason,
                solver=solver,
                evidence_provenance=provenance,
            )
            return RavResult(
                financebench_id=example.financebench_id,
                question=example.question,
                answer=VIOLATION_MESSAGE,
                final_status="VIOLATED",
                first_pass_solver_status="UNSAT",
                gold_answer=example.answer,
                verified=False,
                abstained=False,
                grounding=_grounding_tier(verification_ir),
                rule_id=f"formalized_{schema.metric}",
                metric=schema.metric,
                certificate=certificate,
                claim=claim,
                smtlib=smtlib,
                repair_history=repair_history,
                repair_rounds=len(repair_history),
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieval_plan=retrieval_plan,
                verification_ir=verification_ir,
                verification_checks=verification_checks,
                diagnostics=diagnostics,
                verifier_certificate=_verifier_certificate(
                    example=example,
                    final_status="VIOLATED",
                    answer=VIOLATION_MESSAGE,
                    claim=claim,
                    schema=schema,
                    ir=verification_ir,
                    certificate=certificate,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieval_plan=retrieval_plan,
                    validation=validation,
                    solver=solver,
                    consistency_solver=consistency_solver,
                    verification_checks=verification_checks,
                    smtlib=smtlib,
                ),
            )

        self._progress("abstain", solver_status=solver.solver_status)
        verification_checks["mathematical_validity"] = _math_check(
            status="failed",
            valid=False,
            smt_status=validation.smt_status,
            solver_status=solver.solver_status,
            reason=solver.error,
        )
        diagnostics = _diagnostics(
            solver.solver_status,
            claim,
            schema,
            solver=solver,
            evidence_provenance=provenance,
        )
        return self._abstain(
            example,
            solver.solver_status,
            claim,
            schema,
            smtlib,
            diagnostics,
            certificate,
            retrieved_chunk_ids,
            retrieval_plan,
            verification_ir,
            verification_checks=verification_checks,
            verifier_certificate=_verifier_certificate(
                example=example,
                final_status="ABSTAIN",
                answer=ABSTENTION_MESSAGE,
                claim=claim,
                schema=schema,
                ir=verification_ir,
                certificate=certificate,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieval_plan=retrieval_plan,
                validation=validation,
                solver=solver,
                consistency_solver=consistency_solver,
                verification_checks=verification_checks,
                smtlib=smtlib,
                failure_reason=solver.error or solver.solver_status,
            ),
        )

    def _formalization_pass(
        self,
        example: FinanceBenchExample,
        chunks: list[EvidenceChunk],
        retrieval_plan: Optional[RetrievalPlan],
    ) -> _FormalizationPass:
        self._progress("generate_answer", chunks=len(chunks))
        try:
            answer = _generate_answer(self.answer_generator, example.question, chunks, retrieval_plan)
        except Exception as exc:
            raise RuntimeError(f"answer_generation_failed:{type(exc).__name__}:{exc}") from exc

        self._progress("formalize")
        try:
            certificate = _formalize(self.formalizer, example.question, chunks, answer, retrieval_plan)
        except Exception as exc:
            fallback = _finqa_program_formalization_fallback(example, chunks, answer, exc)
            if fallback is not None:
                self._progress("finqa_program_formalization_fallback", reason=f"{type(exc).__name__}:{exc}")
                return fallback
            raise
        certificate = prune_unused_certificate_facts(certificate)
        certificate = normalize_certificate_quantities(certificate, chunks, strict=False)
        answer_spec = answer_spec_from_question(example.question, answer)
        claim, schema = certificate_to_claim_schema(certificate)
        certificate, claim, schema = apply_answer_spec(certificate, claim, schema, answer_spec, answer)
        certificate, claim, schema, output_contract = apply_output_contract(
            certificate, claim, schema, answer_spec, answer
        )
        certificate, claim, schema = _reconcile_certificate_consistency(certificate, claim, schema)
        verification_ir = ir_from_certificate(certificate, claim, schema)
        verified_answer = answer
        validation_answer = answer
        provenance = certificate_provenance(certificate)

        if self.config.certificate_grounding_check:
            self._progress("validate_certificate", metric=schema.metric)
            certificate_validation = self.certificate_validator.validate(
                certificate,
                chunks,
                validation_answer,
                question=example.question,
                retrieval_plan=retrieval_plan,
                require_units=not self.config.numeric_only_verification,
            )
            if not certificate_validation.valid:
                self._progress("repair_certificate", reason=certificate_validation.reason)
                try:
                    certificate = self.formalizer.repair(certificate, certificate_validation.reason, chunks)
                    certificate = prune_unused_certificate_facts(certificate)
                    certificate = normalize_certificate_quantities(certificate, chunks, strict=False)
                    claim, schema = certificate_to_claim_schema(certificate)
                    certificate, claim, schema = apply_answer_spec(certificate, claim, schema, answer_spec, answer)
                    certificate, claim, schema, output_contract = apply_output_contract(
                        certificate, claim, schema, answer_spec, answer
                    )
                    certificate, claim, schema = _reconcile_certificate_consistency(certificate, claim, schema)
                    verification_ir = ir_from_certificate(certificate, claim, schema)
                    verified_answer = answer
                    validation_answer = answer
                    provenance = certificate_provenance(certificate)
                    certificate_validation = self.certificate_validator.validate(
                        certificate,
                        chunks,
                        validation_answer,
                        question=example.question,
                        retrieval_plan=retrieval_plan,
                        require_units=not self.config.numeric_only_verification,
                    )
                except Exception:
                    pass
        else:
            self._progress("skip_certificate_grounding", metric=schema.metric)
            certificate_validation = CertificateValidationResult(
                valid=True,
                reason="certificate_grounding_disabled",
            )

        source_table = _source_table_for_example(example)
        if source_table:
            try:
                verification_ir = attach_table_calculations(verification_ir, source_table)
                table_grounding = verification_ir.xbrl_calculations or {}
                self._progress(
                    "finqa_table_grounding",
                    status=table_grounding.get("status", ""),
                    bindings=len(table_grounding.get("bindings") or []),
                )
            except Exception as exc:
                verification_ir.xbrl_calculations = {
                    "enabled": True,
                    "status": "table_grounding_error",
                    "source": "finqa_table",
                    "grounding_status": "TABLE_ERROR",
                    "diagnostics": [f"{type(exc).__name__}:{exc}"],
                    "bindings": [],
                    "constraints": [],
                    "declared_variables": [],
                }
                self._progress("finqa_table_grounding", status="table_grounding_error", error=f"{type(exc).__name__}:{exc}")
        elif self.config.xbrl_calculation_dir is not None:
            if self.config.xbrl_llm_binding:
                self._progress("xbrl_llm_binding", status="deprecated_using_registry_linkbase")
            try:
                verification_ir = attach_xbrl_calculations(
                    verification_ir,
                    example.doc_name,
                    self.config.xbrl_calculation_dir,
                    ignore_unit_matching=self.config.numeric_only_verification,
                )
                xbrl = verification_ir.xbrl_calculations or {}
                self._progress(
                    "xbrl_linkbase",
                    status=xbrl.get("status", ""),
                    bindings=len(xbrl.get("bindings") or []),
                    constraints=len(xbrl.get("constraints") or []),
                )
            except Exception as exc:
                verification_ir.xbrl_calculations = {
                    "enabled": True,
                    "status": "xbrl_linkbase_error",
                    "doc_name": example.doc_name,
                    "diagnostics": [f"{type(exc).__name__}:{exc}"],
                    "bindings": [],
                    "constraints": [],
                    "declared_variables": [],
                }
                self._progress("xbrl_linkbase", status="xbrl_linkbase_error", error=f"{type(exc).__name__}:{exc}")

        # Document grounding: normally a fallback for prose/non-GAAP facts with no
        # structured binding. Some FinanceBench direct-lookup policies explicitly prefer
        # the cited document row over XBRL because the question asks for a displayed line
        # item and companyfacts may bind a narrower taxonomy concept instead.
        prior_xbrl = verification_ir.xbrl_calculations or {}
        prefer_document = _policy_prefers_document_grounding(
            self.policy_checker.registry,
            verification_ir.metric,
        )
        if prefer_document or not (prior_xbrl.get("bindings")):
            try:
                document_ir = attach_document_calculations(verification_ir, chunks)
                dg = document_ir.xbrl_calculations or {}
                if dg.get("bindings"):
                    if prefer_document and prior_xbrl.get("bindings"):
                        dg["replaced_grounding_status"] = prior_xbrl.get("grounding_status", "")
                        dg["replaced_grounding_source"] = prior_xbrl.get("source", "")
                    verification_ir = document_ir
                    self._progress(
                        "document_grounding",
                        status=dg.get("status", ""),
                        bindings=len(dg.get("bindings") or []),
                    )
                elif prior_xbrl:
                    # Document grounding added nothing; keep the more informative XBRL/
                    # table diagnostic rather than overwriting it with the empty result.
                    verification_ir.xbrl_calculations = prior_xbrl
            except Exception as exc:
                verification_ir.xbrl_calculations = prior_xbrl
                self._progress("document_grounding", status="document_grounding_error", error=f"{type(exc).__name__}:{exc}")

        return _FormalizationPass(
            answer=answer,
            certificate=certificate,
            claim=claim,
            schema=schema,
            verification_ir=verification_ir,
            verified_answer=verified_answer,
            validation_answer=validation_answer,
            provenance=provenance,
            certificate_validation=certificate_validation,
            output_contract=output_contract,
        )

    def _adaptive_retrieval_retry(
        self,
        example: FinanceBenchExample,
        retrieval_plan: Optional[RetrievalPlan],
        chunks: list[EvidenceChunk],
        reason: str,
        certificate: Optional[VerificationCertificate] = None,
    ) -> Optional[tuple[list[EvidenceChunk], _FormalizationPass]]:
        if not self.config.adaptive_retrieval:
            return None
        if retrieval_plan is None or not retrieval_plan.facts:
            return None
        if not hasattr(self.evidence_retriever, "expand_for_facts"):
            return None

        target_facts = _target_facts_for_failure(reason, retrieval_plan, certificate)
        if not target_facts:
            return None

        self._progress(
            "adaptive_retrieve",
            reason=reason,
            facts=[fact.name for fact in target_facts],
        )
        expanded_chunks = self.evidence_retriever.expand_for_facts(
            example.question,
            retrieval_plan,
            target_facts,
            chunks,
            company=example.company,
            doc_name=example.doc_name,
            top_k=self.config.evidence_top_k,
            financebench_id=example.financebench_id if self.config.oracle_retrieval else "",
        )
        if [chunk.chunk_id for chunk in expanded_chunks] == [chunk.chunk_id for chunk in chunks]:
            return None

        self._record_evidence(
            example,
            retrieval_plan,
            expanded_chunks,
            f"adaptive_retrieved:{_short_reason(reason)}",
        )
        try:
            pass_result = self._formalization_pass(example, expanded_chunks, retrieval_plan)
        except Exception as exc:
            self._progress("adaptive_retrieval_failed", error=f"{type(exc).__name__}:{exc}")
            return None
        return expanded_chunks, pass_result

    def _unverified_formula(
        self,
        example: FinanceBenchExample,
        first_status: str,
        claim: Claim,
        schema: VerificationSchema,
        smtlib: str,
        diagnostics: DiagnosticTrace,
        certificate: Optional[VerificationCertificate] = None,
        retrieved_chunk_ids: Optional[list] = None,
        retrieval_plan=None,
        verification_ir=None,
        verification_checks: Optional[dict[str, Any]] = None,
        verifier_certificate: Optional[dict[str, Any]] = None,
    ) -> RavResult:
        return RavResult(
            financebench_id=example.financebench_id,
            question=example.question,
            answer=ABSTENTION_MESSAGE,
            final_status="ABSTAIN",
            first_pass_solver_status=first_status,
            gold_answer=example.answer,
            verified=False,
            abstained=True,
            grounding=_grounding_tier(verification_ir) if verification_ir is not None else "llm_only",
            rule_id=f"formalized_{schema.metric}",
            metric=schema.metric,
            certificate=certificate,
            claim=claim,
            smtlib=smtlib,
            diagnostics=diagnostics,
            retrieved_chunk_ids=retrieved_chunk_ids or [],
            retrieval_plan=retrieval_plan,
            verification_ir=verification_ir,
            verification_checks=verification_checks
            or _finalize_unreached_checks(
                _initial_verification_checks(self.config.policy_semantic_check),
                diagnostics.schema_or_smt_error or first_status,
            ),
            verifier_certificate=verifier_certificate or {},
        )

    def _abstain(
        self,
        example: FinanceBenchExample,
        first_status: str,
        claim: Claim,
        schema: VerificationSchema,
        smtlib: str,
        diagnostics: DiagnosticTrace,
        certificate: Optional[VerificationCertificate] = None,
        retrieved_chunk_ids: Optional[list] = None,
        retrieval_plan=None,
        verification_ir=None,
        verification_checks: Optional[dict[str, Any]] = None,
        verifier_certificate: Optional[dict[str, Any]] = None,
    ) -> RavResult:
        return RavResult(
            financebench_id=example.financebench_id,
            question=example.question,
            answer=ABSTENTION_MESSAGE,
            final_status="ABSTAIN",
            first_pass_solver_status=first_status,
            gold_answer=example.answer,
            abstained=True,
            rule_id=f"formalized_{schema.metric}",
            metric=schema.metric,
            certificate=certificate,
            claim=claim,
            smtlib=smtlib,
            diagnostics=diagnostics,
            retrieved_chunk_ids=retrieved_chunk_ids or [],
            retrieval_plan=retrieval_plan,
            verification_ir=verification_ir,
            verification_checks=verification_checks
            or _finalize_unreached_checks(
                _initial_verification_checks(self.config.policy_semantic_check),
                diagnostics.schema_or_smt_error or first_status,
            ),
            verifier_certificate=verifier_certificate or {},
        )

    def _abstain_early(self, example: FinanceBenchExample, error: str, answer: str = "",
                       retrieved_chunk_ids: Optional[list] = None, retrieval_plan=None) -> RavResult:
        return RavResult(
            financebench_id=example.financebench_id,
            question=example.question,
            answer=answer or ABSTENTION_MESSAGE,
            final_status="ABSTAIN",
            first_pass_solver_status="INVALID",
            gold_answer=example.answer,
            abstained=True,
            diagnostics=DiagnosticTrace(
                solver_status="INVALID",
                schema_or_smt_error=error,
            ),
            retrieved_chunk_ids=retrieved_chunk_ids or [],
            retrieval_plan=retrieval_plan,
            verification_checks=_finalize_unreached_checks(
                _initial_verification_checks(self.config.policy_semantic_check),
                error,
            ),
        )

    def _progress(self, stage: str, **details) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage, details)

    def _record_evidence(
        self,
        example: FinanceBenchExample,
        retrieval_plan: Optional[RetrievalPlan],
        chunks: list[EvidenceChunk],
        status: str,
    ) -> None:
        if self.evidence_callback is None:
            return
        try:
            self.evidence_callback(example, retrieval_plan, list(chunks), status)
        except Exception as exc:
            self._progress("evidence_trace_failed", error=f"{type(exc).__name__}:{exc}")

    def _run_reconciler_shadow(
        self,
        example: FinanceBenchExample,
        verification_ir: VerificationIR,
    ) -> dict[str, Any] | None:
        if self.config.verification_mode != "reconciler-shadow":
            return None
        if self.reconciler_runner is None:
            return {
                "enabled": True,
                "mode": "reconciler_shadow",
                "status": "skipped",
                "valid": None,
                "reason": "missing_reconciler_runner",
            }
        try:
            self._progress("reconciler_shadow")
            return self.reconciler_runner.run(example, verification_ir)
        except Exception as exc:
            return {
                "enabled": True,
                "mode": "reconciler_shadow",
                "status": "error",
                "valid": None,
                "reason": f"{type(exc).__name__}:{exc}",
            }


def _oracle_chunks_from_example(example: FinanceBenchExample) -> list:
    """Build a single EvidenceChunk from all gold evidence passages in a FinanceBenchExample.

    All passages are concatenated into one block so the LLM receives the full
    oracle evidence as unified context with no retrieval selection.

    FinanceBench provides ``evidence`` as a list of dicts with keys:
    ``evidence_text``, ``doc_name``, ``evidence_page_num``, and optionally
    ``evidence_text_full_page``.  We prefer ``evidence_text`` (the exact
    passage) and fall back to ``evidence_text_full_page``.
    """
    parts = []
    for ev in example.evidence or []:
        if not isinstance(ev, dict):
            continue
        text = ev.get("evidence_text") or ev.get("evidence_text_full_page") or ""
        if text:
            parts.append(text)
    if not parts:
        return []
    return [
        EvidenceChunk(
            chunk_id=f"{example.financebench_id or 'example'}:oracle",
            doc_name=example.doc_name or "",
            page=None,
            text="\n\n".join(parts),
            source_type="oracle_evidence",
            company=example.company,
            financebench_id=example.financebench_id,
        )
    ]


def _has_finqa_program(example: FinanceBenchExample) -> bool:
    raw = getattr(example, "raw", None) or {}
    qa = raw.get("qa") if isinstance(raw, dict) else None
    return isinstance(qa, dict) and bool(qa.get("program"))


def _initial_verification_checks(policy_semantic_enabled: bool) -> dict[str, dict[str, Any]]:
    semantic_status = "pending" if policy_semantic_enabled else "skipped"
    semantic_reason = "" if policy_semantic_enabled else "policy_semantic_check_disabled"
    return {
        "mathematical_validity": {
            "enabled": True,
            "status": "pending",
            "valid": None,
            "reason": "",
        },
        "semantic_validity": {
            "enabled": policy_semantic_enabled,
            "status": semantic_status,
            "valid": None,
            "reason": semantic_reason,
        },
        "policy_alignment": {
            "enabled": policy_semantic_enabled,
            "status": semantic_status,
            "valid": None,
            "reason": semantic_reason,
        },
        "formula_authority": {
            "enabled": True,
            "status": "pending",
            "valid": None,
            "reason": "",
        },
        "output_contract": {
            "enabled": True,
            "status": "pending",
            "valid": None,
            "reason": "",
        },
    }


def _finalize_unreached_checks(checks: dict[str, dict[str, Any]], reason: str) -> dict[str, dict[str, Any]]:
    out = {name: dict(check) for name, check in checks.items()}
    for check in out.values():
        if check.get("status") == "pending":
            check["status"] = "skipped"
            check["valid"] = None
            check["reason"] = f"not_reached:{reason}"
    return out


def _formula_authority_result(
    example: FinanceBenchExample,
    ir: VerificationIR,
    schema: VerificationSchema,
    registry,
    certificate_validation: CertificateValidationResult | None = None,
    *,
    ignore_unit_matching: bool = False,
    reconciler: Any = None,
    chunks: Any = None,
    use_finqa_program: bool = True,
    use_generic_operations: bool = True,
) -> dict[str, Any]:
    finqa_program = resolve_finqa_program(example, ir, schema) if use_finqa_program else None
    if finqa_program is not None:
        if finqa_program.get("status") == "applied":
            check = {
                "enabled": True,
                "status": "passed",
                "valid": True,
                "reason": finqa_program.get("reason", "finqa_program_resolved"),
                "source": "finqa_qa_program",
                "program": finqa_program.get("program", ""),
                "program_re": finqa_program.get("program_re", ""),
                "formula": finqa_program.get("formula", ""),
                "computed_value": finqa_program.get("computed_value"),
                "raw_computed_value": finqa_program.get("raw_computed_value"),
                "output_scale": finqa_program.get("output_scale"),
                "output_scale_reason": finqa_program.get("output_scale_reason", ""),
                "n_operands": finqa_program.get("n_operands", 0),
                "n_bindings": finqa_program.get("n_bindings", 0),
            }
            return {"valid": True, "verification_ir": finqa_program["verification_ir"], "check": check}
        return _formula_authority_failure(
            finqa_program.get("reason", "finqa_program_failed"),
            finqa_program,
        )

    absence_reason = _absence_fact_authority_reason(ir, certificate_validation)
    if absence_reason:
        return _formula_authority_failure(absence_reason)

    resolved = resolve_policy_formula(
        ir,
        registry,
        ignore_unit_matching=ignore_unit_matching,
        question=example.question,
        evidence_text="\n".join(chunk.text for chunk in (chunks or [])),
    )
    if resolved.get("status") == "applied":
        revised_ir = _apply_policy_metadata_to_ir(
            replace(ir, formula=str(resolved["formula"])),
            resolved,
        )
        metadata = resolved.get("policy_metadata") or {}
        check = {
            "enabled": True,
            "status": "passed",
            "valid": True,
            "reason": resolved.get("reason", "policy_formula_resolved"),
            "source": "policy_registry",
            "policy_id": resolved.get("policy_id", ""),
            "formula": resolved.get("formula", ""),
            "template": resolved.get("template", ""),
            "role_bindings": resolved.get("role_bindings", {}),
            "computed_value": resolved.get("computed_value"),
            "claim_error": resolved.get("claim_error"),
            "policy_source_type": resolved.get("policy_source_type", ""),
            "policy_source_refs": resolved.get("policy_source_refs", []),
        }
        for key in ("claim_unit", "computed_unit", "source_priority"):
            if key in metadata:
                check[key] = metadata[key]
        if ignore_unit_matching:
            check["unit_matching"] = "disabled_numeric_only"
        return {"valid": True, "verification_ir": revised_ir, "check": check}

    # No named registry policy. Before abstaining, try the generic-operation
    # authority: FinQA arithmetic (a period change, a percentage change) has a
    # universal definition that is legitimate, instance-independent external
    # knowledge (the form of the computation, not the gold derivation). Authorize
    # iff the operation binds structurally (period-ordered operands) and the
    # grounded facts reproduce the canonical form.
    if use_generic_operations and resolved.get("status") == "no_policy":
        generic = resolve_generic_operation(ir, example.question, registry=registry,
            evidence_text="\n".join(chunk.text for chunk in (chunks or [])))
        if generic is not None:
            revised_ir = replace(ir, formula=str(generic["formula"]))
            check = {
                "enabled": True,
                "status": "passed",
                "valid": True,
                "reason": f"generic_operation_resolved:{generic['operation']}",
                "source": "generic_operation",
                "operation": generic["operation"],
                "technique": generic.get("technique", ""),
                "formula": generic["formula"],
                "computed_value": generic["computed_value"],
                "variables": generic["variables"],
                "policy_source_type": "standard_financial_analysis_operation_definition",
                "policy_source_refs": list(generic.get("source_refs") or []),
                "source_note": generic.get("source_note", ""),
            }
            return {"valid": True, "verification_ir": revised_ir, "check": check}

    # No registry policy for this metric (typically a company-specific non-GAAP
    # measure). Try to authorize the certificate's bridge formula against the
    # filing's own Reg-G reconciliation: extract the measure's published total
    # (grounded) and authorize iff formula(grounded facts) reproduces it.
    if reconciler is not None and chunks:
        try:
            doc = reconciler.resolve(ir, example.question, chunks)
        except Exception:
            doc = None
        if doc is not None:
            check = {
                "enabled": True,
                "status": "passed",
                "valid": True,
                "reason": "document_reconciliation_resolved",
                "source": "document_reconciliation",
                "formula": doc["formula"],
                "computed_value": doc["computed_value"],
                "published_total": doc["published_total"],
                "policy_source_type": "filing_non_gaap_reconciliation_reg_g",
                "policy_source_refs": [
                    {
                        "title": "Filing non-GAAP reconciliation (SEC Regulation G)",
                        "quote": doc["source_quote"],
                        "chunk_id": doc["chunk_id"],
                    }
                ],
            }
            return {"valid": True, "verification_ir": ir, "check": check}

    if not ignore_unit_matching:
        unit_reason = _monetary_ratio_unit_reason(example, ir)
        if unit_reason:
            return _formula_authority_failure(unit_reason, resolved)

    return _formula_authority_failure(
        resolved.get("reason", "no_registry_formula_for_metric"),
        resolved,
    )


def _apply_policy_metadata_to_ir(ir: VerificationIR, resolved: dict[str, Any]) -> VerificationIR:
    metadata = resolved.get("policy_metadata") or {}
    updates: dict[str, Any] = {}
    claim_unit = str(metadata.get("claim_unit") or "").strip()
    computed_unit = str(metadata.get("computed_unit") or "").strip()
    if claim_unit:
        updates["claim_unit"] = claim_unit
    if computed_unit:
        updates["computed_unit"] = computed_unit
    elif claim_unit:
        updates["computed_unit"] = claim_unit
    return replace(ir, **updates) if updates else ir


_UNIT_SCALE_FACTORS = (1e-3, 1e3, 1e-6, 1e6, 1e-2, 1e2)


def _reconcile_certificate_consistency(certificate, claim, schema):
    """Make the certificate internally consistent: the formula must actually
    produce the claimed value.

    The formalizer sometimes emits the claim at a different unit scale than its
    formula yields (e.g. claims 13.2 billion while the grounded fact + formula
    yield 13,200 million) and/or mislabels a monetary result as a "ratio". When
    the mismatch is a clean unit-scale factor (×/÷ 1000, 1e6, 100), rescale the
    formula so the proof is self-consistent, and correct a ratio label on a
    monetary result. Genuine (non-unit-scale) inconsistencies are left untouched
    for the solver to reject.
    """
    formula = schema.formula or certificate.formula
    try:
        variables = formula_variables(formula)
        values = {f.name: float(f.value) for f in certificate.facts if f.name in variables}
        computed = evaluate_formula(formula, values)
    except (FormulaError, SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return certificate, claim, schema

    target = float(claim.claimed_value)
    money_vars = {
        f.name for f in certificate.facts
        if _is_money_unit(f.unit) or _is_money_unit(f.raw_unit)
    }
    money_result = result_is_money(formula, money_vars)

    new_formula = formula
    denom = max(abs(computed), abs(target), 1.0)
    if abs(computed - target) / denom > 1e-6 and abs(computed) > 1e-12:
        ratio = target / computed
        for factor in _UNIT_SCALE_FACTORS:
            if abs(ratio - factor) <= 1e-3 * factor:
                new_formula = f"({formula}) * {factor:g}"
                break

    claim_unit = (schema.claim_unit or certificate.claim.unit or "")
    new_unit = claim_unit
    if money_result and claim_unit.strip().lower() == "ratio":
        new_unit = "USD"

    if new_formula == formula and new_unit == claim_unit:
        return certificate, claim, schema
    certificate = replace(
        certificate,
        formula=new_formula,
        claim=replace(certificate.claim, unit=new_unit),
    )
    schema = replace(schema, formula=new_formula, claim_unit=new_unit, computed_unit=new_unit)
    return certificate, claim, schema


def _policy_prefers_document_grounding(registry: Any, metric: str) -> bool:
    try:
        policy = registry.find_policy(metric)
    except Exception:
        return False
    metadata = getattr(policy, "metadata", {}) if policy is not None else {}
    return str((metadata or {}).get("source_priority") or "").strip().lower() == "document"


def _augment_smt_with_formula_authority_failure(smtlib: str, check: dict[str, Any]) -> str:
    reason = str(check.get("reason") or "formula_authority_failed")
    return _augment_smt_with_forced_failure(
        smtlib,
        reason,
        "formula_authority_failed_",
        "Formula authority failed",
    )


def _augment_smt_with_output_contract_failure(smtlib: str, check: dict[str, Any]) -> str:
    reason = str(check.get("reason") or "output_contract_failed")
    return _augment_smt_with_forced_failure(
        smtlib,
        reason,
        "output_contract_failed_",
        "Output contract failed",
    )


def _augment_smt_with_forced_failure(
    smtlib: str,
    reason: str,
    prefix: str,
    label: str,
) -> str:
    name = prefix + _short_reason(reason)
    block = "\n".join([
        "",
        f"; {label}: {reason}. Force UNSAT so Z3 catches the invalid proof.",
        f"(assert (! false :named {name}))",
    ])
    match = re.search(r"\(\s*check-sat\s*\)", smtlib)
    if match is None:
        return smtlib.rstrip() + "\n" + block + "\n"
    return smtlib[: match.start()].rstrip() + "\n" + block + "\n\n" + smtlib[match.start():].lstrip()


def _absence_fact_authority_reason(
    ir: VerificationIR,
    certificate_validation: CertificateValidationResult | None,
) -> str:
    try:
        active_names = formula_variables(ir.formula)
    except Exception:
        active_names = set(ir.facts)
    absence_facts = [
        fact for name, fact in ir.facts.items()
        if name in active_names and str(fact.fact_type or "").startswith("absence")
    ]
    if not absence_facts:
        return ""
    if certificate_validation is None:
        return "absence_fact_missing_certificate_grounding"
    if not certificate_validation.valid:
        reason = certificate_validation.reason or "certificate_validation_failed"
        return f"absence_fact_not_grounded:{reason}"
    return ""


# Most formula-authority failures mean "the formula is not authorized", not "the
# answer is false". Those must block strong verification without manufacturing a
# false UNSAT. A small set of failures are hard proof failures: e.g. an absence-zero
# claim without grounded absence evidence, an impossible unit interpretation, or an
# explicitly failed policy role constraint. Those are solver-enforced so the result is
# a real non-verified violation rather than an accidental SAT on an invalid proof.
_AUTHORITY_CONTRADICTION_PREFIXES: tuple[str, ...] = (
    "absence_fact_missing_certificate_grounding",
    "absence_fact_not_grounded",
    # NOTE: "ambiguous_program_operand" is intentionally NOT here. It means a gold
    # FinQA program operand value appears multiple times and couldn't be uniquely
    # bound — an inability to apply the gold-program cross-check, not a contradiction.
    # Forcing a violation on it rejected correct answers (e.g. 51.2-47.4=3.8=gold).
    # As a soft failure it falls back to the standard grounded-formula SMT check.
    "finqa_program_has_no_grounded_operands",
    "monetary_claim_unit_is_ratio",
    "policy_role_constraints_failed",
    "program_operand_not_grounded",
    "program_operand_reuse_exhausted",
    "unsupported_program_operation",
    "unsupported_program_token",
)


def _authority_failure_forces_violation(reason: str) -> bool:
    return str(reason or "").startswith(_AUTHORITY_CONTRADICTION_PREFIXES)


def _formula_authority_unverified_reason(checks: dict[str, dict[str, Any]]) -> str:
    reason = (checks.get("formula_authority") or {}).get("reason") or "formula_authority_unresolved"
    return f"formula_authority_unresolved:{reason}"


def _formula_authority_failure(reason: str, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    check = {
        "enabled": True,
        "status": "failed",
        "valid": False,
        "reason": reason,
    }
    if details:
        if "status" in details:
            check["policy_resolution_status"] = details["status"]
        for key in (
            "policy_id",
            "role_bindings",
            "template",
            "formula",
            "variables",
            "policy_source_type",
            "policy_source_refs",
            "source",
            "program",
            "program_re",
        ):
            if key in details:
                check[key] = details[key]
    return {"valid": False, "check": check}



def _monetary_ratio_unit_reason(example: FinanceBenchExample, ir: VerificationIR) -> str:
    if (ir.claim_unit or "").strip().lower() != "ratio":
        return ""
    if any((fact.fact_type or "").strip() == "absence_implies_zero" for fact in ir.facts.values()):
        if abs(float(ir.claimed_value or 0.0)) <= max(float(ir.tolerance or 0.0), 1e-12):
            return ""
    text = f"{example.question} {ir.metric}".lower()
    ratio_terms = (
        " ratio",
        "margin",
        "rate",
        "percent",
        "percentage",
        "coverage",
        "turnover",
        "yield",
        "return on",
        "roe",
        "roa",
        "cagr",
    )
    if any(term in text for term in ratio_terms):
        return ""
    money_terms = (
        "amount",
        "cash",
        "proceeds",
        "gain",
        "cost",
        "costs",
        "expense",
        "expenses",
        "revenue",
        "sales",
        "asset",
        "assets",
        "liability",
        "liabilities",
        "debt",
        "dividend",
        "dividends",
        "capex",
        "capital expenditure",
        "income",
        "payment",
        "payments",
        "$",
        "usd",
    )
    if any(term in text for term in money_terms):
        return "monetary_claim_unit_is_ratio"
    return ""


def _needs_xbrl_consistency_check(ir: VerificationIR) -> bool:
    xbrl = getattr(ir, "xbrl_calculations", None) or {}
    return bool(xbrl.get("constraints"))


def _has_grounded_fact_bindings(ir: VerificationIR) -> bool:
    xbrl = getattr(ir, "xbrl_calculations", None) or {}
    return bool(xbrl.get("bindings"))


def _strip_claim_assertions(smtlib: str) -> str:
    smtlib, _ = _remove_named_assertion(smtlib, "claim_upper")
    smtlib, _ = _remove_named_assertion(smtlib, "claim_lower")
    return smtlib


def _strip_llm_evidence_for_bound_facts(smtlib: str, ir: VerificationIR) -> str:
    """Drop LLM-extracted value assertions for independently bound facts.

    Once a fact is bound to an authoritative source value (EDGAR/XBRL or a FinQA
    table cell), the source binding should be the sole constraint on that fact.
    Only bound facts are stripped; unbound facts keep their LLM evidence so they
    cannot become free variables that satisfy any claim.
    """
    xbrl = getattr(ir, "xbrl_calculations", None) or {}
    bound_facts = {
        b.get("fact_name")
        for b in (xbrl.get("bindings") or [])
        if b.get("fact_name") and float(b.get("binding_tolerance") or 0.0) == 0.0
    }
    for fact_name in bound_facts:
        smtlib, _ = _remove_named_assertion(smtlib, f"evidence_{fact_name}")
    return smtlib


def _grounding_tier(ir: VerificationIR) -> str:
    """Return the strongest data grounding tier used by the SMT verdict."""
    xbrl = getattr(ir, "xbrl_calculations", None) or {}
    bindings = xbrl.get("bindings") or []
    constraints = xbrl.get("constraints") or []
    sources = {str(binding.get("source") or "") for binding in bindings}
    if bindings and (xbrl.get("source") == "finqa_table" or "finqa_table" in sources):
        return "finqa_table_grounded"
    if bindings and (xbrl.get("source") == "source_document" or "source_document" in sources):
        # Inputs corroborated against the retrieved source text (cited-figure provenance
        # + value), but not against a structured XBRL/table source. Weaker than XBRL.
        return "document_grounded"
    if bindings and constraints:
        return "xbrl_linkbase_grounded"
    if bindings:
        return "xbrl_instance_grounded"
    return "llm_only"


def _source_table_for_example(example: FinanceBenchExample) -> list[list[Any]]:
    raw = getattr(example, "raw", None) or {}
    table = raw.get("table") if isinstance(raw, dict) else None
    if not isinstance(table, list) or len(table) < 2:
        return []
    rows = [row for row in table if isinstance(row, list) and row]
    return rows if len(rows) >= 2 else []


def _remove_named_assertion(smtlib: str, name: str) -> tuple[str, bool]:
    out: list[str] = []
    removed = False
    index = 0
    while index < len(smtlib):
        if smtlib[index] != "(":
            out.append(smtlib[index])
            index += 1
            continue
        end = _sexpr_end(smtlib, index)
        if end is None:
            out.append(smtlib[index])
            index += 1
            continue
        form = smtlib[index:end]
        if form.lstrip().startswith("(assert") and f":named {name}" in form:
            removed = True
        else:
            out.append(form)
        index = end
    return "".join(out), removed


def _sexpr_end(text: str, start: int) -> Optional[int]:
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == ";":
            newline = text.find("\n", index)
            if newline == -1:
                return None
            index = newline + 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def _math_check(
    *,
    status: str,
    valid: Optional[bool],
    reason: str = "",
    smt_status: str = "",
    solver_status: str = "",
    first_pass_solver_status: str = "",
    repair_rounds: int | None = None,
) -> dict[str, Any]:
    check = {
        "enabled": True,
        "status": status,
        "valid": valid,
        "reason": reason,
    }
    if smt_status:
        check["smt_status"] = smt_status
    if solver_status:
        check["solver_status"] = solver_status
    if first_pass_solver_status:
        check["first_pass_solver_status"] = first_pass_solver_status
    if repair_rounds is not None:
        check["repair_rounds"] = repair_rounds
    return check


def _checks_from_policy_validation(policy_validation, policy) -> dict[str, dict[str, Any]]:
    if policy_validation.status == "no_policy":
        skipped = {
            "enabled": True,
            "status": "skipped",
            "valid": None,
            "reason": policy_validation.reason,
            "variable_concepts": policy_validation.variable_concepts,
        }
        return {
            "semantic_validity": dict(skipped),
            "policy_alignment": dict(skipped),
        }

    status = "passed" if policy_validation.valid else "failed"
    semantic_check = {
        "enabled": True,
        "status": status,
        "valid": policy_validation.valid,
        "reason": policy_validation.reason,
        "policy_id": policy_validation.policy_id,
        "variable_concepts": policy_validation.variable_concepts,
        "role_bindings": policy_validation.role_bindings,
    }
    policy_check = {
        "enabled": True,
        "status": status,
        "valid": policy_validation.valid,
        "reason": policy_validation.reason,
        "policy_id": policy_validation.policy_id,
        "matched_template": policy_validation.matched_template,
        "role_bindings": policy_validation.role_bindings,
    }
    if policy is not None:
        policy_check["policy_source"] = policy.source
        policy_check["policy_source_type"] = policy.source_type
        policy_check["policy_source_refs"] = list(policy.source_refs)
    return {
        "semantic_validity": semantic_check,
        "policy_alignment": policy_check,
    }


def _verifier_certificate(
    *,
    example: FinanceBenchExample,
    final_status: str,
    answer: str,
    claim: Claim,
    schema: VerificationSchema,
    ir: VerificationIR,
    certificate: Optional[VerificationCertificate],
    retrieved_chunk_ids: list,
    retrieval_plan: Optional[RetrievalPlan],
    validation,
    solver: Optional[SolverResult],
    verification_checks: dict[str, Any],
    smtlib: str,
    consistency_solver: Optional[SolverResult] = None,
    failure_reason: str = "",
    revised_claim: Optional[Claim] = None,
    revised_ir: Optional[VerificationIR] = None,
    revised_solver: Optional[SolverResult] = None,
    revised_smtlib: str = "",
) -> dict[str, Any]:
    effective_claim = revised_claim or claim
    effective_ir = revised_ir or ir
    effective_solver = revised_solver or solver or consistency_solver
    solver_status = effective_solver.solver_status if effective_solver is not None else "INVALID"
    consistency_status = consistency_solver.solver_status if consistency_solver is not None else "not_run"
    smt_status = validation.smt_status if validation is not None else "not_validated"
    computed_value = _compute_expected(claim.variables, schema)

    decision_reason = _decision_reason(final_status, solver_status, smt_status, failure_reason)
    facts = []
    for name in sorted(effective_ir.facts):
        fact = effective_ir.facts[name]
        facts.append({
            "name": name,
            "value": fact.value,
            "unit": fact.unit,
            "period": fact.period,
            "row_label": fact.row_label,
            "column": fact.column,
            "chunk_id": fact.chunk_id,
            "source_quote": fact.source_quote,
            "evidence_assertion": f"evidence_{name}",
        })

    return {
        "certificate_type": "verifiqa_verifier_certificate",
        "version": 1,
        "financebench_id": example.financebench_id,
        "question": example.question,
        "answer": answer,
        "decision": {
            "final_status": final_status,
            "verified": final_status in {"VERIFIED", "REPAIRED_VERIFIED"},
            "reason": decision_reason,
        },
        "claim": {
            "metric": effective_claim.metric,
            "claimed_value": effective_claim.claimed_value,
            "unit": effective_ir.claim_unit,
            "period": effective_claim.period or effective_ir.period,
            "tolerance": effective_ir.tolerance,
            "tolerance_source": effective_ir.tolerance_source,
        },
        "evidence_bindings": facts,
        "formula": {
            "expression": effective_ir.formula,
            "computed_variable": f"computed_{effective_ir.metric}",
            "computed_value": computed_value,
            "computed_unit": effective_ir.computed_unit or effective_ir.claim_unit,
            "assertion_name": f"formula_{effective_ir.metric}",
        },
        "smt": {
            "query_type": effective_ir.query_type,
            "semantics": "For XBRL-backed SMT, SAT on the consistency query means P/E/R/formula are coherent; UNSAT there means an extraction-linkbase conflict. After consistency is SAT, UNSAT on the violation query means no counterexample exists outside tolerance; SAT means a violating model exists.",
            "smt_validation_status": smt_status,
            "smt_validation_reason": getattr(validation, "reason", ""),
            "solver_status": solver_status,
            "consistency_solver_status": consistency_status,
            "violation_solver_status": solver.solver_status if solver is not None else "not_run",
            "smt_path": solver.smt_path if solver is not None else (consistency_solver.smt_path if consistency_solver is not None else ""),
            "consistency_smt_path": consistency_solver.smt_path if consistency_solver is not None else "",
            "revised_smt_path": revised_solver.smt_path if revised_solver is not None else "",
            "first_pass_solver_status": solver.solver_status if solver is not None else "INVALID",
            "revised_solver_status": revised_solver.solver_status if revised_solver is not None else "",
            "has_smtlib": bool(smtlib),
            "has_revised_smtlib": bool(revised_smtlib),
            "counterexample_model": solver.model if solver is not None else "",
            "unsat_core": solver.unsat_core if solver is not None else [],
            "xbrl_linkbase": _xbrl_summary(effective_ir),
        },
        "retrieval": {
            "retrieved_chunk_ids": list(retrieved_chunk_ids or []),
            "plan": _retrieval_plan_summary(retrieval_plan),
        },
        "checks": verification_checks,
        "source_certificate": _source_certificate_summary(certificate),
    }


def _xbrl_summary(ir: VerificationIR) -> dict[str, Any]:
    xbrl = getattr(ir, "xbrl_calculations", None) or {}
    if not xbrl:
        return {"enabled": False}
    return {
        "enabled": bool(xbrl.get("enabled", True)),
        "status": xbrl.get("status", ""),
        "doc_name": xbrl.get("doc_name", ""),
        "instance_path": xbrl.get("instance_path", ""),
        "calculation_path": xbrl.get("calculation_path", ""),
        "n_bindings": len(xbrl.get("bindings") or []),
        "n_constraints": len(xbrl.get("constraints") or []),
        "n_declared_variables": len(xbrl.get("declared_variables") or []),
        "instantiated_constraints": xbrl.get("instantiated_constraints", 0),
        "connected_constraints": xbrl.get("connected_constraints", 0),
        "grounding_status": xbrl.get("grounding_status", ""),
        "source": xbrl.get("source", ""),
        "binding_sources": xbrl.get("binding_sources") or [],
        "bindings": xbrl.get("bindings") or [],
        "diagnostics": xbrl.get("diagnostics") or [],
    }


def _decision_reason(final_status: str, solver_status: str, smt_status: str, failure_reason: str) -> str:
    if final_status in {"VERIFIED", "REPAIRED_VERIFIED"} and solver_status == "SAT" and smt_status == "valid":
        return "SMT was validated against the typed IR and Z3 proved the claim check SAT."
    if final_status == "VIOLATED":
        return "Z3 returned UNSAT: the claimed value is inconsistent with the evidence and formula."
    if failure_reason:
        return failure_reason
    if solver_status == "UNSAT":
        return "Z3 returned UNSAT for the claim check."
    if solver_status == "UNKNOWN":
        return "Z3 returned UNKNOWN for the claim check."
    if smt_status != "valid":
        return "SMT did not pass validation against the typed IR."
    return solver_status or final_status


def _retrieval_plan_summary(plan: Optional[RetrievalPlan]) -> dict[str, Any]:
    if plan is None:
        return {}
    return {
        "metric": plan.metric,
        "reason": plan.reason,
        "facts": [
            {
                "name": fact.name,
                "aliases": list(fact.aliases),
                "period": fact.period,
                "statement": fact.statement,
            }
            for fact in plan.facts
        ],
    }


def _source_certificate_summary(certificate: Optional[VerificationCertificate]) -> dict[str, Any]:
    if certificate is None:
        return {}
    return {
        "claim_metric": certificate.claim.metric,
        "claim_value": certificate.claim.claimed_value,
        "claim_unit": certificate.claim.unit,
        "formula": certificate.formula,
        "calculation": certificate.calculation,
        "tolerance": certificate.tolerance,
        "fact_names": [fact.name for fact in certificate.facts],
    }


def _diagnostics(
    solver_status: str,
    claim: Claim,
    schema: VerificationSchema,
    error: str = "",
    solver: Optional[SolverResult] = None,
    evidence_provenance: Optional[dict] = None,
) -> DiagnosticTrace:
    return DiagnosticTrace(
        solver_status=solver_status,
        schema_or_smt_error=error or (solver.error if solver else ""),
        counterexample_model=solver.model if solver and solver.solver_status == "SAT" else "",
        expected_value=_compute_expected(claim.variables, schema),
        claimed_value=claim.claimed_value,
        evidence_facts=dict(claim.variables),
        evidence_provenance=evidence_provenance or {},
        retrieved_formula=schema.formula,
    )


def _target_facts_for_failure(
    reason: str,
    retrieval_plan: RetrievalPlan,
    certificate: Optional[VerificationCertificate] = None,
) -> list[RetrievalFact]:
    if not retrieval_plan.facts:
        return []

    target_names = _fact_names_from_failure(reason)
    if target_names:
        matched = [
            fact for fact in retrieval_plan.facts
            if any(_matches_plan_fact(fact.name, target_name) for target_name in target_names)
        ]
        if matched:
            return matched

    if certificate is not None:
        certificate_fact_names = {fact.name for fact in certificate.facts}
        missing_from_certificate = [
            fact for fact in retrieval_plan.facts
            if not any(_matches_plan_fact(fact.name, cert_name) for cert_name in certificate_fact_names)
        ]
        if missing_from_certificate:
            return missing_from_certificate

    lower = (reason or "").lower()
    if (
        "missing" in lower
        or "not_verifiable" in lower
        or "source_quote" in lower
        or "value_not_in_chunk" in lower
        or "unknown_chunk_id" in lower
    ):
        return list(retrieval_plan.facts)
    return []


def _fact_names_from_failure(reason: str) -> list[str]:
    if not reason:
        return []
    patterns = (
        r"formula_variable_missing_fact:([A-Za-z0-9_]+)",
        r"unused_fact:([A-Za-z0-9_]+)",
        r"fact_missing_unit:([A-Za-z0-9_]+)",
        r"fact_missing_source_quote:([A-Za-z0-9_]+)",
        r"fact_missing_chunk_id:([A-Za-z0-9_]+)",
        r"unknown_chunk_id:([A-Za-z0-9_]+):",
        r"source_quote_not_in_chunk:([A-Za-z0-9_]+)",
        r"fact_value_not_in_chunk:([A-Za-z0-9_]+)",
        r"absence_zero_not_allowed_by_question:([A-Za-z0-9_]+)",
        r"absence_not_supported:([A-Za-z0-9_]+)",
        r"absence_zero_no_absence_language_in_quote:([A-Za-z0-9_]+)",
        r"absence_zero_retrieval_failure_not_filing_absence:([A-Za-z0-9_]+)",
        r"fact_source_statement_mismatch:([A-Za-z0-9_]+):",
    )
    names: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, reason):
            name = match.group(1)
            if name not in names:
                names.append(name)
    return names


def _matches_plan_fact(plan_fact_name: str, target_name: str) -> bool:
    plan_base = _base_fact_name(plan_fact_name)
    target_base = _base_fact_name(target_name)
    if not plan_base or not target_base:
        return False
    return plan_base == target_base or plan_base in target_base or target_base in plan_base


def _base_fact_name(name: str) -> str:
    base = re.sub(r"(^|_)(?:fy)?(?:19|20)\d{2}(?=$|_)", "_", (name or "").lower())
    base = re.sub(r"_+", "_", base).strip("_")
    return base


def _short_reason(reason: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", (reason or "unknown"))
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return (sanitized or "unknown")[:120]


def _compute_expected(variables: Dict[str, float], schema: VerificationSchema) -> Optional[float]:
    try:
        return evaluate_formula(schema.formula, variables)
    except (FormulaError, Exception):
        return None


def _generate_answer(answer_generator, question: str, chunks, retrieval_plan):
    if _accepts_keyword(answer_generator.generate, "retrieval_plan"):
        return answer_generator.generate(question, chunks, retrieval_plan=retrieval_plan)
    return answer_generator.generate(question, chunks)


def _formalize(formalizer, question: str, chunks, answer: str, retrieval_plan):
    if _accepts_keyword(formalizer.formalize, "retrieval_plan"):
        return formalizer.formalize(question, chunks, answer, retrieval_plan=retrieval_plan)
    return formalizer.formalize(question, chunks, answer)


def _finqa_program_formalization_fallback(
    example: FinanceBenchExample,
    chunks: list[EvidenceChunk],
    answer: str,
    error: Exception,
) -> _FormalizationPass | None:
    raw = getattr(example, "raw", None) or {}
    qa = raw.get("qa") if isinstance(raw, dict) else None
    if not isinstance(qa, dict) or not qa.get("program"):
        return None
    parsed = _first_scalar_claim_number(answer)
    if parsed is None:
        return None
    claimed_value, token_has_percent = parsed
    answer_spec = answer_spec_from_question(example.question, answer)
    unit = "percent" if token_has_percent else (answer_spec.expected_unit if answer_spec.expected_unit != "unspecified" else "")
    metric = _fallback_metric_from_question(example.question)
    fact = CertificateFact(
        name="finqa_claim_placeholder",
        value=float(claimed_value),
        unit=unit,
        source_quote=answer,
        chunk_id=chunks[0].chunk_id if chunks else "",
        fact_type="numeric",
        raw_value=float(claimed_value),
        raw_unit=unit,
        source_scale="ones",
    )
    certificate = VerificationCertificate(
        claim=CertificateClaim(
            metric=metric,
            claimed_value=float(claimed_value),
            unit=unit,
            period="",
        ),
        facts=[fact],
        formula="finqa_claim_placeholder",
        calculation=f"fallback scalar claim from answer after formalization error: {type(error).__name__}:{error}",
        tolerance=answer_spec.tolerance,
    )
    claim, schema = certificate_to_claim_schema(certificate)
    certificate, claim, schema = apply_answer_spec(certificate, claim, schema, answer_spec, answer)
    certificate, claim, schema, output_contract = apply_output_contract(
        certificate, claim, schema, answer_spec, answer
    )
    verification_ir = ir_from_certificate(certificate, claim, schema)
    verified_answer = answer
    return _FormalizationPass(
        answer=answer,
        certificate=certificate,
        claim=claim,
        schema=schema,
        verification_ir=verification_ir,
        verified_answer=verified_answer,
        validation_answer=answer,
        provenance={},
        certificate_validation=CertificateValidationResult(
            valid=True,
            reason=f"finqa_program_fallback_after_formalization_error:{type(error).__name__}:{error}",
        ),
        output_contract=output_contract,
    )


def _first_scalar_claim_number(answer: str) -> tuple[float, bool] | None:
    candidates: list[tuple[float, bool]] = []
    for raw in re.findall(r"\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?", answer or ""):
        token = raw.strip()
        negative = token.startswith("(") and token.endswith(")")
        token = token.strip("()").replace("$", "").replace(",", "")
        has_percent = token.endswith("%")
        if has_percent:
            token = token[:-1]
        try:
            value = float(token)
        except ValueError:
            continue
        if negative:
            value = -value
        candidates.append((value, has_percent))
    for value, has_percent in candidates:
        if not _looks_like_year(value, has_percent):
            return value, has_percent
    return candidates[0] if candidates else None


def _looks_like_year(value: float, has_percent: bool) -> bool:
    if has_percent or not float(value).is_integer():
        return False
    year = int(value)
    return 1900 <= year <= 2100


def _fallback_metric_from_question(question: str) -> str:
    words = re.findall(r"[a-z0-9]+", (question or "").lower())
    stop = {"what", "was", "were", "is", "are", "the", "for", "and", "in", "of", "to", "a", "an"}
    parts = [word for word in words if word not in stop][:8]
    return "finqa_" + ("_".join(parts) or "program_answer")


def _accepts_keyword(callable_obj, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == keyword
        for parameter in parameters
    )


def _generate_smt(smt_generator, ir: VerificationIR, claim: Claim, schema: VerificationSchema) -> tuple[str, str]:
    """Returns (base_smtlib, augmented_smtlib). base is LLM-only; augmented adds source bindings."""
    xbrl = getattr(ir, "xbrl_calculations", None) or {}
    if xbrl.get("formula_source") == "finqa_qa_program":
        smtlib = render_ir_smt(ir)
    elif hasattr(smt_generator, "generate_ir"):
        smtlib = smt_generator.generate_ir(ir)
    else:
        smtlib = smt_generator.generate(claim, schema)
    base = _ensure_qf_nra_logic(smtlib)
    augmented = augment_smt_with_xbrl(base, ir)
    # Independent canonical-formula constraint (when the metric has a policy): catches a
    # wrong formula the LLM encoded as a Z3 counterexample, the same way XBRL bindings
    # catch a wrong input. Additive — a no-op for metrics without a policy.
    augmented = augment_smt_with_policy_formula(augmented, ir)
    return base, augmented


def _ensure_qf_nra_logic(smtlib: str) -> str:
    text = (smtlib or "").strip()
    if re.search(r"\(\s*set-logic\s+QF_NRA\s*\)", text):
        return text
    if re.search(r"\(\s*set-logic\s+", text):
        return text
    return f"(set-logic QF_NRA)\n{text}"


def _revised_answer(claim: Claim, schema: VerificationSchema) -> str:
    metric = schema.metric.replace("_", " ")
    unit = (schema.claim_unit or schema.computed_unit or "").strip()
    value = _format_value_with_unit(claim.claimed_value, unit, schema.precision_digits)
    return f"The verified {metric} is {value}."


def _answer_from_certificate(
    certificate: VerificationCertificate,
    claim: Claim,
    schema: VerificationSchema,
    *,
    include_exact: bool = False,
) -> str:
    answer = _revised_answer(claim, schema)
    if include_exact:
        answer = f"{answer} Exact claimed value: {claim.claimed_value:.12g}."
    calculation = (certificate.calculation or "").strip()
    if calculation:
        return f"{answer} Calculation: {calculation}."
    return answer


def _format_value_with_unit(value: float, unit: str, precision_digits: Optional[int]) -> str:
    formatted = _format_repaired_value(value, precision_digits)
    unit = (unit or "").strip()
    if unit == "percent":
        return f"{formatted}%"
    if unit and unit != "ratio":
        return f"{formatted} {unit}"
    return formatted


def _format_repaired_value(value: float, precision_digits: Optional[int], min_sig_figs: int = 4) -> str:
    if precision_digits is None:
        return f"{value:.12g}"
    # Keep at least ~min_sig_figs significant figures, so a correct small value (0.10935)
    # is never rounded to "0" by a coarse precision_digits that was tied to the LLM's
    # original (low-precision) answer. precision_digits acts only as a *minimum* decimals.
    sig_decimals = 0
    if value != 0 and Decimal(str(value)).is_finite():
        magnitude = Decimal(str(abs(value))).adjusted()  # floor(log10|value|)
        sig_decimals = max(0, min_sig_figs - 1 - magnitude)
    decimals = max(precision_digits, sig_decimals) if precision_digits > 0 else sig_decimals
    if decimals <= 0:
        return str(int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)))
    formatted = f"{value:.{decimals}f}"
    if "." in formatted:  # drop trailing zeros added by the significant-figure floor
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted
