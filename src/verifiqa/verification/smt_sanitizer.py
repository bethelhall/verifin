from __future__ import annotations

import re

from verifiqa.types import Claim, SmtValidationResult, VerificationIR, VerificationSchema
from verifiqa.verification.ir import ir_from_claim_schema


class SmtSanitizer:
    """Minimal structural check on LLM-generated SMT.

    Only two checks:
      1. Evidence presence — every required fact has a named evidence assertion.
      2. Formula name presence — a named formula assertion exists (content not checked).

    Formula correctness and violation-query shape are caught by Z3 counterexamples,
    not by the sanitizer.
    """

    def validate_ir(self, smtlib: str, ir: VerificationIR) -> SmtValidationResult:
        try:
            return self._validate_ir(smtlib, ir)
        except ValueError as exc:
            return SmtValidationResult(smt_status="invalid", reason=str(exc))

    def validate(
        self,
        smtlib: str,
        claim: Claim,
        schema: VerificationSchema,
    ) -> SmtValidationResult:
        return self.validate_ir(smtlib, ir_from_claim_schema(claim, schema))

    def _validate_ir(self, smtlib: str, ir: VerificationIR) -> SmtValidationResult:
        if not smtlib.strip():
            raise ValueError("empty_smt")
        body = _strip_comments(smtlib)

        for required in sorted(ir.facts):
            if f":named evidence_{required}" not in body:
                raise ValueError(f"missing_named_evidence_assertion:{required}")
        for evidence in sorted(getattr(ir, "evidence_facts", {}) or {}):
            if f":named evidence_{evidence}" not in body:
                raise ValueError(f"missing_named_evidence_assertion:{evidence}")

        if f":named formula_{ir.metric}" not in body:
            raise ValueError("missing_named_formula_assertion")

        return SmtValidationResult(smt_status="valid")


def _strip_comments(smtlib: str) -> str:
    return re.sub(r";[^\n]*", "", smtlib)
