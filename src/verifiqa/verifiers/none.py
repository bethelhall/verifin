from __future__ import annotations

from verifiqa.experiments.schemas import AnswerOutcome, SelectedEvidence, VerificationOutcome
from verifiqa.types import FinanceBenchExample


class NoVerifier:
    def verify(
        self,
        example: FinanceBenchExample,
        evidence: SelectedEvidence,
        answer: AnswerOutcome,
    ) -> VerificationOutcome:
        abstained = bool(answer.error)
        return VerificationOutcome(
            final_answer=answer.answer,
            verifier="none",
            verified=False,
            abstained=abstained,
            final_status="ABSTAIN" if abstained else "ANSWERED",
            failure_reason=answer.error,
        )
