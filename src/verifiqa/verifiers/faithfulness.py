from __future__ import annotations

from verifiqa.answerers.cot import format_evidence
from verifiqa.experiments.schemas import AnswerOutcome, SelectedEvidence, VerificationOutcome
from verifiqa.generation.llm_client import ChatMessage
from verifiqa.types import ABSTENTION_MESSAGE, FinanceBenchExample


_PROMPT = """\
Check whether the proposed answer is faithful to the provided evidence.

Use only the evidence. This is a support/faithfulness check, not a helpfulness
or style judgment. Reject answers that contain unsupported numbers, use numbers
from the wrong period, apply an unsupported unit scale, or make a calculation
that is not entailed by the evidence.

Respond with JSON only:
{{
  "verdict": "SUPPORTED" or "UNSUPPORTED",
  "reason": "<one sentence>"
}}

Question:
{question}

Evidence:
{evidence}

Proposed answer:
{answer}
"""


class FaithfulnessVerifier:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def verify(
        self,
        example: FinanceBenchExample,
        evidence: SelectedEvidence,
        answer: AnswerOutcome,
    ) -> VerificationOutcome:
        if answer.error or not evidence.chunks:
            return VerificationOutcome(
                final_answer=ABSTENTION_MESSAGE,
                verifier="faithfulness",
                verified=False,
                abstained=True,
                final_status="ABSTAIN",
                failure_reason=answer.error or evidence.status,
            )
        response = self.llm_client.chat(
            [ChatMessage(role="user", content=_PROMPT.format(
                question=example.question,
                evidence=format_evidence(evidence),
                answer=answer.answer,
            ))],
            temperature=0.0,
            stage="experiment_faithfulness",
        ).strip()
        supported = _supported(response)
        return VerificationOutcome(
            final_answer=answer.answer if supported else ABSTENTION_MESSAGE,
            verifier="faithfulness",
            verified=supported,
            abstained=not supported,
            final_status="VERIFIED" if supported else "ABSTAIN",
            verified_answer=answer.answer if supported else "",
            failure_reason="" if supported else "faithfulness_rejected",
            raw={"faithfulness_response": response},
        )


def _supported(response: str) -> bool:
    normalized = response.upper()
    if '"VERDICT"' in normalized or "VERDICT" in normalized:
        return "SUPPORTED" in normalized and "UNSUPPORTED" not in normalized
    return "SUPPORTED" in normalized and "UNSUPPORTED" not in normalized
