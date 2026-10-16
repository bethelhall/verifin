from __future__ import annotations

from verifiqa.answerers.cot import format_evidence
from verifiqa.experiments.schemas import AnswerOutcome, SelectedEvidence, VerificationOutcome
from verifiqa.generation.llm_client import ChatMessage
from verifiqa.types import ABSTENTION_MESSAGE, FinanceBenchExample


_PROMPT = """\
You are a financial calculation verifier. Determine whether the proposed answer
to the question is numerically correct based solely on the provided evidence.

Steps:
1. Identify the relevant numbers in the evidence.
2. Perform the calculation yourself from scratch.
3. Compare your result to the proposed answer.

Respond with exactly one of:
  VERDICT: CORRECT
  VERDICT: INCORRECT

Then in one sentence explain why.

Question:
{question}

Evidence:
{evidence}

Proposed answer:
{answer}
"""


class LlmJudgeVerifier:
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
                verifier="llm_judge",
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
            stage="experiment_llm_judge",
        ).strip()
        correct = _parse_verdict(response)
        return VerificationOutcome(
            final_answer=answer.answer if correct else ABSTENTION_MESSAGE,
            verifier="llm_judge",
            verified=correct,
            abstained=not correct,
            final_status="VERIFIED" if correct else "ABSTAIN",
            verified_answer=answer.answer if correct else "",
            failure_reason="" if correct else "llm_judge_rejected",
            raw={"judge_response": response},
        )


def _parse_verdict(response: str) -> bool:
    for line in response.splitlines():
        normalized = line.strip().upper()
        if "VERDICT:" in normalized:
            return "CORRECT" in normalized and "INCORRECT" not in normalized
    head = response[:300].upper()
    if "VERDICT: INCORRECT" in head:
        return False
    if "VERDICT: CORRECT" in head:
        return True
    return False
