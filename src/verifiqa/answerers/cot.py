from __future__ import annotations

from verifiqa.experiments.schemas import AnswerOutcome, SelectedEvidence
from verifiqa.generation.llm_client import ChatMessage
from verifiqa.types import ABSTENTION_MESSAGE, FinanceBenchExample


_PROMPT = """\
Answer the following financial question using only the evidence provided.
Show the calculation steps needed to support the answer, then state the final
numeric answer clearly.

If the evidence is insufficient, say so rather than guessing.

Question:
{question}

Evidence:
{evidence}
"""


class CotAnswerer:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def answer(self, example: FinanceBenchExample, evidence: SelectedEvidence) -> AnswerOutcome:
        if not evidence.chunks:
            return AnswerOutcome(
                answer=ABSTENTION_MESSAGE,
                answerer="cot",
                error=evidence.status,
            )
        content = self.llm_client.chat(
            [ChatMessage(role="user", content=_PROMPT.format(
                question=example.question,
                evidence=format_evidence(evidence),
            ))],
            temperature=0.0,
            stage="experiment_cot_answer",
        ).strip()
        return AnswerOutcome(
            answer=content or ABSTENTION_MESSAGE,
            answerer="cot",
            error="" if content else "empty_answer",
        )


def format_evidence(evidence: SelectedEvidence) -> str:
    parts = []
    for chunk in evidence.chunks:
        page = "" if chunk.page is None else f" page={chunk.page}"
        parts.append(f"[{chunk.chunk_id}{page}]\n{chunk.text}")
    return "\n\n".join(parts)
