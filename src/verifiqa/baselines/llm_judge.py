from __future__ import annotations

from verifiqa.baselines.vanilla_rag import _chunks_from_example
from verifiqa.generation.llm_client import ChatMessage
from verifiqa.retrieval.evidence_retriever import EvidenceRetriever
from verifiqa.types import FinanceBenchExample, RavResult, ABSTENTION_MESSAGE


_ANSWER_PROMPT = (
    "Answer the following financial question using only the evidence provided. "
    "Show your calculation steps, then state the final numeric answer clearly.\n\n"
    "Question: {question}\n\n"
    "Evidence:\n{evidence}"
)

_JUDGE_PROMPT = """\
You are a financial calculation verifier. Determine whether the proposed answer \
to the question is numerically correct based solely on the provided evidence.

Steps:
1. Identify the relevant numbers in the evidence.
2. Perform the calculation yourself from scratch.
3. Compare your result to the proposed answer (accept within 5% relative tolerance).

Respond with exactly one of:
  VERDICT: CORRECT
  VERDICT: INCORRECT

Then in one sentence explain why.

Question: {question}

Evidence:
{evidence}

Proposed answer: {answer}
"""


def _parse_verdict(response: str) -> bool:
    """Return True if the judge said CORRECT, False otherwise."""
    for line in response.splitlines():
        line = line.strip().upper()
        if "VERDICT:" in line:
            return "CORRECT" in line and "INCORRECT" not in line
    # Fallback: look for the words anywhere in the first 300 chars
    head = response[:300].upper()
    if "VERDICT: INCORRECT" in head:
        return False
    if "VERDICT: CORRECT" in head:
        return True
    # If the judge didn't follow the format, default to INCORRECT (conservative)
    return False


class LlmJudgeBaseline:
    """LLM-as-judge baseline: CoT answer → LLM critic → VERIFIED or ABSTAIN.

    Mirrors the critic-agent approach in Tan et al. (2025) / VERAFI.
    The same LLM both answers the question and then judges its own answer,
    making this a fair apples-to-apples comparison with VerifiQA's Z3 judge.

    Outcome labels:
      VERIFIED  — judge said CORRECT   (verified=True,  abstained=False)
      ABSTAIN   — judge said INCORRECT (verified=False, abstained=True)
    """

    def __init__(
        self,
        evidence_retriever: EvidenceRetriever,
        llm_client=None,
        evidence_callback=None,
        oracle_retrieval: bool = False,
    ):
        self.evidence_retriever = evidence_retriever
        self.llm_client = llm_client
        self.evidence_callback = evidence_callback
        self.oracle_retrieval = oracle_retrieval

    def run_example(self, example: FinanceBenchExample) -> RavResult:
        # --- evidence ---
        if self.oracle_retrieval and example.evidence:
            chunks = _chunks_from_example(example)
        else:
            chunks = self.evidence_retriever.retrieve(
                example.question, example.company, example.doc_name
            )

        if self.evidence_callback is not None:
            self.evidence_callback(
                example, None, chunks,
                "retrieved" if chunks else "no_retrieved_evidence",
            )

        if not chunks:
            return RavResult(
                financebench_id=example.financebench_id,
                question=example.question,
                gold_answer=example.answer,
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                verified=False,
                final_status="ABSTAIN",
                first_pass_solver_status="N/A",
            )

        evidence_text = "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)

        if self.llm_client is None:
            return RavResult(
                financebench_id=example.financebench_id,
                question=example.question,
                gold_answer=example.answer,
                answer=evidence_text,
                abstained=False,
                verified=False,
                final_status="ANSWERED",
                first_pass_solver_status="N/A",
            )

        # --- Pass 1: generate answer ---
        answer = self.llm_client.chat(
            [ChatMessage(role="user", content=_ANSWER_PROMPT.format(
                question=example.question,
                evidence=evidence_text,
            ))],
            temperature=0.0,
            stage="llm_judge_answer",
        ).strip()

        if not answer:
            return RavResult(
                financebench_id=example.financebench_id,
                question=example.question,
                gold_answer=example.answer,
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                verified=False,
                final_status="ABSTAIN",
                first_pass_solver_status="N/A",
            )

        # --- Pass 2: judge ---
        judge_response = self.llm_client.chat(
            [ChatMessage(role="user", content=_JUDGE_PROMPT.format(
                question=example.question,
                evidence=evidence_text,
                answer=answer,
            ))],
            temperature=0.0,
            stage="llm_judge_verdict",
        ).strip()

        verdict_correct = _parse_verdict(judge_response)

        return RavResult(
            financebench_id=example.financebench_id,
            question=example.question,
            gold_answer=example.answer,
            answer=answer,
            abstained=not verdict_correct,
            verified=verdict_correct,
            final_status="VERIFIED" if verdict_correct else "ABSTAIN",
            first_pass_solver_status="LLM_JUDGE",
        )

    # Legacy compatibility
    def run(self, example: FinanceBenchExample) -> str:
        return self.run_example(example).answer
