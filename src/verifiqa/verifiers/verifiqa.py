from __future__ import annotations

from verifiqa.experiments.schemas import AnswerOutcome, SelectedEvidence, VerificationOutcome
from verifiqa.formalization.formalizer import Formalizer
from verifiqa.generation.answer_generator import AnswerGenerator
from verifiqa.pipeline import CounterexampleRavPipeline, PipelineConfig
from verifiqa.retrieval.planner import RetrievalPlanner
from verifiqa.types import ABSTENTION_MESSAGE, EvidenceChunk, FinanceBenchExample, RetrievalPlan, to_jsonable
from verifiqa.verification.smt_generator import SmtGenerator


class VerifiqaVerifier:
    def __init__(
        self,
        llm_client,
        strict: bool = False,
        policy_semantic_check: bool = False,
        certificate_grounding_check: bool = True,
    ):
        self.llm_client = llm_client
        self.strict = strict
        self.policy_semantic_check = policy_semantic_check
        self.certificate_grounding_check = certificate_grounding_check

    def verify(
        self,
        example: FinanceBenchExample,
        evidence: SelectedEvidence,
        answer: AnswerOutcome,
    ) -> VerificationOutcome:
        if answer.error or not evidence.chunks:
            return VerificationOutcome(
                final_answer=ABSTENTION_MESSAGE,
                verifier=self._name,
                verified=False,
                abstained=True,
                final_status="ABSTAIN",
                failure_reason=answer.error or evidence.status,
            )
        runner = CounterexampleRavPipeline(
            evidence_retriever=_FixedEvidenceRetriever(evidence.chunks),
            answer_generator=_FixedAnswerGenerator(answer.answer),
            formalizer=Formalizer(self.llm_client),
            retrieval_planner=_FixedRetrievalPlanner(evidence.retrieval_plan) if evidence.retrieval_plan else None,
            smt_generator=SmtGenerator(self.llm_client),
            config=PipelineConfig(
                evidence_top_k=len(evidence.chunks),
                oracle_retrieval=False,
                numeric_only=False,
                adaptive_retrieval=False,
                verify_original_answer=self.strict,
                allow_sat_repair=not self.strict,
                policy_semantic_check=self.policy_semantic_check,
                certificate_grounding_check=self.certificate_grounding_check,
            ),
        )
        result = runner.run_example(example)
        corrected_answer = None
        if result.verified and result.answer != answer.answer:
            corrected_answer = result.answer
        reason = ""
        if result.diagnostics is not None:
            reason = result.diagnostics.schema_or_smt_error
        return VerificationOutcome(
            final_answer=result.answer,
            verifier=self._name,
            verified=result.verified,
            abstained=result.abstained,
            final_status=result.final_status,
            verified_answer=result.answer if result.verified else "",
            corrected_answer=corrected_answer,
            failure_reason=reason,
            verification_checks=result.verification_checks,
            raw={"rav_result": to_jsonable(result)},
        )

    @property
    def _name(self) -> str:
        return "verifiqa_strict" if self.strict else "verifiqa_full"


class _FixedAnswerGenerator(AnswerGenerator):
    def __init__(self, answer: str):
        self.answer = answer

    def generate(self, question: str, evidence_chunks, retrieval_plan=None) -> str:
        return self.answer


class _FixedRetrievalPlanner(RetrievalPlanner):
    def __init__(self, plan: RetrievalPlan | None):
        self.fixed_plan = plan

    def plan(self, question: str) -> RetrievalPlan:
        return self.fixed_plan or RetrievalPlan()


class _FixedEvidenceRetriever:
    def __init__(self, chunks: list[EvidenceChunk]):
        self.chunks = list(chunks)

    def has_document(self, doc_name: str) -> bool:
        return True

    def retrieve(self, question: str, company: str = "", doc_name: str = "", top_k: int = 5, financebench_id: str = ""):
        return self.chunks[:top_k]

    def retrieve_with_plan(
        self,
        question: str,
        plan: RetrievalPlan,
        company: str = "",
        doc_name: str = "",
        top_k: int = 5,
        financebench_id: str = "",
    ):
        return self.chunks[:top_k]
