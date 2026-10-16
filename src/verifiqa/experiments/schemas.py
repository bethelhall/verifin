from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from verifiqa.types import EvidenceChunk, RetrievalPlan


@dataclass(frozen=True)
class MethodSpec:
    name: str
    evidence_mode: str
    answerer: str
    verifier: str
    description: str = ""


@dataclass
class SelectedEvidence:
    chunks: list[EvidenceChunk]
    status: str
    retrieval_plan: Optional[RetrievalPlan] = None


@dataclass
class AnswerOutcome:
    answer: str
    answerer: str
    raw: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class VerificationOutcome:
    final_answer: str
    verifier: str
    verified: bool = False
    abstained: bool = False
    final_status: str = "ANSWERED"
    verified_answer: str = ""
    corrected_answer: Optional[str] = None
    failure_reason: str = ""
    verification_checks: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    method: str
    evidence_mode: str
    answerer: str
    verifier: str
    financebench_id: str
    question: str
    gold_answer: str
    draft_answer: str
    final_answer: str
    verified_answer: str = ""
    corrected_answer: Optional[str] = None
    verified: bool = False
    abstained: bool = False
    correct: Optional[bool] = None
    draft_correct: Optional[bool] = None
    final_status: str = "ANSWERED"
    evidence_status: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    evidence_pages: list[dict[str, Any]] = field(default_factory=list)
    retrieval_plan: Optional[RetrievalPlan] = None
    llm_calls: int = 0
    runtime_seconds: float = 0.0
    failure_reason: str = ""
    verification_checks: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _to_jsonable(asdict(value))
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value
