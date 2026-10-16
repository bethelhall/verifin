from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


ABSTENTION_MESSAGE = "The generated answer could not be verified against retrieved financial evidence."
UNVERIFIED_FORMULA_MESSAGE = "The generated answer could not be verified because the formula was not authorized."
VIOLATION_MESSAGE = "The claimed value was not verified: Z3 found a counterexample outside tolerance."


@dataclass
class FinanceBenchExample:
    financebench_id: str
    question: str
    answer: str = ""
    evidence: Any = None
    justification: str = ""
    question_type: str = ""
    question_reasoning: str = ""
    company: str = ""
    doc_name: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceChunk:
    chunk_id: str
    doc_name: str
    page: Optional[int]
    text: str
    source_type: str = "filing"
    company: str = ""
    financebench_id: str = ""


@dataclass
class RetrievalFact:
    name: str
    aliases: List[str] = field(default_factory=list)
    period: str = ""
    statement: str = ""


@dataclass
class RetrievalPlan:
    metric: str = ""
    facts: List[RetrievalFact] = field(default_factory=list)
    reason: str = ""


@dataclass
class VerificationSchema:
    metric: str
    formula: str
    allowed_variables: List[str]
    required_evidence: List[str]
    allowed_operators: List[str]
    tolerance: float
    unit_policy: str
    period_policy: str
    query_type: str = "counterexample"
    fact_units: Dict[str, str] = field(default_factory=dict)
    claim_unit: str = ""
    computed_unit: str = ""
    precision_digits: Optional[int] = None
    tolerance_source: str = ""


@dataclass
class VerificationFact:
    name: str
    value: float
    unit: str
    fact_type: str = "numeric"
    raw_value: Optional[float] = None
    raw_unit: str = ""
    source_scale: str = ""
    source_scale_quote: str = ""
    source_quote: str = ""
    chunk_id: str = ""
    period: str = ""
    row_label: str = ""
    column: str = ""
    concept: str = ""  # authoritative XBRL concept qname; bind directly when set
    absence_scope: Dict[str, str] = field(default_factory=dict)
    source_section: str = ""
    source_column: str = ""
    source_table_id: str = ""
    xbrl_context_id: str = ""
    source_ref: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationIR:
    metric: str
    formula: str
    facts: Dict[str, VerificationFact]
    claimed_value: float
    claim_unit: str
    tolerance: float
    computed_unit: str = ""
    precision_digits: Optional[int] = None
    tolerance_source: str = ""
    period: str = ""
    query_type: str = "counterexample"
    evidence_facts: Dict[str, VerificationFact] = field(default_factory=dict)
    xbrl_calculations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim:
    metric: str
    claimed_value: float
    period: str = ""
    company: str = ""
    variables: Dict[str, float] = field(default_factory=dict)


@dataclass
class CertificateClaim:
    metric: str
    claimed_value: float
    unit: str = ""
    period: str = ""
    reported_value: Optional[float] = None


@dataclass
class Quantity:
    raw_value: float
    raw_unit: str
    source_scale: str
    source_scale_quote: str
    value: float
    unit: str


@dataclass
class CertificateFact:
    name: str
    value: float
    unit: str
    source_quote: str
    chunk_id: str
    period: str = ""
    row_label: str = ""
    column: str = ""
    fact_type: str = "numeric"
    raw_value: Optional[float] = None
    raw_unit: str = ""
    source_scale: str = ""
    source_scale_quote: str = ""
    # "as_reported" keeps the value's sign as printed; "magnitude" means the
    # canonical value is the absolute magnitude (e.g. capex reported as a negative
    # cash outflow but used as a positive quantity). Empty -> infer heuristically.
    sign_convention: str = ""
    absence_scope: Dict[str, str] = field(default_factory=dict)


@dataclass
class VerificationCertificate:
    claim: CertificateClaim
    facts: List[CertificateFact]
    formula: str
    calculation: str = ""
    tolerance: float = 0.01
    verifiable: bool = True
    reason: str = ""


@dataclass
class AnswerSpec:
    expected_unit: str = "unspecified"
    value_scale: float = 1.0
    tolerance: float = 0.01
    precision_digits: Optional[int] = None
    precision_kind: str = "default"
    percent_expected: bool = False
    percentage_points_expected: bool = False
    source: str = "question"
    tolerance_source: str = "default"


@dataclass
class SmtValidationResult:
    smt_status: str
    reason: str = ""


@dataclass
class SolverResult:
    solver_status: str
    raw_output: str = ""
    model: str = ""
    error: str = ""
    smt_path: str = ""
    unsat_core: list = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.solver_status == "UNSAT"


@dataclass
class DiagnosticTrace:
    solver_status: str
    schema_or_smt_error: str = ""
    counterexample_model: str = ""
    expected_value: Optional[float] = None
    claimed_value: Optional[float] = None
    evidence_facts: Dict[str, float] = field(default_factory=dict)
    evidence_provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    retrieved_formula: str = ""


@dataclass
class RavResult:
    financebench_id: str
    question: str
    answer: str
    final_status: str
    first_pass_solver_status: str
    gold_answer: str = ""
    verified: bool = False
    abstained: bool = False
    # Grounding tier: "xbrl_linkbase_grounded" = XBRL facts connected to filing
    # calculation-linkbase constraints; "xbrl_instance_grounded" = XBRL facts only;
    # "llm_only" = no usable XBRL binding supported the SMT verdict.
    grounding: str = "llm_only"
    rule_id: str = ""
    metric: str = ""
    certificate: Optional[VerificationCertificate] = None
    claim: Optional[Claim] = None
    revised_claim: Optional[Claim] = None
    smtlib: str = ""
    revised_smtlib: str = ""
    repair_history: list = field(default_factory=list)
    repair_rounds: int = 0
    diagnostics: Optional[DiagnosticTrace] = None
    retrieved_chunk_ids: list = field(default_factory=list)
    retrieval_plan: Optional[RetrievalPlan] = None
    verification_ir: Optional[VerificationIR] = None
    verification_checks: Dict[str, Any] = field(default_factory=dict)
    verifier_certificate: Dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value
