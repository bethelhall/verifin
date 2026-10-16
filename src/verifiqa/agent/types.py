from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RoleSpec:
    name: str
    aliases: list[str] = field(default_factory=list)
    period: str = ""
    # Authoritative XBRL concept qname (e.g. "us-gaap:RepaymentsOfLongTermDebt") when the
    # role is derived from a known calc-linkbase child — binds directly, skipping fuzzy
    # name→concept resolution that collapses distinct debt/flow roles onto one concept.
    concept: str = ""
    xbrl_context_id: str = ""  # Only independently reconstructed schema plans may pin this.
    source_ref: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimSpec:
    metric: str
    formula: str
    roles: list[RoleSpec]
    claim_unit: str
    tolerance: float
    formula_source: str  # "policy_registry" | "generic_operation" | "no_formula"
    period: str = ""
    operation: Optional[str] = None
    # LLM-parsed claim from the answer string (replaces regex _parse_number)
    claimed_value: Optional[float] = None
    claimed_unit: str = ""
    # Question/evidence-only planning provenance; never derived from the answer.
    planning_source: str = ""
    source_document_sha256: str = ""
    source_schema_sha256: str = ""
    source_report_unit: str = ""  # Printed display unit, never inferred from numeric agreement.
    source_selection_id: str = ""  # Digest of a revalidated question/source plan, never the answer.


@dataclass
class ClaimedAnswer:
    value: Optional[float]
    unit: str = ""


@dataclass
class GroundedFact:
    name: str
    value: float
    unit: str
    source_quote: str
    period: str = ""
    row_label: str = ""
    fact_type: str = "numeric"
    concept: str = ""  # authoritative XBRL concept qname carried from the RoleSpec
    source_value: Optional[float] = None  # deterministic source cell before normalization
    source_unit: str = ""
    source_section: str = ""  # explicit table/section heading, not neighboring narrative
    source_column: str = ""  # original dated or typed source-column header
    source_table_id: str = ""  # local table boundary; equal cells in distinct tables stay distinct
    xbrl_context_id: str = ""
    source_ref: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    question: str
    answer: str
    status: str  # VERIFIED | VIOLATED | ABSTAIN
    metric: str = ""
    formula: str = ""
    formula_source: str = ""
    facts: dict[str, GroundedFact] = field(default_factory=dict)
    claimed_value: Optional[float] = None
    smtlib: str = ""
    solver_status: str = ""
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Historical callers may still construct the old fourth status. Expose
        # only an abstention while retaining why the formula was withheld.
        if self.status == "UNVERIFIED_FORMULA":
            self.status = "ABSTAIN"
            self.failure_reason = self.failure_reason or "formula_unverified"
            self.diagnostics = {**(self.diagnostics or {}), "abstention_category": "unverified_formula"}
