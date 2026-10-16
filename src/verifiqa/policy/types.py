from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SemanticConcept:
    concept_id: str
    aliases: list[str]
    xbrl_concepts: list[str]
    unit_kind: str = "money"
    source_type: str = ""
    source_refs: list[dict] = field(default_factory=list)


@dataclass
class FormulaPolicy:
    policy_id: str
    metric: str
    formula_templates: list[str]
    roles: dict[str, Any]
    metric_aliases: list[str] = field(default_factory=list)
    source_type: str = ""
    source_refs: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.source_type


@dataclass
class PolicyCheckResult:
    status: str  # "no_policy" | "passed" | "failed"
    valid: bool
    reason: str
    policy_id: str | None = None
    variable_concepts: dict[str, str] = field(default_factory=dict)
    role_bindings: dict[str, str] = field(default_factory=dict)
    matched_template: str | None = None
