from __future__ import annotations

from typing import TYPE_CHECKING

from verifiqa.formulas.evaluator import formula_variables

from .registry import DEFAULT_POLICY_REGISTRY, PolicyRegistry
from .types import PolicyCheckResult

if TYPE_CHECKING:
    from verifiqa.types import VerificationIR


class PolicySemanticChecker:
    def __init__(self, registry: PolicyRegistry | None = None):
        self.registry = registry or DEFAULT_POLICY_REGISTRY

    def validate_ir(self, ir: VerificationIR, *, question: str = "", evidence_text: str = "") -> PolicyCheckResult:
        policy = self.registry.find_policy(ir.metric)
        variable_concepts = self.registry.infer_concepts(*ir.facts.keys())

        if policy is None:
            return PolicyCheckResult(
                status="no_policy",
                valid=True,
                reason="no_policy_for_metric",
                variable_concepts=variable_concepts,
            )

        if policy.metadata.get("requires_question_binding"):
            from verifiqa.verification.policy_formula import resolve_policy_formula
            resolved = resolve_policy_formula(ir, self.registry, question=question, evidence_text=evidence_text)
            valid = resolved.get("status") == "applied"
            return PolicyCheckResult(status="passed" if valid else "failed", valid=valid,
                reason=str(resolved.get("reason", "operation_roles_unresolved_or_ambiguous")),
                policy_id=policy.policy_id, variable_concepts=variable_concepts,
                role_bindings=resolved.get("role_bindings", {}), matched_template=resolved.get("formula"))

        # Validate only facts that the effective formula actually uses. Extra facts in
        # the source LLM certificate should not make an approved registry formula fail.
        try:
            active_fact_names = [name for name in sorted(formula_variables(ir.formula)) if name in ir.facts]
        except Exception:
            active_fact_names = sorted(ir.facts)

        role_bindings: dict[str, str] = {}
        unmatched: list[str] = []

        for var_name in active_fact_names:
            concept_id = variable_concepts.get(var_name)
            matched_role = _find_role(var_name, concept_id, policy.roles)
            if matched_role is None:
                unmatched.append(var_name)
            elif matched_role not in role_bindings:
                role_bindings[matched_role] = var_name

        if unmatched:
            return PolicyCheckResult(
                status="failed",
                valid=False,
                reason="formula_not_allowed_by_policy",
                policy_id=policy.policy_id,
                variable_concepts=variable_concepts,
                role_bindings=role_bindings,
            )

        matched_template = policy.formula_templates[0] if policy.formula_templates else None
        return PolicyCheckResult(
            status="passed",
            valid=True,
            reason="formula_matches_policy",
            policy_id=policy.policy_id,
            variable_concepts=variable_concepts,
            role_bindings=role_bindings,
            matched_template=matched_template,
        )


def _find_role(var_name: str, concept_id: str | None, roles: dict) -> str | None:
    """Return the role name that accepts this variable, or None."""
    for role_name, role_def in roles.items():
        allowed_concepts = role_def.get("concept_ids", []) if isinstance(role_def, dict) else []
        if concept_id and concept_id in allowed_concepts:
            return role_name
    return None
