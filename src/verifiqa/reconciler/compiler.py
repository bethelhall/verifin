from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from verifiqa.formulas.evaluator import (
    FormulaError,
    format_number,
    formula_assertion_smt,
    formula_domain_constraints_smt,
    formula_variables,
)
from verifiqa.policy.registry import PolicyRegistry
from verifiqa.policy.types import FormulaPolicy
from verifiqa.types import VerificationFact, VerificationIR

from .context import GroundedSourceFact, ReconcilerContext
from .spec import ReconcilerSpec


@dataclass
class CompiledReconcilerCheck:
    status: str
    reason: str = ""
    smtlib: str = ""
    formula_id: str = ""
    formula_expression: str = ""
    claim_value: float | None = None
    claim_unit: str = ""
    absolute_tolerance: float | None = None
    relative_tolerance: float | None = None
    bound_facts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_reconciler_spec(
    spec: ReconcilerSpec,
    context: ReconcilerContext,
    ir: VerificationIR,
    registry: PolicyRegistry,
    *,
    relative_tolerance: float = 0.01,
) -> CompiledReconcilerCheck:
    formula_id = spec.formula.strip()
    if formula_id == "none":
        return CompiledReconcilerCheck(status="skipped", reason="formula_none")

    try:
        formula_expression = _select_formula_expression(formula_id, spec.fact_bindings, registry)
        used_variables = sorted(formula_variables(formula_expression))
    except (FormulaError, ValueError, SyntaxError) as exc:
        return CompiledReconcilerCheck(status="invalid", reason=f"formula_selection_failed:{exc}")

    source_index = _SourceIndex(context.source_facts)
    facts: dict[str, VerificationFact] = {}
    bound_records: list[dict[str, Any]] = []
    for variable in used_variables:
        pointer = spec.fact_bindings.get(variable)
        if not pointer:
            return CompiledReconcilerCheck(status="invalid", reason=f"missing_binding:{variable}")
        source_fact, reason = source_index.resolve(pointer)
        if source_fact is None:
            return CompiledReconcilerCheck(status="invalid", reason=f"binding_not_resolved:{variable}:{reason}")
        if not source_fact.grounded:
            return CompiledReconcilerCheck(status="invalid", reason=f"binding_not_grounded:{variable}:{pointer}")
        facts[variable] = VerificationFact(
            name=variable,
            value=float(source_fact.value),
            unit=source_fact.unit,
            source_quote=f"reconciler_source:{source_fact.source_id}",
            chunk_id=source_fact.source_id,
            period=source_fact.period,
            row_label=source_fact.concept,
        )
        bound_records.append(
            {
                "variable": variable,
                "source_id": source_fact.source_id,
                "fact_name": source_fact.fact_name,
                "concept": source_fact.concept,
                "period": source_fact.period,
                "unit": source_fact.unit,
                "source": source_fact.source,
                "value": float(source_fact.value),
            }
        )

    claim_value = float(ir.claimed_value) * float(spec.claim_scale)
    abs_tolerance = _absolute_tolerance(claim_value, ir, relative_tolerance)
    check_ir = VerificationIR(
        metric=ir.metric,
        formula=formula_expression,
        facts=facts,
        claimed_value=claim_value,
        claim_unit=spec.claim_unit or ir.claim_unit,
        tolerance=abs_tolerance,
        computed_unit=spec.claim_unit or ir.computed_unit or ir.claim_unit,
        precision_digits=None,
        tolerance_source=f"reconciler_relative:{relative_tolerance:g}",
        period=ir.period,
        query_type="reconciler_consistency",
    )
    try:
        smtlib = render_reconciler_smt(check_ir)
    except (FormulaError, SyntaxError) as exc:
        return CompiledReconcilerCheck(status="invalid", reason=f"smt_render_failed:{exc}")

    return CompiledReconcilerCheck(
        status="compiled",
        smtlib=smtlib,
        formula_id=formula_id,
        formula_expression=formula_expression,
        claim_value=claim_value,
        claim_unit=check_ir.claim_unit,
        absolute_tolerance=abs_tolerance,
        relative_tolerance=relative_tolerance,
        bound_facts=bound_records,
    )


def render_reconciler_smt(ir: VerificationIR) -> str:
    metric_var = _symbol(f"computed_{ir.metric}")
    fact_names = sorted(ir.facts)
    lines = [
        "(set-logic QF_NRA)",
        "(set-option :produce-unsat-cores true)",
        "(set-option :produce-models true)",
        "",
    ]
    for variable in [metric_var] + fact_names:
        lines.append(f"(declare-const {variable} Real)")
    lines.append("")
    for name in fact_names:
        lines.append(
            f"(assert (! (= {name} {format_number(ir.facts[name].value)}) :named {_symbol('reconciler_evidence_' + name)}))"
        )
    lines.append("")
    lines.append(
        f"(assert (! {formula_assertion_smt(metric_var, ir.formula, ir.computed_unit or ir.claim_unit)} :named {_symbol('reconciler_formula_' + ir.metric)}))"
    )
    for index, constraint in enumerate(
        formula_domain_constraints_smt(metric_var, ir.formula, ir.computed_unit or ir.claim_unit)
    ):
        lines.append(f"(assert (! {constraint} :named {_symbol(f'reconciler_domain_{ir.metric}_{index}')}))")

    claimed = format_number(float(ir.claimed_value))
    tolerance = format_number(float(ir.tolerance))
    lines.append("")
    lines.append(f"(assert (! (<= (- {metric_var} {claimed}) {tolerance}) :named reconciler_claim_upper))")
    lines.append(f"(assert (! (<= (- {claimed} {metric_var}) {tolerance}) :named reconciler_claim_lower))")
    lines.append("")
    lines.append("(check-sat)")
    lines.append("(get-unsat-core)")
    lines.append("(get-model)")
    return "\n".join(lines)


def _select_formula_expression(
    formula_id: str,
    fact_bindings: dict[str, str],
    registry: PolicyRegistry,
) -> str:
    if formula_id == "identity":
        if len(fact_bindings) != 1:
            raise ValueError("identity_requires_one_fact_binding")
        return next(iter(fact_bindings))

    policy = _policy_by_id(registry, formula_id)
    if policy is None:
        raise ValueError(f"unknown_policy_id:{formula_id}")
    binding_names = set(fact_bindings)
    for template in policy.formula_templates:
        variables = formula_variables(template)
        if variables <= binding_names:
            return template
    raise ValueError(f"no_formula_template_covered_by_bindings:{formula_id}")


def _policy_by_id(registry: PolicyRegistry, policy_id: str) -> FormulaPolicy | None:
    for policy in registry.policies:
        if policy.policy_id == policy_id:
            return policy
    return None


def _absolute_tolerance(claim_value: float, ir: VerificationIR, relative_tolerance: float) -> float:
    rel = abs(float(claim_value)) * abs(float(relative_tolerance))
    if rel > 0:
        return rel
    try:
        return max(abs(float(ir.tolerance)), 1e-9)
    except (TypeError, ValueError):
        return 1e-9


def _symbol(value: str, limit: int = 180) -> str:
    symbol = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    symbol = re.sub(r"_+", "_", symbol).strip("_")
    if not symbol:
        symbol = "symbol"
    if symbol[0].isdigit():
        symbol = f"v_{symbol}"
    return symbol[:limit]


class _SourceIndex:
    def __init__(self, source_facts: list[GroundedSourceFact]):
        self._by_source_id = _unique_index(source_facts, lambda fact: fact.source_id)
        self._by_fact_name = _unique_index(source_facts, lambda fact: fact.fact_name)
        self._by_concept = _unique_index(source_facts, lambda fact: fact.concept)

    def resolve(self, pointer: str) -> tuple[GroundedSourceFact | None, str]:
        pointer = (pointer or "").strip()
        for label, index in (
            ("source_id", self._by_source_id),
            ("concept", self._by_concept),
            ("fact_name", self._by_fact_name),
        ):
            value = index.get(pointer)
            if value == "AMBIGUOUS":
                return None, f"ambiguous_{label}:{pointer}"
            if isinstance(value, GroundedSourceFact):
                return value, label
        return None, f"unknown_pointer:{pointer}"


def _unique_index(
    facts: list[GroundedSourceFact],
    key_fn,
) -> dict[str, GroundedSourceFact | str]:
    index: dict[str, GroundedSourceFact | str] = {}
    for fact in facts:
        key = str(key_fn(fact) or "").strip()
        if not key:
            continue
        if key in index:
            index[key] = "AMBIGUOUS"
        else:
            index[key] = fact
    return index
