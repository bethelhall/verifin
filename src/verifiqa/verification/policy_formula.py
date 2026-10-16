"""Canonical-formula grounding for independently verified metric formulas.

XBRL/table/document grounding gives the solver independent values for metric inputs.
This module supplies the matching independent anchor for formulas: for metrics that
have a canonical definition in the policy registry, resolve that definition into the
IR's fact names and require the claim to match the canonical computation.
"""
from __future__ import annotations

import re
from typing import Any

from verifiqa.formulas.evaluator import (
    FormulaError,
    evaluate_formula,
    formula_assertion_smt,
    formula_domain_constraints_smt,
)
from verifiqa.policy.registry import DEFAULT_POLICY_REGISTRY, PolicyRegistry
from verifiqa.types import VerificationFact, VerificationIR

def augment_smt_with_policy_formula(
    smtlib: str, ir: VerificationIR, registry: PolicyRegistry | None = None
) -> str:
    """Append an independent canonical-formula constraint when available."""
    resolved = resolve_policy_formula(ir, registry)
    if resolved.get("status") != "applied":
        return smtlib
    if ir.claimed_value is None:
        return smtlib

    declared = set(re.findall(r"\(declare-const\s+([A-Za-z_][A-Za-z0-9_]*)", smtlib))
    variables = set(resolved.get("variables") or [])
    if not variables <= declared:
        return smtlib

    claim_value = ir.claimed_value
    claim_s = _format_number(claim_value)
    tol = _format_number(ir.tolerance if ir.tolerance is not None else 0.0)

    formula = str(resolved.get("formula") or "")
    try:
        formula_eq = formula_assertion_smt(_policy_value_var(ir.metric), formula, ir.claim_unit or ir.computed_unit or "")
        guards = formula_domain_constraints_smt(_policy_value_var(ir.metric), formula, ir.claim_unit or ir.computed_unit or "")
    except FormulaError:
        return smtlib

    metric = ir.metric
    policy_value = _policy_value_var(metric)
    closeness = f"(and (<= (- {policy_value} {claim_s}) {tol}) (<= (- {claim_s} {policy_value}) {tol}))"
    parts = [formula_eq, *guards, closeness]
    body = f"(and {' '.join(parts)})" if len(parts) > 1 else parts[0]
    lines = [
        "",
        f"; Independent canonical formula for {metric} from the policy registry "
        f"({resolved.get('policy_id', '')}): the claim must match this form, else UNSAT.",
        f"(declare-const {policy_value} Real)",
        f"(assert (! {body} :named policy_formula_{metric}))",
    ]
    return _insert_before_check_sat(smtlib, "\n".join(lines))


def resolve_policy_formula(
    ir: VerificationIR,
    registry: PolicyRegistry | None = None,
    *,
    ignore_unit_matching: bool = False,
    question: str = "",
    evidence_text: str = "",
) -> dict[str, Any]:
    """Resolve the registry's canonical metric formula into this IR's fact names.

    The registry may carry scale variants. Pick the resolvable template whose
    recomputed value is closest to the claimed answer, using source-grounded values
    when XBRL/table/document bindings are available.
    """
    registry = registry or DEFAULT_POLICY_REGISTRY
    policy = registry.find_policy(ir.metric)
    if policy is None or not policy.formula_templates:
        result = {
            "status": "no_policy",
            "valid": False,
            "reason": "no_policy_for_metric",
        }
        # Surface the closest registry metric so a naming/coverage gap is visible
        # in diagnostics rather than silently abstaining.
        nearest = None
        if hasattr(registry, "nearest_policy_metric"):
            nearest = registry.nearest_policy_metric(ir.metric)
        if nearest:
            result["nearest_policy_metric"] = nearest
        return result

    if policy.metadata.get("requires_question_binding"):
        if not question:
            return {"status": "unresolved", "valid": False,
                    "reason": "operation_requires_question_binding", "policy_id": policy.policy_id}
        from .generic_ops import resolve_generic_operation
        resolved = resolve_generic_operation(ir, question, registry=registry, evidence_text=evidence_text)
        expected = {"arithmetic_mean": "average", "ratio": "generic_ratio"}.get(
            policy.metadata.get("resolver_operation"), policy.metadata.get("resolver_operation"))
        if resolved is None or resolved["operation"] != expected:
            return {"status": "unresolved", "valid": False,
                    "reason": "operation_roles_unresolved_or_ambiguous", "policy_id": policy.policy_id}
        return {**resolved, "valid": True, "reason": "operation_formula_resolved",
                "metric": policy.metric, "template": resolved["formula"],
                "role_bindings": {k: v for k, v in resolved.items() if isinstance(v, str) and v in ir.facts},
                "policy_source_type": policy.source_type, "policy_source_refs": policy.source_refs,
                "policy_metadata": policy.metadata}

    role_to_var, role_issues = _resolve_roles(
        policy,
        ir,
        registry,
        ignore_unit_matching=ignore_unit_matching,
    )
    if not role_to_var:
        return {
            "status": "unresolved",
            "valid": False,
            "reason": role_issues[0] if role_issues else "policy_roles_unresolved_or_ambiguous",
            "policy_id": policy.policy_id,
            "role_issues": role_issues,
        }

    source_values = _authoritative_values(
        ir,
        policy,
        role_to_var,
        registry,
        ignore_unit_matching=ignore_unit_matching,
    )
    candidates: list[dict[str, Any]] = []
    for template in policy.formula_templates:
        identifiers = _identifiers(template)
        if not identifiers or not identifiers <= set(role_to_var):
            continue
        formula = _substitute_template(template, role_to_var)
        if not formula:
            continue
        variables = sorted(_identifiers(formula))
        if not set(variables) <= set(source_values):
            continue
        candidate = {
            "template": template,
            "formula": formula,
            "variables": variables,
        }
        try:
            computed = evaluate_formula(formula, {name: source_values[name] for name in variables})
            candidate["computed_value"] = computed
            candidate["claim_error"] = (
                abs(float(ir.claimed_value) - computed)
                if ir.claimed_value is not None
                else None
            )
        except (FormulaError, ZeroDivisionError, ValueError):
            candidate["computed_value"] = None
            candidate["claim_error"] = None
        candidates.append(candidate)

    if not candidates:
        return {
            "status": "unresolved",
            "valid": False,
            "reason": role_issues[0] if role_issues else "policy_templates_unresolved",
            "policy_id": policy.policy_id,
            "role_bindings": role_to_var,
            "role_issues": role_issues,
        }

    metadata = getattr(policy, "metadata", {}) or {}
    if metadata.get("nonnegative_output"):
        nonnegative = [
            c for c in candidates
            if c.get("computed_value") is None or float(c["computed_value"]) >= -1e-12
        ]
        if nonnegative:
            candidates = nonnegative

    candidates.sort(
        key=lambda c: (
            float("inf") if c.get("claim_error") is None else float(c["claim_error"]),
            len(c.get("formula") or ""),
        )
    )
    chosen = candidates[0]
    return {
        "status": "applied",
        "valid": True,
        "reason": "policy_formula_resolved",
        "policy_id": policy.policy_id,
        "metric": policy.metric,
        "policy_source_type": policy.source_type,
        "policy_source_refs": list(policy.source_refs),
        "policy_metadata": dict(metadata),
        "template": chosen["template"],
        "formula": chosen["formula"],
        "role_bindings": role_to_var,
        "variables": chosen["variables"],
        "computed_value": chosen.get("computed_value"),
        "claim_error": chosen.get("claim_error"),
    }


# -- role resolution ---------------------------------------------------------

def _resolve_roles(
    policy: Any,
    ir: VerificationIR,
    registry: PolicyRegistry,
    *,
    ignore_unit_matching: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """Map each policy role to one IR fact variable, by concept and role constraints."""
    variable_concepts = registry.infer_concepts(*ir.facts.keys())
    role_to_var: dict[str, str] = {}
    issues: list[str] = []
    for role_name, role_def in policy.roles.items():
        allowed = role_def.get("concept_ids", []) if isinstance(role_def, dict) else []
        concept_matches = [v for v in ir.facts if variable_concepts.get(v) in allowed]
        matches = [
            v for v in concept_matches
            if _fact_satisfies_role_constraints(
                ir.facts[v],
                role_name,
                role_def,
                registry,
                ignore_unit_matching=ignore_unit_matching,
            )
        ]
        if concept_matches and not matches:
            issues.append(f"policy_role_constraints_failed:{role_name}")
        elif not matches and _forbidden_substitute_present(ir, role_def):
            issues.append(f"policy_role_constraints_failed:{role_name}")
        specific_matches = _specific_role_matches(role_name, matches)
        if specific_matches:
            matches = specific_matches
        if len(matches) > 1:
            issues.append(f"policy_role_ambiguous:{role_name}")
            continue
        if matches:
            role_to_var[role_name] = matches[0]
    return role_to_var, issues


def _identifiers(expr: str) -> set[str]:
    function_names = {"cagr", "cagr_percent", "coverage_ratio"}
    return {
        token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
        if token not in function_names
    }


def _specific_role_matches(role_name: str, matches: list[str]) -> list[str]:
    if len(matches) <= 1:
        return []
    role_years = set(re.findall(r"(?:19|20)\d{2}", role_name or ""))
    if role_years:
        year_matches = [name for name in matches if role_years <= set(re.findall(r"(?:19|20)\d{2}", name))]
        if year_matches:
            return year_matches

    period_match = _period_role_match(role_name, matches)
    if period_match:
        return period_match

    role = _norm(role_name)
    if not role:
        return []
    role_tokens = _specific_tokens(role)
    if role_tokens:
        token_matches = [
            name for name in matches
            if role_tokens <= _specific_tokens(_norm(name))
        ]
        if token_matches:
            return token_matches

    out = []
    for name in matches:
        normalized = _norm(name)
        if role == normalized or role in normalized or normalized in role:
            out.append(name)
    return out


def _period_role_match(role_name: str, matches: list[str]) -> list[str]:
    ordered = _matches_ordered_by_year(matches)
    if len(ordered) <= 1:
        return []
    role = _norm(role_name)
    y_match = re.search(r"(?:^| )y([1-9])(?: |$)", role)
    if y_match:
        index = int(y_match.group(1)) - 1
        return [ordered[index]] if index < len(ordered) else []
    if any(token in role.split() for token in {"start", "begin", "beginning", "prior", "previous"}):
        return [ordered[0]]
    if any(token in role.split() for token in {"end", "ending", "current", "latest", "new"}):
        return [ordered[-1]]
    return []


def _matches_ordered_by_year(matches: list[str]) -> list[str]:
    keyed: list[tuple[int, str]] = []
    for name in matches:
        years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", name)]
        if years:
            keyed.append((years[-1], name))
    if len(keyed) != len(matches):
        return []
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [name for _year, name in keyed]


def _specific_tokens(text: str) -> set[str]:
    generic = {"and", "of", "the", "to", "in", "from", "with", "a", "an", "fy", "assets", "asset", "amount"}
    return {
        token for token in re.findall(r"[a-z0-9]+", text or "")
        if token not in generic and not re.fullmatch(r"(?:19|20)\d{2}", token)
    }


def _substitute_template(template: str, role_to_var: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return role_to_var.get(token, token)

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", repl, template)


def _authoritative_values(
    ir: VerificationIR,
    policy: Any,
    role_to_var: dict[str, str],
    registry: PolicyRegistry,
    *,
    ignore_unit_matching: bool = False,
) -> dict[str, float]:
    values = {name: float(fact.value) for name, fact in ir.facts.items()}
    var_to_role = {var: role for role, var in role_to_var.items()}
    xbrl = getattr(ir, "xbrl_calculations", None) or {}
    source_priority = str((getattr(policy, "metadata", {}) or {}).get("source_priority") or "").strip().lower()
    for binding in xbrl.get("bindings") or []:
        if source_priority == "document" and str(binding.get("source") or "") != "source_document":
            continue
        fact_name = binding.get("fact_name")
        if not fact_name:
            continue
        role_name = var_to_role.get(str(fact_name))
        if role_name and not _binding_satisfies_role(
            binding,
            ir.facts.get(str(fact_name)),
            policy.roles.get(role_name, {}),
            registry,
            ignore_unit_matching=ignore_unit_matching,
        ):
            continue
        if float(binding.get("binding_tolerance") or 0.0) > 0.0:
            continue
        try:
            source_value = float(binding.get("xbrl_value"))
        except (TypeError, ValueError):
            continue
        try:
            multiplier = float(binding.get("binding_multiplier", 1.0))
        except (TypeError, ValueError):
            multiplier = 1.0
        values[str(fact_name)] = source_value * multiplier
    return values


def _forbidden_substitute_present(ir: VerificationIR, role_def: Any) -> bool:
    if not isinstance(role_def, dict):
        return False
    forbidden = role_def.get("forbidden_substitute_variables_any") or []
    if not forbidden:
        return False
    haystack = " ".join([ir.formula or "", *ir.facts.keys()]).lower()
    return any(_contains_phrase(haystack, str(name)) for name in forbidden)


def _fact_satisfies_role_constraints(
    fact: VerificationFact,
    role_name: str,
    role_def: Any,
    registry: PolicyRegistry,
    *,
    ignore_unit_matching: bool = False,
) -> bool:
    if not isinstance(role_def, dict):
        return True

    expected_kind = _role_unit_kind(role_def, registry)
    if (
        not ignore_unit_matching
        and expected_kind
        and not _fact_unit_compatible(expected_kind, fact)
    ):
        return False

    if fact.fact_type.startswith("absence") and role_def.get("allow_absence_fact") is False:
        return False

    source_text = " ".join(value for value in (fact.row_label, fact.source_quote) if value)
    required = role_def.get("required_source_text_any") or []
    if required and source_text and not any(_contains_phrase(source_text, phrase) for phrase in required):
        return False

    forbidden = role_def.get("forbidden_source_text_any") or []
    if forbidden and source_text and any(_contains_phrase(source_text, phrase) for phrase in forbidden):
        return False

    return True


def _binding_satisfies_role(
    binding: dict[str, Any],
    fact: VerificationFact | None,
    role_def: Any,
    registry: PolicyRegistry,
    *,
    ignore_unit_matching: bool = False,
) -> bool:
    if ignore_unit_matching:
        return True
    if not isinstance(role_def, dict):
        return True
    expected_kind = _role_unit_kind(role_def, registry)
    if not expected_kind:
        return True
    actual_kind = _binding_unit_kind(binding)
    if actual_kind != "unknown":
        return _unit_kind_compatible(expected_kind, actual_kind)
    if fact is not None:
        return _fact_unit_compatible(expected_kind, fact)
    return False


def _fact_unit_compatible(expected_kind: str, fact: VerificationFact) -> bool:
    candidates = [fact.unit or "", fact.raw_unit or ""]
    return any(
        _unit_kind_compatible(expected_kind, _unit_kind_from_text(candidate))
        for candidate in candidates
        if candidate
    )


def _role_unit_kind(role_def: dict[str, Any], registry: PolicyRegistry) -> str:
    explicit = str(role_def.get("unit_kind") or "").strip().lower()
    if explicit:
        return explicit
    kinds = []
    for concept_id in role_def.get("concept_ids", []) or []:
        concept = registry.concepts.get(concept_id)
        if concept and concept.unit_kind:
            kinds.append(concept.unit_kind)
    kinds = [kind for kind in kinds if kind and kind != "unknown"]
    return kinds[0] if kinds and all(kind == kinds[0] for kind in kinds) else ""


def _binding_unit_kind(binding: dict[str, Any]) -> str:
    measures = binding.get("unit_measures") or []
    if measures:
        return _unit_kind_from_text(" ".join(str(value) for value in measures if value))
    return _unit_kind_from_text(str(binding.get("unit_ref", "")))


def _unit_kind_from_text(value: str) -> str:
    text = (value or "").lower()
    if not text:
        return "unknown"
    if "per share" in text or "/share" in text or "per-share" in text:
        return "per_share"
    if "percent" in text or "percentage" in text or "%" in text:
        return "percent"
    if "ratio" in text or "pure" in text or "xbrli:pure" in text:
        return "ratio"
    if "usd" in text or "$" in text or "iso4217" in text:
        return "money"
    return "unknown"


def _unit_kind_compatible(expected: str, actual: str) -> bool:
    expected = (expected or "").lower()
    actual = (actual or "").lower()
    if not expected or expected == "unknown" or not actual or actual == "unknown":
        return True
    if expected == actual:
        return True
    if expected == "percent" and actual == "ratio":
        return True
    if expected == "ratio" and actual == "percent":
        return True
    return False


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _norm(text)
    normalized_phrase = _norm(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text) is not None


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[,\.&]", " ", s)
    s = re.sub(r"[\s_\-]+", " ", s)
    return s.strip()


# -- infix -> SMT s-expression ---------------------------------------------

class _TranslationError(Exception):
    pass


_TOKEN = re.compile(r"\s*(?:(?P<num>\d+\.?\d*)|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[()+\-*/]))")


def _infix_to_smt(expr: str, var_map: dict[str, str]) -> str:
    """Translate a simple arithmetic infix expression into an SMT s-expression."""
    tokens = _tokenize(expr)
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def advance() -> str:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def parse_expr() -> str:
        node = parse_term()
        while peek() in ("+", "-"):
            op = advance()
            rhs = parse_term()
            node = f"({op} {node} {rhs})"
        return node

    def parse_term() -> str:
        node = parse_factor()
        while peek() in ("*", "/"):
            op = advance()
            rhs = parse_factor()
            node = f"({op} {node} {rhs})"
        return node

    def parse_factor() -> str:
        tok = peek()
        if tok is None:
            raise _TranslationError("unexpected end of expression")
        if tok == "-":
            advance()
            return f"(- {parse_factor()})"
        if tok == "(":
            advance()
            node = parse_expr()
            if peek() != ")":
                raise _TranslationError("missing closing paren")
            advance()
            return node
        advance()
        if re.fullmatch(r"\d+\.?\d*", tok):
            return tok
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            if tok not in var_map:
                raise _TranslationError(f"unresolved identifier {tok}")
            return var_map[tok]
        raise _TranslationError(f"unexpected token {tok}")

    node = parse_expr()
    if pos != len(tokens):
        raise _TranslationError("trailing tokens")
    return node


def _tokenize(expr: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        m = _TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            raise _TranslationError(f"bad token at {pos}: {expr[pos:]!r}")
        tokens.append(next(g for g in m.groups() if g is not None))
        pos = m.end()
    return tokens


def _denominators(template: str, var_map: dict[str, str]) -> list[str]:
    """SMT sub-expressions that appear as division denominators."""
    out: list[str] = []
    tokens = _tokenize(template)
    pos = 0
    while pos < len(tokens):
        if tokens[pos] == "/" and pos + 1 < len(tokens):
            rhs = tokens[pos + 1]
            if rhs == "(":
                depth, j = 0, pos + 1
                while j < len(tokens):
                    if tokens[j] == "(":
                        depth += 1
                    elif tokens[j] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                try:
                    out.append(_infix_to_smt(" ".join(tokens[pos + 1:j + 1]), var_map))
                except _TranslationError:
                    pass
            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rhs) and rhs in var_map:
                out.append(var_map[rhs])
            elif re.fullmatch(r"\d+\.?\d*", rhs):
                pass
        pos += 1
    return out


# -- helpers ----------------------------------------------------------------

def _format_number(value: Any) -> str:
    f = float(value)
    if f < 0:
        return f"(- {_format_number(abs(f))})"
    text = f"{f:.12g}"
    if "e" in text.lower():
        text = f"{f:.12f}".rstrip("0").rstrip(".")
    return text or "0"


def _policy_value_var(metric: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", metric or "metric").strip("_") or "metric"
    if not re.match(r"[A-Za-z_]", slug):
        slug = f"metric_{slug}"
    return f"policy_value_{slug}"


def _insert_before_check_sat(smtlib: str, block: str) -> str:
    idx = smtlib.rfind("(check-sat)")
    if idx == -1:
        return smtlib + "\n" + block + "\n"
    return smtlib[:idx] + block + "\n\n" + smtlib[idx:]
