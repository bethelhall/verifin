"""Resolve complete named metric definitions before their sub-operations.

The question, registered templates and independently grounded role identities
select the formula. Neither the proposed formula nor claimed value selects it.
"""
from __future__ import annotations

import ast
import math
import re

from verifiqa.formulas.evaluator import evaluate_formula, normalize_formula
from verifiqa.policy.registry import DEFAULT_POLICY_REGISTRY
from .policy_formula import _identifiers, _resolve_roles, _substitute_template


def _tokens(text):
    text = text.lower().replace("_", " ").replace("%", " percent ")
    text = re.sub(r"\b(?:fy\s*)?(?:19|20)\d{2}\b", " ", text)
    text = re.sub(r"\b(\d+)\s*yr\b", r"\1 year", text)
    text = re.sub(r"\bavg\b", "average", text)
    text = re.sub(r"\bpercentage\b|\bpercents\b", "percent", text)
    stop = {"of", "the", "a", "as", "for", "in", "on", "to", "and", "fy"}
    return frozenset(word[:-1] if len(word) > 3 and word.endswith("s") and word != "gross" else word
                     for word in re.findall(r"[a-z0-9]+", text) if word not in stop)


def requested_named_metric_policy(question):
    """Identify the requested complete named metric without a model label."""
    registry = DEFAULT_POLICY_REGISTRY
    # This path covers composed named metrics. Direct lookups and two-input
    # arithmetic retain their existing authorization paths.
    eligible = [policy for policy in registry.policies
                if not policy.metadata.get("requires_question_binding")
                and any(len(_identifiers(template)) >= 3 for template in policy.formula_templates)]
    head = question.split("?", 1)[0]
    head = re.split(r"\b(?:is defined as|defined as|define\b)", head, maxsplit=1, flags=re.I)[0]
    words = _tokens(head)
    matches = [(policy, tokens) for policy in eligible
               for alias in (policy.metric, *policy.metric_aliases)
               if (tokens := _tokens(alias)) and tokens <= words]
    maximal = [(policy, tokens) for policy, tokens in matches
               if not any(tokens < other for _, other in matches)]
    if len({policy.policy_id for policy, _ in maximal}) != 1:
        return None
    target = set().union(*(tokens for _, tokens in maximal))
    if (words & {"change", "increase", "decrease", "difference", "growth"}) - target:
        return None
    if words & {"average", "mean"} and not target & {"average", "mean"}:
        return None
    return maximal[0][0]


def _requested_policy(ir, question):
    requested = requested_named_metric_policy(question)
    proposed = DEFAULT_POLICY_REGISTRY.find_policy(ir.metric)
    return requested if requested is not None and proposed is not None and requested.policy_id == proposed.policy_id else None


def _numeric_target(question):
    head = question.split("?", 1)[0]
    if re.search(r"\b(?:which|who|rank|ranking|highest|lowest|biggest|smallest|winner|outcome|comparison)\b", head, re.I):
        return False
    return bool(re.search(r"\bwhat\s+(?:is|was|are|were)\b|\bhow\s+much\b|\b(?:calculate|compute)\b", head, re.I))


def _periods(ir, question, role_bindings):
    from .periods import fact_period_conflict, fact_period_key, quarter_number
    years = sorted({int(year) for year in re.findall(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", question, re.I)})
    if not years or re.search(r"\bquarter\b|\bQ[1-4]\b", question, re.I):
        return False
    if re.search(r"\b3\s*(?:year|yr)\b", question, re.I) and len(years) == 2 and years[-1] - years[0] == 2:
        years = list(range(years[0], years[-1] + 1))
    observed = {}
    for name, fact in ir.facts.items():
        key = fact_period_key(name, fact)
        period_text = f"{fact.period} {fact.column} {fact.source_column}"
        if (not key or fact_period_conflict(name, fact)
                or quarter_number(f"{name} {period_text}")
                or re.search(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|[1-9]|10|11)[ -]+months?\b"
                             r"|\b(?:semiannual|half[- ]year|quarter(?:ly)?)\b", period_text, re.I)):
            return False
        observed[name] = key[0]
    if set(observed.values()) != set(years):
        return False
    for role, name in role_bindings.items():
        ordinal = re.search(r"(?:^|_)y([1-9])(?:_|$)", role)
        expected = (years[int(ordinal.group(1)) - 1] if ordinal and int(ordinal.group(1)) <= len(years)
                    else years[0] if re.search(r"(?:^|_)(?:start|beginning|prior|previous)(?:_|$)", role)
                    else years[-1])
        if observed[name] != expected:
            return False
    return True


def _source_basis(ir, policy, bindings):
    from .money_units import money_currency, unrecognized_currency_unit
    units = {re.sub(r"\s+", " ", fact.unit.lower()).strip() for fact in ir.facts.values()}
    currencies = {money_currency(fact.unit) for fact in ir.facts.values()}
    if len(units) != 1 or len(currencies) != 1 or None in currencies:
        return False
    for role, name in bindings.items():
        fact = ir.facts[name]
        if unrecognized_currency_unit(fact.unit):
            return False
        expected = set(policy.roles[role].get("concept_ids", ()))
        allowed_qnames = {qname for concept in expected if concept in DEFAULT_POLICY_REGISTRY.concepts
                          for qname in DEFAULT_POLICY_REGISTRY.concepts[concept].xbrl_concepts}
        if fact.concept and fact.concept not in allowed_qnames:
            return False
        label = fact.row_label
        actual = DEFAULT_POLICY_REGISTRY.infer_concepts(label).get(label)
        if ((actual is not None and actual not in expected)
                or (actual not in expected and fact.concept not in allowed_qnames)):
            return False
        if "PpeLike" in expected:
            basis = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label + " " + fact.concept)
            if re.search(r"\bgross\b|\baccumulated\b", basis, re.I):
                return False
            if fact.concept and fact.concept.split(":")[-1] != "PropertyPlantAndEquipmentNet":
                return False
    return True


def _outflow_formula(formula, ir, policy, bindings, question, evidence):
    normalized = re.sub(r"\s+", " ", evidence).lower()
    for role, name in bindings.items():
        if "CapitalExpendituresLike" not in policy.roles[role].get("concept_ids", ()):
            continue
        fact = ir.facts[name]
        if fact.value >= 0:
            continue
        label = fact.row_label.lower()
        quote = re.sub(r"\s+", " ", fact.source_quote).lower()
        from verifiqa.agent.table_fact_extractor import _parse_number
        negative_tokens = re.findall(r"\(\s*\d[\d,]*(?:\.\d+)?\s*\)|-\s*\d[\d,]*(?:\.\d+)?", quote)
        raw = fact.raw_value if fact.raw_value is not None else fact.value
        witnessed_negative = any(_parse_number(token) is not None
                                 and math.isclose(_parse_number(token), raw, rel_tol=1e-12, abs_tol=1e-12)
                                 for token in negative_tokens)
        if (not re.search(r"\bcapex\b|\bcapital expenditures?\b", question, re.I)
                or not re.search(r"\bcash\s+flows?\b", normalized)
                or not quote or quote not in normalized
                or not re.search(r"capital expenditures?|purchases? of property", label)
                or not witnessed_negative):
            return None
        formula = re.sub(r"\b" + re.escape(name) + r"\b", "(-" + name + ")", formula)
    return formula


def _definition_matches(formula, ir, policy, bindings, question):
    """Parse only explicit, fully bound arithmetic definitions in the question."""
    definitions = re.findall(r"\b(?:is\s+)?defined\s+as\s*:?\s*([^.?]+)", question, re.I)
    if not definitions:
        return not re.search(r"\bdefine\b.+?\bas\b", question, re.I)
    from .periods import fact_period_key
    aliases = {}
    for role, name in bindings.items():
        aliases[name] = {alias for concept in policy.roles[role].get("concept_ids", ())
                         if concept in DEFAULT_POLICY_REGISTRY.concepts
                         for alias in DEFAULT_POLICY_REGISTRY.concepts[concept].aliases}

    def lookup(phrase, year):
        matches = [name for name, names in aliases.items()
                   if any(_tokens(phrase) == _tokens(alias) for alias in names)
                   and fact_period_key(name, ir.facts[name])[0] == int(year)]
        if len(matches) != 1:
            raise ValueError("ambiguous_definition_role")
        return matches[0]

    for definition in definitions:
        expression = definition.strip()
        def aggregate(match):
            before, after = lookup(match.group("measure"), match.group("first")), lookup(match.group("measure"), match.group("last"))
            return (f"(({before} + {after}) / 2)" if match.group("kind").lower() == "average"
                    else f"({after} - {before})")
        try:
            expression = re.sub(r"\b(?P<kind>average|change in)\s+(?P<measure>[A-Za-z& ,]+?)\s+between\s+"
                                r"(?:FY\s*)?(?P<first>(?:19|20)\d{2})\s+and\s+(?:FY\s*)?(?P<last>(?:19|20)\d{2})\b",
                                aggregate, expression, flags=re.I)
        except ValueError:
            return False
        for name, names in aliases.items():
            year = fact_period_key(name, ir.facts[name])[0]
            for alias in sorted(names, key=len, reverse=True):
                phrase = r"\s+".join(re.escape(part) for part in alias.split())
                expression = re.sub(r"\b(?:FY\s*)?" + str(year) + r"\s+" + phrase + r"\b", name, expression, flags=re.I)
        # Undated primitives are usable only when the entire requested
        # population contains one observation with that semantic identity.
        unique = {}
        for name, names in aliases.items():
            for alias in names:
                unique.setdefault(alias, set()).add(name)
        for alias in sorted(unique, key=len, reverse=True):
            if len(unique[alias]) == 1:
                phrase = r"\s+".join(re.escape(part) for part in alias.split())
                expression = re.sub(r"\b" + phrase + r"\b", next(iter(unique[alias])), expression, flags=re.I)
        expression = re.sub(r"\bdivided by\b", "/", expression, flags=re.I)
        expression = re.sub(r"\bplus\b", "+", expression, flags=re.I)
        expression = re.sub(r"\b(?:minus|less)\b", "-", expression, flags=re.I)
        try:
            if ast.dump(ast.parse(expression, mode="eval"), include_attributes=False) != ast.dump(ast.parse(formula, mode="eval"), include_attributes=False):
                return False
        except (ValueError, SyntaxError):
            return False
    return True


def resolve_named_metric_operation(ir, question, *, evidence_text=""):
    """Return one canonical, complete, question-compatible metric expression."""
    policy = _requested_policy(ir, question)
    if policy is None or not _numeric_target(question):
        return None
    from verifiqa.answer_spec import answer_spec_from_question
    expected_unit = answer_spec_from_question(question).expected_unit
    if expected_unit and expected_unit != ir.claim_unit:
        return None
    from verifiqa.types import VerificationFact
    from .xbrl_scope import requested_scopes
    from .money_units import explicit_money_currencies, money_currency
    scopes = set(requested_scopes(VerificationFact("", 0, "", row_label=question)))
    if any(set(requested_scopes(VerificationFact("", 0, "", row_label=fact.row_label,
                    source_section=fact.source_section))) != scopes for fact in ir.facts.values()):
        return None
    requested_currency = explicit_money_currencies(question)
    if requested_currency and any(money_currency(fact.unit) not in requested_currency for fact in ir.facts.values()):
        return None
    from .operation_bindings import question_scope_matches, source_sign_basis_supported
    from .question_measure_types import question_input_types_match
    if (not question_input_types_match(ir, question, None)
            or not question_scope_matches(ir, question, None)
            or not source_sign_basis_supported(ir, question, None, evidence_text)):
        return None
    mapping, _issues = _resolve_roles(policy, ir, DEFAULT_POLICY_REGISTRY)
    candidates = {}
    for template in policy.formula_templates:
        active = _identifiers(template)
        if not active <= set(mapping):
            continue
        bindings = {role: mapping[role] for role in active}
        if (set(bindings.values()) != set(ir.facts) or len(set(bindings.values())) != len(bindings)
                or not _periods(ir, question, bindings) or not _source_basis(ir, policy, bindings)):
            continue
        formula = _outflow_formula(_substitute_template(template, mapping), ir, policy, bindings, question, evidence_text)
        if formula is not None and _definition_matches(formula, ir, policy, bindings, question):
            candidates[normalize_formula(formula)] = (formula, bindings)
    if len(candidates) != 1:
        return None
    formula, bindings = next(iter(candidates.values()))
    try:
        computed = evaluate_formula(formula, {name: fact.value for name, fact in ir.facts.items()})
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    if not math.isfinite(computed):
        return None
    return {"formula": formula, "operation": "named_metric", "variables": sorted(ir.facts),
            "computed_value": computed, "policy_id": policy.policy_id, "role_bindings": bindings,
            "source_note": policy.metadata.get("formula_source_note", ""),
            "source_refs": list(policy.source_refs)}
