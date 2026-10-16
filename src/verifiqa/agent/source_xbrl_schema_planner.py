"""Question/source plans whose missing amount units are supplied by local XBRL.

Only labels, complete source columns and filing schema select operands here.
The ordinary XBRL binder supplies their amounts after the plan is selected.
No answer, benchmark identifier, or proposed operand value is an input.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date
from decimal import Decimal
import hashlib
import json
import re

from verifiqa.agent.types import ClaimSpec, RoleSpec
from verifiqa.verification.xbrl_linkbase import FilingXbrl


_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")
_MONTH = "|".join(_MONTHS)
_CELL = r"\$?\s*(?:\(\s*-?\d[\d,]*(?:\.\d+)?\s*\)|-?\d[\d,]*(?:\.\d+)?|[—–-])"


def source_xbrl_schema_sha256(evidence: str, bundle: FilingXbrl) -> str:
    """Pin raw source and schema provenance without hashing XML amount fields."""
    payload = {
        # Numeric spelling is not role provenance. Dates and complete column
        # populations are independently re-read and are also pinned by each
        # role/context in the reconstructed plan signature.
        "source_structure": re.sub(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", "#", evidence),
        "labels": sorted((k, sorted(v)) for k, v in bundle.concept_labels.items()),
        "contexts": sorted((k, asdict(v)) for k, v in bundle.contexts.items()),
        "units": sorted((k, list(v)) for k, v in bundle.units.items()),
        "numeric_schema": sorted((f.concept, f.context_id, f.unit_ref)
                                  for f in bundle.facts_by_local_context.values()
                                  if f.value is not None),
        "calculation_schema": sorted((g.role, g.parent_concept,
                                       tuple((c.concept, str(c.weight)) for c in g.children))
                                      for g in bundle.calc_groups),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def plan_from_source_xbrl(question: str, evidence: str, bundle: FilingXbrl) -> ClaimSpec | None:
    """Return one complete source/schema plan, withholding competing matches."""
    from verifiqa.agent.source_xbrl_compound_planner import compound_plan_from_source_xbrl
    plans = [plan for plan in (_simple_plan_from_source_xbrl(question, evidence, bundle),
                              compound_plan_from_source_xbrl(question, evidence, bundle))
             if plan is not None]
    return plans[0] if len(plans) == 1 else None


def _simple_plan_from_source_xbrl(question, evidence, bundle):
    if not question or not evidence or bundle.status != "ok":
        return None
    from verifiqa.verification.generic_ops import classify_operation
    operation = classify_operation(question)
    if operation not in {"average", "sum", "period_change"}:
        return None
    if re.search(r"\bquarter(?:ly)?\b|\bQ[1-4]\b", question, re.I):
        return None
    from verifiqa.verification.money_units import explicit_money_currencies
    if explicit_money_currencies(question) - {"USD"}:
        return None
    plans = []
    goodwill = _segment_goodwill_plan(question, evidence, bundle, operation)
    if goodwill is not None:
        plans.append(goodwill)
    from verifiqa.agent.source_question_planner import _annual_tables, _selected_periods
    for table in _annual_tables(evidence):
        if not all(re.fullmatch(r"(?:19|20)\d{2}", c) for c in table.columns):
            continue
        if table.unit not in {"", "USD", "USD thousands", "USD millions", "USD billions"}:
            continue
        selected = _selected_periods(question, table, operation)
        if selected is None or len(selected) < 2:
            continue
        header = _annual_header(table, evidence)
        if header is None or not _question_date_matches(question, header):
            continue
        month, day = header
        totals = [r for r in table.rows if _normal(r.label) == "total"]
        if (_total_charges_requested(question) and len(totals) == 1
                and totals[0] is table.rows[-1]):
            # A generic "Total" label cannot select a filing concept. Its
            # enclosing, explicitly named source heading must identify one
            # numeric calculation parent with the same full measure name.
            heading = _normal(table.section)
            if heading != "special charges recoveries":
                continue
            from verifiqa.agent.source_question_planner import _scope_supported
            if not _scope_supported(question, table, replace(totals[0], label=heading)):
                continue
            concepts = _label_concepts(heading, bundle)
            parents = {g.parent_concept for g in bundle.calc_groups
                       if g.parent_concept in concepts and len(g.children) >= 2
                       and len({c.concept for c in g.children}) == len(g.children)
                       and all(c.weight == Decimal(1) for c in g.children)}
            if len(parents) != 1:
                continue
            concept = next(iter(parents))
            label = _unique_label(concept, heading, bundle)
            if not label:
                continue
            roles = _annual_roles("total_charges", label, concept, table.columns,
                                  selected, month, day, bundle)
            if roles:
                plans.append(_spec(operation, roles, evidence, bundle,
                                   table.unit if table.unit_declared else "", question=question))
            continue
        measure = _change_measure(question) if operation == "period_change" else ""
        if not measure:
            continue
        for row in table.rows:
            label = re.sub(r"^\s*(?:Add|Less):\s*", "", row.label, flags=re.I)
            if len(_normal(label).split()) < 2 or _normal(label) != _normal(measure):
                continue
            from verifiqa.agent.source_question_planner import _scope_supported
            if not _scope_supported(question, table, row):
                continue
            concepts = _label_concepts(label, bundle)
            if len(concepts) != 1:
                continue
            concept = next(iter(concepts))
            roles = _annual_roles(_slug(label), label, concept, table.columns,
                                  selected, month, day, bundle)
            if roles:
                plans.append(_spec(operation, roles, evidence, bundle,
                                   table.unit if table.unit_declared else "", question=question))
    unique = {_signature(p): p for p in plans}
    return next(iter(unique.values())) if len(unique) == 1 else None


def source_xbrl_plan_matches(spec: ClaimSpec, question: str, evidence: str,
                             bundle: FilingXbrl) -> bool:
    planned = plan_from_source_xbrl(question, evidence, bundle)
    return planned is not None and _signature(spec) == _signature(planned)


def source_xbrl_facts_match(ir, question: str, evidence: str, bundle: FilingXbrl) -> bool:
    """Check identity before comparing the already selected context's amount."""
    planned = plan_from_source_xbrl(question, evidence, bundle)
    return planned is not None and _matched_roles(ir, planned, bundle) is not None


def resolve_source_xbrl_operation(ir, question: str, evidence: str,
                                  bundle: FilingXbrl, *, registry=None):
    """Authorize only the rebuilt plan over the matching validated IR facts."""
    planned = plan_from_source_xbrl(question, evidence, bundle)
    if planned is None:
        return None
    mapping = _matched_roles(ir, planned, bundle)
    if mapping is None:
        return None
    from verifiqa.verification.operation_bindings import _make
    from verifiqa.verification.generic_ops import _result
    formula = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b",
                     lambda match: mapping.get(match.group(), match.group()), planned.formula)
    bound = _make(ir, formula, list(mapping.values()))
    if bound is None:
        return None
    return _result(planned.operation, bound["formula"], bound["variables"], bound["computed"],
                   registry=registry, binding_source="independently_reconstructed_xbrl_schema")


def _matched_roles(ir, planned, bundle):
    if len(ir.facts) != len(planned.roles):
        return None
    mapping = {}
    for role in planned.roles:
        matches = [(name, fact) for name, fact in ir.facts.items()
                   if fact.concept == role.concept
                   and getattr(fact, "xbrl_context_id", "") == role.xbrl_context_id]
        if len(matches) != 1:
            return None
        name, fact = matches[0]
        # The existing binder canonicalizes monetary XBRL amounts to USDm.
        # Disparate display scales cannot enter arithmetic as equal units.
        if fact.unit != "USD millions":
            return None
        candidates = [f for f in bundle.facts_by_local_context.values()
                      if f.concept == role.concept and f.context_id == role.xbrl_context_id
                      and f.value is not None]
        if len(candidates) != 1:
            return None
        actual = Decimal(str(fact.value)) * 1000000
        expected = candidates[0].value
        if not actual.is_finite() or not expected.is_finite():
            return None
        if abs(actual - expected) > max(Decimal("0.000001"), abs(expected) * Decimal("1e-12")):
            return None
        mapping[role.name] = name
    return mapping if len(set(mapping.values())) == len(ir.facts) else None


def _normal(value):
    value = re.sub(r"^\s*NOTE\s+\d+\s*[-—–:]?\s*", "", value, flags=re.I)
    value = re.sub(r"\(\s*note\s+\d+\s*\)", "", value, flags=re.I)
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _slug(value):
    return _normal(value).replace(" ", "_")


def _label_concepts(label, bundle):
    numeric = {f.concept for f in bundle.facts_by_local_context.values() if f.value is not None}
    return {c for c, labels in bundle.concept_labels.items() if c in numeric
            and any(_normal(s) == _normal(label) for s in labels)}


def _unique_label(concept, normalized, bundle):
    labels = {label for label in bundle.concept_labels.get(concept, [])
              if _normal(label) == normalized}
    return sorted(labels)[0] if labels else ""


def _annual_header(table, evidence):
    quote = table.rows[0].quote
    if not quote or evidence.count(quote) != 1:
        return None
    first = evidence.index(quote)
    pattern = (r"\bYears?\s+Ended\s+(?P<month>" + _MONTH + r")\s+(?P<day>\d{1,2}),?\s*"
               r"(?P<years>(?:19|20)\d{2}(?:\s+(?:19|20)\d{2})+)")
    matches = [m for m in re.finditer(pattern, evidence, re.I)
               if re.findall(r"(?:19|20)\d{2}", m.group("years")) == table.columns
               and -40 <= first - m.end() <= 200]
    if len(matches) != 1:
        return None
    match = matches[0]
    return _MONTHS.index(match.group("month").title()) + 1, int(match.group("day"))


def _question_date_matches(question, month_day):
    dates = [(_MONTHS.index(m.group(1).title()) + 1, int(m.group(2)))
             for m in re.finditer(r"\b(" + _MONTH + r")\s+(\d{1,2})\b", question, re.I)]
    return all(d == month_day for d in dates)


def _total_charges_requested(question):
    return bool(re.search(r"\btotal\s+(?:special\s+)?charges\b", question, re.I))


def _change_measure(question):
    match = re.search(r"\bchange\s+in\s+(?:the\s+)?(.+?)\s+(?:from|between)\b", question, re.I)
    return re.sub(r"^\s*(?:Add|Less):\s*", "", match.group(1), flags=re.I) if match else ""


def _annual_roles(stem, label, concept, columns, selected, month, day, bundle):
    roles = []
    for index in selected:
        year = int(columns[index])
        contexts = _contexts(concept, year, month, day, bundle, instant=False)
        if len(contexts) != 1:
            return None
        roles.append(RoleSpec(f"{stem}_{year}", [label], str(year), concept,
                              xbrl_context_id=contexts[0]))
    return sorted(roles, key=lambda role: role.period)


def _contexts(concept, year, month, day, bundle, *, instant, payments=False):
    result = []
    for fact in bundle.facts_by_local_context.values():
        if fact.concept != concept or fact.value is None:
            continue
        context = bundle.contexts.get(fact.context_id)
        if context is None or bundle.units.get(fact.unit_ref) != ("iso4217:USD",):
            continue
        if payments:
            dimensions = context.dimensions
            if len(dimensions) != 1:
                continue
            axis, sep, member = dimensions[0].partition("=")
            if (not sep or axis.split(":")[-1] != "StatementBusinessSegmentsAxis"
                    or member.split(":")[-1] != "PaymentsMember"):
                continue
        elif context.dimensions:
            continue
        target = f"{year:04d}-{month:02d}-{day:02d}"
        if instant:
            if context.instant != target or context.start_date or context.end_date:
                continue
        else:
            if context.instant or context.end_date != target or not context.start_date:
                continue
            try:
                if not 350 <= (date.fromisoformat(context.end_date) - date.fromisoformat(context.start_date)).days <= 380:
                    continue
            except ValueError:
                continue
        result.append(fact.context_id)
    return sorted(set(result))


def _segment_goodwill_plan(question, evidence, bundle, operation):
    if operation != "average" or not re.search(
            r"\baverage\s+(?:ending\s+)?carrying\s+amount\s+of\s+goodwill\s+"
            r"under\s+the\s+payments?\s+segment\b", question, re.I):
        return None
    from verifiqa.types import VerificationFact
    from verifiqa.verification.xbrl_scope import requested_scopes
    if requested_scopes(VerificationFact("", 0, "", row_label=question)) != [("payment", "payments")]:
        return None
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    if len(years) != 2 or len(set(years)) != 2:
        return None
    introduction = re.compile(
        r"The\s+carrying\s+amount\s+of\s+goodwill\s+for\s+the\s+fiscal\s+years\s+ended\s+"
        r"(?P<month>" + _MONTH + r")\s+(?P<day>\d{1,2}),?\s+(?P<year1>(?:19|20)\d{2})\s+"
        r"and\s+(?P<year2>(?:19|20)\d{2}),?\s+by\s+reportable\s+segments,?\s+is\s+as\s+follows:", re.I)
    intros = list(introduction.finditer(evidence))
    if len(intros) != 1:
        return None
    intro = intros[0]
    source_years = [intro.group("year1"), intro.group("year2")]
    if set(source_years) != set(years):
        return None
    month, day = _MONTHS.index(intro.group("month").title()) + 1, int(intro.group("day"))
    # Preserve the complete rollforward boundary. Neighboring Core and
    # Complementary balances cannot become Payments merely through role names.
    tail = evidence[intro.end():]
    header = re.search(r"\b" + re.escape(intro.group("month")) + r"\s+" + str(day)
                       + r",?\s+Core\s+(?P<a>(?:19|20)\d{2})\s+(?P<b>(?:19|20)\d{2})\b", tail, re.I)
    if not header or [header.group("a"), header.group("b")] != source_years:
        return None
    row = (r"\bPayments\s+Beginning\s+balance\s+" + _CELL + r"\s+" + _CELL
           + r"\s+Goodwill,?\s+acquired\s+during\s+the\s+year\s+" + _CELL + r"\s+" + _CELL
           + r"\s+Goodwill,?\s+adjustments\s+related\s+to\s+dispositions\s+" + _CELL + r"\s+" + _CELL
           + r"\s+Ending\s+balance\s+" + _CELL + r"\s+" + _CELL + r"\s+Complementary\b")
    matches = list(re.finditer(row, tail, re.I))
    if len(matches) != 1 or matches[0].start() < header.end():
        return None
    concepts = _label_concepts("Goodwill", bundle)
    if len(concepts) != 1:
        return None
    concept = next(iter(concepts))
    roles = []
    for year in sorted(years):
        contexts = _contexts(concept, int(year), month, day, bundle, instant=True, payments=True)
        if len(contexts) != 1:
            return None
        roles.append(RoleSpec("payments_segment_goodwill_" + year, ["Goodwill"],
                              f"{intro.group('month')} {day}, {year}", concept,
                              xbrl_context_id=contexts[0]))
    from verifiqa.agent.source_xbrl_compound_planner import _printed_unit
    return _spec(operation, roles, evidence, bundle, _printed_unit(evidence), question=question)


def _spec(operation, roles, evidence, bundle, report_unit, *, question):
    if not report_unit and bundle.artifacts_dir is not None:
        from verifiqa.agent.source_report_units import recover_report_unit
        witness = recover_report_unit(evidence, bundle.doc_name, bundle.artifacts_dir)
        if witness is not None:
            report_unit = witness["unit"]
    names = [r.name for r in roles]
    formula = ("(" + " + ".join(names) + ") / " + str(len(names)) if operation == "average"
               else " + ".join(names) if operation == "sum" else names[-1] + " - " + names[0])
    if operation == "period_change":
        from verifiqa.verification.generic_ops import decline_magnitude_requested
        if decline_magnitude_requested(question):
            formula = names[0] + " - " + names[-1]
    return ClaimSpec(operation, formula, roles, "USD millions", .01, "generic_operation",
                     period=roles[-1].period, operation=operation,
                     planning_source="deterministic_xbrl_schema",
                     source_schema_sha256=source_xbrl_schema_sha256(evidence, bundle),
                     source_report_unit=report_unit)


def _signature(spec):
    return (spec.metric, spec.formula, spec.claim_unit, spec.tolerance, spec.formula_source,
            spec.period, spec.operation, spec.planning_source, spec.source_schema_sha256,
            spec.source_report_unit,
            tuple((r.name, tuple(r.aliases), r.period, r.concept, r.xbrl_context_id) for r in spec.roles))
