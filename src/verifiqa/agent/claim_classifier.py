"""Phase A: question → ClaimSpec, independent of the proposed answer.
generic_ops classification runs first (deterministic, no LLM). The result is
passed as a hint so the LLM only needs to identify the specific metric and
periods — not discover the operation type from scratch.
"""

from __future__ import annotations
import ast
import json
import re
import subprocess
from verifiqa.agent.types import ClaimedAnswer, ClaimSpec, RoleSpec
from verifiqa.agent.utils import extract_json
from verifiqa.answer_spec import answer_spec_from_question
from verifiqa.formulas.evaluator import formula_variables, normalize_formula
from verifiqa.generation.llm_client import ChatMessage
from verifiqa.policy import DEFAULT_POLICY_REGISTRY
from verifiqa.verification.generic_ops import classify_operation

_DECOMPOSE_PROMPT = """\
Classify a financial question to set up formal arithmetic verification.

You are resolving the metric, formula, roles, and expected answer unit requested by the
question. You must use only the Question, Evidence, deterministic operation hint, and
policy registry. You must not use, infer from, or anticipate any proposed answer.

Pre-classified operation hint (deterministic): {operation}

Policy registry (metric → formula → role names):
{registry_index}

Return JSON only — no markdown, no explanation:
{{
  "verifiable": "<true|false>",
  "metric": "<snake_case metric name>",
  "formula": "<arithmetic formula using the role variable names>",
  "roles": [
    {{"name": "<variable_name>", "aliases": ["<row label 1>", "<row label 2>"], "period": "<year or quarter>"}}
  ],
  "claim_unit": "<percent|ratio|days|USD millions|USD billions|count|other>",
  "formula_source": "<policy_registry|generic_operation|document_derived|no_formula>"
}}

Rules for verifiable / formula:
- verifiable: set to "false" ONLY when the question cannot be computed from source-reported
  quantities using an authorized operation. Numeric ages, compensation, share counts,
  disclosed percentages and maturity buckets are eligible, as well as financial statement rows.
  If verifiable = "false", set formula_source = "no_formula", formula = "", roles = [].
- First check whether the question requests a named metric in the registry. An
  operation hint can describe only an inner part of that metric: average PP&E
  in a turnover denominator does not make turnover an arithmetic mean.
  Preserve the named metric's complete applicable formula, expanding its
  question-selected periods into roles. Include only roles used by that template.
  With an active operation hint, use formula_source = "generic_operation" so
  the complete named formula is independently authorized after grounding.
- Otherwise, if pre-classified operation is any registered arithmetic operation:
  formula_source = "generic_operation"
  Identify the metric and extract all relevant periods or roles from the question.
  percent_change:  formula = (var_end - var_start) / var_start * 100
                   Name roles as <metric>_<year> (e.g. revenue_2021, revenue_2022).
  period_change:   formula = var_end - var_start
                   Name roles as <metric>_<year>.
  proportion:      formula = part_var / base_var * 100
  average:         formula = (v1 + v2 + ... + vN) / N  where N is the count of periods
                   Name roles as <metric>_<year> for each year mentioned.
                   If question says "from 2013 to 2015" that is 3 years: v1=2013, v2=2014, v3=2015.
  generic_ratio:   formula = numerator_var / denominator_var
                   Name roles from the question grammar ("ratio of X to Y" → numerator=X, denominator=Y).
  sum / selected_three_row_sum: add the complete question-selected set of grounded rows or periods.
  line_item_difference: first requested item minus second (high minus low for a price range).
  percentage_point_change: later reported percentage minus earlier; convert ratios to percent first.
  exclusion_or_remaining_amount: whole minus the explicitly excluded or remaining component.
  difference_of_changes: subtract the earlier change from the later change, or bind the two named changes.
  change_of_growth_rates: later growth percentage minus earlier, using disclosed rates or three periods.
  share_of_a_change: component change / total change * 100, with the same two dates for both.
  sum_then_ratio: sum the named numerator components, then divide by the question's denominator.
  quantity_times_price: eligible option quantity * exercise price; exclude zero-price restricted units.
  net_gains_losses: gains minus positive gross-loss magnitude, or gains plus a signed disclosed loss.
  change_of_ratios: distinguish percentage-point difference from relative percentage change; do not guess.
  Expand sum(values) and count(values) into explicit variable additions and a literal count.
  Never emit sum/count function calls. Identify complete membership from the source table.
  Preserve exact quarters, dates, maturity buckets, product/geographic scope, and basic/diluted distinctions.
- If pre-classified operation is "none": match to the registry above.
  Copy the formula template verbatim. Use exact role names from the registry.
  formula_source = "policy_registry" for named metrics; use "generic_operation" for entries
  marked requires_question_binding, after binding and expanding their complete operand set.
- If the evidence explicitly defines a formula for a non-GAAP or custom metric
  (e.g. "Adjusted EBITDA is defined as Net Income plus Depreciation and Amortization"):
  use that exact definition, formula_source = "document_derived".
  Do NOT use document_derived to infer or construct formulas from visible numbers.
- If none of the above apply: formula_source = "no_formula", formula = "", roles = [].
- roles must list every variable that appears in the formula.
- Every reported observation must be a role variable; never copy source numbers into the formula.
- aliases: exact source row or column labels, never numeric values or cells containing a proposed answer.
- claim_unit is the unit expected by the question/formula, not any proposed answer.
  DPO, DSO, DIO and cash conversion cycle have output unit "days"; the monetary
  units of their financial inputs do not describe the result.
- A difference between line items, or a dated change in a reported measure, uses the
  registered arithmetic difference unless the question explicitly requests percent/rate change.
  An undated decline with no defined comparison or unit remains ambiguous; prefer no_formula.

Question: {question}
{evidence_section}"""


_CLAIM_PARSE_PROMPT = """\
Extract the single final numeric value claimed by the Answer for an already resolved
verification target.

Do not choose or modify the formula. Do not decide whether the answer is correct.
Use the resolved metric/formula only to identify which number in the answer is the
final claimed value. Ignore source facts, years, intermediate values, and calculations
unless the final answer itself is one of those numbers.

Return JSON only — no markdown, no explanation:
{{
  "claimed_value": <number or null>,
  "claimed_unit": "<displayed unit: e.g. USD, USD thousands, USD millions, USD billions, EUR, TWD thousands, percent, ratio, days, shares, count, other>"
}}

Rules:
- If the answer contains no concrete numeric final answer, set claimed_value to null.
- If the answer says evidence is insufficient, unavailable, not provided, or cannot calculate,
  set claimed_value to null even if the answer includes other numbers.
- If the answer contains several numbers, return the number that directly answers the question,
  not input facts or intermediate values.
- If the question asks about a ratio or level ("what is the quick ratio?", "is the margin
  improving?"), claimed_value is the end-period ratio/level, not a YoY change unless the question
  explicitly asks for the change.
- claimed_unit must describe claimed_value specifically. For example, if the answer says
  "$5 thousand", use "USD thousands"; "$8.7 billion", use "USD billions";
  "-$97,000", use claimed_value=-97000 and "USD"; "12.4%", use "percent".
- Preserve the displayed currency and scale. Do not relabel a literal amount with the
  expected claim unit or convert its number. Preserve negative signs before currency symbols.
- A final number followed by "days" has unit "days". A note that financial inputs
  are in USD millions does not change the final number's day unit.
- For a currency-only number whose display scale is ambiguous in context, use "other";
  do not choose a scale by agreement with a calculation or an expected answer.
- Do not parse company-name tokens as numbers ("3M" is a company, not the claimed value).

Question: {question}
Resolved metric: {metric}
Resolved formula: {formula}
Expected claim unit: {claim_unit}
Answer: {answer}
"""


def decompose_question(
    question: str,
    llm_client,
    evidence_text: str = "",
    *,
    temperature: float = 0.0,
    source_pdf_path: str = "",
    source_xbrl_bundle=None,
    source_plan_cache_dir=None,
) -> ClaimSpec:
    operation = classify_operation(question) or "none"
    spec = answer_spec_from_question(question)
    # A broad PPE alias must never turn an unresolved gross/net distinction
    # into a model-selected cost subtotal. Applicable source ambiguity is fatal.
    from verifiqa.agent.source_ppe_basis import resolve_ppe_basis
    ppe_applies, ppe_plan = resolve_ppe_basis(question, evidence_text)
    if ppe_applies and ppe_plan is None:
        return ClaimSpec("property_and_equipment", "", [], "", spec.tolerance,
                         "no_formula", operation="ambiguous_operation")
    known = _known_claim_spec(question, spec, operation)
    if known is not None and known.formula_source == "policy_registry":
        return known

    # A complete source population can authorize a deterministic arithmetic
    # plan before model decomposition. The planner never receives the answer.
    if evidence_text:
        from verifiqa.agent.source_plan_cache import resolve_source_plan
        source_plan = resolve_source_plan(question, evidence_text, source_pdf_path=source_pdf_path,
            source_xbrl_bundle=source_xbrl_bundle, cache_dir=source_plan_cache_dir)
        if source_plan is not None:
            return source_plan
    if known is not None:
        return known

    # Pass the full evidence into classification. Document-derived and ad hoc
    # direct formulas often need row labels and visible table structure.
    prompt = _DECOMPOSE_PROMPT.format(
        operation=operation,
        registry_index=_format_registry(),
        question=question,
        evidence_section=_format_evidence_section(evidence_text),
    )
    raw = llm_client.chat(
        [ChatMessage(role="user", content=prompt)],
        temperature=temperature,
        stage="question_decomposition",
    ).strip()

    data = json.loads(extract_json(raw))

    fallback = _direct_lookup_fallback(
        question=question,
        answer="",
        spec=spec,
        operation=operation,
        claimed_value=None,
        claimed_unit="",
    )

    # LLM-determined verifiability: if false, treat as no_formula regardless of other fields
    if str(data.get("verifiable", "true")).strip().lower() == "false":
        if fallback is not None:
            return fallback
        return ClaimSpec(
            metric=str(data.get("metric", "unknown")).strip(),
            formula="",
            roles=[],
            claim_unit="",
            tolerance=spec.tolerance,
            formula_source="no_formula",
            period=_extract_period(question),
            operation=None,
        )

    roles = [
        RoleSpec(
            name=str(r.get("name", "")).strip(),
            aliases=[str(a) for a in (r.get("aliases") or [])],
            period=str(r.get("period", "")).strip(),
        )
        for r in (data.get("roles") or [])
        if r.get("name")
    ]
    formula = str(data.get("formula", "")).strip()
    formula_source = str(data.get("formula_source", "no_formula")).strip()
    policy = DEFAULT_POLICY_REGISTRY.find_policy(str(data.get("metric", "")))
    if policy and policy.metadata.get("requires_question_binding") and operation != "none":
        formula_source = "generic_operation"
    if fallback is not None and (
        formula_source == "no_formula" or not formula or not roles
    ):
        return fallback
    claim_spec = ClaimSpec(
        metric=str(data.get("metric", "unknown")).strip(),
        formula=formula,
        roles=roles,
        claim_unit=("days" if spec.expected_unit == "days"
                    else str(data.get("claim_unit", "")).strip()),
        tolerance=spec.tolerance,
        formula_source=formula_source,
        period=_extract_period(question),
        operation=operation if operation != "none" else None,
    )
    if _ambiguous_change_without_unit(question, claim_spec):
        return ClaimSpec(
            metric=claim_spec.metric,
            formula="",
            roles=[],
            claim_unit="",
            tolerance=spec.tolerance,
            formula_source="no_formula",
            period=claim_spec.period,
            operation="ambiguous_operation",
        )
    return claim_spec


def decompose_question_consensus(
    question: str,
    llm_client,
    evidence_text: str = "",
    *,
    temperatures: tuple[float, float, float] = (0.0, 0.2, 0.4),
    source_pdf_path: str = "",
    source_xbrl_bundle=None,
    source_plan_cache_dir=None,
) -> tuple[ClaimSpec | None, dict, str]:
    """Run three independent question decompositions and require semantic agreement.

    This is intentionally conservative. It only accepts when normalized metric,
    operation, output unit, period, roles, and formulas agree. Disagreement means
    the verifier should abstain instead of proving one possibly-wrong
    interpretation downstream.
    """
    runs = []
    specs: list[ClaimSpec] = []
    for index, temperature in enumerate(temperatures, start=1):
        try:
            spec = decompose_question(
                question,
                llm_client,
                evidence_text=evidence_text,
                temperature=temperature,
                source_pdf_path=source_pdf_path,
                source_xbrl_bundle=source_xbrl_bundle,
                **({"source_plan_cache_dir": source_plan_cache_dir} if source_plan_cache_dir is not None else {}),
            )
            specs.append(spec)
            runs.append({
                "index": index,
                "temperature": temperature,
                "spec": _claim_spec_to_dict(spec),
            })
        except Exception as exc:
            runs.append({
                "index": index,
                "temperature": temperature,
                "error": f"{type(exc).__name__}:{exc}",
            })
            diagnostics = {
                "claimspec_consensus": {
                    "runs": runs,
                    "agreed": False,
                    "failure_reason": "decomposition_error",
                }
            }
            return specs[0] if specs else None, diagnostics, "claimspec_consensus_failed"

    agreed, reason = _claim_specs_semantically_agree(specs)
    diagnostics = {
        "claimspec_consensus": {
            "runs": runs,
            "agreed": agreed,
            "failure_reason": "" if agreed else reason,
        }
    }
    return specs[0] if specs else None, diagnostics, "" if agreed else "claimspec_consensus_failed"


def parse_claimed_answer(
    question: str,
    answer: str,
    claim_spec: ClaimSpec,
    llm_client,
) -> ClaimedAnswer:
    prompt = _CLAIM_PARSE_PROMPT.format(
        question=question,
        metric=claim_spec.metric,
        formula=claim_spec.formula,
        claim_unit=claim_spec.claim_unit or "other",
        answer=answer or "(not provided)",
    )
    raw = llm_client.chat(
        [ChatMessage(role="user", content=prompt)],
        temperature=0.0,
        stage="claim_value_parse",
    ).strip()
    data = json.loads(extract_json(raw))
    value = _parse_claimed_value(data.get("claimed_value"))
    unit = str(data.get("claimed_unit", "")).strip()
    if value is None:
        return ClaimedAnswer(value=None, unit=unit)
    if answer_spec_from_question(question).expected_unit == "days":
        from verifiqa.agent.claim_unit_evidence import explicit_claim_day_unit
        literal = explicit_claim_day_unit(answer, value)
        unit = literal or ("days" if unit.lower() in {"", "other", "unspecified", "day", "days"} else unit)
    return ClaimedAnswer(value=value, unit=unit or _infer_claim_unit(question, answer, ""))


def classify(question: str, llm_client, evidence_text: str = "", answer: str = "") -> ClaimSpec:
    """Compatibility wrapper for the former single-stage classifier API."""
    claim_spec = decompose_question(question, llm_client, evidence_text=evidence_text)
    if answer:
        claimed = parse_claimed_answer(question, answer, claim_spec, llm_client)
        claim_spec.claimed_value = claimed.value
        claim_spec.claimed_unit = claimed.unit
    return claim_spec



def _parse_claimed_value(raw) -> float | None:
    """Coerce the LLM's claimed_value output to float, or None."""
    if raw is None or str(raw).strip().lower() in ("null", "none", ""):
        return None
    try:
        return float(str(raw).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


def _claim_specs_semantically_agree(specs: list[ClaimSpec]) -> tuple[bool, str]:
    if len(specs) != 3:
        return False, "expected_three_specs"
    first = specs[0]

    # Check metric, operation, unit, period — but not roles, which are checked after
    # formula equivalence so that semantically identical formulas with different role
    # variable names (net_income_end vs net_income_2022) still agree.
    for key in ("metric", "operation", "claim_unit", "period", "source_document_sha256", "source_schema_sha256", "source_report_unit"):
        first_val = _claim_spec_signature(first)[key]
        for spec in specs[1:]:
            if _claim_spec_signature(spec)[key] != first_val:
                return False, f"{key}_mismatch"

    no_formula = [not (spec.formula or "").strip() or spec.formula_source == "no_formula" for spec in specs]
    if any(no_formula):
        return (True, "") if all(no_formula) else (False, "formula_presence_mismatch")

    # Formula equivalence before role names — Z3 checks algebraic structure regardless of
    # variable naming, so net_income_end and net_income_2022 still agree here.
    for spec in specs[1:]:
        if not _formulas_equivalent(first.formula, spec.formula):
            return False, "formula_mismatch"

    # Role names after formula — normalized to strip temporal suffixes so _end == _2022 == _y1,
    # but a genuine metric mismatch (revenue vs gross_profit) is still caught.
    first_roles = tuple(_role_signatures(first.roles))
    for spec in specs[1:]:
        if tuple(_role_signatures(spec.roles)) != first_roles:
            return False, "role_name_mismatch"

    return True, ""


def _claim_spec_signature(spec: ClaimSpec) -> dict:
    return {
        "metric": _normalize_identifier(spec.metric),
        "operation": _normalize_operation(spec.operation),
        "claim_unit": _normalize_unit(spec.claim_unit),
        "period": _normalize_period(spec.period),
        "source_document_sha256": spec.source_document_sha256,
        "source_schema_sha256": spec.source_schema_sha256,
        "source_report_unit": spec.source_report_unit,
        "roles": tuple(_role_signatures(spec.roles)),
    }


def _claim_spec_to_dict(spec: ClaimSpec) -> dict:
    return {
        "source_selection_id": spec.source_selection_id,
        "metric": spec.metric,
        "formula": spec.formula,
        "source_document_sha256": spec.source_document_sha256,
        "source_schema_sha256": spec.source_schema_sha256,
        "source_report_unit": spec.source_report_unit,
        "roles": [
            {"name": role.name, "aliases": list(role.aliases), "period": role.period,
             "concept": role.concept, "xbrl_context_id": role.xbrl_context_id,
             "source_ref": role.source_ref}
            for role in spec.roles
        ],
        "claim_unit": spec.claim_unit,
        "tolerance": spec.tolerance,
        "formula_source": spec.formula_source,
        "period": spec.period,
        "operation": spec.operation,
        "planning_source": spec.planning_source,
    }


def _role_signatures(roles: list[RoleSpec]) -> list[tuple[str, str, str, str]]:
    return sorted(
        (
            _normalize_role_name(role.name),
            _normalize_period(role.period),
            role.concept,
            role.xbrl_context_id,
        )
        for role in roles
    )


def _normalize_role_name(name: str) -> str:
    """Strip temporal suffixes so net_income_end == net_income_2022 == net_income_y1."""
    text = _normalize_identifier(name)
    # Remove trailing year (4 digits)
    text = re.sub(r"_\d{4}$", "", text)
    # Remove trailing period markers: _y1 _y2 _y3 _yr1 _yr2 _end _start _prior _current _previous
    text = re.sub(r"_(y|yr)\d+$", "", text)
    text = re.sub(r"_(end|start|prior|current|previous|begin)$", "", text)
    return text.strip("_") or text


def _normalize_identifier(value: str) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text

def _normalize_operation(value: str | None) -> str:
    return _normalize_identifier(value or "")

def _normalize_period(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\bfiscal year\s*", "", text)  # "fiscal year 2022" → "2022"
    text = re.sub(r"\bfy(?=\d)", "", text)         # "FY2022" → "2022", "Q2 FY2023" → "Q2 2023"
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _normalize_unit(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if "%" in text or "percent" in text or "percentage" in text:
        return "percent"
    if "ratio" in text or text in {"rate", "multiple"}:
        return "ratio"
    if "usd" in text or "$" in text or "dollar" in text:
        if "billion" in text:
            return "usd_billions"
        if "thousand" in text:
            return "usd_thousands"
        if "million" in text:
            return "usd_millions"
        return "usd"
    if "share" in text or "count" in text or "unit" in text:
        return "count"
    return _normalize_identifier(text)


def _formula_variables_normalized(formula: str) -> set[str]:
    try:
        tree = ast.parse(normalize_formula(formula), mode="eval")
    except SyntaxError:
        return set()
    return {
        _normalize_identifier(node.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }


def _formulas_equivalent(formula_a: str, formula_b: str) -> bool:
    if _canonical_formula_text(formula_a) == _canonical_formula_text(formula_b):
        return True
    try:
        tree_a = ast.parse(normalize_formula(formula_a), mode="eval")
        tree_b = ast.parse(normalize_formula(formula_b), mode="eval")
    except Exception:
        return False
    if not _formula_ast_supported(tree_a.body) or not _formula_ast_supported(tree_b.body):
        return False
    return _formulas_equivalent_with_z3(tree_a, tree_b)


def _canonical_formula_text(formula: str) -> str:
    return re.sub(r"\s+", "", normalize_formula(formula or "")).lower()


def _formula_ast_supported(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _formula_ast_supported(node.operand)
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div))
            and _formula_ast_supported(node.left)
            and _formula_ast_supported(node.right)
        )
    return False


def _formulas_equivalent_with_z3(tree_a: ast.Expression, tree_b: ast.Expression) -> bool:
    variables = sorted(
        {
            _normalize_identifier(node.id)
            for tree in (tree_a, tree_b)
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
    )
    denominators: list[str] = []
    expr_a = _ast_to_smt(tree_a.body, denominators)
    expr_b = _ast_to_smt(tree_b.body, denominators)
    declarations = "\n".join(f"(declare-const {name} Real)" for name in variables)
    domain = "\n".join(f"(assert (not (= {denominator} 0)))" for denominator in denominators)
    smtlib = "\n".join(
        part
        for part in [
            "(set-logic QF_NRA)",
            declarations,
            domain,
            f"(assert (not (= {expr_a} {expr_b})))",
            "(check-sat)",
        ]
        if part
    )
    try:
        proc = subprocess.run(
            ["z3", "-in"],
            input=smtlib,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    first_line = (proc.stdout or "").strip().splitlines()[0:1]
    return bool(first_line and first_line[0].strip().lower() == "unsat")


def _ast_to_smt(node: ast.AST, denominators: list[str]) -> str:
    if isinstance(node, ast.Name):
        return _normalize_identifier(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _smt_number(float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(- {_ast_to_smt(node.operand, denominators)})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _ast_to_smt(node.operand, denominators)
    if isinstance(node, ast.BinOp):
        left = _ast_to_smt(node.left, denominators)
        right = _ast_to_smt(node.right, denominators)
        if isinstance(node.op, ast.Add):
            return f"(+ {left} {right})"
        if isinstance(node.op, ast.Sub):
            return f"(- {left} {right})"
        if isinstance(node.op, ast.Mult):
            return f"(* {left} {right})"
        if isinstance(node.op, ast.Div):
            denominators.append(right)
            return f"(/ {left} {right})"
    raise ValueError(f"unsupported formula node:{type(node).__name__}")


def _smt_number(value: float) -> str:
    if value < 0:
        return f"(- {_smt_number(abs(value))})"
    if value.is_integer():
        return str(int(value))
    return format(value, ".17g")


def _ambiguous_change_without_unit(question: str, claim_spec: ClaimSpec) -> bool:
    text = _normalize_question(question)
    if not re.search(r"\b(?:change|increase|decrease|decline|drop|difference)\b", text):
        return False
    if "%" in text:
        return False
    if re.search(
        r"\b(?:percent|percentage|rate|ratio|amount|dollars?|usd|millions?|billions?|"
        r"thousands?|percentage points?|basis points?)\b",
        text,
    ):
        return False
    operation = classify_operation(question)
    if operation in {"line_item_difference", "exclusion_or_remaining_amount", "difference_of_changes"}:
        return False
    if "difference" in text or re.search(r"\bhow much\b", text):
        return False
    if len(set(re.findall(r"\b(?:19|20)\d{2}\b", text))) >= 2 and re.search(r"\b(?:between|from|to)\b", text):
        return False
    if re.search(r"\b(?:between|from|to)\b", text):
        quarters = set(re.findall(r"\bq[1-4]\b", text))
        months = set(re.findall(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", text))
        if len(quarters) >= 2 or len(months) >= 2:
            return False
    formula = re.sub(r"\s+", "", claim_spec.formula or "").lower()
    if not formula or "/" in formula or "*100" in formula:
        return False
    return "-" in formula and len(claim_spec.roles) >= 2


def _format_evidence_section(evidence_text: str) -> str:
    evidence_text = (evidence_text or "").strip()
    if not evidence_text:
        return "Evidence: (not provided)"
    return f"Evidence:\n{evidence_text}"


def _direct_lookup_fallback(
    *,
    question: str,
    answer: str,
    spec,
    operation: str,
    claimed_value: float | None,
    claimed_unit: str,
) -> ClaimSpec | None:
    """Build a one-role identity formula for simple extraction questions.

    This is deliberately used only as a fallback after the classifier failed to
    provide a usable formula. Calculation questions should still go through the
    registry/generic-operation paths.
    """
    if operation != "none" and not _allow_direct_lookup_despite_operation(question, operation):
        return None
    target = _direct_lookup_target(question)
    if not target:
        return None

    policy = DEFAULT_POLICY_REGISTRY.find_policy(target)
    if policy is not None and _is_direct_lookup_policy(policy):
        formula = policy.formula_templates[0].strip()
        role_names = list(policy.roles.keys())
        return ClaimSpec(
            metric=policy.metric,
            formula=formula,
            roles=[
                RoleSpec(
                    name=role_name,
                    aliases=_aliases_for_policy_role(role_name, policy.roles.get(role_name, {})),
                    period=_extract_period(question),
                )
                for role_name in role_names
            ],
            claim_unit=_infer_claim_unit(question, answer, claimed_unit),
            tolerance=spec.tolerance,
            formula_source="direct_lookup",
            period=_extract_period(question),
            operation="direct_lookup",
            claimed_value=claimed_value,
            claimed_unit=claimed_unit,
        )

    role_name = _slug(target)
    if not role_name:
        return None
    aliases = _direct_lookup_aliases(target, role_name)
    return ClaimSpec(
        metric=role_name,
        formula=role_name,
        roles=[
            RoleSpec(
                name=role_name,
                aliases=aliases,
                period=_extract_period(question),
            )
        ],
        claim_unit=_infer_claim_unit(question, answer, claimed_unit),
        tolerance=spec.tolerance,
        formula_source="direct_lookup",
        period=_extract_period(question),
        operation="direct_lookup",
        claimed_value=claimed_value,
        claimed_unit=claimed_unit,
    )


def _direct_lookup_target(question: str) -> str:
    text = _normalize_question(question)
    if not text or _looks_like_calculation_question(text):
        return ""

    patterns = (
        r"\bwhat\s+(?:is|was|were|are)\s+(?:the\s+)?(?P<target>.+)$",
        r"\bhow\s+(?:much|many)\s+(?:is|was|were|are|did|does)?\s*(?:the\s+)?(?P<target>.+)$",
    )
    target = ""
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            target = match.group("target")
            break
    if not target:
        return ""

    target = _strip_direct_lookup_target(target)
    return target if len(target.split()) <= 12 else ""


def _allow_direct_lookup_despite_operation(question: str, operation: str) -> bool:
    """Permit fallback for direct reported gains mis-tagged as period changes.

    The generic operation classifier intentionally runs broad for FinQA-style
    arithmetic. For FinanceBench text extraction, "amount of gain" is often a
    reported line item, not a delta to compute.
    """
    if operation != "period_change":
        return False
    text = _normalize_question(question)
    return bool(re.search(r"\b(?:amount|quantity|value)\s+of\s+(?:the\s+)?gain\b", text))


def _looks_like_calculation_question(text: str) -> bool:
    calculation_markers = (
        " as a percentage of ",
        " percentage of ",
        " percent of ",
        " percentage change ",
        " percent change ",
        " change in ",
        " change of ",
        " increase in ",
        " decrease in ",
        " decline in ",
        " difference ",
        " ratio ",
        " turnover ",
        " margin ",
        " return on ",
        " cagr ",
        " growth ",
        " average of ",
        " divided by ",
        " defined as ",
        " if ",
    )
    if "/" in text:
        return True
    return any(marker in f" {text} " for marker in calculation_markers)


def _strip_direct_lookup_target(target: str) -> str:
    target = re.sub(r"\([^)]*\)", " ", target)
    target = re.sub(r"\b(?:fy|fiscal year|fiscal)\s*(?:19|20)\d{2}\b", " ", target)
    target = re.sub(r"\b(?:19|20)\d{2}\b", " ", target)
    target = re.sub(r"\b(?:usd|dollars?|millions?|billions?|thousands?|in)\b", " ", target)
    target = re.split(
        r"\b(?:for|as a result of|resulting from|related to|as of|at the end of|at|during|from|between|using|based on|according to|that|which)\b",
        target,
        maxsplit=1,
    )[0]
    target = re.sub(
        r"^(?:the\s+)?(?:(?:amount|quantity|value|balance|level)\s+of\s+(?:the\s+)?)",
        "",
        target,
    )
    target = re.sub(r"\b(?:amount|quantity|value|balance|level)\b$", "", target)
    target = re.sub(r"[^a-z0-9&/\- ]+", " ", target)
    target = re.sub(r"\s+", " ", target).strip()
    return target


def _is_direct_lookup_policy(policy) -> bool:
    if len(policy.roles) != 1 or not policy.formula_templates:
        return False
    role_name = next(iter(policy.roles.keys()))
    formula = policy.formula_templates[0].strip()
    return bool(re.fullmatch(rf"{re.escape(role_name)}(?:\s*/\s*(?:1000|1000000))?", formula))


def _aliases_for_policy_role(role_name: str, role_def: dict) -> list[str]:
    aliases = {role_name.replace("_", " ")}
    for concept_id in role_def.get("concept_ids", []) or []:
        concept = DEFAULT_POLICY_REGISTRY.concepts.get(concept_id)
        if concept is not None:
            aliases.update(concept.aliases)
    return sorted(aliases)


def _direct_lookup_aliases(target: str, role_name: str) -> list[str]:
    aliases = {target, role_name.replace("_", " ")}
    without_fillers = re.sub(r"\b(?:amount|quantity|value|balance|level)\b", " ", target)
    without_fillers = re.sub(r"\s+", " ", without_fillers).strip()
    if without_fillers:
        aliases.add(without_fillers)
    return sorted(aliases)


def _infer_claim_unit(question: str, answer: str, claimed_unit: str) -> str:
    if claimed_unit:
        return claimed_unit
    if answer_spec_from_question(question).expected_unit == "days":
        return "days"
    text = f"{question} {answer}".lower()
    if "%" in text or "percent" in text:
        return "percent"
    if "$" in text or "usd" in text or "dollar" in text:
        if "billion" in text:
            return "USD billions"
        if "thousand" in text:
            return "USD thousands"
        return "USD millions" if "million" in text else "USD"
    if "share" in text or "unit" in text:
        return "count"
    return "other"


def _slug(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    tokens = [
        token
        for token in tokens
        if token not in {"the", "a", "an", "of", "and", "to", "for", "in"}
    ]
    return "_".join(tokens)
    try:
        return float(str(raw).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


def _format_registry() -> str:
    entries = []
    for p in DEFAULT_POLICY_REGISTRY.policies:
        template = p.formula_templates[0] if p.formula_templates else ""
        try:
            active_roles = formula_variables(template) if template else set()
        except (ValueError, SyntaxError):
            active_roles = set(p.roles)
        entries.append({
            "metric": p.metric,
            "aliases": list(p.metric_aliases[:3]),
            "formula": template,
            "roles": [name for name in p.roles if name in active_roles],
            "requires_question_binding": bool(p.metadata.get("requires_question_binding")),
        })
    return json.dumps(entries, indent=2)


def _known_claim_spec(question: str, spec, operation: str) -> ClaimSpec | None:
    text = _normalize_question(question)
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)

    if (
        "gross margin decline" in text
        and len(years) >= 2
        and "from" in text
    ):
        end_year, start_year = years[0], years[1]
        start = f"gross_margin_percentage_{start_year}"
        end = f"gross_margin_percentage_{end_year}"
        return ClaimSpec(
            metric="gross_margin_percentage_decline",
            formula=f"{start} - {end}",
            roles=[
                RoleSpec(
                    name=start,
                    aliases=["gross margin percentage", "gross margin"],
                    period=start_year,
                ),
                RoleSpec(
                    name=end,
                    aliases=["gross margin percentage", "gross margin"],
                    period=end_year,
                ),
            ],
            claim_unit="percent",
            tolerance=spec.tolerance,
            formula_source="generic_operation",
            period=end_year,
            operation="period_change",
        )

    if (
        "percentage change" in text
        and "carrying amount of loan receivable" in text
        and "net of" in text
        and "allowance" in text
    ):
        return ClaimSpec(
            metric="carrying_amount_loan_receivable_net_pct_change",
            formula=(
                "((loan_receivable_end - allowance_end) - "
                "(loan_receivable_start - allowance_start)) / "
                "(loan_receivable_start - allowance_start) * 100"
            ),
            roles=[
                RoleSpec(
                    name="loan_receivable_start",
                    aliases=["carrying amount of loan receivable"],
                    period="beginning balance",
                ),
                RoleSpec(
                    name="allowance_start",
                    aliases=["allowance"],
                    period="beginning balance",
                ),
                RoleSpec(
                    name="loan_receivable_end",
                    aliases=["carrying amount of loan receivable"],
                    period=_extract_period(question) or "ending balance",
                ),
                RoleSpec(
                    name="allowance_end",
                    aliases=["allowance"],
                    period=_extract_period(question) or "ending balance",
                ),
            ],
            claim_unit="percent",
            tolerance=spec.tolerance,
            formula_source="generic_operation",
            period=_extract_period(question),
            operation="percent_change",
        )

    return None


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").lower()).strip()




def _extract_period(question: str) -> str:
    years = re.findall(
        r"\b(?:fy|fiscal\s+year|fiscal)?\s*((?:19|20)\d{2})\b",
        question,
        flags=re.IGNORECASE,
    )
    return years[-1] if years else ""
