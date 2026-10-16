from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Iterable, Tuple

from verifiqa.formulas.evaluator import FormulaError, formula_variables
from verifiqa.generation.llm_client import ChatMessage
from verifiqa.policy.registry import DEFAULT_POLICY_REGISTRY, PolicyRegistry
from verifiqa.types import (
    CertificateClaim,
    CertificateFact,
    Claim,
    EvidenceChunk,
    RetrievalPlan,
    VerificationCertificate,
    VerificationSchema,
)


_PROMPT = """\
Convert the retrieved filing evidence and draft answer into a verification certificate.

Return JSON only, no markdown.

If the question can be checked numerically from the retrieved evidence, return:
{{
  "verifiable": true,
  "claim": {{
    "metric": "<snake_case_metric>",
    "claimed_value": <numeric answer value>,
    "unit": "<answer unit, e.g. USD millions, USD billions, ratio, percent>",
    "period": "<period from the answer/question, if any>",
    "reported_value": <optional displayed rounded value from the answer>
  }},
    "facts": [
    {{
      "name": "<snake_case_variable>",
      "fact_type": "<numeric|absence_implies_zero>",
      "raw_value": <number exactly as reported in retrieved evidence>,
      "raw_unit": "<raw unit, e.g. USD, ratio, percent, count>",
      "source_scale": "<ones|thousands|millions|billions>",
      "source_scale_quote": "<exact evidence phrase proving scale, e.g. in thousands; empty only for ones/raw units>",
      "sign_convention": "<as_reported|magnitude>",
      "value": <canonical value used in formulas; monetary facts must be USD millions>,
      "unit": "<canonical fact unit, e.g. USD millions, ratio, percent, count>",
      "source_quote": "<short exact quote copied from retrieved evidence containing the value>",
      "chunk_id": "<one retrieved chunk id exactly as shown>",
      "period": "<period/column for the fact, if any>",
      "row_label": "<table row label or nearby label, if any>",
      "column": "<table column/header, if any>",
      "absence_scope": {{
        "metric": "<metric absent from evidence, only for absence_implies_zero>",
        "period": "<period covered by absence quote, only for absence_implies_zero>",
        "statement": "<statement/scope, only for absence_implies_zero>"
      }}
    }}
  ],
  "formula": "<arithmetic expression over fact names>",
  "calculation": "<short calculation string>",
  "tolerance": <see tolerance rules below>
}}

Only if no numeric certificate can be produced from the retrieved evidence
without inventing unsupported facts, return:
{{
  "verifiable": false,
  "reason": "<brief reason>"
}}

Rules:
- Do not use "verifiable": false merely because the retrieved evidence is
  imperfect or spread across pages. If the retrieved evidence contains the
  numeric facts needed for a formula, emit a certificate and let the verifier
  check it.
- Every formula variable must be one fact name.
- Never create a fact for the quantity the question asks you to compute. A fact
  must be a line item printed in the evidence (its number appears verbatim in a
  chunk). If the answer is itself a derived metric (a total, change, growth,
  difference, ratio, or margin) that is not printed as its own number, do NOT
  bind it as a single fact. Instead bind each printed component as a fact and
  set "formula" to compute the answer from those components. Example: for "real
  change in sales = reported change excluding FX and passthrough", bind
  reported_change, fx_impact, and passthrough as grounded facts and set
  formula to "reported_change - fx_impact - passthrough" — do not bind a
  real_change fact.
- A fact's number must appear in its chunk. Do not invent a source_quote or use
  placeholders like "...", "(blank cell)", or descriptive text for a value that
  is not actually printed in the evidence.
- Do not include unused facts.
- Facts must come from retrieved evidence, not outside knowledge.
- The retrieval plan lists the required facts and acceptable aliases. If a
  retrieved table row matches an alias, it is acceptable evidence for that fact.
- If Registry guidance lists detailed_candidate_policies, you must choose one
  policy from that list whenever it fits the question. Do not create a new
  metric name for a registry-covered metric.
- For a registry-backed certificate, claim.metric must be copied exactly from
  the chosen policy's metric field, or exactly from one of that policy's
  metric_aliases when the alias better matches the question. Never invent a
  new claim.metric when a listed policy applies.
- For a registry-backed certificate, copy the formula field exactly from one of
  the chosen policy's formula_templates. Do not invent, simplify, rename, or
  algebraically rewrite the formula.
- For a registry-backed certificate, every fact name must be exactly one of the
  registry role names used by the selected formula_template. Do not invent
  synonyms or rename evidence into a role it does not satisfy.
- A source row may fill a registry role only when its row_label/source_quote
  satisfies that role's required_source_text_any and does not violate its
  forbidden_source_text_any or forbidden_substitute_variables_any.
- Never bind Operating income/loss to adjusted_ebit unless the source explicitly
  defines that row as Adjusted EBIT. If Adjusted EBIT is not directly reported
  but the registry offers an EBITDAR bridge and the needed rows are present,
  use that bridge template.
- If STRUCTURED TABLE ROWS are present, treat them as the primary evidence.
  They already bind row_label, column/period, value, statement type, unit_scale,
  unit_scale_quote, chunk_id, and source_quote. Prefer them over reparsing raw
  page text.
- When using a structured row, choose the value from the column/period requested
  by the fact or question, set source_quote to the row's source_quote, chunk_id
  to the row's chunk_id, source_scale to row.unit_scale, and source_scale_quote
  to row.unit_scale_quote.
- source_scale_quote must be the table-level or document-level scale declaration
  applying to that row. Do not use the requested answer unit or unrelated prose
  such as "billion" as source_scale_quote for a table reported in millions.
- Use fact_type "numeric" for normal numeric extraction/calculation facts.
- Use fact_type "absence_implies_zero" only when the question explicitly says to
  state 0 if a requested line item is absent/not outlined. The source_quote must
  be copied from the filing and must either contain genuine absence language
  such as "no such costs" or be a statement/table excerpt whose row list omits
  the requested line item. Do not invent words like "no ... is present" unless
  those words appear in the filing.
- For absence_implies_zero facts: value must be 0, raw_value must be null,
  source_scale must be "ones", source_quote must be the absence language or
  omission-witness statement/table excerpt, and absence_scope must identify the
  metric, period, and statement/scope.
- source_quote must be copied from the retrieved text and must contain the numeric value.
- For absence_implies_zero facts, source_quote need not contain a numeric value.
- For monetary facts, raw_value is the as-reported number; value is the canonical USD millions value.
- sign_convention declares how the value's sign relates to the filing. Use
  "magnitude" when the filing prints the amount as a negative/parenthesized cash
  outflow but the metric uses its positive magnitude (e.g. capital expenditures,
  purchases of property/plant/equipment). Use "as_reported" otherwise. This lets
  the verifier check the sign explicitly instead of guessing from the label.
- For absolute monetary line-item quantities such as costs, expenses, revenue,
  assets, liabilities, or cash flow amounts, claim.unit should be "USD millions"
  unless the question asks for another display unit. Do not use "ratio" for an
  absolute monetary quantity.
- If the table says "in thousands", source_scale must be "thousands" and value = raw_value / 1000.
- If the table says "in millions", source_scale must be "millions" and value = raw_value.
- If the table says "in billions", source_scale must be "billions" and value = raw_value * 1000.
- If the quote is a raw dollar amount, source_scale must be "ones" and value = raw_value / 1000000.
- source_scale_quote must be copied from the retrieved evidence when source_scale is not "ones".
- If a percentage/ratio table cell is visually blank and the answer is "flat" or 0, use raw_value/value 0 only when row_label and column identify the blank cell and source_quote contains the row context.
- chunk_id must match one retrieved chunk id.
- If the draft answer contains a numeric answer, claimed_value must match that
  displayed answer. Never replace it with an evidence-supported corrected value.
- If the retrieved evidence implies a different numeric answer, keep the draft
  answer's claimed_value and build the evidence/formula certificate so the SMT
  solver can catch the mismatch.
- If the draft answer contains no numeric answer, return verifiable false with
  reason "no_numeric_answer"; do not invent a corrected claimed_value.
- Use only +, -, *, /, parentheses, numeric constants, snake_case variables,
  and the explicit functions cagr(end_value, start_value, years),
  cagr_percent(end_value, start_value, years), or
  coverage_ratio(numerator, denominator) in formula.
- For CAGR questions, always use cagr_percent(...) when the answer is in
  percent units and cagr(...) when the answer is a ratio. Do not use ^ or **.
- For ratios, claimed_value should use decimal form unless the question asks for percent.
- Do not include units inside numeric values.
- Tolerance: set based on the rounding precision requested in the question:
  - "round to zero decimal places" or nearest integer → 0.5
  - "round to one decimal place" → 0.05
  - "round to two decimal places" → 0.005
  - "round to three decimal places" → 0.0005
  - no rounding instruction → 0.01

Question:
{question}

Draft answer:
{answer}

Retrieval plan:
{retrieval_plan}

Registry guidance:
{registry_guidance}

Retrieved filing evidence:
{evidence}
"""


_REPAIR_PROMPT = """\
The verification certificate below was rejected with this error:

  {error}

Return a corrected certificate JSON. Fix only what the error describes. Do not change anything else.

If the error is fact_value_not_in_chunk or source_quote_not_in_chunk for a fact,
that fact's number is not printed in the cited evidence. If that fact is a
derived/computed quantity (a total, change, growth, difference, ratio, or
margin), remove it and instead bind each printed component as a grounded fact,
then rewrite "formula" to compute the answer from those components. Only keep a
fact if its exact number appears in a retrieved chunk.

Original certificate:
{certificate_json}

Registry guidance:
{registry_guidance}

Retrieved filing evidence (for reference):
{evidence}
"""

# Sent as a follow-up user turn when the formula contains cagr()/cagr_percent()
# with a variable years argument (only literal integers are supported).
_CAGR_VAR_YEARS_HINT = (
    "Your formula used cagr() or cagr_percent() with a variable as the years argument. "
    "The years parameter must be a literal integer constant (e.g. cagr_percent(end, start, 5)). "
    "If the question asks for a simple cumulative percentage return — not a compound annual "
    "growth rate — use arithmetic directly: (end - start) / start * 100. "
    "Return a corrected certificate JSON only."
)

_NO_ABSTAIN_HINT = (
    "If the draft answer contains a numeric answer and the retrieved evidence contains "
    "the facts needed to build a formula, emit an evidence-grounded certificate for "
    "that displayed claimed_value. Do not invent or substitute a corrected answer. "
    "If the draft answer contains no numeric answer, return verifiable false with "
    "reason no_numeric_answer. Return certificate JSON only."
)


class FormalizationError(ValueError):
    pass


# JSON schema enforced via the Messages API `output_config.format`. With this the
# model is guaranteed to return one object of this exact shape — enums, required
# fields, and number types are validated API-side, so the parsing layer can trust
# the output instead of coercing/repairing free text.
_NULLABLE_NUMBER = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_FACT_PROPERTIES = {
    "name": {"type": "string"},
    "fact_type": {"type": "string", "enum": ["numeric", "absence_implies_zero"]},
    "raw_value": _NULLABLE_NUMBER,
    "raw_unit": {"type": "string"},
    "source_scale": {"type": "string", "enum": ["ones", "thousands", "millions", "billions"]},
    "source_scale_quote": {"type": "string"},
    "sign_convention": {"type": "string", "enum": ["as_reported", "magnitude"]},
    "value": {"type": "number"},
    "unit": {"type": "string"},
    "source_quote": {"type": "string"},
    "chunk_id": {"type": "string"},
    "period": {"type": "string"},
    "row_label": {"type": "string"},
    "column": {"type": "string"},
    "absence_scope": {
        "type": "object",
        "properties": {
            "metric": {"type": "string"},
            "period": {"type": "string"},
            "statement": {"type": "string"},
        },
        "required": ["metric", "period", "statement"],
        "additionalProperties": False,
    },
}
_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "metric": {"type": "string"},
        "claimed_value": {"type": "number"},
        "unit": {"type": "string"},
        "period": {"type": "string"},
        "reported_value": _NULLABLE_NUMBER,
    },
    "required": ["metric", "claimed_value", "unit", "period", "reported_value"],
    "additionalProperties": False,
}
_CERTIFICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "verifiable": {"type": "boolean"},
        "reason": {"type": "string"},
        "claim": {"anyOf": [_CLAIM_SCHEMA, {"type": "null"}]},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _FACT_PROPERTIES,
                "required": list(_FACT_PROPERTIES.keys()),
                "additionalProperties": False,
            },
        },
        "formula": {"type": "string"},
        "calculation": {"type": "string"},
        "tolerance": {"type": "number"},
    },
    "required": ["verifiable", "reason", "claim", "facts", "formula", "calculation", "tolerance"],
    "additionalProperties": False,
}
_OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": _CERTIFICATE_SCHEMA}}


class Formalizer:
    """LLM formalizer that emits an evidence-grounded verification certificate."""

    def __init__(self, llm_client, policy_registry: PolicyRegistry | None = None):
        self.llm_client = llm_client
        self.policy_registry = policy_registry or DEFAULT_POLICY_REGISTRY

    def formalize(
        self,
        question: str,
        evidence_chunks: Iterable[EvidenceChunk],
        answer: str,
        retrieval_plan: RetrievalPlan | None = None,
    ) -> VerificationCertificate:
        chunks = list(evidence_chunks)
        prompt = _PROMPT.format(
            question=question,
            answer=answer,
            retrieval_plan=_format_retrieval_plan(retrieval_plan),
            registry_guidance=_format_registry_guidance(question, retrieval_plan, self.policy_registry),
            evidence=_format_evidence(chunks),
        )
        content = self.llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            stage="formalization",
            output_config=_OUTPUT_CONFIG,
        ).strip()
        if not content:
            raise FormalizationError("empty_llm_formalization")
        try:
            return certificate_from_text(content)
        except FormalizationError as exc:
            if "not_verifiable" in str(exc):
                retry_content = self.llm_client.chat(
                    [
                        ChatMessage(role="user", content=prompt),
                        ChatMessage(role="assistant", content=content),
                        ChatMessage(role="user", content=_NO_ABSTAIN_HINT),
                    ],
                    temperature=0.0,
                    stage="formalization_no_abstain_retry",
                    output_config=_OUTPUT_CONFIG,
                ).strip()
                if not retry_content:
                    raise
                try:
                    return certificate_from_text(retry_content)
                except FormalizationError as retry_exc:
                    if "cagr_years_must_be_constant" not in str(retry_exc):
                        raise retry_exc
                    content = retry_content
                    exc = retry_exc
            # The LLM used cagr()/cagr_percent() with a variable years argument.
            # Only literal integer years are supported (needed for SMT encoding).
            # Give the model one corrective turn to switch to plain arithmetic.
            if "cagr_years_must_be_constant" not in str(exc):
                raise
            retry_content = self.llm_client.chat(
                [
                    ChatMessage(role="user", content=prompt),
                    ChatMessage(role="assistant", content=content),
                    ChatMessage(role="user", content=_CAGR_VAR_YEARS_HINT),
                ],
                temperature=0.0,
                stage="formalization_cagr_retry",
                output_config=_OUTPUT_CONFIG,
            ).strip()
            if not retry_content:
                raise
            return certificate_from_text(retry_content)

    def repair(
        self,
        certificate: VerificationCertificate,
        error: str,
        evidence_chunks: Iterable[EvidenceChunk],
    ) -> VerificationCertificate:
        chunks = list(evidence_chunks)
        prompt = _REPAIR_PROMPT.format(
            error=error,
            certificate_json=json.dumps(_certificate_to_dict(certificate), indent=2),
            registry_guidance=_format_registry_guidance(certificate.claim.metric, None, self.policy_registry),
            evidence=_format_evidence(chunks),
        )
        content = self.llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            stage="formalization_repair",
            output_config=_OUTPUT_CONFIG,
        ).strip()
        if not content:
            raise FormalizationError("empty_llm_repair")
        return certificate_from_text(content)


def certificate_from_text(text: str) -> VerificationCertificate:
    try:
        data = json.loads(_extract_json(text), strict=False)
    except (json.JSONDecodeError, ValueError) as exc:
        raise FormalizationError(f"invalid_certificate_json:{exc}") from exc
    return certificate_from_json(data)


def certificate_from_json(data: dict) -> VerificationCertificate:
    if not isinstance(data, dict):
        raise FormalizationError("certificate_must_be_object")
    if data.get("verifiable") is False:
        reason = str(data.get("reason", "not_verifiable"))
        raise FormalizationError(f"not_verifiable:{reason}")

    claim = _coerce_claim(data.get("claim"))
    facts = _coerce_facts(data.get("facts"))
    if not facts:
        raise FormalizationError("missing_facts")
    formula = str(data.get("formula", "")).strip()
    if not formula:
        raise FormalizationError("missing_formula")
    calculation = str(data.get("calculation", "")).strip()
    tolerance = _coerce_number(data.get("tolerance", 0.01), "tolerance")
    if tolerance <= 0:
        raise FormalizationError("non_positive_tolerance")

    try:
        variables = formula_variables(formula)
    except (SyntaxError, FormulaError, ValueError) as exc:
        raise FormalizationError(f"invalid_formula:{exc}") from exc
    if not variables:
        raise FormalizationError("formula_has_no_variables")

    fact_names = {fact.name for fact in facts}
    missing = sorted(variable for variable in variables if variable not in fact_names)
    if missing:
        raise FormalizationError(f"formula_variable_missing_fact:{missing[0]}")
    facts = [fact for fact in facts if fact.name in variables]

    return VerificationCertificate(
        claim=claim,
        facts=facts,
        formula=formula,
        calculation=calculation,
        tolerance=tolerance,
    )


def prune_unused_certificate_facts(certificate: VerificationCertificate) -> VerificationCertificate:
    variables = formula_variables(certificate.formula)
    facts = [fact for fact in certificate.facts if fact.name in variables]
    if len(facts) == len(certificate.facts):
        return certificate
    return replace(certificate, facts=facts)


def certificate_to_claim_schema(certificate: VerificationCertificate) -> Tuple[Claim, VerificationSchema]:
    variables = sorted(formula_variables(certificate.formula))
    facts = {fact.name: float(fact.value) for fact in certificate.facts if fact.name in variables}
    fact_units = {fact.name: fact.unit for fact in certificate.facts if fact.name in variables}
    computed = f"computed_{certificate.claim.metric}"
    return (
        Claim(
            metric=certificate.claim.metric,
            claimed_value=float(certificate.claim.claimed_value),
            period=certificate.claim.period,
            variables=facts,
        ),
        VerificationSchema(
            metric=certificate.claim.metric,
            formula=certificate.formula,
            allowed_variables=variables + [computed],
            required_evidence=variables,
            allowed_operators=["+", "-", "*", "/", "=", "<=", ">=", "<", ">", "or", "and"],
            tolerance=float(certificate.tolerance),
            unit_policy="certificate_supplied",
            period_policy="certificate_supplied",
            query_type="counterexample",
            fact_units=fact_units,
            claim_unit=certificate.claim.unit,
            computed_unit=certificate.claim.unit,
            tolerance_source="certificate_supplied",
        ),
    )


def _coerce_claim(value) -> CertificateClaim:
    if not isinstance(value, dict):
        raise FormalizationError("claim_must_be_object")
    metric = _require_name(value.get("metric"), "metric")
    claimed_value = _coerce_number(value.get("claimed_value"), "claimed_value")
    reported = value.get("reported_value")
    return CertificateClaim(
        metric=metric,
        claimed_value=claimed_value,
        unit=str(value.get("unit", "")).strip(),
        period=str(value.get("period", "")).strip(),
        reported_value=None if reported in (None, "") else _coerce_number(reported, "reported_value"),
    )


def _coerce_facts(value) -> list[CertificateFact]:
    if not isinstance(value, list):
        raise FormalizationError("facts_must_be_list")
    facts = []
    names = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise FormalizationError(f"fact_must_be_object:{index}")
        name = _require_name(item.get("name"), "fact_name")
        if name in names:
            raise FormalizationError(f"duplicate_fact:{name}")
        names.add(name)
        facts.append(
            CertificateFact(
                name=name,
                value=_coerce_number(item.get("value"), f"fact_value:{name}"),
                unit=str(item.get("unit", "")).strip(),
                source_quote=str(item.get("source_quote", "")).strip(),
                chunk_id=str(item.get("chunk_id", "")).strip(),
                period=str(item.get("period", "")).strip(),
                row_label=str(item.get("row_label", "")).strip(),
                column=str(item.get("column", "")).strip(),
                fact_type=_coerce_fact_type(item.get("fact_type", "numeric")),
                raw_value=_coerce_optional_number(item.get("raw_value"), f"raw_value:{name}"),
                raw_unit=str(item.get("raw_unit", "")).strip(),
                source_scale=str(item.get("source_scale", "")).strip(),
                source_scale_quote=str(item.get("source_scale_quote", "")).strip(),
                sign_convention=_coerce_sign_convention(item.get("sign_convention")),
                absence_scope=_coerce_absence_scope(item.get("absence_scope")),
            )
        )
    return facts


def _format_evidence(evidence_chunks: Iterable[EvidenceChunk]) -> str:
    parts = []
    for chunk in evidence_chunks:
        page = "" if chunk.page is None else f" page={chunk.page}"
        parts.append(f"[{chunk.chunk_id}{page}]\n{chunk.text}")
    return "\n\n".join(parts)


def _format_retrieval_plan(plan: RetrievalPlan | None) -> str:
    if plan is None or not plan.facts:
        return "No explicit plan."
    lines = []
    if plan.metric:
        lines.append(f"metric: {plan.metric}")
    for fact in plan.facts:
        aliases = ", ".join(fact.aliases) if fact.aliases else ""
        parts = [f"- {fact.name}"]
        if fact.period:
            parts.append(f"period={fact.period}")
        if fact.statement:
            parts.append(f"statement={fact.statement}")
        if aliases:
            parts.append(f"aliases=[{aliases}]")
        lines.append(" ".join(parts))
    if plan.reason:
        lines.append(f"reason: {plan.reason}")
    return "\n".join(lines)


def _format_registry_guidance(
    question_or_metric: str,
    plan: RetrievalPlan | None,
    registry: PolicyRegistry,
    *,
    limit: int = 6,
) -> str:
    policies = _matching_registry_policies(question_or_metric, plan, registry, limit=limit)
    if not policies:
        return "No matching registry policy. Use evidence-grounded names only."
    payload = {
        "registry_index": [
            {
                "policy_id": policy.policy_id,
                "metric": policy.metric,
                "metric_aliases": list(policy.metric_aliases),
            }
            for policy in registry.policies
        ],
        "detailed_candidate_policies": [],
    }
    for policy in policies:
        metadata = getattr(policy, "metadata", {}) or {}
        payload["detailed_candidate_policies"].append(
            {
                "policy_id": policy.policy_id,
                "metric": policy.metric,
                "metric_aliases": list(policy.metric_aliases),
                "formula_templates": list(policy.formula_templates),
                "roles": policy.roles,
                "formula_source_note": metadata.get("formula_source_note", ""),
            }
        )
    return json.dumps(payload, indent=2, sort_keys=True)


def _matching_registry_policies(
    question_or_metric: str,
    plan: RetrievalPlan | None,
    registry: PolicyRegistry,
    *,
    limit: int,
):
    candidates = []
    seen = set()

    def add(policy):
        if policy is not None and policy.policy_id not in seen:
            seen.add(policy.policy_id)
            candidates.append(policy)

    if plan and plan.metric:
        add(registry.find_policy(plan.metric))
    add(registry.find_policy(question_or_metric or ""))

    text = _registry_match_text(question_or_metric, plan)
    for policy in registry.policies:
        if len(candidates) >= limit:
            break
        names = [policy.metric, *policy.metric_aliases]
        if any(_policy_name_in_text(name, text) for name in names):
            add(policy)

    return candidates[:limit]


def _registry_match_text(question_or_metric: str, plan: RetrievalPlan | None) -> str:
    pieces = [question_or_metric or ""]
    if plan is not None:
        pieces.append(plan.metric or "")
        for fact in plan.facts:
            pieces.append(fact.name)
            pieces.extend(fact.aliases or [])
    return _registry_norm(" ".join(pieces))


def _policy_name_in_text(name: str, text: str) -> bool:
    normalized = _registry_norm(name)
    if not normalized:
        return False
    tokens = [tok for tok in normalized.split() if tok not in {"the", "of", "and", "to", "in", "from", "for"}]
    if len(tokens) <= 1:
        return normalized in text.split()
    return " ".join(tokens) in text


def _registry_norm(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(r"[,\.\-/]+", " ", value)
    value = re.sub(r"[_\s]+", " ", value)
    return value.strip()


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object")
    return text[start: end + 1]


def _coerce_number(value, field: str) -> float:
    if isinstance(value, bool):
        raise FormalizationError(f"{field}_must_be_number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        percent = stripped.endswith("%")
        if percent:
            stripped = stripped[:-1].strip()
        try:
            number = float(stripped)
        except ValueError as exc:
            raise FormalizationError(f"{field}_must_be_number") from exc
        return number / 100.0 if percent else number
    raise FormalizationError(f"{field}_must_be_number")


def _coerce_optional_number(value, field: str) -> float | None:
    if value in (None, ""):
        return None
    return _coerce_number(value, field)


def _coerce_fact_type(value) -> str:
    fact_type = str(value or "numeric").strip()
    if fact_type not in {"numeric", "absence_implies_zero"}:
        raise FormalizationError(f"invalid_fact_type:{fact_type}")
    return fact_type


def _coerce_sign_convention(value) -> str:
    convention = str(value or "").strip().lower()
    if convention in {"", "as_reported", "magnitude"}:
        return convention
    raise FormalizationError(f"invalid_sign_convention:{convention}")


def _coerce_absence_scope(value) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise FormalizationError("absence_scope_must_be_object")
    return {str(k): str(v).strip() for k, v in value.items() if str(v).strip()}


_METRIC_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def _require_name(value, field: str) -> str:
    name = str(value or "").strip()
    if field == "metric":
        name = _normalize_metric_name(name)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise FormalizationError(f"invalid_{field}:{name}")
    return name


def _normalize_metric_name(name: str) -> str:
    if not name:
        return name
    if re.fullmatch(r"\d+_[A-Za-z0-9_]+", name):
        number, rest = name.split("_", 1)
        word = _METRIC_NUMBER_WORDS.get(number)
        if word is not None:
            return f"{word}_{rest}"
    if name[0].isdigit():
        return f"metric_{name}"
    return name


def _certificate_to_dict(certificate: VerificationCertificate) -> dict:
    return {
        "verifiable": True,
        "claim": {
            "metric": certificate.claim.metric,
            "claimed_value": certificate.claim.claimed_value,
            "unit": certificate.claim.unit,
            "period": certificate.claim.period,
            "reported_value": certificate.claim.reported_value,
        },
        "facts": [
            {
                "name": f.name,
                "fact_type": f.fact_type,
                "raw_value": f.raw_value,
                "raw_unit": f.raw_unit,
                "source_scale": f.source_scale,
                "source_scale_quote": f.source_scale_quote,
                "value": f.value,
                "unit": f.unit,
                "source_quote": f.source_quote,
                "chunk_id": f.chunk_id,
                "period": f.period,
                "row_label": f.row_label,
                "column": f.column,
                "absence_scope": f.absence_scope,
            }
            for f in certificate.facts
        ],
        "formula": certificate.formula,
        "calculation": certificate.calculation,
        "tolerance": certificate.tolerance,
    }
