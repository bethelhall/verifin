from __future__ import annotations

import json
import re
from decimal import Decimal

from verifiqa.formulas.evaluator import (
    format_number,
    formula_assertion_smt,
    formula_domain_constraints_smt,
)
from verifiqa.generation.llm_client import ChatMessage
from verifiqa.types import Claim, VerificationIR, VerificationSchema
from verifiqa.verification.ir import ir_from_claim_schema

_EXAMPLE = """\
--- EXAMPLE ---
Input:
{
  "metric": "quick_ratio",
  "computed_variable": "computed_quick_ratio",
  "formula": "(current_assets - inventory) / current_liabilities",
  "facts": {
    "current_assets": {"value": 5000.0, "unit": "USD millions"},
    "current_liabilities": {"value": 2700.0, "unit": "USD millions"},
    "inventory": {"value": 800.0, "unit": "USD millions"}
  },
  "claim": {"value": 1.5555555555555556, "unit": "ratio", "tolerance": 0.01},
  "allowed_variables": ["computed_quick_ratio", "current_assets", "current_liabilities", "inventory"],
  "required_facts": ["current_assets", "current_liabilities", "inventory"]
}

Output:
(set-logic QF_NRA)
(set-option :produce-unsat-cores true)
(set-option :produce-models true)

(declare-const computed_quick_ratio Real)
(declare-const current_assets Real)
(declare-const current_liabilities Real)
(declare-const inventory Real)

(assert (! (= current_assets 5000.0) :named evidence_current_assets))
(assert (! (= current_liabilities 2700.0) :named evidence_current_liabilities))
(assert (! (= inventory 800.0) :named evidence_inventory))

(assert (! (= computed_quick_ratio (/ (- current_assets inventory) current_liabilities)) :named formula_quick_ratio))

(assert (! (or (> current_liabilities 0) (< current_liabilities 0)) :named denom_nonzero))

(assert (! (<= (- computed_quick_ratio 1.5555555555555556) 0.01) :named claim_upper))
(assert (! (<= (- 1.5555555555555556 computed_quick_ratio) 0.01) :named claim_lower))

(check-sat)
(get-unsat-core)
(get-model)
--- END EXAMPLE ---"""


_REPAIR_PROMPT = """\
The SMT-LIB query below was generated from the verification IR, run through Z3, \
and Z3 returned UNSAT.

UNSAT means no assignment simultaneously satisfies the evidence, formula, and claim \
assertions. This could mean:
(A) The SMT encoding has an error — wrong formula expression, wrong evidence value, \
or missing/incorrect domain constraint.
(B) The claimed value is genuinely wrong given the evidence.

Unsatisfiable core (the named assertions that are contradictory):
{unsat_core}

Your task: diagnose, then either fix or declare CANNOT_REPAIR.

Step 1 — Check the formula assertion.
  Verify the formula assertion matches the IR formula string exactly.
  If not, fix it.

Step 2 — Check evidence values.
  Verify each evidence assertion uses the exact IR value.
  If not, fix it.

Step 3 — Check domain constraints.
  Every division denominator in the formula needs:
    (or (> <denominator_expr> 0) (< <denominator_expr> 0))
  where <denominator_expr> is the exact sub-expression from the formula, not simplified.
  Add or correct any missing/wrong domain constraints.

Step 4 — Decide.
  If you found and fixed an encoding error: output the corrected SMT-LIB only. \
No markdown, no explanation.
  If the claim is genuinely wrong (evidence and formula are correctly encoded \
and the claim cannot hold): output exactly: CANNOT_REPAIR

IR:
{payload}

Original SMT:
{smtlib}

Unsat core:
{unsat_core}
"""

_ANSWER_REPAIR_PROMPT = """\
The Z3 SMT solver verified that {metric} = {computed_value} given the grounded evidence.

Your earlier answer claimed {metric} = {claimed_value}{unit_str}, which is outside \
the allowed tolerance ±{tolerance}.

Try again.

Output only that number. No explanation, no units, no other text.
"""

_PROMPT = """\
Generate SMT-LIB from the typed verification IR below.
Return SMT-LIB only. No markdown, no explanation.

You are a compiler from verification IR to SMT-LIB, not a financial reasoning
agent. Encode the IR literally. Do not infer, complete, repair, simplify, or
reinterpret the formula from the metric name, question, facts, or claimed value.
Never choose an arithmetic operation because it makes the claim SAT.

Semantics: assert that the claimed value is within tolerance.
SAT = claim verified. UNSAT = claim is inconsistent with evidence or formula.

Requirements:
- First command: (set-logic QF_NRA)
- Second: (set-option :produce-unsat-cores true)
- Third: (set-option :produce-models true)
- declare-const only (never declare-fun); declare exactly allowed_variables, all Real.
- One named evidence fact per required_facts variable.
- One named formula constraint named formula_<metric> that encodes the formula
  string literally.
- Add required formula domain constraints, including one assertion for each division denominator:
  (or (> <denominator> 0) (< <denominator> 0)).
- Two named claim assertions: claim_upper asserts (computed - claimed) <= tolerance,
  claim_lower asserts (claimed - computed) <= tolerance.
  Together they assert |computed - claimed| <= tolerance.
- Do not assert computed_<metric> equals the claimed value.
- Use fact values, claim.value, tolerance, and formula constants exactly as provided.
- Do not invent an operation, intermediate expression, or formula variable that
  is not present in the formula string.
- Write every numeric literal in SMT-LIB decimal form, never scientific notation.
  Example: use 0.00005, not 5e-05.
- Introduce no variables, constants, functions, assertions, or assumptions not in the IR.
- All facts are already normalized to canonical units. Do not perform unit conversion in SMT.
- Formula language:
  - +, -, *, / are normal arithmetic over Real variables.
  - cagr_percent(end, start, years): assert that start * (1 + computed_<metric>/100)^years = end,
    and add domain assertion (> computed_<metric> -100).
  - cagr(end, start, years): assert that start * (1 + computed_<metric>)^years = end,
    and add domain assertion (> computed_<metric> -1).
  - Expand powers by repeated multiplication; do not use unsupported pow/expt/^ operators.
- End with (check-sat), (get-unsat-core), and (get-model).

{example}

Input:
{payload}
"""

_INVALID_REPAIR_PROMPT = """\
The SMT-LIB you produced was rejected by the Z3 parser with this error:
{error}

Invalid SMT-LIB you produced:
{bad_smt}

Regenerate correct SMT-LIB for the same IR below, fixing only the syntax/parse error.
Encode the IR literally and follow every requirement from the original task (same
logic, same declared variables, same evidence/formula/claim assertions, decimal
literals only, end with (check-sat) (get-unsat-core) (get-model)). Do not change the
intended meaning or make the claim SAT. Return SMT-LIB only, no markdown, no prose.

Input:
{payload}
"""


class SmtGenerator:
    """Generate SMT-LIB with an LLM from a typed verification IR."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_ir(self, ir: VerificationIR) -> str:
        if self.llm_client is None:
            raise ValueError("missing_llm_client_for_smt_generation")
        prompt = _PROMPT.format(
            example=_EXAMPLE,
            payload=_prompt_payload_json(_payload_ir(ir)),
        )
        content = self.llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            stage="smt_generation",
            max_tokens=1024,
            
        ).strip()
        if not content:
            raise ValueError("empty_llm_smt")
        return _extract_smt(content)

    def repair(self, ir: VerificationIR, original_smtlib: str, unsat_core: list[str]) -> str | None:
        """Feed the Z3 unsat-core back to the LLM for self-diagnosis and repair.

        Returns the repaired SMT-LIB string, or None if the LLM determines the
        claim is genuinely wrong (CANNOT_REPAIR).
        """
        if self.llm_client is None:
            raise ValueError("missing_llm_client_for_smt_repair")
        core_str = " ".join(unsat_core) if unsat_core else "(no core returned)"
        prompt = _REPAIR_PROMPT.format(
            payload=_prompt_payload_json(_payload_ir(ir)),
            smtlib=original_smtlib,
            unsat_core=core_str,
        )
        content = self.llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            stage="smt_repair",
            max_tokens=1500,
        ).strip()
        if not content or "CANNOT_REPAIR" in content:
            return None
        return _extract_smt(content)

    def repair_invalid(self, ir: VerificationIR, bad_smtlib: str, error: str) -> str | None:
        """Regenerate SMT-LIB once when the Z3 parser rejected it (INVALID), feeding the
        parse error and the broken SMT back so the model fixes the syntax. Returns the
        regenerated SMT-LIB, or None if the model produced nothing."""
        if self.llm_client is None:
            raise ValueError("missing_llm_client_for_smt_repair")
        prompt = _INVALID_REPAIR_PROMPT.format(
            error=(error or "(no error text)")[:600],
            bad_smt=(bad_smtlib or "")[:2500],
            payload=_prompt_payload_json(_payload_ir(ir)),
        )
        content = self.llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            stage="smt_repair_invalid",
            max_tokens=1024,
        ).strip()
        if not content:
            return None
        return _extract_smt(content)

    def repair_answer(self, ir: VerificationIR, counterexample: str) -> float | None:
        """Feed the Z3-computed value back to the LLM to correct its numeric answer.

        Returns the LLM's corrected value, or None if the counterexample cannot be parsed.
        The caller should check whether the corrected value is within ir.tolerance of
        the Z3-computed value before accepting it as REPAIRED_VERIFIED.
        """
        if self.llm_client is None:
            return None
        computed = extract_computed_value(counterexample, ir.metric)
        if computed is None:
            return None
        unit_str = f" {ir.claim_unit}" if ir.claim_unit else ""
        prompt = _ANSWER_REPAIR_PROMPT.format(
            metric=ir.metric,
            claimed_value=ir.claimed_value,
            unit_str=unit_str,
            tolerance=ir.tolerance,
            computed_value=round(computed, 6),
        )
        content = self.llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            stage="answer_repair",
            max_tokens=64,
        ).strip()
        return _parse_float_answer(content)

    def generate(self, claim: Claim, schema: VerificationSchema) -> str:
        return self.generate_ir(ir_from_claim_schema(claim, schema))


def extract_computed_value(counterexample: str, metric: str) -> float | None:
    """Extract the computed_<metric> value from a Z3 model string.

    Handles decimal literals and Z3 rational fractions like (/ 1458175.0 15536.0).
    Uses a balanced-paren extractor so nested expressions like (/ num den) parse correctly.
    """
    pattern = rf'\(define-fun\s+computed_{re.escape(metric)}\s+\(\)\s+Real\s+'
    m = re.search(pattern, counterexample)
    if not m:
        return None
    val_str = _next_sexp(counterexample, m.end())
    if val_str is None:
        return None
    return _parse_z3_value(val_str)


def _next_sexp(text: str, start: int) -> str | None:
    """Return the next complete S-expression (atom or parenthesized) starting at start."""
    i = start
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    if i >= n:
        return None
    if text[i] != '(':
        j = i
        while j < n and not text[j].isspace() and text[j] not in ('(', ')'):
            j += 1
        return text[i:j] if j > i else None
    depth = 0
    j = i
    while j < n:
        if text[j] == '(':
            depth += 1
        elif text[j] == ')':
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
        j += 1
    return None


def _parse_z3_value(val_str: str) -> float | None:
    val_str = val_str.strip()
    try:
        return float(val_str)
    except ValueError:
        pass
    # Rational: (/ NUM DEN)
    m = re.fullmatch(r'\(/\s+([\d\.]+)\s+([\d\.]+)\s*\)', val_str)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    # Negation: (- EXPR)
    m = re.fullmatch(r'\(-\s+(.*)\)', val_str, re.DOTALL)
    if m:
        inner = _parse_z3_value(m.group(1).strip())
        if inner is not None:
            return -inner
    # Last-resort: first numeric token
    m = re.search(r'-?[\d]+\.?[\d]*', val_str)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


def _parse_float_answer(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    m = re.search(r'-?[\d]+\.?[\d]*', text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


_JSON_SCI_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?\d+")


def _prompt_payload_json(payload: dict) -> str:
    return _normalize_json_numeric_literals(json.dumps(payload, indent=2, sort_keys=True))


def _normalize_json_numeric_literals(text: str) -> str:
    out: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue

        match = _JSON_SCI_NUMBER_RE.match(text, index)
        if match is not None:
            out.append(format_number(float(match.group(0))))
            index = match.end()
            continue

        out.append(char)
        index += 1
    return "".join(out)

def _payload_ir(ir: VerificationIR) -> dict:
    required = sorted(ir.facts)
    computed = f"computed_{ir.metric}"
    # The IR claim has already been bound to the displayed LLM answer by the
    # output contract. SMT must use it exactly; hidden rounding here would verify a
    # value the model did not actually output.
    claim_value = ir.claimed_value
    return {
        "metric": ir.metric,
        "computed_variable": computed,
        "formula": ir.formula,
        "facts": {
            name: {
                "value": ir.facts[name].value,
                "unit": ir.facts[name].unit,
            }
            for name in required
        },
        "claim": {
            "value": claim_value,
            "unit": ir.claim_unit,
            "tolerance": ir.tolerance,
            "precision_digits": ir.precision_digits,
            "tolerance_source": ir.tolerance_source,
        },
        "computed_unit": ir.computed_unit or ir.claim_unit,
        "allowed_variables": [computed] + required,
        "required_facts": required,
        "query_type": ir.query_type,
    }



def render_ir_smt(ir: VerificationIR, *, include_claim: bool = True) -> str:
    """Render the arithmetic IR, optionally checking its claim-free consistency.

    Omitting the claim removes only its two tolerance assertions. Evidence,
    formula and domain assertions remain identical; the claimed value and its
    tolerance need not even be numeric for this independent consistency query.
    """

    metric_var = f"computed_{ir.metric}"
    fact_names = sorted(ir.facts)
    declared = [metric_var] + fact_names
    lines = [
        "(set-logic QF_NRA)",
        "(set-option :produce-unsat-cores true)",
        "(set-option :produce-models true)",
        "",
    ]
    for variable in declared:
        lines.append(f"(declare-const {variable} Real)")
    lines.append("")
    exact_bound_values = {}
    for binding in ir.xbrl_calculations.get("bindings", []):
        name = binding.get("fact_name")
        exact = binding.get("xbrl_smt_value")
        if name in ir.facts and exact is not None and not binding.get("binding_tolerance"):
            value = Decimal(str(exact)) * Decimal(str(binding.get("binding_multiplier") or 1.0))
            # Keep rounded or independently extracted evidence intact when it has
            # not been replaced by this authoritative binding.
            if float(value) == ir.facts[name].value:
                exact_bound_values[name] = format(value, "f")
    for name in fact_names:
        value = exact_bound_values.get(name, format_number(ir.facts[name].value))
        lines.append(
            f"(assert (! (= {name} {value}) :named evidence_{name}))"
        )
    lines.append("")
    lines.append(
        f"(assert (! {formula_assertion_smt(metric_var, ir.formula, ir.computed_unit or ir.claim_unit)} :named formula_{ir.metric}))"
    )
    for index, constraint in enumerate(
        formula_domain_constraints_smt(metric_var, ir.formula, ir.computed_unit or ir.claim_unit)
    ):
        lines.append(f"(assert (! {constraint} :named domain_{ir.metric}_{index}))")

    if include_claim:
        claimed = format_number(float(ir.claimed_value))
        tolerance = format_number(ir.tolerance)
        lines.append("")
        lines.append(f"(assert (! (<= (- {metric_var} {claimed}) {tolerance}) :named claim_upper))")
        lines.append(f"(assert (! (<= (- {claimed} {metric_var}) {tolerance}) :named claim_lower))")
    lines.append("")
    lines.append("(check-sat)")
    lines.append("(get-unsat-core)")
    lines.append("(get-model)")
    return "\n".join(lines)

def _extract_smt(text: str) -> str:
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 3:
            block = parts[1].strip()
            lines = block.splitlines()
            if lines and lines[0].strip().lower() in {"smt", "smt2", "smt-lib", "smtlib"}:
                lines = lines[1:]
            stripped = "\n".join(lines).strip()
        else:
            stripped = stripped.replace("```", "").strip()

    start = re.search(r"\(\s*set-logic\b", stripped)
    if start is None:
        return stripped

    smt = stripped[start.start():].strip()
    commands = list(re.finditer(r"\(\s*(?:check-sat|get-unsat-core|get-model)\s*\)", smt))
    if commands:
        return _normalize_smt_numeric_literals(smt[:commands[-1].end()].strip())
    return _normalize_smt_numeric_literals(smt)


def _normalize_smt_numeric_literals(smt: str) -> str:
    return re.sub(
        r"(?<![A-Za-z0-9_:.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)[eE][-+]?\d+(?![A-Za-z0-9_:.])",
        lambda match: format_number(float(match.group(0))),
        smt,
    )
