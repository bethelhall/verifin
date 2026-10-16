from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any

from verifiqa.formulas.evaluator import format_number
from verifiqa.types import FinanceBenchExample, VerificationFact, VerificationIR, VerificationSchema


class FinqaProgramError(ValueError):
    pass


@dataclass(frozen=True)
class _Compiled:
    formula: str
    value: float


@dataclass(frozen=True)
class _SourceCandidate:
    source: str
    source_label: str
    context_id: str
    row_idx: int | None
    col_idx: int | None
    row_label: str
    column: str
    quote: str
    raw_token: str
    raw_value: float
    canonical_value: float
    has_percent: bool
    fallback: bool = False


@dataclass(frozen=True)
class _Operand:
    raw_text: str
    raw_value: float
    value: float
    has_percent: bool


def resolve_finqa_program(
    example: FinanceBenchExample,
    ir: VerificationIR,
    schema: VerificationSchema,
) -> dict[str, Any] | None:
    """Resolve FinQA's gold executable program into an authoritative verifier IR.

    FinanceBench gets formula authority from a metric registry. FinQA already ships
    per-question executable programs, so those programs should replace the
    LLM-written formula and the program operands should be grounded to the source
    table/text evidence.
    """

    raw = getattr(example, "raw", None) or {}
    qa = raw.get("qa") if isinstance(raw, dict) else None
    if not isinstance(qa, dict) or not qa.get("program"):
        return None

    try:
        resolver = _FinqaProgramResolver(example, ir, schema)
        return resolver.resolve()
    except FinqaProgramError as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "source": "finqa_qa_program",
            "program": qa.get("program") or "",
            "program_re": qa.get("program_re") or "",
        }


class _FinqaProgramResolver:
    def __init__(self, example: FinanceBenchExample, ir: VerificationIR, schema: VerificationSchema):
        self.example = example
        self.ir = ir
        self.schema = schema
        self.raw = getattr(example, "raw", None) or {}
        self.qa = self.raw.get("qa") or {}
        self.program = str(self.qa.get("program") or "").strip()
        self.program_re = str(self.qa.get("program_re") or "").strip()
        self.table = _source_table(self.raw)
        self.candidates = _source_candidates(self.raw, self.table)
        self.used_candidates: set[str] = set()
        self.literal_counts = Counter(_literal_key(text) for text in _program_numeric_literals(self.program))
        self.literal_seen: defaultdict[str, int] = defaultdict(int)
        self.facts: dict[str, VerificationFact] = {}
        self.bindings: list[dict[str, Any]] = []
        self.diagnostics: list[str] = []

    def resolve(self) -> dict[str, Any]:
        steps = _split_top_level(self.program)
        if not steps:
            raise FinqaProgramError("finqa_program_empty")

        previous: list[_Compiled] = []
        for step in steps:
            previous.append(self._compile(step, previous))
        compiled = previous[-1]

        output_scale, scale_reason = _output_scale(
            self.qa,
            self.ir,
            compiled.value,
            compiled.formula,
        )
        formula = compiled.formula
        computed = compiled.value
        if output_scale != 1.0:
            formula = f"({formula}) * {format_number(output_scale)}"
            computed *= output_scale

        if not self.facts:
            raise FinqaProgramError("finqa_program_has_no_grounded_operands")

        xbrl_calculations = {
            "enabled": True,
            "status": "ok",
            "source": "finqa_program",
            "formula_source": "finqa_qa_program",
            "program": self.program,
            "program_re": self.program_re,
            "bindings": self.bindings,
            "constraints": [],
            "declared_variables": sorted({binding["xbrl_variable"] for binding in self.bindings}),
            "dropped_constraints": [],
            "instantiated_constraints": 0,
            "connected_constraints": 0,
            "evidence_fact_variables": [],
            "diagnostics": self.diagnostics,
            "grounding_status": "FINQA_PROGRAM_GROUNDED",
            "binding_sources": sorted({binding.get("source", "") for binding in self.bindings if binding.get("source")}),
            "computed_value": computed,
            "raw_computed_value": compiled.value,
            "output_scale": output_scale,
            "output_scale_reason": scale_reason,
        }

        revised_ir = replace(
            self.ir,
            formula=formula,
            facts=dict(self.facts),
            computed_unit=self.ir.computed_unit or self.ir.claim_unit,
            xbrl_calculations=xbrl_calculations,
        )
        return {
            "status": "applied",
            "reason": "finqa_program_resolved",
            "source": "finqa_qa_program",
            "program": self.program,
            "program_re": self.program_re,
            "formula": formula,
            "computed_value": computed,
            "raw_computed_value": compiled.value,
            "output_scale": output_scale,
            "output_scale_reason": scale_reason,
            "n_operands": len(self.facts),
            "n_bindings": len(self.bindings),
            "verification_ir": revised_ir,
        }

    def _compile(self, text: str, previous: list[_Compiled]) -> _Compiled:
        text = text.strip()
        call = _parse_call(text)
        if call is not None:
            op, args = call
            return self._compile_call(op, args, previous)

        if text.startswith("#"):
            try:
                index = int(text[1:])
            except ValueError as exc:
                raise FinqaProgramError(f"invalid_program_reference:{text}") from exc
            try:
                return previous[index]
            except IndexError as exc:
                raise FinqaProgramError(f"program_reference_out_of_range:{text}") from exc

        if text.startswith("const_"):
            grounded = self._ground_const_if_unique(text)
            if grounded is not None:
                return grounded
            value = _parse_const(text)
            return _Compiled(format_number(value), value)

        operand = _parse_operand(text)
        if operand is not None:
            return self._ground_operand(operand)

        raise FinqaProgramError(f"unsupported_program_token:{text}")

    def _compile_call(self, op: str, args: list[str], previous: list[_Compiled]) -> _Compiled:
        if op in {"add", "subtract", "multiply", "divide"}:
            if len(args) != 2:
                raise FinqaProgramError(f"{op}_requires_two_args")
            left = self._compile(args[0], previous)
            right = self._compile(args[1], previous)
            if op == "add":
                return _Compiled(f"({left.formula} + {right.formula})", left.value + right.value)
            if op == "subtract":
                return _Compiled(f"({left.formula} - {right.formula})", left.value - right.value)
            if op == "multiply":
                return _Compiled(f"({left.formula} * {right.formula})", left.value * right.value)
            if right.value == 0:
                raise FinqaProgramError("finqa_program_division_by_zero")
            return _Compiled(f"({left.formula} / {right.formula})", left.value / right.value)

        if op == "exp":
            if len(args) != 2:
                raise FinqaProgramError("exp_requires_two_args")
            base = self._compile(args[0], previous)
            exponent = self._compile(args[1], previous)
            if not float(exponent.value).is_integer() or exponent.value < 0 or exponent.value > 50:
                raise FinqaProgramError("unsupported_exp_exponent")
            n = int(exponent.value)
            if n == 0:
                return _Compiled("1", 1.0)
            terms = [base.formula for _ in range(n)]
            return _Compiled(f"({' * '.join(terms)})", base.value ** n)

        if op == "greater":
            if len(args) != 2:
                raise FinqaProgramError("greater_requires_two_args")
            left = self._compile(args[0], previous)
            right = self._compile(args[1], previous)
            self.diagnostics.append("greater_resolved_deterministically")
            return _Compiled("1" if left.value > right.value else "0", 1.0 if left.value > right.value else 0.0)

        if op in {"table_sum", "table_average", "table_min", "table_max"}:
            return self._compile_table_aggregate(op, args)

        raise FinqaProgramError(f"unsupported_program_operation:{op}")

    def _ground_const_if_unique(self, text: str) -> _Compiled | None:
        value = _parse_const(text)
        operand = _Operand(raw_text=text, raw_value=value, value=value, has_percent=False)
        matches = [candidate for candidate in self.candidates if _candidate_matches(candidate, operand)]
        primary = [candidate for candidate in matches if not candidate.fallback]
        if primary:
            matches = primary
        unused = [candidate for candidate in matches if candidate.context_id not in self.used_candidates]
        if len(unused) != 1:
            return None
        return self._ground_operand(operand, candidate=unused[0])

    def _ground_operand(self, operand: _Operand, candidate: _SourceCandidate | None = None) -> _Compiled:
        if candidate is None:
            candidate = self._select_candidate(operand)
        else:
            self.used_candidates.add(candidate.context_id)
        index = len(self.facts)
        name = f"finqa_operand_{index}"
        unit = "ratio" if operand.has_percent else ""
        fact = VerificationFact(
            name=name,
            value=operand.value,
            unit=unit,
            fact_type="numeric",
            raw_value=candidate.raw_value,
            raw_unit="percent" if candidate.has_percent else "",
            source_scale="ones",
            source_scale_quote="%" if candidate.has_percent else "",
            source_quote=candidate.quote,
            chunk_id=f"{self.example.financebench_id}:source:{candidate.source_label}",
            row_label=candidate.row_label,
            column=candidate.column,
        )
        self.facts[name] = fact
        self.bindings.append(_binding_for_fact(name, operand.value, candidate))
        return _Compiled(name, operand.value)

    def _compile_table_aggregate(self, op: str, args: list[str]) -> _Compiled:
        if len(args) != 2:
            raise FinqaProgramError(f"{op}_requires_label_and_selector")
        label = args[0].strip()
        if not label or label.lower() == "none":
            raise FinqaProgramError(f"{op}_missing_row_label")
        cells = self._table_row_numeric_cells(label)
        if not cells:
            raise FinqaProgramError(f"{op}_row_not_grounded:{_slug(label)}")

        compiled: list[_Compiled] = []
        for candidate in cells:
            index = len(self.facts)
            name = f"finqa_operand_{index}"
            value = candidate.raw_value
            fact = VerificationFact(
                name=name,
                value=value,
                unit="",
                fact_type="numeric",
                raw_value=candidate.raw_value,
                raw_unit="percent" if candidate.has_percent else "",
                source_scale="ones",
                source_scale_quote="%" if candidate.has_percent else "",
                source_quote=candidate.quote,
                chunk_id=f"{self.example.financebench_id}:source:{candidate.source_label}",
                row_label=candidate.row_label,
                column=candidate.column,
            )
            self.facts[name] = fact
            self.bindings.append(_binding_for_fact(name, value, candidate))
            compiled.append(_Compiled(name, value))

        values = [item.value for item in compiled]
        formulas = [item.formula for item in compiled]
        if op == "table_sum":
            return _Compiled(_sum_formula(formulas), sum(values))
        if op == "table_average":
            return _Compiled(f"({_sum_formula(formulas)} / {len(values)})", sum(values) / len(values))
        if op == "table_min":
            selected = min(range(len(values)), key=lambda idx: values[idx])
            self.diagnostics.append(f"table_min_resolved_deterministically:{_slug(label)}")
            return compiled[selected]
        selected = max(range(len(values)), key=lambda idx: values[idx])
        self.diagnostics.append(f"table_max_resolved_deterministically:{_slug(label)}")
        return compiled[selected]

    def _table_row_numeric_cells(self, label: str) -> list[_SourceCandidate]:
        if len(self.table) < 2:
            return []
        wanted = _tokens(label)
        matches: list[int] = []
        for row_idx, row in enumerate(self.table[1:], start=1):
            if not row:
                continue
            have = _tokens(str(row[0]))
            if have and (have == wanted or wanted <= have):
                matches.append(row_idx)
        if len(matches) != 1:
            return []
        row_idx = matches[0]
        header = self.table[0]
        row = self.table[row_idx]
        out: list[_SourceCandidate] = []
        for col_idx in range(1, len(row)):
            cell = str(row[col_idx])
            numbers = _numbers_in_source(cell)
            if not numbers:
                continue
            # A table cell holds a single value. FinQA writes negatives as "-13 ( 13 )"
            # (the signed value plus a parenthesized absolute repeat), which parses as two
            # numbers (-13 and +13); summing/averaging both cancels the magnitude to ~0.
            # Take only the first, correctly-signed number per cell.
            raw_value, canonical_value, has_percent, raw_token = numbers[0]
            out.append(_table_candidate(
                self.table,
                header,
                row,
                row_idx,
                col_idx,
                0,
                raw_token,
                raw_value,
                canonical_value,
                has_percent,
                fallback=False,
            ))
        return out

    def _select_candidate(self, operand: _Operand) -> _SourceCandidate:
        matches = [candidate for candidate in self.candidates if _candidate_matches(candidate, operand)]
        primary = [candidate for candidate in matches if not candidate.fallback]
        if primary:
            matches = primary
        if not matches:
            question_constant = self._question_constant_candidate(operand)
            if question_constant is not None:
                key = _literal_key(operand.raw_text)
                self.literal_seen[key] += 1
                self.used_candidates.add(question_constant.context_id)
                return question_constant
            raise FinqaProgramError(f"program_operand_not_grounded:{operand.raw_text}")

        key = _literal_key(operand.raw_text)
        ordered = sorted(matches, key=lambda c: (c.fallback, c.source_label, c.row_idx or -1, c.col_idx or -1, c.context_id))
        unused = [candidate for candidate in ordered if candidate.context_id not in self.used_candidates]
        if len(unused) == 1:
            selected = unused[0]
        elif len(ordered) == 1 and self.literal_counts.get(key, 0) > 1:
            selected = ordered[0]
        elif len(ordered) == self.literal_counts.get(key, 0):
            index = self.literal_seen[key]
            if index >= len(ordered):
                selected = ordered[-1]
            else:
                selected = ordered[index]
        elif _candidates_share_value(ordered):
            # All ambiguous candidates hold the same value, so the bound value is identical
            # regardless of which cell is chosen — provenance is immaterial to verification.
            # Prefer an unused candidate; otherwise reuse the first.
            selected = unused[0] if unused else ordered[0]
        else:
            raise FinqaProgramError(f"ambiguous_program_operand:{operand.raw_text}:{len(ordered)}")

        self.literal_seen[key] += 1
        self.used_candidates.add(selected.context_id)
        return selected

    def _question_constant_candidate(self, operand: _Operand) -> _SourceCandidate | None:
        question = " ".join(
            str(value or "")
            for value in (
                getattr(self.example, "question", ""),
                self.qa.get("question", "") if isinstance(self.qa, dict) else "",
            )
        )
        if not _approved_question_constant(operand, question):
            return None
        context_id = f"question_constant_{_literal_key(operand.raw_text)}"
        return _SourceCandidate(
            source="finqa_question",
            source_label="question",
            context_id=context_id,
            row_idx=None,
            col_idx=None,
            row_label="question",
            column="",
            quote=question.strip(),
            raw_token=operand.raw_text,
            raw_value=operand.raw_value,
            canonical_value=operand.value,
            has_percent=operand.has_percent,
            fallback=False,
        )


def _candidates_share_value(candidates: list[_SourceCandidate], eps: float = 1e-9) -> bool:
    """True if every candidate carries the same canonical value (binding is immaterial)."""
    if not candidates:
        return False
    first = candidates[0].canonical_value
    return all(abs(c.canonical_value - first) <= eps for c in candidates)


def _binding_for_fact(name: str, fact_value: float, candidate: _SourceCandidate) -> dict[str, Any]:
    multiplier = 1.0
    if candidate.raw_value != 0:
        multiplier = fact_value / candidate.raw_value
    xbrl_var = f"finqa_src_{name}"
    return {
        "fact_name": name,
        "xbrl_variable": xbrl_var,
        "xbrl_value": candidate.raw_value,
        "xbrl_smt_value": format_number(candidate.raw_value),
        "concept": f"{candidate.source}:{candidate.source_label}",
        "context_id": candidate.context_id,
        "context_period": {
            "instant": "",
            "start_date": "",
            "end_date": "",
            "dimensions": [],
        },
        "unit_ref": "percent" if candidate.has_percent else "",
        "unit_measures": [],
        "scale": 1.0,
        "binding_multiplier": multiplier,
        "binding_kind": "same_sign",
        "source": candidate.source,
    }


def _source_candidates(raw: dict[str, Any], table: list[list[Any]]) -> list[_SourceCandidate]:
    qa = raw.get("qa") or {}
    gold = qa.get("gold_inds") or {}
    pre_text = raw.get("pre_text") or []
    post_text = raw.get("post_text") or []
    all_text = pre_text + post_text
    candidates: list[_SourceCandidate] = []
    header = table[0] if table else []

    gold_table_rows: set[int] = set()
    for key in sorted(gold):
        if key.startswith("table_"):
            row_idx = _parse_index(key)
            if row_idx is not None and row_idx < len(table):
                gold_table_rows.add(row_idx)
                row = table[row_idx]
                for col_idx, cell in enumerate(row):
                    for number_idx, parsed in enumerate(_numbers_in_source(str(cell))):
                        raw_value, canonical_value, has_percent, raw_token = parsed
                        candidates.append(_table_candidate(
                            table,
                            header,
                            row,
                            row_idx,
                            col_idx,
                            number_idx,
                            raw_token,
                            raw_value,
                            canonical_value,
                            has_percent,
                            fallback=False,
                        ))
        elif key.startswith("text_"):
            text_idx = _parse_index(key)
            if text_idx is not None and text_idx < len(all_text):
                text = str(all_text[text_idx])
                for number_idx, parsed in enumerate(_numbers_in_source(text)):
                    raw_value, canonical_value, has_percent, raw_token = parsed
                    candidates.append(_text_candidate(
                        key,
                        text,
                        number_idx,
                        raw_token,
                        raw_value,
                        canonical_value,
                        has_percent,
                        fallback=False,
                    ))

    fallback_start = 0 if table and _row_has_number(table[0]) else 1
    for row_idx, row in enumerate(table[fallback_start:], start=fallback_start):
        if row_idx in gold_table_rows:
            continue
        for col_idx, cell in enumerate(row):
            for number_idx, parsed in enumerate(_numbers_in_source(str(cell))):
                raw_value, canonical_value, has_percent, raw_token = parsed
                candidates.append(_table_candidate(
                    table,
                    header,
                    row,
                    row_idx,
                    col_idx,
                    number_idx,
                    raw_token,
                    raw_value,
                    canonical_value,
                    has_percent,
                    fallback=True,
                ))
    return candidates


def _row_has_number(row: list[Any]) -> bool:
    return any(_numbers_in_source(str(cell)) for cell in row)


def _table_candidate(
    table: list[list[Any]],
    header: list[Any],
    row: list[Any],
    row_idx: int,
    col_idx: int,
    number_idx: int,
    raw_token: str,
    raw_value: float,
    canonical_value: float,
    has_percent: bool,
    *,
    fallback: bool,
) -> _SourceCandidate:
    row_label = str(row[0]).strip() if row else ""
    column = str(header[col_idx]).strip() if col_idx < len(header) else ""
    source_label = f"table_{row_idx}"
    return _SourceCandidate(
        source="finqa_table",
        source_label=source_label,
        context_id=f"finqa_table_r{row_idx}_c{col_idx}_n{number_idx}",
        row_idx=row_idx,
        col_idx=col_idx,
        row_label=row_label,
        column=column,
        quote=_format_table_row(header, row),
        raw_token=raw_token,
        raw_value=raw_value,
        canonical_value=canonical_value,
        has_percent=has_percent,
        fallback=fallback,
    )


def _text_candidate(
    source_label: str,
    text: str,
    number_idx: int,
    raw_token: str,
    raw_value: float,
    canonical_value: float,
    has_percent: bool,
    *,
    fallback: bool,
) -> _SourceCandidate:
    return _SourceCandidate(
        source="source_document",
        source_label=source_label,
        context_id=f"finqa_text_{_slug(source_label)}_n{number_idx}",
        row_idx=None,
        col_idx=None,
        row_label="",
        column="",
        quote=text.strip(),
        raw_token=raw_token,
        raw_value=raw_value,
        canonical_value=canonical_value,
        has_percent=has_percent,
        fallback=fallback,
    )


def _candidate_matches(candidate: _SourceCandidate, operand: _Operand) -> bool:
    if operand.has_percent:
        return _close(candidate.canonical_value, operand.value)
    return _close(candidate.raw_value, operand.raw_value) or _close(candidate.canonical_value, operand.value)


def _approved_question_constant(operand: _Operand, question: str) -> bool:
    if operand.has_percent:
        return False
    text = (question or "").lower()
    raw = str(operand.raw_text or "").lower().removeprefix("const_")
    raw = raw.replace(",", "")
    if not re.search(rf"(?<!\d){re.escape(raw)}(?!\d)", text):
        return False
    if _close(operand.raw_value, 365.0):
        return "day" in text and "year" in text
    return False


def _output_scale(qa: dict[str, Any], ir: VerificationIR, raw_value: float, formula: str) -> tuple[float, str]:
    unit = (ir.claim_unit or ir.computed_unit or "").lower()
    if unit not in {"percent", "percentage_points"}:
        return 1.0, "claim_unit_not_percent"

    answer_numbers = _numbers_in_answer(str(qa.get("answer") or ""))
    if answer_numbers:
        best_raw = min(abs(number - raw_value) for number in answer_numbers)
        best_scaled = min(abs(number - raw_value * 100.0) for number in answer_numbers)
        if best_scaled + 1e-9 < best_raw:
            return 100.0, "finqa_answer_percent_presentation"
        return 1.0, "finqa_answer_raw_presentation"

    compact = re.sub(r"\s+", "", formula)
    if "*100" in compact or "100*" in compact:
        return 1.0, "formula_already_percent_scaled"
    if abs(raw_value) <= 5:
        return 100.0, "percent_claim_ratio_program_output"
    return 1.0, "percent_claim_raw_program_output"


def _numbers_in_answer(text: str) -> list[float]:
    return [raw for raw, _canonical, _has_percent, _token in _numbers_in_source(text)]


def _program_numeric_literals(program: str) -> list[str]:
    literals: list[str] = []
    for token in re.findall(r"(?<![A-Za-z_#])[-+]?\d[\d,]*(?:\.\d+)?%?", program or ""):
        literals.append(token)
    return literals


def _source_table(raw: dict[str, Any]) -> list[list[Any]]:
    table = raw.get("table") if isinstance(raw, dict) else None
    if not isinstance(table, list) or len(table) < 2:
        return []
    return [row for row in table if isinstance(row, list) and row]


def _split_top_level(text: str) -> list[str]:
    out: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise FinqaProgramError("unbalanced_program_parentheses")
        elif char == "," and depth == 0:
            item = text[start:index].strip()
            if item:
                out.append(item)
            start = index + 1
    if depth != 0:
        raise FinqaProgramError("unbalanced_program_parentheses")
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _parse_call(text: str) -> tuple[str, list[str]] | None:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if not match or not text.endswith(")"):
        return None
    op = match.group(1)
    inner = text[match.end():-1]
    return op, _split_top_level(inner)


def _parse_const(text: str) -> float:
    suffix = text[len("const_"):]
    if suffix.startswith("m"):
        suffix = "-" + suffix[1:]
    try:
        return float(suffix)
    except ValueError as exc:
        raise FinqaProgramError(f"invalid_program_constant:{text}") from exc


def _parse_operand(text: str) -> _Operand | None:
    token = text.strip()
    parsed = _parse_number_token(token)
    if parsed is None:
        return None
    raw_value, canonical_value, has_percent = parsed
    return _Operand(raw_text=token, raw_value=raw_value, value=canonical_value, has_percent=has_percent)


def _numbers_in_source(text: str) -> list[tuple[float, float, bool, str]]:
    out: list[tuple[float, float, bool, str]] = []
    pattern = r"\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?"
    for raw in re.findall(pattern, text or ""):
        parsed = _parse_number_token(raw)
        if parsed is None:
            continue
        raw_value, canonical_value, has_percent = parsed
        out.append((raw_value, canonical_value, has_percent, raw))
    return out


def _parse_number_token(token: str) -> tuple[float, float, bool] | None:
    text = str(token or "").strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").strip()
    text = text.replace("$", "").replace(",", "")
    has_percent = text.endswith("%")
    if has_percent:
        text = text[:-1]
    if not text:
        return None
    try:
        raw_value = float(text)
    except ValueError:
        return None
    if negative:
        raw_value = -raw_value
    canonical = raw_value / 100.0 if has_percent else raw_value
    return raw_value, canonical, has_percent


def _format_table_row(header: list[Any], row: list[Any]) -> str:
    if not header:
        return " | ".join(str(value).strip() for value in row)
    pairs = []
    for col_idx, value in enumerate(row):
        text = str(value).strip()
        label = str(header[col_idx]).strip() if col_idx < len(header) else ""
        pairs.append(f"{label}: {text}" if label else text)
    return " | ".join(pairs)


def _sum_formula(formulas: list[str]) -> str:
    if not formulas:
        return "0"
    if len(formulas) == 1:
        return formulas[0]
    return "(" + " + ".join(formulas) + ")"


def _parse_index(key: str) -> int | None:
    try:
        return int(key.rsplit("_", 1)[-1])
    except (TypeError, ValueError, IndexError):
        return None


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token[:-1] if len(token) > 3 and token.endswith("s") else token
        for token in re.findall(r"[a-z]+|[0-9]+", (text or "").lower())
        if token not in {"and", "or", "of", "the", "a", "an", "in", "to", "for", "none"}
    )


def _literal_key(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text or "").strip("_")[:80] or "unknown"


def _close(left: float, right: float) -> bool:
    if not (math.isfinite(left) and math.isfinite(right)):
        return False
    return abs(left - right) <= max(1e-9, 1e-6 * max(abs(left), abs(right), 1.0))
