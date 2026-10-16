"""Table-cell grounding for table-QA datasets (FinQA).

FinanceBench grounds facts against EDGAR XBRL concepts; FinQA answers come from a
specific source table, so the authoritative value is a *table cell*. This module is
the table analogue of the XBRL binder: it resolves each IR fact to a cell by matching
the fact's row label to a table row and its column/period cues to a source-table
column — entirely value-blind (the cell is chosen by labels, never by the
LLM-extracted number) — and emits bindings in the same shape as the XBRL path so
all downstream machinery (SMT
augmentation, catch, repair, grounding tier) is reused unchanged.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from verifiqa.types import VerificationFact, VerificationIR


_STOP = frozenset({"and", "or", "of", "the", "a", "an", "in", "to", "for"})
_SCALE_FACTORS = {
    "thousand": Decimal("1000"),
    "thousands": Decimal("1000"),
    "million": Decimal("1000000"),
    "millions": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "billions": Decimal("1000000000"),
}


def attach_table_calculations(ir: VerificationIR, table: list[list[Any]]) -> VerificationIR:
    """Bind every numeric IR fact to a source-table cell, value-blind.

    Sets ir.xbrl_calculations to a binding result in the same format the XBRL path
    produces. Requires *all* numeric facts to bind (full grounding) — a partial bind
    would mix table-grounded and LLM-extracted evidence, so it falls back to no
    bindings (llm_only) rather than ground a corrupt mixture.
    """
    numeric_facts = {
        name for name, fact in ir.facts.items() if not _is_absence_zero_fact(fact)
    }
    bindings: list[dict[str, Any]] = []
    diagnostics: list[str] = []

    for name in sorted(numeric_facts):
        binding = _bind_fact_table(ir.facts[name], table)
        if binding is None:
            diagnostics.append(f"unbound_fact:{name}")
        else:
            bindings.append(binding)

    bound_names = {b["fact_name"] for b in bindings}
    if bindings and bound_names == numeric_facts:
        ir.xbrl_calculations = _ok_result(bindings, diagnostics)
    else:
        status = "partial_table_bindings" if bindings else "no_table_bindings"
        ir.xbrl_calculations = {
            "enabled": True,
            "status": status,
            "bindings": [],
            "constraints": [],
            "declared_variables": [],
            "diagnostics": diagnostics,
            "source": "finqa_table",
            "grounding_status": "PARTIAL_TABLE_BINDINGS" if bindings else "NO_TABLE_BINDINGS",
            "binding_sources": [],
            "partial_bindings": sorted(bound_names),
        }
    return ir


def _bind_fact_table(fact: VerificationFact, table: list[list[Any]]) -> dict[str, Any] | None:
    if not table or len(table) < 2:
        return None
    header = [str(c or "") for c in table[0]]
    usable_cols = _usable_columns(table)
    col_years = _column_years(header)
    target_year = _fact_year(fact)

    col = _select_column(fact, header, usable_cols, col_years, target_year)
    if col is None:
        return None

    row_idx = _select_row(fact, table, col)
    if row_idx is None:
        return None

    value = _parse_number(str(table[row_idx][col]))
    if value is None:
        return None

    multiplier = _binding_multiplier(fact, header, col)
    var = re.sub(r"[^A-Za-z0-9_]", "_", f"tbl_{fact.name}_r{row_idx}c{col}")
    decimal_value = Decimal(str(value))
    return {
        "fact_name": fact.name,
        "xbrl_variable": var,
        "xbrl_value": float(value),
        "xbrl_smt_value": format(decimal_value.normalize(), "f"),
        "concept": f"table:row[{row_idx}]:{str(table[row_idx][0]).strip()[:40]}",
        "context_id": f"table_r{row_idx}_c{col}",
        "context_period": {
            "instant": "",
            "start_date": "",
            "end_date": col_years.get(col, "") or target_year or "",
            "dimensions": [],
        },
        "unit_ref": fact.unit or "",
        "unit_measures": [],
        "scale": float(multiplier),
        "binding_multiplier": float(multiplier),
        "binding_kind": "same_sign",
        "source": "finqa_table",
    }


def _usable_columns(table: list[list[Any]]) -> list[int]:
    header = table[0] if table else []
    usable: list[int] = []
    for ci in range(1, len(header)):
        for row in table[1:]:
            if isinstance(row, list) and ci < len(row) and _parse_number(str(row[ci])) is not None:
                usable.append(ci)
                break
    return usable


def _column_years(header: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for ci, cell in enumerate(header):
        years = re.findall(r"(?:19|20)\d{2}", cell)
        if years:
            out[ci] = years[-1]
    return out


def _select_column(
    fact: VerificationFact,
    header: list[str],
    usable_cols: list[int],
    col_years: dict[int, str],
    target_year: str | None,
) -> int | None:
    if not usable_cols:
        return None

    candidates = list(usable_cols)
    if target_year:
        year_cols = [ci for ci in usable_cols if col_years.get(ci) == target_year]
        if year_cols:
            candidates = year_cols

    column_matches = _rank_header_matches(fact.column or "", header, candidates)
    if column_matches:
        best_score = column_matches[0][0]
        best = [ci for score, ci in column_matches if score == best_score]
        return best[0] if len(best) == 1 else None

    if target_year and len(candidates) == 1:
        return candidates[0]
    if not target_year and len(usable_cols) == 1:
        return usable_cols[0]
    if len(candidates) == 1 and not _tokens(fact.column or ""):
        return candidates[0]
    return None


def _rank_header_matches(column_text: str, header: list[str], candidates: list[int]) -> list[tuple[tuple[int, int], int]]:
    wanted = _tokens(_strip_year(column_text), keep_years=False)
    if not wanted:
        return []

    matches: list[tuple[tuple[int, int], int]] = []
    for ci in candidates:
        if ci >= len(header):
            continue
        have = _tokens(_strip_year(header[ci]), keep_years=False)
        if not have:
            continue
        if wanted <= have:
            score = (0, len(have - wanted))
        elif have <= wanted:
            score = (1, len(wanted - have))
        else:
            overlap = len(wanted & have)
            if not overlap or overlap / max(1, len(wanted)) < 0.67:
                continue
            score = (2, len(wanted | have) - overlap)
        matches.append((score, ci))
    return sorted(matches)


def _select_row(fact: VerificationFact, table: list[list[Any]], col: int) -> int | None:
    label = fact.row_label or _strip_year(fact.name)
    normalized_label = _normalized_label(label)
    if not normalized_label:
        return None

    exact: list[int] = []
    for ri in range(1, len(table)):
        row = table[ri]
        if isinstance(row, list) and col < len(row) and _normalized_label(str(row[0])) == normalized_label:
            exact.append(ri)
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None

    fact_tokens = _tokens(label, keep_years=True)
    if not fact_tokens:
        return None

    best: tuple[int, int] | None = None
    tie = False
    for ri in range(1, len(table)):
        row = table[ri]
        if not isinstance(row, list) or col >= len(row):
            continue
        row_tokens = _tokens(str(row[0]), keep_years=True)
        if not row_tokens or not (fact_tokens <= row_tokens):
            continue
        extra = len(row_tokens - fact_tokens)
        if best is None or extra < best[0]:
            best, tie = (extra, ri), False
        elif extra == best[0]:
            tie = True
    if best is None or tie:
        return None
    return best[1]


# ── helpers ──────────────────────────────────────────────────────────────────

def _ok_result(bindings: list[dict[str, Any]], diagnostics: list[str]) -> dict[str, Any]:
    return {
        "enabled": True,
        "status": "ok",
        "bindings": bindings,
        "constraints": [],
        "declared_variables": sorted({b["xbrl_variable"] for b in bindings}),
        "dropped_constraints": [],
        "instantiated_constraints": 0,
        "connected_constraints": 0,
        "evidence_fact_variables": [],
        "diagnostics": diagnostics,
        "source": "finqa_table",
        "grounding_status": "TABLE_OK",
        "binding_sources": ["finqa_table"],
    }


def _binding_multiplier(fact: VerificationFact, header: list[str], col: int) -> Decimal:
    source_parts = []
    if header:
        source_parts.append(header[0])
    if col < len(header):
        source_parts.append(header[col])
    if fact.source_scale:
        source_parts.append(fact.source_scale)
    if fact.source_scale_quote:
        source_parts.append(fact.source_scale_quote)

    source_scale = _scale_factor(" ".join(source_parts))
    if source_scale is None:
        return Decimal("1")
    target_scale = _scale_factor(fact.unit or "") or Decimal("1")
    return source_scale / target_scale


def _scale_factor(text: str) -> Decimal | None:
    normalized = (text or "").lower()
    for key in ("billions", "billion", "millions", "million", "thousands", "thousand"):
        if re.search(rf"\b{key}\b", normalized):
            return _SCALE_FACTORS[key]
    return None


def _tokens(text: str, keep_years: bool = False) -> frozenset[str]:
    # Split letters and digits into separate tokens so OCR-mashed labels like
    # "tier 1capital" tokenize the same as "tier 1 capital".
    raw = re.findall(r"[a-z]+|[0-9]+", (text or "").lower())
    tokens = []
    for token in raw:
        if token in _STOP:
            continue
        if not keep_years and re.fullmatch(r"(?:19|20)\d{2}", token):
            continue
        tokens.append(_normalize_token(token))
    return frozenset(tokens)


def _normalize_token(token: str) -> str:
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _normalized_label(text: str) -> str:
    return " ".join(sorted(_tokens(text, keep_years=True)))


def _strip_year(name: str) -> str:
    return re.sub(r"_?(?:fy)?(?:19|20)\d{2}", " ", name or "").replace("_", " ")


def _fact_year(fact: VerificationFact) -> str | None:
    for source in (fact.name, fact.period or "", fact.column or ""):
        years = re.findall(r"(?:19|20)\d{2}", source)
        if years:
            return years[-1]
    return None


def _parse_number(cell: str) -> float | None:
    s = cell.strip().replace("$", "").replace(",", "").replace("%", "").strip()
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip()
    if not s:
        return None
    try:
        value = float(Decimal(s))
    except (InvalidOperation, ValueError):
        return None
    return -value if negative else value


def _is_absence_zero_fact(fact: VerificationFact) -> bool:
    return (fact.fact_type or "").strip() == "absence_implies_zero" and abs(float(fact.value)) <= 1e-12
