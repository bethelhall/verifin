"""Source-document grounding — the independent-input check for facts XBRL can't reach.

The XBRL/table binders give the audit engine an authoritative *structured* source for a
fact. Many facts have no such source: non-GAAP and narrative figures that live only as
prose in the filing (e.g. "we expect to incur costs of approximately $700 million"). For
those the only independent source is the **source document itself** — the retrieved
evidence text, which is ground-truth corpus text, not produced by the answer LLM.

This module is a *corroboration* tier, not a catch tier. It confirms that the figure the
LLM used actually appears in the cited real source, then binds the fact to that value so
the verdict carries an independent provenance signal (`document_grounded`) instead of
`llm_only` (checked only against the LLM's own numbers). It deliberately does **not** try
to hard-catch prose transcription errors: bare prose figures carry no reliable scale
context ("4,200" may mean 4,200 or 4,200 million), so binding a re-scaled number could
manufacture false violations. Hard catching stays with the structured sources (XBRL,
companyfacts, tables) where scale is unambiguous. Here the check is scale-invariant —
the figure's significant digits must occur in the source — which corroborates without
risking a false catch.

Bindings are emitted in the same shape as the XBRL path, so SMT augmentation, evidence
stripping, and the grounding tier are all reused unchanged.
"""
from __future__ import annotations

import re
from decimal import Context, Decimal, InvalidOperation
from typing import Any, Iterable

from verifiqa.types import VerificationFact, VerificationIR

# Round to this many significant digits before taking a signature, to erase binary
# float artifacts (1.4393719999999999 -> 1.439372) without touching real precision.
_SIGNIFICANT_PRECISION = Context(prec=12)


def attach_document_calculations(
    ir: VerificationIR, chunks: Iterable[Any]
) -> VerificationIR:
    """Corroborate every numeric IR fact against the retrieved source text.

    Mirrors ``attach_table_calculations``: requires *all* numeric facts to corroborate
    (full grounding) — a partial grounding would mix document-corroborated and
    LLM-asserted evidence, so it falls back to no bindings (llm_only) rather than ground
    a corrupt mixture.
    """
    source = _source_text(chunks)
    source_figures = _figure_digit_set(source)
    numeric_facts = {
        name for name, fact in ir.facts.items() if not _is_absence_zero_fact(fact)
    }
    bindings: list[dict[str, Any]] = []
    diagnostics: list[str] = []

    for name in sorted(numeric_facts):
        binding, diag = _ground_fact(ir.facts[name], source_figures)
        if diag:
            diagnostics.append(diag)
        if binding is not None:
            bindings.append(binding)

    bound_names = {b["fact_name"] for b in bindings}
    if bindings and bound_names == numeric_facts:
        ir.xbrl_calculations = _ok_result(bindings, diagnostics)
    else:
        status = "partial_document_bindings" if bindings else "no_document_bindings"
        ir.xbrl_calculations = {
            "enabled": True,
            "status": status,
            "grounding_status": "PARTIAL_DOCUMENT_BINDINGS" if bindings else "NO_DOCUMENT_BINDINGS",
            "source": "source_document",
            "bindings": [],
            "constraints": [],
            "declared_variables": [],
            "diagnostics": diagnostics,
            "binding_sources": [],
            "partial_bindings": sorted(bound_names),
        }
    return ir


def _ground_fact(
    fact: VerificationFact, source_figures: set[str]
) -> tuple[dict[str, Any] | None, str]:
    if not (fact.source_quote or "").strip():
        return None, f"no_source_quote:{fact.name}"

    digits = _significant_digits(fact.value)
    if digits is None:
        return None, f"no_value:{fact.name}"

    # Provenance, scale-invariant: a figure with the same significant digits as the
    # LLM's value must occur in the real source text. A fabricated or mis-transcribed
    # figure does not, and the fact is left llm_only rather than corroborated.
    if digits not in source_figures:
        return None, f"figure_not_in_source:{fact.name}"

    value = float(fact.value)
    decimal_value = Decimal(str(fact.value))
    var = re.sub(r"[^A-Za-z0-9_]", "_", f"doc_{fact.name}")
    return {
        "fact_name": fact.name,
        "xbrl_variable": var,
        "xbrl_value": value,
        "xbrl_smt_value": format(decimal_value.normalize(), "f"),
        "concept": f"document:{(fact.chunk_id or 'evidence')}",
        "context_id": f"doc_{re.sub(r'[^A-Za-z0-9_]', '_', fact.chunk_id or 'evidence')}",
        "context_period": {
            "instant": "",
            "start_date": "",
            "end_date": fact.period or "",
            "dimensions": [],
        },
        "unit_ref": fact.unit or "",
        "unit_measures": [],
        "scale": 1.0,
        "binding_multiplier": 1.0,
        "binding_kind": "same_sign",
        "source": "source_document",
    }, ""


# ── significant-digit matching ───────────────────────────────────────────────

def _significant_digits(value: Any) -> str | None:
    """Scale- and sign-invariant digit signature of a number.

    Drops the sign, decimal point, and leading/trailing zeros so that 0.594954,
    594,954, and 594954 all share the signature "594954" regardless of the unit scale
    each side uses (a value of 12,645 in a "$ millions" table corroborates a fact stored
    as 12645000000). Returns None for non-numeric or all-zero values.
    """
    try:
        dec = _SIGNIFICANT_PRECISION.create_decimal(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError):
        return None
    digits = re.sub(r"\D", "", format(abs(dec), "f"))
    digits = digits.strip("0")
    return digits or None


def _figure_digit_set(text: str) -> set[str]:
    """Significant-digit signatures of every numeric figure in the source text."""
    out: set[str] = set()
    for token in re.findall(r"\d[\d,]*\.?\d*", text or ""):
        if _is_year_token(token):
            continue
        sig = _significant_digits(token.replace(",", ""))
        if sig:
            out.add(sig)
    return out


# ── helpers ──────────────────────────────────────────────────────────────────

def _source_text(chunks: Iterable[Any]) -> str:
    return "\n".join(str(getattr(c, "text", "") or "") for c in (chunks or []))


def _ok_result(bindings: list[dict[str, Any]], diagnostics: list[str]) -> dict[str, Any]:
    return {
        "enabled": True,
        "status": "ok",
        "grounding_status": "DOCUMENT_OK",
        "source": "source_document",
        "bindings": bindings,
        "constraints": [],
        "declared_variables": sorted({b["xbrl_variable"] for b in bindings}),
        "dropped_constraints": [],
        "instantiated_constraints": 0,
        "connected_constraints": 0,
        "evidence_fact_variables": [],
        "binding_sources": ["source_document"],
        "diagnostics": diagnostics,
    }


def _is_year_token(token: str) -> bool:
    digits = token.strip().strip("()$").replace(",", "")
    return bool(re.fullmatch(r"(?:19|20)\d{2}", digits))


def _is_absence_zero_fact(fact: VerificationFact) -> bool:
    return (fact.fact_type or "").strip() == "absence_implies_zero" and abs(float(fact.value)) <= 1e-12
