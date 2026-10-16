from __future__ import annotations

import json
import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from verifiqa.generation.llm_client import ChatMessage
from verifiqa.types import VerificationIR
from verifiqa.verification.xbrl_candidate_gen import CandidateHandle, generate_candidates
from verifiqa.verification.xbrl_linkbase import (
    EdgarCompanyFactsCache,
    _candidate_concept_locals,
    _format_decimal,
    _manifest_entry,
    _unit_scale,
    _xbrl_variable,
    _years_for_fact,
)


# Braces that are NOT format placeholders must be doubled.
_BINDER_PROMPT = """\
You are an XBRL binding specialist. For each financial fact variable in the IR, \
select the best-matching XBRL concept from the candidates provided.

Guidelines:
- Match source_quote and row_label to the candidate label and description.
- Prefer the us-gaap namespace unless the filing is IFRS.
- For income-statement or cash-flow facts, prefer duration candidates over instant ones.
- The candidates have already been filtered to the target fiscal year. \
Select based on semantic match, not period alignment.
- Do not use llm_extracted_value to pick a candidate; it may be wrong. \
Match by label/description only.
- If no candidate is a clear match, set selected_candidate_id to null.

Respond with a JSON object only — no explanation, no markdown.

Format:
{{
  "bindings": [
    {{
      "fact_name": "<variable name>",
      "selected_candidate_id": "<candidate_id string or null>",
      "confidence": <0.0-1.0>,
      "reason": "<one short sentence>"
    }}
  ]
}}

IR:
{ir_json}

Facts to bind:
{facts_json}
"""


class XbrlLlmBinder:
    """Bind IR facts to EDGAR companyfacts values using LLM-driven candidate selection."""

    def __init__(self, llm_client, edgar_dir: Path):
        self.llm_client = llm_client
        self.edgar_dir = edgar_dir

    def bind_facts(self, ir: VerificationIR, doc_name: str, artifacts_dir: Path) -> VerificationIR:
        """Replace ir.xbrl_calculations with LLM-selected EDGAR bindings.

        If only a subset of facts can be bound (partial binding), the bindings list
        is cleared and status is set to "partial_bindings" so Z3 does not operate
        on mixed LLM-extracted + XBRL-grounded evidence.
        """
        from .periods import fact_period_conflict, fact_period_key
        from .xbrl_scope import requested_scopes
        if any(fact_period_conflict(name, fact) for name, fact in ir.facts.items()):
            ir.xbrl_calculations = _empty_result(doc_name, "conflicting_period_metadata")
            return ir
        needs_local = any(requested_scopes(f) or
                          ((key := fact_period_key(name, f)) is not None and key[1] != 0)
                          for name, f in ir.facts.items())
        if needs_local:
            bound = {b.get("fact_name") for b in ir.xbrl_calculations.get("bindings", [])}
            if not set(ir.facts) <= bound:
                ir.xbrl_calculations = _empty_result(doc_name, "scope_or_subannual_period_requires_local_binding")
            return ir
        entry = _manifest_entry(doc_name, artifacts_dir)
        cik = ""
        if entry:
            cik = str(entry.get("cik") or (entry.get("xbrl") or {}).get("cik") or "").lstrip("0")

        if not cik:
            ir.xbrl_calculations = _empty_result(doc_name, "missing_cik")
            return ir

        companyfacts = EdgarCompanyFactsCache.get(
            cik,
            self.edgar_dir,
            allow_network=bool(os.environ.get("SEC_USER_AGENT", "").strip()),
        )
        if not companyfacts:
            ir.xbrl_calculations = _empty_result(doc_name, "missing_companyfacts")
            return ir

        # Generate candidates for every numeric IR fact. Absence-implies-zero
        # facts are evidence omissions, not numeric XBRL facts, so requiring an
        # XBRL binding for them creates false partial-bind failures.
        facts_payload: list[dict[str, Any]] = []
        all_value_maps: dict[str, dict[str, dict[str, Any]]] = {}

        cand_diagnostics: dict[str, dict] = {}
        absence_witnesses: list[dict[str, Any]] = []

        for fact_name, fact in ir.facts.items():
            if _is_absence_zero_fact(fact):
                absence_witnesses.append(_absence_witness_to_dict(fact))
                cand_diagnostics[fact_name] = {"skipped": "absence_implies_zero"}
                continue

            target_years = _years_for_fact(fact, ir)
            seed_concepts = _candidate_concept_locals(fact, None)
            candidates, value_map, cand_diag = generate_candidates(
                fact_name=fact_name,
                source_quote=fact.source_quote or "",
                row_label=fact.row_label or "",
                target_years=target_years,
                companyfacts=companyfacts,
                metric=ir.metric,
                seed_concept_locals=seed_concepts,
                top_k=15,
            )
            all_value_maps[fact_name] = value_map
            cand_diagnostics[fact_name] = cand_diag
            facts_payload.append({
                "fact_name": fact_name,
                "source_quote": (fact.source_quote or "")[:300],
                "row_label": fact.row_label or "",
                "llm_extracted_value": fact.value,
                "unit": fact.unit,
                "period": fact.period or "",
                "candidates": [_handle_to_dict(h) for h in candidates],
            })

        if not facts_payload:
            if absence_witnesses and len(absence_witnesses) == len(ir.facts):
                ir.xbrl_calculations = _ok_result(
                    doc_name=doc_name,
                    bindings=[],
                    declared=[],
                    diagnostics=[],
                    cand_diagnostics=cand_diagnostics,
                    absence_witnesses=absence_witnesses,
                )
                return ir
            ir.xbrl_calculations = _empty_result(doc_name, "no_facts")
            return ir

        ir_summary = {
            "metric": ir.metric,
            "formula": ir.formula,
            "period": ir.period or "",
        }
        prompt = _BINDER_PROMPT.format(
            ir_json=json.dumps(ir_summary, indent=2),
            facts_json=json.dumps(facts_payload, indent=2),
        )

        llm_response = ""
        try:
            llm_response = self.llm_client.chat(
                [ChatMessage(role="user", content=prompt)],
                temperature=0.0,
                stage="xbrl_llm_binding",
                max_tokens=1024,
            ).strip()
        except Exception as exc:
            ir.xbrl_calculations = _empty_result(doc_name, f"llm_error:{type(exc).__name__}")
            return ir

        selections = _parse_llm_response(llm_response)
        if selections is None:
            ir.xbrl_calculations = _empty_result(doc_name, "llm_parse_error", llm_response[:200])
            return ir

        bindings: list[dict[str, Any]] = []
        diagnostics: list[str] = []
        bound_facts: set[str] = set()

        for sel in selections:
            fact_name = sel.get("fact_name") or ""
            cand_id = sel.get("selected_candidate_id")
            if not fact_name or not cand_id:
                continue

            fact = ir.facts.get(fact_name)
            if fact is None:
                diagnostics.append(f"unknown_fact:{fact_name}")
                continue

            value_map = all_value_maps.get(fact_name, {})
            entry_data = value_map.get(cand_id)
            if entry_data is None:
                diagnostics.append(f"unknown_candidate:{fact_name}:{cand_id}")
                continue

            binding = _resolve_binding(fact_name, fact.unit, cand_id, entry_data)
            if binding is None:
                diagnostics.append(f"resolve_failed:{fact_name}:{cand_id}")
                continue

            binding["_llm_confidence"] = sel.get("confidence", 0.0)
            binding["_llm_reason"] = sel.get("reason", "")
            bindings.append(binding)
            bound_facts.add(fact_name)

        numeric_facts = {
            name
            for name, fact in ir.facts.items()
            if not _is_absence_zero_fact(fact)
        }
        unbound = sorted(numeric_facts - bound_facts)

        if unbound:
            # Partial binding: mixed evidence would corrupt the SMT query.
            for f in unbound:
                cd = cand_diagnostics.get(f, {})
                n_kept = cd.get("kept", -1)
                units = cd.get("allowed_units", [])
                diagnostics.append(
                    f"unbound_fact:{f}:candidates_kept={n_kept}:allowed_units={units}"
                )
            ir.xbrl_calculations = {
                "enabled": True,
                "status": "partial_bindings",
                "grounding_status": "NO_SAFE_BINDING",
                "doc_name": doc_name,
                "bindings": [],
                "constraints": [],
                "declared_variables": [],
                "dropped_constraints": [],
                "instantiated_constraints": 0,
                "connected_constraints": 0,
                "evidence_fact_variables": [],
                "diagnostics": diagnostics,
                "candidate_diagnostics": cand_diagnostics,
                "absence_witnesses": absence_witnesses,
                "source": "llm_binding",
                "partial_bindings": [b["fact_name"] for b in bindings],
            }
            return ir

        # Full binding: preserve the LLM/text-extracted IR value. The XBRL
        # value is attached as an independent witness and checked later by SMT.
        for binding in bindings:
            fact = ir.facts.get(binding["fact_name"])
            if fact is not None:
                binding["llm_original_value"] = fact.value

        declared = sorted({b["xbrl_variable"] for b in bindings})
        ir.xbrl_calculations = _ok_result(
            doc_name=doc_name,
            bindings=bindings,
            declared=declared,
            diagnostics=diagnostics,
            cand_diagnostics=cand_diagnostics,
            absence_witnesses=absence_witnesses,
        )
        return ir


# ── helpers ─────────────────────────────────────────────────────────────────

def _handle_to_dict(h: CandidateHandle) -> dict[str, Any]:
    return {
        "candidate_id": h.candidate_id,
        "concept": h.concept,
        "label": h.label,
        "description": h.description,
        "period_type": h.period_type,
        "fiscal_year": h.fiscal_year,
        "period_end": h.period_end,
        "period_start": h.period_start,
    }


def _parse_llm_response(text: str) -> list[dict[str, Any]] | None:
    stripped = text.strip()
    if "```" in stripped:
        parts = stripped.split("```")
        if len(parts) >= 3:
            stripped = parts[1].strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
    try:
        data = json.loads(stripped)
        return data.get("bindings") or []
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r'\{.*\}', stripped, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                return data.get("bindings") or []
            except (json.JSONDecodeError, AttributeError):
                pass
    return None


def _resolve_binding(
    fact_name: str,
    fact_unit: str,
    candidate_id: str,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    raw_val = entry.get("val")
    if raw_val is None:
        return None

    namespace = entry.get("_namespace", "")
    concept_name = entry.get("_concept_name", "")
    period_end = str(entry.get("end") or "")
    period_start = str(entry.get("_period_start") or "")
    period_type = entry.get("_period_type", "duration")
    accn = str(entry.get("accn") or "")

    qname = f"{namespace}:{concept_name}" if namespace else concept_name
    context_id = f"edgar_{period_end}_{accn}" if accn else f"edgar_{period_end}"

    scale = _unit_scale(("USD",), fact_unit)
    xbrl_decimal = Decimal(str(raw_val)) / Decimal(str(scale)) if scale else Decimal(str(raw_val))
    xbrl_value = float(xbrl_decimal)

    if period_type == "instant":
        context_period = {
            "instant": period_end,
            "start_date": "",
            "end_date": period_end,
            "dimensions": [],
        }
    else:
        context_period = {
            "instant": "",
            "start_date": period_start,
            "end_date": period_end,
            "dimensions": [],
        }

    return {
        "fact_name": fact_name,
        "xbrl_variable": _xbrl_variable(qname, context_id),
        "xbrl_value": xbrl_value,
        "xbrl_smt_value": _format_decimal(xbrl_decimal),
        "concept": qname,
        "context_id": context_id,
        "context_period": context_period,
        "unit_ref": "USD",
        "unit_measures": ["USD"],
        "scale": scale,
        "source": "llm_binding",
        "candidate_id": candidate_id,
    }


def _is_absence_zero_fact(fact) -> bool:
    return (fact.fact_type or "").strip() == "absence_implies_zero" and abs(float(fact.value)) <= 1e-12


def _absence_witness_to_dict(fact) -> dict[str, Any]:
    return {
        "fact_name": fact.name,
        "value": fact.value,
        "unit": fact.unit,
        "source_quote": fact.source_quote,
        "chunk_id": fact.chunk_id,
        "period": fact.period,
        "row_label": fact.row_label,
        "column": fact.column,
        "absence_scope": dict(fact.absence_scope or {}),
        "source": "evidence_omission",
    }


def _ok_result(
    *,
    doc_name: str,
    bindings: list[dict[str, Any]],
    declared: list[str],
    diagnostics: list[str],
    cand_diagnostics: dict[str, dict],
    absence_witnesses: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "enabled": True,
        "status": "ok",
        "grounding_status": "INSTANCE_ONLY" if bindings else "EVIDENCE_OMISSION",
        "doc_name": doc_name,
        "bindings": bindings,
        "constraints": [],
        "declared_variables": declared,
        "dropped_constraints": [],
        "instantiated_constraints": 0,
        "connected_constraints": 0,
        "evidence_fact_variables": [],
        "diagnostics": diagnostics,
        "candidate_diagnostics": cand_diagnostics,
        "absence_witnesses": absence_witnesses,
        "source": "llm_binding",
    }


def _empty_result(doc_name: str, status: str, detail: str = "") -> dict[str, Any]:
    return {
        "enabled": True,
        "status": status,
        "grounding_status": "NO_SAFE_BINDING",
        "doc_name": doc_name,
        "bindings": [],
        "constraints": [],
        "declared_variables": [],
        "dropped_constraints": [],
        "instantiated_constraints": 0,
        "connected_constraints": 0,
        "evidence_fact_variables": [],
        "diagnostics": [detail] if detail else [],
        "source": "llm_binding",
    }
