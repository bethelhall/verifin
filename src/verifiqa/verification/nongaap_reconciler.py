"""Document-sourced formula authority for company-specific non-GAAP metrics.

A non-GAAP metric (e.g. "comparable constant-currency growth", "adjusted EBITDA
margin") has no universal formula, so the policy registry can't authorize it and
the verifier abstains with ``no_policy_for_metric``. But SEC Regulation G requires
the company to publish a reconciliation of every non-GAAP measure in the filing —
which includes the measure's own published value.

This module authorizes the formalizer's bridge formula by an independent check
against that published figure: extract the metric's PUBLISHED value from the
filing (grounded), and authorize the formula iff ``formula(grounded facts)``
reproduces it. The LLM proposes the components and bridge; the verifier confirms
they reconcile to the company's own published total. The formula authority comes
from the filing (Reg G), not the LLM's free choice — preserving the trust boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from verifiqa.formulas.evaluator import FormulaError, evaluate_formula, formula_variables
from verifiqa.generation.llm_client import ChatMessage
from verifiqa.types import EvidenceChunk, VerificationIR
from verifiqa.units import numbers_in_text


_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "source_quote": {"type": "string"},
        "chunk_id": {"type": "string"},
    },
    "required": ["found", "value", "source_quote", "chunk_id"],
    "additionalProperties": False,
}
_OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": _SCHEMA}}

_PROMPT = """\
The quantity below is a company-specific non-GAAP measure. Under SEC Regulation G,
the filing publishes this measure's value in a reconciliation table.

Question: {question}
Metric: {metric}

From the retrieved evidence, find the measure's PUBLISHED value exactly as printed
in the filing's reconciliation (e.g. the "Comparable Constant Currency Growth %"
total row). Return:
- found: true only if the measure's own published value is present in the evidence
- value: that number exactly as printed (no rescaling)
- source_quote: the exact text containing it (row label + value)
- chunk_id: the retrieved chunk id it came from

If the measure's published value is not present, return found=false.
Return JSON only.

Retrieved evidence:
{evidence}
"""


@dataclass
class PublishedTotal:
    value: float
    source_quote: str
    chunk_id: str


class NonGaapReconciler:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def published_total(
        self, metric: str, question: str, chunks: list[EvidenceChunk]
    ) -> Optional[PublishedTotal]:
        """Extract the metric's published value from the filing, grounded to a chunk."""
        prompt = _PROMPT.format(
            question=question,
            metric=metric,
            evidence=_format_evidence(chunks),
        )
        try:
            content = self.llm_client.chat(
                [ChatMessage(role="user", content=prompt)],
                temperature=0.0,
                stage="nongaap_published_total",
                output_config=_OUTPUT_CONFIG,
            ).strip()
            data = _parse_json(content)
        except Exception:
            return None
        if not isinstance(data, dict) or not data.get("found") or data.get("value") is None:
            return None
        try:
            value = float(data["value"])
        except (TypeError, ValueError):
            return None
        chunk_id = str(data.get("chunk_id", "")).strip()
        chunk = next((c for c in chunks if c.chunk_id == chunk_id), None)
        if chunk is None:
            return None
        # Ground it: the value must actually appear in the cited chunk, and the
        # chunk must mention the measure (so we don't grab an unrelated number).
        if not _number_in_text(value, chunk.text):
            return None
        if not _mentions_metric(metric, chunk.text):
            return None
        return PublishedTotal(value=value, source_quote=str(data.get("source_quote", "")), chunk_id=chunk_id)

    def resolve(
        self, ir: VerificationIR, question: str, chunks: list[EvidenceChunk]
    ) -> Optional[dict]:
        """Authorize ir.formula iff it reconciles to the filing's published total.

        Returns a formula-authority payload (formula, computed_value, grounded
        published-total citation) or None to leave the metric abstaining.
        """
        try:
            values = {name: float(f.value) for name, f in ir.facts.items()}
            needed = formula_variables(ir.formula)
        except (FormulaError, SyntaxError, ValueError, TypeError):
            return None
        if not needed or not needed.issubset(values):
            return None
        # Reconciliation is only an INDEPENDENT check when the formula combines
        # >=2 distinct grounded facts into a total stated separately in the filing.
        # A single-fact pass-through trivially equals its own "published total"
        # (same number, same sentence) -> circular, rubber-stamps the LLM's value
        # (e.g. Upjohn $700M claim verified against the same "$700 million" quote
        # while gold was 77.78). Require a genuine multi-component bridge.
        if len(needed) < 2:
            return None
        try:
            computed = evaluate_formula(ir.formula, values)
        except (FormulaError, ZeroDivisionError, ValueError, TypeError):
            return None

        total = self.published_total(ir.metric, question, chunks)
        if total is None:
            return None
        tol = max(float(ir.tolerance or 0.0), 0.5)
        if abs(computed - total.value) > tol:
            return None  # formula does NOT reproduce the company's published total -> don't authorize
        return {
            "formula": ir.formula,
            "computed_value": computed,
            "published_total": total.value,
            "source_quote": total.source_quote,
            "chunk_id": total.chunk_id,
        }


def _format_evidence(chunks: list[EvidenceChunk]) -> str:
    return "\n\n".join(f"[chunk_id: {c.chunk_id}]\n{c.text}" for c in chunks)


def _parse_json(text: str):
    import json

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0)) if match else None


def _number_in_text(value: float, text: str) -> bool:
    for number in numbers_in_text(text):
        if abs(number - value) <= max(1e-6, abs(value) * 1e-6):
            return True
    return False


def _mentions_metric(metric: str, text: str) -> bool:
    stop = {"percent", "total", "fy", "the", "of", "and", "value", "amount", "growth"}
    tokens = [t for t in re.split(r"[^a-z0-9]+", metric.lower()) if len(t) > 2 and t not in stop]
    if not tokens:
        return True
    lowered = text.lower()
    hits = sum(1 for t in tokens if t in lowered)
    return hits >= max(1, len(tokens) // 2)
