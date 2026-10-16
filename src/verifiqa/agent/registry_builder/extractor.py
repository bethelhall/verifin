"""LLM-based policy extraction from source text."""
from __future__ import annotations

import json

from verifiqa.agent.utils import extract_json
from verifiqa.generation.llm_client import ChatMessage


_PROMPT = """\
Extract a formal metric policy from the financial definition text below.

Return JSON only — no markdown:
{{
  "metric": "<snake_case canonical metric name>",
  "metric_aliases": ["<alias 1>", "<alias 2>"],
  "formula_templates": ["<formula using snake_case variable names>"],
  "roles": {{
    "<var_name>": {{"description": "<what this variable is>"}},
    ...
  }},
  "claim_unit": "<percent|ratio|USD|count|other>",
  "source_quote": "<verbatim sentence(s) from the text that state the formula>",
  "formula_note": "<one sentence explaining what the formula computes>"
}}

Rules:
- formula_templates: use only arithmetic operators (+, -, *, /). Variable names must be snake_case.
- If the source gives multiple equivalent formulas (e.g. using average vs end-of-period), include both.
- roles: one entry per variable in the formula. Description is what that variable represents.
- source_quote must be copied verbatim from the source text below.
- If the text does not contain a clear arithmetic formula, return {{"found": false}}.

Metric name hint: {metric}
Source URL: {source_url}

Source text:
{source_text}
"""


def extract_policy(
    metric: str,
    source_text: str,
    source_url: str,
    source_name: str,
    llm_client,
) -> dict | None:
    """Extract a structured policy from source text. Returns None if no formula found."""
    prompt = _PROMPT.format(
        metric=metric,
        source_url=source_url,
        source_text=source_text,
    )
    raw = llm_client.chat(
        [ChatMessage(role="user", content=prompt)],
        temperature=0.0,
        stage="registry_extraction",
    ).strip()

    data = json.loads(extract_json(raw))
    if data.get("found") is False or "formula_templates" not in data:
        return None

    data["source_refs"] = [{"name": source_name, "url": source_url}]
    data["source_quote"] = data.get("source_quote", "")
    return data
