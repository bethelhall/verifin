from __future__ import annotations

import json

from .context import ReconcilerContext


_PROMPT = """\
You are the value-blind Reconciler for a financial QA verifier.

Your job is to emit a Verification Spec that contains only pointers and one scale.
You must not compute the answer. You must not decide pass/fail. You must not invent
formula text or use any source value. Source values and the claim value are masked.

Rules:
- Return only one JSON object.
- fact_bindings maps registry formula variables to source_id, concept, or fact_name
  strings that appear in source_facts_no_values.
- formula must be one policy_id from formula_registry, or "identity", or "none".
- claim_scale is a numeric multiplier applied to the masked claim value; use 1 when
  the claim is already in the same scale as the source facts.
- claim_unit is the unit after claim_scale is applied.
- If no registry formula fits the question, return formula "none".

Input:
{payload}
"""


def build_reconciler_prompt(context: ReconcilerContext) -> str:
    payload = json.dumps(context.prompt_payload, indent=2, sort_keys=True)
    return _PROMPT.format(payload=payload)
