"""Orchestrates fetch → extract → cross-check → output policy JSON."""
from __future__ import annotations

import json
import re
from pathlib import Path

from verifiqa.agent.registry_builder.extractor import extract_policy
from verifiqa.agent.registry_builder.fetcher import fetch_text
from verifiqa.agent.utils import extract_json
from verifiqa.policy import DEFAULT_POLICY_REGISTRY


_SOURCES_FILE = Path(__file__).parent / "metric_sources.json"


def build_registry(
    llm_client,
    out_path: Path,
    sources_file: Path = _SOURCES_FILE,
    skip_existing: bool = True,
    progress=print,
) -> list[dict]:
    """
    For each metric in sources_file:
      1. Fetch each source page
      2. LLM extracts a policy from the text
      3. Cross-check: if two sources agree on the formula → high_confidence
      4. Map roles to XBRL concept IDs using existing registry
      5. Write to out_path

    Returns the list of built policies.
    """
    sources = json.loads(sources_file.read_text())
    existing_metrics = {p.metric for p in DEFAULT_POLICY_REGISTRY.policies}

    policies = []
    for entry in sources:
        metric = entry["metric"]

        if skip_existing and metric in existing_metrics:
            progress(f"  skip {metric} (already in registry)")
            continue

        progress(f"  building {metric} ...")
        extractions = []

        for src in entry.get("sources", []):
            text = fetch_text(src["url"], local_file=src.get("local_file"))
            if not text:
                progress(f"    fetch failed: {src['url']}")
                continue

            try:
                policy = extract_policy(
                    metric=metric,
                    source_text=text,
                    source_url=src["url"],
                    source_name=src["name"],
                    llm_client=llm_client,
                )
            except Exception as exc:
                progress(f"    extract error ({src['name']}): {exc}")
                continue

            if policy:
                extractions.append(policy)
                progress(f"    {src['name']}: {policy.get('formula_templates', [])}")

        if not extractions:
            progress(f"    → no formula found, skipping")
            continue

        merged = _merge(metric, entry.get("aliases", []), extractions)
        merged["roles"] = _map_concept_ids(merged["roles"], llm_client=llm_client)
        policies.append(merged)
        progress(f"    → confidence={merged['confidence']} formula={merged['formula_templates']}")

    out_path.write_text(json.dumps(policies, indent=2))
    progress(f"\nWrote {len(policies)} policies to {out_path}")
    return policies


def _merge(metric: str, aliases: list[str], extractions: list[dict]) -> dict:
    """Merge extractions from multiple sources. Agreement → high_confidence."""
    formulas_per_source = [
        _normalize_formula(f)
        for e in extractions
        for f in (e.get("formula_templates") or [])
    ]

    # Deduplicate preserving order
    seen, unique_formulas = set(), []
    for f in formulas_per_source:
        if f not in seen:
            seen.add(f)
            unique_formulas.append(f)

    # Confidence: high if ≥2 sources produced at least one common formula
    source_formula_sets = [
        {_normalize_formula(f) for f in (e.get("formula_templates") or [])}
        for e in extractions
    ]
    cross_agreement = any(
        bool(s1 & s2)
        for i, s1 in enumerate(source_formula_sets)
        for s2 in source_formula_sets[i + 1:]
    )
    confidence = "high" if cross_agreement else "single_source"

    # Merge roles from all extractions
    roles: dict = {}
    for e in extractions:
        for name, info in (e.get("roles") or {}).items():
            if name not in roles:
                roles[name] = info

    # Collect all source refs and quotes
    source_refs = []
    source_quotes = []
    for e in extractions:
        source_refs.extend(e.get("source_refs") or [])
        q = e.get("source_quote", "")
        if q:
            source_quotes.append(q)

    return {
        "policy_id": f"auto_{metric}",
        "metric": metric,
        "metric_aliases": list({a for e in extractions for a in (e.get("metric_aliases") or [])} | set(aliases)),
        "formula_templates": unique_formulas,
        "roles": roles,
        "claim_unit": extractions[0].get("claim_unit", ""),
        "source_type": "auto_extracted",
        "confidence": confidence,
        "source_refs": source_refs,
        "source_quotes": source_quotes,
        "metadata": {
            "formula_note": extractions[0].get("formula_note", ""),
        },
    }


def _normalize_formula(formula: str) -> str:
    """Normalize formula for comparison: remove spaces, lowercase."""
    return re.sub(r"\s+", "", formula.lower())


def _map_concept_ids(roles: dict, llm_client=None) -> dict:
    """Map role names to XBRL concept IDs.

    Uses the existing registry first (exact/fuzzy match). Falls back to an LLM
    concept-picker when the registry has no match and an llm_client is provided.
    Roles with no mapping get concept_ids=[] — verification still works, XBRL
    binding is simply skipped for that role.
    """
    concept_index = [
        {"concept_id": c.concept_id, "aliases": c.aliases[:5]}
        for c in DEFAULT_POLICY_REGISTRY.concepts.values()
    ]
    mapped = {}
    for name, info in roles.items():
        concept_map = DEFAULT_POLICY_REGISTRY.infer_concepts(name)
        concept_ids = list(concept_map.values())

        if not concept_ids and llm_client is not None:
            concept_ids = _llm_map_concept(name, info.get("description", ""), concept_index, llm_client)

        mapped[name] = {**info, "concept_ids": concept_ids}
    return mapped


def _llm_map_concept(role_name: str, description: str, concept_index: list, llm_client) -> list[str]:
    """Ask the LLM to pick the closest concept ID from the index, or return []."""
    from verifiqa.generation.llm_client import ChatMessage
    prompt = (
        f"Pick the single best matching concept_id from the list below for this financial variable.\n"
        f"Variable: {role_name}\nDescription: {description}\n\n"
        f"Concept index (concept_id → aliases):\n{json.dumps(concept_index, indent=2)}\n\n"
        f'Return JSON only: {{"concept_id": "<id>"}} or {{"concept_id": null}} if no match.'
    )
    try:
        raw = llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            stage="concept_mapping",
        ).strip()
        data = json.loads(extract_json(raw))
        cid = data.get("concept_id")
        return [cid] if cid else []
    except Exception:
        return []
