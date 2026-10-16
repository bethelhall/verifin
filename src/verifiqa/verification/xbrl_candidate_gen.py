from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any


_STOP_WORDS = frozenset({"and", "or", "of", "the", "a", "an", "in", "to", "by", "for", "net"})

# Financial jargon → EDGAR label keyword expansions.
# Keys are matched as whole words (word-boundary) against the combined search text.
# Values are space-separated label words added to the search token set.
_ALIASES: dict[str, str] = {
    "capex": "payments acquire property plant equipment capital expenditures",
    "capital expenditure": "payments acquire property plant equipment",
    "capital expenditures": "payments acquire property plant equipment",
    "restructuring": "restructuring charges restructuring expenses restructuring related",
    "inventories": "inventory inventories merchandise finished goods",
    "inventory": "inventories inventory merchandise finished goods",
    "ppe": "property plant equipment fixed assets",
    "operating cash flow": "net cash provided used operating activities",
    "cash from operations": "net cash provided used operating activities",
    "cogs": "cost goods sold cost revenue cost sales",
    "cost of goods sold": "cost goods sold cost revenue",
    "sga": "selling general administrative expenses",
    "selling general administrative": "selling general administrative expenses",
    "depreciation amortization": "depreciation amortization impairment",
    "goodwill impairment": "goodwill impairment loss",
    "income before tax": "income loss before income tax",
    "pretax income": "income loss before income tax provision",
    "provision for income taxes": "income tax expense benefit provision",
    "income tax expense": "income tax expense benefit provision",
}


@dataclass
class CandidateHandle:
    candidate_id: str    # "namespace__concept__period_end__accn", opaque key for lookup
    concept: str         # qname, e.g. "us-gaap:NetIncomeLoss"
    label: str           # human-readable label from companyfacts
    description: str     # truncated description (max 150 chars)
    period_type: str     # "duration" | "instant"
    fiscal_year: int     # fy from the entry
    period_end: str      # "YYYY-MM-DD"
    period_start: str    # "YYYY-MM-DD" for duration, "" for instant
    unit_type: str       # EDGAR unit string, e.g. "USD", "pure", "shares"
    score: float = 0.0   # lexical relevance score, not shown to LLM


def expected_unit_types(fact_name: str, metric: str, row_label: str) -> frozenset[str]:
    """Infer plausible EDGAR unit type(s) for this individual fact variable."""
    fact_text = _normalize_text(f"{fact_name} {row_label}")
    metric_text = _normalize_text(metric)

    # Only use metric-level ratio words if the variable itself appears to be
    # the reported final metric, not a component like revenue or tax expense.
    same_as_metric = fact_text.strip() == metric_text.strip() or metric_text in fact_text

    # Use word-boundary matching to avoid "ratio" matching inside "operations",
    # "margin" inside "margins" (ok) but not "percentage" inside other words, etc.
    def _has_word(text: str, *words: str) -> bool:
        return any(bool(re.search(r"\b" + w + r"\b", text)) for w in words)

    if _has_word(fact_text, "per share", "eps", "earnings per share"):
        return frozenset({"USD/shares"})

    if _has_word(fact_text, "fair value", "intrinsic value", "compensation expense", "compensation cost"):
        return frozenset({"USD"})

    if _has_word(fact_text,
                 "shares outstanding", "share count", "weighted average shares",
                 "diluted shares", "basic shares", "common shares", "stock options",
                 "restricted stock units", "restricted share awards", "nonvested common stock"):
        return frozenset({"shares"})

    if _has_word(fact_text, "rate", "ratio", "margin", "percentage", "percent", "yield"):
        return frozenset({"pure"})

    if same_as_metric and _has_word(metric_text, "rate", "ratio", "margin", "percentage", "percent", "yield"):
        return frozenset({"pure"})

    return frozenset({"USD"})


def generate_candidates(
    *,
    fact_name: str,
    source_quote: str,
    row_label: str,
    target_years: set[str],
    companyfacts: dict[str, Any],
    metric: str = "",
    seed_concept_locals: set[str] | None = None,
    top_k: int = 15,
) -> tuple[list[CandidateHandle], dict[str, dict[str, Any]], dict[str, Any]]:
    """Generate ranked candidate handles for one IR fact.

    Returns (handles, value_map, diagnostics).
    value_map maps candidate_id → raw companyfacts entry (enriched with _* fields)
    for value retrieval after LLM selection. Handles are sorted by descending relevance
    score; values are NOT included.
    """
    search_toks = _search_tokens(source_quote, row_label, fact_name)
    allowed_units = expected_unit_types(fact_name, metric, row_label)
    seed_locals = {local for local in (seed_concept_locals or set()) if local}

    diag: dict[str, Any] = {
        "concepts_scanned": 0,
        "dropped_no_label": 0,
        "dropped_no_lexical_overlap": 0,
        "dropped_unit": 0,
        "dropped_form": 0,
        "dropped_fp": 0,
        "dropped_fy": 0,
        "dropped_period_shape": 0,
        "dropped_comparative": 0,
        "allowed_units": sorted(allowed_units),
        "seed_concepts": sorted(seed_locals),
        "kept_seed": 0,
        "kept": 0,
    }

    # (score, duration_days, namespace, concept_name, enriched_entry)
    scored: list[tuple[float, int, str, str, dict[str, Any]]] = []

    for namespace, concepts in (companyfacts.get("facts") or {}).items():
        for concept_name, concept_data in concepts.items():
            diag["concepts_scanned"] += 1
            is_seed = concept_name in seed_locals
            label = (concept_data.get("label") or "").strip()
            if not label and not is_seed:
                diag["dropped_no_label"] += 1
                continue
            if not label:
                label = _split_camel(concept_name)

            description = (concept_data.get("description") or "").strip()
            # Expanded text used only for overlap detection; scoring uses label only
            # so that long descriptions don't dilute the Jaccard score.
            expanded_toks = _label_tokens(
                f"{label} {description[:200]} {_split_camel(concept_name)}"
            )
            if not is_seed and not (search_toks & expanded_toks):
                diag["dropped_no_lexical_overlap"] += 1
                continue

            label_toks = _label_tokens(label)
            score = _jaccard(search_toks, label_toks)
            if is_seed:
                # Registry concepts are curated semantic matches. Keep them at
                # the top of the menu even if the filing label uses sparse or
                # unfamiliar wording.
                score += 10.0

            # Collect best entry per (period_end, accn) — latest filed wins within each key.
            best: dict[str, dict[str, Any]] = {}

            for unit_type, entries in (concept_data.get("units") or {}).items():
                if unit_type not in allowed_units:
                    diag["dropped_unit"] += 1
                    continue
                for entry in entries:
                    if entry.get("form", "").upper() not in {"10-K", "10-K/A"}:
                        diag["dropped_form"] += 1
                        continue

                    if str(entry.get("fp") or "").upper() != "FY":
                        diag["dropped_fp"] += 1
                        continue

                    fy = str(entry.get("fy") or "")
                    if target_years and fy not in target_years:
                        diag["dropped_fy"] += 1
                        continue

                    period_start = str(entry.get("start") or "")
                    period_end = str(entry.get("end") or "")
                    if not period_end:
                        continue

                    if period_start:
                        if not _is_full_year(period_start, period_end):
                            diag["dropped_period_shape"] += 1
                            continue
                        period_type = "duration"
                        # Exclude comparative periods: in a FY2019 10-K, entries for
                        # 2017 and 2018 are tagged fy=2019. Keep only facts whose
                        # period_end year matches the declared fiscal year.
                        if fy and period_end[:4] != fy:
                            diag["dropped_comparative"] += 1
                            continue
                    else:
                        period_type = "instant"

                    accn = str(entry.get("accn") or "")
                    # One slot per period_end; latest filed wins (handles 10-K/A amendments).
                    cid_key = period_end
                    existing = best.get(cid_key)
                    filed = str(entry.get("filed") or "")
                    if existing is None or filed >= str(existing.get("filed") or ""):
                        best[cid_key] = {
                            **entry,
                            "_namespace": namespace,
                            "_concept_name": concept_name,
                            "_label": label,
                            "_description": description,
                            "_period_type": period_type,
                            "_period_start": period_start,
                            "_unit_type": unit_type,
                            "_accn": accn,
                        }

            for _cid_key, enriched in best.items():
                p_end = str(enriched.get("end") or "")
                dur = _duration_days(enriched["_period_start"], p_end)
                scored.append((score, dur, namespace, concept_name, enriched))
                diag["kept"] += 1
                if concept_name in seed_locals:
                    diag["kept_seed"] += 1

    # Highest score first; longest duration as tiebreaker (prefer full-year over sub-annual).
    scored.sort(key=lambda x: (-x[0], -x[1]))

    handles: list[CandidateHandle] = []
    value_map: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    for score, _dur, namespace, concept_name, entry in scored:
        if score == 0.0:
            break  # sorted descending; all remaining are also 0, stop early
        period_end = str(entry.get("end") or "")
        accn = entry.get("_accn", "")
        cid = f"{namespace}__{concept_name}__{period_end}__{accn}"
        if cid in seen:
            continue
        seen.add(cid)

        fy_raw = entry.get("fy")
        fy = int(fy_raw) if fy_raw else int(period_end[:4]) if period_end else 0
        desc = entry["_description"]
        handle = CandidateHandle(
            candidate_id=cid,
            concept=f"{namespace}:{concept_name}",
            label=entry["_label"],
            description=desc[:150] if desc else "",
            period_type=entry["_period_type"],
            fiscal_year=fy,
            period_end=period_end,
            period_start=entry["_period_start"],
            unit_type=entry["_unit_type"],
            score=score,
        )
        handles.append(handle)
        value_map[cid] = entry

        if len(handles) >= top_k:
            break

    return handles, value_map, diag


# ── helpers ─────────────────────────────────────────────────────────────────

def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _search_tokens(source_quote: str, row_label: str, fact_name: str) -> frozenset[str]:
    combined = f"{source_quote} {row_label} {fact_name}"
    key_text = combined.lower()

    alias_parts: list[str] = []
    for key, expansion in _ALIASES.items():
        if re.search(r"\b" + re.escape(key) + r"\b", key_text):
            alias_parts.append(expansion)

    return _label_tokens(f"{combined} {' '.join(alias_parts)}")


def _label_tokens(value: str) -> frozenset[str]:
    tokens = re.findall(r"[a-z]+", (value or "").lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS and len(t) > 1)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _split_camel(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", value)


def _is_full_year(start: str, end: str) -> bool:
    days = _duration_days(start, end)
    return 350 <= days <= 380


def _duration_days(start: str, end: str) -> int:
    if not start or not end:
        return 0
    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
        return max(0, (e - s).days)
    except ValueError:
        return 0
