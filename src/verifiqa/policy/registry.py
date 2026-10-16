from __future__ import annotations
import json
import re
from pathlib import Path

from .types import FormulaPolicy, SemanticConcept

class PolicyRegistry:
    def __init__(self, *, concepts: list[SemanticConcept], policies: list[FormulaPolicy]):
        self.concepts = {c.concept_id: c for c in concepts}
        self.policies = list(policies)
        self._alias_map: dict[str, str] = {}
        for concept in concepts:
            for alias in concept.aliases:
                self._alias_map[_norm(alias)] = concept.concept_id

    def operation_policies(self) -> list[FormulaPolicy]:
        return [p for p in self.policies if p.metadata.get("requires_question_binding")]

    def find_policy(self, metric: str) -> FormulaPolicy | None:
        metric_variants = _metric_variants(metric)
        for policy in self.policies:
            policy_names = {_norm(policy.metric)}
            policy_names.update(_norm(a) for a in policy.metric_aliases)
            if metric_variants & policy_names:
                return policy
        # Fallback: match on the normalized token *set* (order-independent). This
        # resolves reordered / decorated names ("3 year average net profit margin"
        # vs "net profit margin 3 year average") without fuzzy subset matching,
        # so it can never conflate a metric with a strict superset (e.g.
        # "net profit margin" must not match "net profit margin 3 year average").
        target = _metric_token_set(metric)
        if target:
            for policy in self.policies:
                for name in (policy.metric, *policy.metric_aliases):
                    if _metric_token_set(name) == target:
                        return policy
        return None

    def nearest_policy_metric(self, metric: str) -> str | None:
        """Best-effort nearest registry metric name for an unresolved metric, for
        diagnostics ('no_policy_for_metric ... did you mean X')."""
        target = _metric_token_set(metric)
        if not target:
            return None
        best_name, best_score = None, 0.0
        for policy in self.policies:
            for name in (policy.metric, *policy.metric_aliases):
                tokens = _metric_token_set(name)
                if not tokens:
                    continue
                score = len(target & tokens) / len(target | tokens)
                if score > best_score:
                    best_name, best_score = policy.metric, score
        return best_name if best_score >= 0.5 else None

    def infer_concepts(self, *variable_names: str) -> dict[str, str]:
        result = {}
        for var in variable_names:
            cid = self._match_variable(var)
            if cid:
                result[var] = cid
        return result

    def _match_variable(self, var: str) -> str | None:
        normalized = _norm(var)

        # 1. Direct match
        if normalized in self._alias_map:
            return self._alias_map[normalized]

        # 2. Strip trailing year token (e.g. "revenue_2019" → "revenue")
        stripped = re.sub(r'\s+(?:q[1-4](?:\s+\d{4}[a-z]?)?|\d{4}[a-z]?(?:\s+q[1-4])?)\s*$', '', normalized).strip()
        if stripped != normalized and stripped in self._alias_map:
            return self._alias_map[stripped]

        # 3. Token-subset: alias tokens ⊆ var tokens (without stop words / year tokens),
        #    alias must account for ≥50% of var tokens (prevents "cash" matching "cash flow ops")
        STOP = frozenset({"and", "of", "the", "to", "in", "from", "with", "a", "an"})
        var_tokens = {
            t for t in re.split(r'\s+', stripped)
            if t and t not in STOP and not re.fullmatch(r'\d{4}', t)
        }
        if not var_tokens:
            return None

        # Distinguishing modifiers change which line item is meant. If the variable
        # carries one the alias lacks, the alias is the wrong (usually broader) concept
        # — e.g. "other noncurrent assets" must not match the "noncurrent assets" total,
        # and "noncurrent liabilities" must not match the "liabilities" total.
        DISTINGUISHING = frozenset({
            "noncurrent", "current", "other", "accumulated", "gross", "accrued",
            "restricted", "unrestricted", "foreign", "domestic", "basic", "diluted",
            "unrecognized", "nonvested", "vested", "projected", "deferred",
            "subscription", "maintenance", "franchise", "proforma",
            "segment", "customer", "supplier", "canada", "taiwan", "singapore",
            "united", "states", "kingdom", "north", "south", "america", "africa",
            "asia", "pacific", "europe", "world", "pro", "forma",
            "municipal", "corporate", "payment", "payments",
        })

        best_cid: str | None = None
        best_score = 0.0
        for alias_norm, cid in self._alias_map.items():
            alias_tokens = {t for t in alias_norm.split() if t and t not in STOP}
            if not alias_tokens:
                continue
            if alias_tokens <= var_tokens:
                if (var_tokens - alias_tokens) & DISTINGUISHING:
                    continue
                score = len(alias_tokens) / len(var_tokens)
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_cid = cid

        return best_cid


# Domain-safe abbreviation expansions so metric/variable names resolve regardless
# of which spelling the LLM emits (e.g. "net_profit_margin_3yr_avg" vs the registry
# key "net_profit_margin_3yr_average"). Applied token-wise during normalization.
_ABBREV = {
    "avg": "average",
    "yr": "year",
    "yrs": "year",
    "pct": "percent",
    "chg": "change",
}


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    # Merge the "non current" negation into a single token so it no longer contains
    # "current" as a subset — otherwise "non current liabilities" wrongly matches the
    # "current liabilities" alias (LiabilitiesCurrent instead of LiabilitiesNoncurrent).
    s = re.sub(r'\bnon current\b', 'noncurrent', s)
    s = re.sub(r'\b(\d+)\s*yr\b', r'\1 year', s)  # "3yr" -> "3 year"
    tokens = [_ABBREV.get(token, token) for token in s.split()]
    return " ".join(tokens).strip()


_METRIC_STOP = frozenset({"and", "of", "the", "to", "in", "from", "with", "a", "an"})


def _metric_token_set(metric: str) -> frozenset[str]:
    """Normalized, order-independent token set for a metric name, dropping stop
    words and bare year/period tokens (e.g. 'fy2017', '2015')."""
    return frozenset(
        token
        for token in _norm(metric).split()
        if token
        and token not in _METRIC_STOP
        and not re.fullmatch(r'(?:fy|q[1-4])?(?:19|20)?\d{2,4}', token)
    )


def _metric_variants(metric: str) -> set[str]:
    """Normalize metric names that carry period/scenario decorations."""
    base = _norm(metric)
    variants = {base}
    queue = [base]
    seen = set(queue)
    patterns = (
        r"^fy\s*(?:19|20)\d{2}\s+",
        r"\s+fy\s*(?:19|20)\d{2}$",
        r"^q[1-4]\s*(?:19|20)?\d{2}\s+",
        r"\s+q[1-4]\s*(?:19|20)?\d{2}$",
        r"\s+(?:19|20)\d{2}$",
        r"\s+fy\s*(?:19|20)\d{2}\s+to\s+fy\s*(?:19|20)\d{2}$",
        r"\s+(?:19|20)\d{2}\s+to\s+(?:19|20)\d{2}$",
        r"\s+\d+\s*year$",
        r"\s+\d+\s*yr$",
    )
    while queue:
        current = queue.pop()
        for pattern in patterns:
            candidate = _norm(re.sub(pattern, " ", current).strip())
            if candidate and candidate not in seen:
                seen.add(candidate)
                variants.add(candidate)
                queue.append(candidate)
    return variants


def _compact(qname: str) -> str:
    return qname.split(':')[-1].lower()


def load_policy_registry(data_dir: Path | None = None) -> PolicyRegistry:
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"

    concepts_path = data_dir / "semantic_concepts.json"
    policies_paths = [data_dir / "metric_policies.json", data_dir / "operation_policies.json"]

    concepts: list[SemanticConcept] = []
    if concepts_path.exists():
        for entry in json.loads(concepts_path.read_text(encoding="utf-8")):
            concepts.append(SemanticConcept(
                concept_id=entry["concept_id"],
                aliases=entry.get("aliases", []),
                xbrl_concepts=entry.get("xbrl_concepts", []),
                unit_kind=entry.get("unit_kind", "money"),
                source_type=entry.get("source_type", ""),
                source_refs=entry.get("source_refs", []),
            ))

    policies: list[FormulaPolicy] = []
    for policies_path in policies_paths:
        if not policies_path.exists():
            continue
        for entry in json.loads(policies_path.read_text(encoding="utf-8")):
            policies.append(FormulaPolicy(
                policy_id=entry["policy_id"],
                metric=entry["metric"],
                formula_templates=entry.get("formula_templates", []),
                roles=entry.get("roles", {}),
                metric_aliases=entry.get("metric_aliases", []),
                source_type=entry.get("source_type", ""),
                source_refs=entry.get("source_refs", []),
                metadata=entry.get("metadata", {}),
            ))

    return PolicyRegistry(concepts=concepts, policies=policies)


DEFAULT_POLICY_REGISTRY = load_policy_registry()
