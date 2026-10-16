from __future__ import annotations

import json
import re
from typing import Iterable

from verifiqa.generation.llm_client import ChatMessage
from verifiqa.types import RetrievalFact, RetrievalPlan


# Registry of derived/ratio metrics whose component facts are known ahead of
# time. Checked first in fallback_retrieval_plan so the deterministic path
# never misses a required fact for these formulas.
_METRIC_DEFINITIONS: dict[str, dict] = {
    "operating_cash_flow_ratio": {
        "keywords": ["operating cash flow ratio"],
        "facts": [
            {
                "name": "cash_from_operations",
                "aliases": [
                    "net cash provided by operating activities",
                    "cash flows from operating activities",
                    "cash from operating activities",
                    "operating cash flow",
                ],
                "statement": "cash flow",
            },
            {
                "name": "current_liabilities",
                "aliases": ["current liabilities", "total current liabilities"],
                "statement": "balance sheet",
            },
        ],
    },
    "free_cash_flow": {
        "keywords": ["free cash flow", " fcf "],
        "facts": [
            {
                "name": "cash_from_operations",
                "aliases": [
                    "net cash provided by operating activities",
                    "cash flows from operating activities",
                    "cash from operating activities",
                    "operating cash flow",
                ],
                "statement": "cash flow",
            },
            {
                "name": "capital_expenditures",
                "aliases": [
                    "capital expenditures",
                    "purchases of property, plant and equipment",
                    "purchases of PP&E",
                ],
                "statement": "cash flow",
            },
        ],
    },
    "current_ratio": {
        "keywords": ["current ratio"],
        "facts": [
            {
                "name": "current_assets",
                "aliases": ["current assets", "total current assets"],
                "statement": "balance sheet",
            },
            {
                "name": "current_liabilities",
                "aliases": ["current liabilities", "total current liabilities"],
                "statement": "balance sheet",
            },
        ],
    },
    "quick_ratio": {
        "keywords": ["quick ratio", "acid-test ratio", "acid test ratio"],
        "facts": [
            {
                "name": "current_assets",
                "aliases": ["current assets", "total current assets"],
                "statement": "balance sheet",
            },
            {
                "name": "inventory",
                "aliases": ["inventory", "inventories"],
                "statement": "balance sheet",
            },
            {
                "name": "current_liabilities",
                "aliases": ["current liabilities", "total current liabilities"],
                "statement": "balance sheet",
            },
        ],
    },
    "gross_profit_margin": {
        "keywords": ["gross profit margin", "gross margin"],
        "facts": [
            {
                "name": "revenue",
                "aliases": ["revenue", "revenues", "total revenues", "net sales"],
                "statement": "income statement",
            },
            {
                "name": "cogs",
                "aliases": ["cost of sales", "cost of goods sold", "cost of revenue"],
                "statement": "income statement",
            },
        ],
    },
    "net_profit_margin": {
        "keywords": ["net profit margin", "net margin"],
        "facts": [
            {
                "name": "net_income",
                "aliases": ["net income", "net earnings", "net income attributable"],
                "statement": "income statement",
            },
            {
                "name": "revenue",
                "aliases": ["revenue", "revenues", "total revenues", "net sales"],
                "statement": "income statement",
            },
        ],
    },
    "return_on_assets": {
        "keywords": ["return on assets", " roa "],
        "facts": [
            {
                "name": "net_income",
                "aliases": ["net income", "net earnings", "net income attributable"],
                "statement": "income statement",
            },
            {
                "name": "total_assets",
                "aliases": ["total assets"],
                "statement": "balance sheet",
            },
        ],
    },
    "return_on_equity": {
        "keywords": ["return on equity", " roe "],
        "facts": [
            {
                "name": "net_income",
                "aliases": ["net income", "net earnings", "net income attributable"],
                "statement": "income statement",
            },
            {
                "name": "shareholders_equity",
                "aliases": [
                    "total stockholders equity",
                    "total shareholders equity",
                    "total equity",
                ],
                "statement": "balance sheet",
            },
        ],
    },
}


_PROMPT = """\
Create a retrieval plan for a numerical financial question.

Return JSON only, no markdown:
{{
  "metric": "<snake_case_metric_or_empty>",
  "facts": [
    {{
      "name": "<snake_case_fact_name>",
      "aliases": ["<row labels or synonym phrases to search>"],
      "period": "<year/quarter/date needed, if any>",
      "statement": "<income statement|balance sheet|cash flow|notes|proxy|other|empty>"
    }}
  ],
  "reason": "<brief explanation>"
}}

Rules:
- Include only facts needed to answer or verify the numerical claim.
- If the question defines a formula, decompose it into the formula variables.
- Include exact row-label aliases likely to appear in filings.
- Include period/year hints separately from aliases.
- Do not invent evidence values.
- For a direct lookup question, return one fact.
- For "per share", "each shareholder", "liquidation value", or bankruptcy
  shareholder-recovery questions, first retrieve reported per-share equity
  metrics such as "tangible book value per share", "TBVPS", and "book value per
  share". Do not use gross total assets divided by shares unless the question
  explicitly defines that formula; common shareholders recover residual equity,
  not total assets before liabilities.
- If the question is not numerical or no facts can be inferred, use an empty facts list.

Question:
{question}
"""


class RetrievalPlanner:
    """Build per-fact retrieval queries before answer generation."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def plan(self, question: str) -> RetrievalPlan:
        if self.llm_client is None:
            return fallback_retrieval_plan(question)
        prompt = _PROMPT.format(question=question)
        try:
            content = self.llm_client.chat(
                [ChatMessage(role="user", content=prompt)],
                temperature=0.0,
                stage="retrieval_planning",
            ).strip()
            return normalize_plan_for_question(question, retrieval_plan_from_text(content))
        except Exception as exc:
            return _fallback_with_reason(question, f"llm_planning_failed:{type(exc).__name__}")


def retrieval_plan_from_text(text: str) -> RetrievalPlan:
    data = json.loads(_extract_json(text))
    return retrieval_plan_from_json(data)


def retrieval_plan_from_json(data: dict) -> RetrievalPlan:
    if not isinstance(data, dict):
        return RetrievalPlan()
    facts = []
    seen = set()
    for item in data.get("facts") or []:
        if not isinstance(item, dict):
            continue
        name = _snake(str(item.get("name", "")))
        if not name or name in seen:
            continue
        aliases = _clean_aliases(item.get("aliases") or [])
        period = str(item.get("period", "")).strip()
        statement = str(item.get("statement", "")).strip()
        facts.append(RetrievalFact(name=name, aliases=aliases, period=period, statement=statement))
        seen.add(name)
    return RetrievalPlan(
        metric=_snake(str(data.get("metric", ""))),
        facts=facts,
        reason=str(data.get("reason", "")).strip(),
    )


def fallback_retrieval_plan(question: str) -> RetrievalPlan:
    """Deterministic fallback used when planner LLM output is unavailable."""

    q = question.lower()
    periods = _periods(question)
    period_str = ", ".join(periods)

    # Check metric definitions registry first — these are derived/ratio metrics
    # whose required facts are fully known and must never be left incomplete.
    for metric_name, defn in _METRIC_DEFINITIONS.items():
        if any(kw in q for kw in defn["keywords"]):
            facts = [
                RetrievalFact(
                    name=f["name"],
                    aliases=_clean_aliases(f["aliases"]),
                    period=period_str,
                    statement=f["statement"],
                )
                for f in defn["facts"]
            ]
            return RetrievalPlan(
                metric=metric_name,
                facts=facts,
                reason="metric_definition_registry",
            )

    shareholder_recovery_question = _is_shareholder_recovery_question(question)
    facts: list[RetrievalFact] = []

    def add(name: str, aliases: Iterable[str], statement: str = "") -> None:
        if any(f.name == name for f in facts):
            return
        facts.append(
            RetrievalFact(
                name=name,
                aliases=_clean_aliases(aliases),
                period=period_str,
                statement=statement,
            )
        )

    if any(term in q for term in ("capex", "capital expenditure", "purchases of property", "purchases of ppe", "purchases of pp&e")):
        add("capital_expenditure", ["capital expenditures", "purchases of property, plant and equipment", "purchases of PP&E"], "cash flow")
    if any(term in q for term in ("revenue", "sales", "net sales", "total revenues")):
        add("revenue", ["revenue", "revenues", "total revenues", "net sales"], "income statement")
    if any(term in q for term in ("cogs", "cost of sales", "cost of goods")):
        add("cogs", ["cost of sales", "cost of goods sold", "cost of revenue"], "income statement")
    if any(term in q for term in ("gross profit", "gross margin")):
        add("gross_profit", ["gross profit", "gross margin"], "income statement")
    if any(term in q for term in ("operating income", "operating margin")):
        add("operating_income", ["operating income", "income from operations", "operating margin"], "income statement")
    if any(term in q for term in ("net income", "net earnings", "roa", "return on assets")):
        add("net_income", ["net income", "net earnings", "net income attributable"], "income statement")
    if any(term in q for term in ("total assets", "assets", "roa", "return on assets")) and not shareholder_recovery_question:
        add("total_assets", ["total assets", "average total assets"], "balance sheet")
    if any(term in q for term in ("ppe", "pp&e", "ppne", "property, plant", "fixed asset", "fixed assets", "plant and equipment")):
        add(
            "ppe",
            [
                "property, plant and equipment",
                "property, plant and equipment, net",
                "property and equipment, net",
                "net property and equipment",
                "PP&E",
                "PPNE",
                "fixed assets",
            ],
            "balance sheet",
        )
    if any(term in q for term in ("current assets", "working capital", "quick ratio", "liquidity")):
        add("current_assets", ["current assets", "total current assets"], "balance sheet")
    if any(term in q for term in ("current liabilities", "working capital", "quick ratio", "liquidity")):
        add("current_liabilities", ["current liabilities", "total current liabilities"], "balance sheet")
    if any(term in q for term in ("inventory", "inventories", "quick ratio", "cash conversion", "inventory turnover")):
        add("inventory", ["inventory", "inventories"], "balance sheet")
    if _mentions_cash_balance(q):
        add("cash", ["cash and cash equivalents", "cash equivalents", "cash"], "balance sheet")
    if any(term in q for term in ("receivable", "receivables", "quick ratio", "dso", "cash conversion")):
        add("receivables", ["accounts receivable", "receivables", "trade receivables"], "balance sheet")
    if any(term in q for term in ("accounts payable", "payables", "dpo", "cash conversion")):
        add("accounts_payable", ["accounts payable", "payables"], "balance sheet")
    if any(term in q for term in ("depreciation", "amortization", "ebitda")):
        add("depreciation_amortization", ["depreciation and amortization", "depreciation", "amortization"], "cash flow")
    if any(term in q for term in ("dividend", "dividends")):
        add("dividends", ["dividends", "cash dividends", "dividends paid"], "cash flow")
    if shareholder_recovery_question:
        add(
            "tangible_book_value_per_share",
            [
                "tangible book value per share",
                "TBVPS",
                "book value per share",
                "market and per common share data",
            ],
            "other",
        )

    metric = facts[0].name if len(facts) == 1 else ""
    return RetrievalPlan(metric=metric, facts=facts, reason="deterministic_keyword_fallback")


def _fallback_with_reason(question: str, reason: str) -> RetrievalPlan:
    plan = fallback_retrieval_plan(question)
    return RetrievalPlan(
        metric=plan.metric,
        facts=plan.facts,
        reason=_append_reason(plan.reason, reason),
    )


def normalize_plan_for_question(question: str, plan: RetrievalPlan) -> RetrievalPlan:
    """Apply deterministic semantic constraints that should not depend on LLM obedience."""

    if not _is_shareholder_recovery_question(question):
        return plan

    per_share_fact = RetrievalFact(
        name="tangible_book_value_per_share",
        aliases=_clean_aliases([
            "tangible book value per share",
            "TBVPS",
            "book value per share",
            "market and per common share data",
        ]),
        period=", ".join(_periods(question)),
        statement="other",
    )
    return RetrievalPlan(
        metric="tangible_book_value_per_share",
        facts=[per_share_fact],
        reason=_append_reason(
            plan.reason,
            "normalized_shareholder_recovery_to_reported_per_share_equity",
        ),
    )


def _is_shareholder_recovery_question(question: str) -> bool:
    q = question.lower()
    return (
        "per share" in q
        or "each shareholder" in q
        or "liquidat" in q
        or "bankrupt" in q
    )


def _append_reason(existing: str, reason: str) -> str:
    existing = (existing or "").strip()
    return f"{existing}; {reason}" if existing else reason


def _mentions_cash_balance(normalized_question: str) -> bool:
    if any(term in normalized_question for term in ("quick ratio", "liquidity")):
        return True
    balance_terms = (
        "cash and cash equivalents",
        "cash equivalents",
        "cash balance",
        "cash on hand",
        "ending cash",
        "year end cash",
    )
    if any(term in normalized_question for term in balance_terms):
        return True
    if "cash" not in normalized_question:
        return False
    return not any(term in normalized_question for term in ("cash flow", "cash flows", "cash-flow"))


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in retrieval plan")
    return text[start: end + 1]


def _clean_aliases(value) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    else:
        raw = [str(item) for item in value if str(item).strip()]
    aliases = []
    seen = set()
    for alias in raw:
        alias = re.sub(r"\s+", " ", alias).strip()
        if alias and alias.lower() not in seen:
            aliases.append(alias)
            seen.add(alias.lower())
    return aliases


def _periods(question: str) -> list[str]:
    periods = []
    for pattern in (r"\bFY\s*(20\d{2})\b", r"\b(20\d{2})\b", r"\bQ[1-4]\s*(?:FY)?\s*(20\d{2})\b"):
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            value = match.group(1)
            if value not in periods:
                periods.append(value)
    return periods


def _snake(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")
