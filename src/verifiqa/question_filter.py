from __future__ import annotations

from verifiqa.types import FinanceBenchExample


_QUALITATIVE_STARTS = (
    "among ",
    "does ",
    "do ",
    "did ",
    "has ",
    "have ",
    "is ",
    "are ",
    "was ",
    "were ",
    "which ",
    "why ",
)

_QUALITATIVE_PHRASES = (
    "advisory vote",
    "capital-intensive",
    "dragged down",
    "explain why",
    "healthy liquidity",
    "if the",
    "improved or declined",
    "improving",
    "key agenda",
    "liquidity profile",
    "nature & purpose",
    "nature of",
    "nature and purpose",
    "please state",
    "primarily due to",
    "purpose of",
    "reasonable",
    "reasonably",
    "state that",
    "shareholder proposal",
    "shareholder vote",
    "the largest",
    "the smallest",
    "highest",
    "lowest",
    "the most",
    "the least",
    "useful metric",
    "vote on",
    "voting results",
    "what was the outcome",
    "what were the outcome",
    "what drove",
    "what drives",
)

_NUMERIC_PHRASES = (
    "answer in",
    "amount",
    "average",
    "basis point",
    "bps",
    "calculate",
    "capital expenditure",
    "capex",
    "cash flow",
    "compute",
    "current ratio",
    "ebitda",
    "free cash flow",
    "gross margin",
    "how many",
    "how much",
    "margin",
    "net debt",
    "net income",
    "net ppne",
    "percent",
    "percentage",
    "portion",
    "proportion",
    "ppne",
    "quick ratio",
    "ratio",
    "return on",
    "roa",
    "round",
    "total assets",
    "usd billion",
    "usd million",
    "what is",
    "what was",
    "what were",
)


def is_numerical_example(example: FinanceBenchExample) -> bool:
    return is_numerical_question(example.question, gold_answer=example.answer)


def is_numerical_question(question: str, gold_answer: str = "") -> bool:
    """
    Return True only for direct numerical FinanceBench questions.

    This intentionally excludes qualitative prompts that mention financial metrics
    but ask for a business judgment, explanation, or category selection.
    """
    text = _normalize(question)
    if not text:
        return False
    if _is_qualitative(text):
        return False
    if any(phrase in text for phrase in _NUMERIC_PHRASES):
        return True
    return False


def _is_qualitative(text: str) -> bool:
    if text.startswith(_QUALITATIVE_STARTS):
        return True
    if " which " in f" {text} ":
        return True
    return any(phrase in text for phrase in _QUALITATIVE_PHRASES)


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("?", " ").split())
