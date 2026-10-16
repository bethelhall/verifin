"""Shared utilities for the agent package."""
from __future__ import annotations


def extract_json(text: str) -> str:
    """Extract the outermost JSON object from an LLM response string."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object found: {text[:200]}")
    return text[start:end + 1]
