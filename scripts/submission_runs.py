"""Authoritative saved-run locations for the six-model submission table.

Scoring policies and source-review inputs retain their own provenance. These
paths select prediction logs only; an absent consolidated log is never replaced
with an older run automatically.
"""
from __future__ import annotations


SUBMISSION_ROOT = "results/new_submissions"
_MODEL_SUFFIXES = {
    "qwen2.5-7b": "qwen2.5_7b",
    "llama-3.1-8b": "llama3.1_8b",
    "qwen3-30b": "qwen3_30b",
    "fino1-identity-pending": "fin_o1",
    "claude-haiku-4.5": "haiku_4_5",
    "gpt-5.5": "gpt5.5",
}
RUNS = {
    dataset: {
        model: f"{SUBMISSION_ROOT}/{dataset}_{suffix}"
        for model, suffix in _MODEL_SUFFIXES.items()
    }
    for dataset in ("xbrlfiling", "financebench", "tatdqa")
}
RUNS["tatdqa"]["fino1-identity-pending"] = f"{SUBMISSION_ROOT}/tatdqa_fin_o1_8b"
