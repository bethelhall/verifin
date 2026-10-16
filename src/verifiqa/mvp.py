from __future__ import annotations

from pathlib import Path
from typing import Optional

from verifiqa.dataset import load_financebench
from verifiqa.formalization.formalizer import Formalizer
from verifiqa.generation.answer_generator import AnswerGenerator
from verifiqa.generation.llm_client import LoggedLlmClient, make_llm_client
from verifiqa.pipeline import CounterexampleRavPipeline, PipelineConfig
from verifiqa.retrieval.evidence_retriever import EvidenceRetriever
from verifiqa.retrieval.planner import RetrievalPlanner
from verifiqa.verification.smt_generator import SmtGenerator


def run_mvp(
    fixture_dir: Path,
    llm_mode: str = "claude",
    model: str = "claude-sonnet-4-20250514",
    vllm_base_url: str = "http://localhost:8000/v1",
    anthropic_base_url: str = "https://api.anthropic.com/v1",
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY",
    max_tokens: int = 2048,
    llm_log_path: Optional[Path] = None,
    llm_terminal_log: bool = True,
):
    examples, chunks = load_financebench(fixture_dir)
    example = examples[0]
    llm = make_llm_client(
        mode=llm_mode,
        config={
            "base_url": vllm_base_url if llm_mode == "vllm" else anthropic_base_url,
            "model": model,
            "api_key_env": anthropic_api_key_env,
            "max_tokens": max_tokens,
        },
    )
    llm = LoggedLlmClient(llm, terminal=llm_terminal_log, jsonl_path=llm_log_path)
    pipeline = CounterexampleRavPipeline(
        evidence_retriever=EvidenceRetriever(chunks),
        answer_generator=AnswerGenerator(llm),
        formalizer=Formalizer(llm),
        retrieval_planner=RetrievalPlanner(llm),
        smt_generator=SmtGenerator(llm),
        config=PipelineConfig(evidence_top_k=3),
    )
    return pipeline.run_example(example)
