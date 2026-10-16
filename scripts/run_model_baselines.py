#!/usr/bin/env python3
"""Run one-word LLM judges on the selected models' complete saved answers.

Each invocation runs FinanceBench and XBRLFiling by default. No new candidate
answers are generated. Use --preflight to check inputs without model calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import submission_runs

JUDGE_PROTOCOL = "full_answer_one_word_v1"
MODELS = {
    "qwen2.5-7b": ("qwen2.5_7b", "vllm", "qwen2.5-7b-tatdqa:latest", 32),
    "qwen3-30b": ("qwen3_30b", "vllm", "qwen3-30b-tatdqa:latest", 32),
    "llama-3.1-8b": ("llama3.1_8b", "vllm", "llama3.1-8b-tatdqa:latest", 32),
    # The Responses budget includes internal reasoning; the visible verdict
    # must still be exactly one word. The effective budget is fingerprinted.
    "gpt-5.5": ("gpt5.5", "openai", "gpt-5.5", 8192),
}
DATASETS = {
    "financebench": ("data/numerical_questions.jsonl", "fb", 67),
    "xbrlfiling": ("data/multicompany_provable/data/provable_calcrequired.jsonl", "xbrl", 600),
}


def input_audit(root: Path, model: str, dataset: str, limit: int = 0) -> dict:
    from scripts import baselines

    relative, _, expected = DATASETS[dataset]
    data = root / relative
    answers = root / submission_runs.RUNS[dataset][model] / "results.jsonl"
    examples = baselines.load_financebench_jsonl(data)
    fixed = baselines.load_answers(answers)
    ids = [example.financebench_id for example in examples]
    if len(ids) != expected or len(set(ids)) != expected or set(fixed) != set(ids):
        raise ValueError(f"{dataset}/{model}: expected {expected} matching unique questions")
    for example in examples:
        if str(fixed[example.financebench_id].get("question") or "").strip() != example.question.strip():
            raise ValueError(f"Candidate question mismatch: {example.financebench_id}")
    selected = examples[:limit] if limit else examples
    missing_text = sum(not fixed[ex.financebench_id]["answer"].strip() for ex in selected)
    missing_evidence = sum(not baselines._evidence_text(ex)[0] for ex in selected)
    eligible = sum(bool(fixed[ex.financebench_id]["answer"].strip())
                   and bool(baselines._evidence_text(ex)[0]) for ex in selected)
    quantities = Counter(fixed[ex.financebench_id]["quantity"]["status"] for ex in selected)
    return {"dataset": dataset, "candidate_model": model, "selected": len(selected),
            "judge_eligible": eligible, "missing_candidate_text": missing_text,
            "missing_evidence": missing_evidence,
            "numeric_quantity_unresolved": quantities["unresolved"],
            "numeric_parser_blocks_judge": False,
            "answers_from": str(answers.relative_to(root)), "judge_protocol": JUDGE_PROTOCOL}


def run_arguments(root: Path, args: argparse.Namespace, dataset: str) -> list[str]:
    suffix, mode, judge, budget = MODELS[args.model]
    relative, prefix, _ = DATASETS[dataset]
    output_root = args.out_root if args.out_root.is_absolute() else root / args.out_root
    output = output_root / suffix / f"{prefix}_judge"
    command = [sys.executable, "-u", str(root / "scripts/baselines.py"),
               "--data", str(root / relative),
               "--answers-from", str(root / submission_runs.RUNS[dataset][args.model] / "results.jsonl"),
               "--baseline", "judge", "--judge-protocol", JUDGE_PROTOCOL,
               "--out", str(output), "--llm-mode", mode, "--model", judge,
               "--max-tokens", str(args.max_tokens or budget), "--resume", "--fail-on-error", "--verbose"]
    if mode == "vllm":
        command.extend(["--vllm-url", args.vllm_url])
    else:
        command.extend(["--openai-reasoning-effort", "low"])
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    return command


def check_saved_errors(path: Path) -> None:
    if not path.exists():
        return
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    failed = [row for row in records if row.get("failure_reason") == "evaluation_error"]
    if failed:
        raise ValueError(f"{len(failed)} saved provider errors in {path}. Resolve provider access, "
                         "then use a new --out-root; resume does not silently retry error rows.")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True, choices=tuple(MODELS),
                     help="Saved answer-generator pool and matching LLM judge")
    cli.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    cli.add_argument("--out-root", type=Path, default=Path("results/new_submissions/baselines"))
    cli.add_argument("--preflight", action="store_true", help="Validate input pools; no model calls or writes")
    cli.add_argument("--vllm-url", default="http://127.0.0.1:11434/v1")
    cli.add_argument("--max-tokens", type=int, default=None,
                     help="Override model budget; only one-word verdicts are accepted")
    cli.add_argument("--limit", type=int, default=0,
                     help="Smoke-test size per dataset; use a separate --out-root")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.limit < 0 or (args.max_tokens is not None and args.max_tokens < 1):
        raise ValueError("Require limit >= 0 and max-tokens >= 1")
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    # Validate every selected pool before starting any potentially long job.
    for dataset in datasets:
        print(json.dumps(input_audit(ROOT, args.model, dataset, args.limit), indent=2), flush=True)
    commands = [run_arguments(ROOT, args, dataset) for dataset in datasets]
    for command in commands:
        output = Path(command[command.index("--out") + 1])
        check_saved_errors(output / "results.jsonl")
        print("Command: " + shlex.join(command), flush=True)
    if args.preflight:
        return 0
    if MODELS[args.model][1] == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise ValueError("Set OPENAI_API_KEY in the environment or root .env before running GPT-5.5.")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        output = Path(command[command.index("--out") + 1])
        check_saved_errors(output / "results.jsonl")
        print(f"Results: {output / 'results.jsonl'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from None
