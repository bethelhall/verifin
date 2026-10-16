"""CLI for the agentic verification pipeline.

Usage:
  python -m verifiqa.agent.cli run --data financebench.jsonl --out results/
  python -m verifiqa.agent.cli ask --question "..." --answer "..." --evidence "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Auto-load .env from project root
_env_file = Path(__file__).parent.parent.parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#"):
            continue
        if "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from verifiqa.agent.agent import AgentConfig, VerificationAgent, _maybe_authorize_xbrl_linkbase_claim
from verifiqa.agent.types import AgentResult, ClaimSpec

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_CODEX_MODEL = "gpt-6.1-sol"
from verifiqa.dataset import chunks_from_examples, load_financebench_jsonl
from verifiqa.dataset_finqa import load_finqa
from verifiqa.eval.candidate_scoring import candidate_matches_gold, claim_quantity
from verifiqa.formulas.evaluator import evaluate_formula
from verifiqa.generation.answer_generator import AnswerGenerator
from verifiqa.generation.llm_client import make_llm_client
from verifiqa.types import EvidenceChunk
from verifiqa.verification.smt_generator import SmtGenerator
from verifiqa.verification.z3_runner import Z3Runner


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-mode", default="claude", choices=["claude", "vllm", "codex", "openai"])
    parser.add_argument("--model", default=None, help="Model override (Codex/OpenAI default: gpt-6.1-sol)")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI executable")
    parser.add_argument("--codex-timeout", type=float, default=300.0,
                        help="Timeout in seconds for each Codex model call (default: 300)")
    parser.add_argument("--codex-reasoning-effort", default="low",
                        choices=["low", "medium", "high", "xhigh"],
                        help="Codex reasoning effort (default: low)")
    parser.add_argument("--openai-reasoning-effort", default="low",
                        choices=["low", "medium", "high", "xhigh"],
                        help="OpenAI Responses API reasoning effort (default: low)")
    parser.add_argument("--openai-max-output-tokens", type=_positive_int, default=8192,
                        help="OpenAI output budget, including reasoning tokens (default: 8192)")


def _llm_config(args) -> dict:
    default_model = _DEFAULT_CODEX_MODEL if args.llm_mode in {"codex", "openai"} else _DEFAULT_MODEL
    cfg: dict = {"model": args.model or default_model}
    if args.llm_mode == "vllm":
        cfg["base_url"] = args.vllm_url
    elif args.llm_mode == "codex":
        cfg.update({
            "executable": args.codex_bin,
            "timeout": args.codex_timeout,
            "reasoning_effort": args.codex_reasoning_effort,
        })
    elif args.llm_mode == "openai":
        if args.openai_max_output_tokens <= 0:
            raise ValueError("OpenAI output token limit must be positive")
        cfg.update({
            "reasoning_effort": args.openai_reasoning_effort,
            "max_output_tokens": args.openai_max_output_tokens,
        })
    return cfg


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="verifiqa-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Run agent on a FinanceBench or FinQA dataset")
    src = run_p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, default=None, help="FinanceBench JSONL with oracle evidence")
    src.add_argument("--finqa-dir", type=Path, default=None, help="FinQA dataset directory (contains dev.json etc.)")
    run_p.add_argument("--finqa-split", default="dev",
                       choices=["train", "dev", "test"], help="FinQA split (default: dev)")
    run_p.add_argument("--finqa-claim-source", default=None,
                       choices=["display_answer", "program_answer"],
                       help="Deprecated: use --claim-source finqa_display_answer or finqa_program_answer")
    run_p.add_argument("--claim-source", default=None,
                       choices=[
                           "generated_answer",
                           "example_answer",
                           "finqa_display_answer",
                           "finqa_program_answer",
                       ],
                       help="Answer to verify: generated LLM answer by default; dataset answers only for verifier-only audits")
    run_p.add_argument("--out", type=Path, required=True, help="Output directory")
    run_p.add_argument("--limit", type=int, default=None, help="Max examples to process")
    run_p.add_argument("--ids", nargs="+", default=None, help="Run only these specific example IDs")
    run_p.add_argument("--resume", action="store_true",
                       help="Skip examples already present in the output results.jsonl and append new ones")
    run_p.add_argument("--scoring-view", choices=["benchmark", "source_reviewed"], default=None,
                       help="TADQA/FinanceBench evaluation view (default: source_reviewed, matching the notebook); "
                            "benchmark preserves original reference targets")
    run_p.add_argument("--scoring-root", type=Path, default=None,
                       help="Repository containing the evaluation policy and reviewed targets")
    run_p.add_argument("--deterministic-smt", action="store_true",
                       help="Render the SMT deterministically from the IR instead of the LLM "
                            "autoformalizer. Always well-formed; removes solver_INVALID abstentions.")
    run_p.add_argument("--repair", action="store_true",
                       help="Retry-once repair (soundness preserved, re-verified each time): on VIOLATED, "
                            "re-ask the LLM with diagnostic feedback (no answer leaked) and adopt the new "
                            "answer only if it verifies; on a Z3-INVALID SMT, regenerate the SMT once with "
                            "the parse error fed back.")
    _add_llm_arguments(run_p)
    run_p.add_argument("--workers", type=int, default=4, help="Parallel fact extraction workers")
    run_p.add_argument("--xbrl-artifacts", type=Path, default=None,
                       help="Pre-cached XBRL artifacts directory (enables calculation-linkbase SMT constraints)")
    run_p.add_argument("--source-plan-cache", type=Path,
                       default=Path(__file__).resolve().parents[3] / "artifacts" / "source_plans",
                       help="Shared source-plan records, independently revalidated on every use")
    run_p.add_argument("--claimspec-consensus", action="store_true",
                       help="Run three question decompositions; abstain unless formula/roles agree semantically")
    run_p.add_argument("--smt-consensus", action="store_true",
                       help="Generate and run three SMT encodings; abstain unless all return the same SAT/UNSAT status")
    run_p.add_argument("--verbose", action="store_true")
    # Real retrieval (RAG). Without --corpus the agent uses oracle evidence (default, unchanged).
    run_p.add_argument("--corpus", type=Path, default=None,
                       help="Filing corpus directory (from build-corpus). Enables real retrieval "
                            "instead of oracle evidence; the verification path is otherwise identical.")
    run_p.add_argument("--top-k", "--evidence-top-k", dest="top_k", type=int, default=0,
                       help="Retrieved chunks per question (corpus mode; default 10)")
    run_p.add_argument("--first-stage-k", type=int, default=0,
                       help="First-stage candidate pool before rerank (corpus mode)")
    run_p.add_argument("--retriever-model", default="",
                       help="SentenceTransformer embedding model (corpus mode; default BAAI/bge-large-en-v1.5)")
    run_p.add_argument("--reranker-model", default="",
                       help="CrossEncoder reranker model (corpus mode)")
    run_p.add_argument("--no-rerank", action="store_true",
                       help="Disable cross-encoder reranking in corpus mode")
    run_p.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="dense",
                       help="First-stage retrieval scoring (corpus mode)")

    # --- setup ---
    setup_p = sub.add_parser("setup", help="Configure API key in .env file")
    setup_p.add_argument("--anthropic-key", metavar="KEY", default=None,
                         help="Anthropic API key (sk-ant-...); prompted if omitted")
    setup_p.add_argument("--vllm-url", metavar="URL", default=None,
                         help="vLLM base URL if using a local model")

    # --- build-registry ---
    reg_p = sub.add_parser("build-registry", help="Auto-build metric policy registry from web sources")
    reg_p.add_argument("--out", type=Path, required=True, help="Output JSON file for new policies")
    reg_p.add_argument("--sources", type=Path, default=None, help="Custom metric_sources.json (default: built-in)")
    reg_p.add_argument("--include-existing", action="store_true", help="Also rebuild metrics already in registry")
    _add_llm_arguments(reg_p)

    # --- ask ---
    ask_p = sub.add_parser("ask", help="Verify a single question/answer against inline evidence")
    ask_p.add_argument("--question", required=True)
    ask_p.add_argument("--answer", required=True)
    ask_p.add_argument("--evidence", required=True, help="Evidence text (or @path to read from file)")
    _add_llm_arguments(ask_p)
    ask_p.add_argument("--workers", type=int, default=4)
    ask_p.add_argument("--claimspec-consensus", action="store_true",
                       help="Run three question decompositions; abstain unless formula/roles agree semantically")
    ask_p.add_argument("--smt-consensus", action="store_true",
                       help="Generate and run three SMT encodings; abstain unless all return the same SAT/UNSAT status")

    args = parser.parse_args(argv)

    if args.command == "setup":
        _cmd_setup(args)
    elif args.command == "build-registry":
        _cmd_build_registry(args)
    elif args.command == "run":
        _cmd_run(args)
    elif args.command == "ask":
        _cmd_ask(args)


def _cmd_build_registry(args) -> None:
    from verifiqa.agent.registry_builder.builder import build_registry, _SOURCES_FILE
    cfg = _llm_config(args)
    llm = make_llm_client(mode=args.llm_mode, config=cfg)

    sources_file = args.sources or _SOURCES_FILE
    print(f"Sources: {sources_file}")
    print(f"Output:  {args.out}")
    print()
    policies = build_registry(
        llm_client=llm,
        out_path=args.out,
        sources_file=sources_file,
        skip_existing=not args.include_existing,
        progress=print,
    )
    print(f"\nDone. {len(policies)} policies written.")
    high = sum(1 for p in policies if p.get("confidence") == "high")
    print(f"  High confidence (2+ sources agree): {high}")
    print(f"  Single source:                      {len(policies) - high}")


def _cmd_setup(args) -> None:
    env_path = Path(__file__).parent.parent.parent.parent / ".env"

    # Read existing entries
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    anthropic_key = args.anthropic_key
    if not anthropic_key:
        current = existing.get("ANTHROPIC_API_KEY", "")
        hint = f" [current: {current[:12]}…]" if current else ""
        anthropic_key = input(f"Anthropic API key (sk-ant-...){hint}: ").strip() or current

    if args.vllm_url:
        existing["VLLM_BASE_URL"] = args.vllm_url

    if anthropic_key:
        existing["ANTHROPIC_API_KEY"] = anthropic_key

    if not existing:
        print("Nothing to save.")
        return

    lines = [f"{k}={v}" for k, v in existing.items()]
    env_path.write_text("\n".join(lines) + "\n")
    os.environ["ANTHROPIC_API_KEY"] = existing.get("ANTHROPIC_API_KEY", "")

    print(f"Saved to {env_path}")
    for k, v in existing.items():
        masked = v[:12] + "…" if len(v) > 12 else v
        print(f"  {k} = {masked}")

    # Smoke-test the key
    key = existing.get("ANTHROPIC_API_KEY", "")
    if key and key.startswith("sk-ant-"):
        print("\nVerifying key with Anthropic API...")
        try:
            from verifiqa.generation.llm_client import make_llm_client
            from verifiqa.generation.llm_client import ChatMessage
            llm = make_llm_client("claude", {"api_key": key, "max_tokens": "16"})
            llm.chat([ChatMessage(role="user", content="reply: ok")],
                     temperature=0.0, stage="setup_check")
            print("  Key valid.")
        except Exception as exc:
            print(f"  Warning: key check failed — {exc}")


def _make_agent(args) -> VerificationAgent:
    cfg = _llm_config(args)
    if args.llm_mode == "vllm":
        cfg["base_url"] = args.vllm_url
        # Local Ollama/vLLM need no key ("EMPTY"); hosted OpenAI-compatible endpoints
        # (e.g. the HuggingFace router) need one -- pick it up from the environment.
        cfg["api_key"] = os.environ.get("VLLM_API_KEY") or os.environ.get("HF_TOKEN") or "EMPTY"
    llm = make_llm_client(mode=args.llm_mode, config=cfg)
    return VerificationAgent(
        llm_client=llm,
        smt_generator=SmtGenerator(llm),
        z3_runner=Z3Runner(),
        config=AgentConfig(
            max_workers=args.workers,
            xbrl_artifacts_dir=getattr(args, "xbrl_artifacts", None),
            source_plan_cache_dir=getattr(args, "source_plan_cache", None),
            require_claimspec_consensus=getattr(args, "claimspec_consensus", False),
            require_smt_consensus=getattr(args, "smt_consensus", False),
            repair_invalid_smt=getattr(args, "repair", False),
            deterministic_smt=getattr(args, "deterministic_smt", False),
        ),
        progress_callback=_noop,
    )


def _cmd_run(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    summary_path = args.out / "summary.json"
    run_settings = None
    run_manifest = None
    manifest_path = args.out / "run_manifest.json"
    provider_label = {"codex": "Codex", "openai": "OpenAI"}.get(args.llm_mode)
    if provider_label:
        run_settings = {
            "mode": args.llm_mode, **_llm_config(args),
            "claim_source": _requested_claim_source(args),
            "deterministic_smt": args.deterministic_smt,
            "workers": args.workers,
        }
        if results_path.exists() and results_path.stat().st_size and not args.resume:
            raise ValueError(f"{provider_label} output already contains results; use --resume or a new output directory")
        if manifest_path.exists():
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if run_manifest.get("settings") != run_settings:
                raise ValueError(f"{provider_label} run settings differ from the saved run; use a new output directory")
    smt_dir = args.out / "smt"
    smt_dir.mkdir(exist_ok=True)

    if args.finqa_dir:
        examples = load_finqa(args.finqa_dir, split=args.finqa_split,
                              limit=args.limit if args.limit else 0, numeric_only=True)
    else:
        examples = load_financebench_jsonl(args.data)
        if args.limit:
            examples = examples[: args.limit]

    scorer, scoring_view = _run_scoring_context(args)
    counts: dict[str, int] = {}
    rows: list[dict] = []
    t0 = time.time()
    done_ids: set[str] = set()
    previous_by_id: dict[str, dict] = {}
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prev = json.loads(line)
            row_id = prev.get("id")
            if row_id in done_ids and scorer is not None:
                from verifiqa.eval.financebench_scoring import FinanceBenchScorer
                if isinstance(scorer, FinanceBenchScorer):
                    raise ValueError(f"Duplicate FinanceBench result: {row_id}")
                from verifiqa.eval.tatdqa_scoring import scoring_signature
                if scoring_signature(previous_by_id[row_id]) != scoring_signature(prev):
                    raise ValueError(f"Conflicting duplicate result: {row_id}")
            if row_id and row_id not in done_ids:
                done_ids.add(row_id)
                previous_by_id[row_id] = prev
                rows.append(prev)
                counts[prev.get("status", "")] = counts.get(prev.get("status", ""), 0) + 1
        if scorer is not None:
            scorer.score_rows(rows)  # Validate frozen inputs before any model work.
        _print(args, f"Resuming: {len(done_ids)} examples already complete, skipping them")

    if run_settings is not None and rows:
        saved_metadata = (run_manifest or {}).get("llm_run")
        if not saved_metadata or any(row.get("llm_run") != saved_metadata for row in rows):
            raise ValueError(f"Existing results lack matching {provider_label} run metadata; use a new output directory")

    pending = any(ex.financebench_id not in done_ids
                  and (not args.ids or ex.financebench_id in args.ids) for ex in examples)
    agent = _make_agent(args) if pending else None
    answer_generator = AnswerGenerator(agent.llm) if agent is not None else None
    run_metadata = None
    if run_settings is not None:
        if agent is not None:
            run_metadata = {"mode": args.llm_mode, **agent.llm.run_metadata}
            if run_manifest and run_manifest.get("llm_run") != run_metadata:
                raise ValueError(f"{provider_label} runtime differs from the saved run; use a new output directory")
        elif run_manifest:
            run_metadata = run_manifest["llm_run"]
        if run_metadata is not None and run_manifest is None:
            manifest_path.write_text(json.dumps({
                "settings": run_settings, "llm_run": run_metadata,
            }, indent=2) + "\n", encoding="utf-8")

    # Optional real retrieval. When --corpus is given, evidence comes from a retriever
    # over the filing corpus instead of the example's oracle chunks; the verification
    # path (claim, grounding, deterministic SMT, Z3) is untouched.
    retriever = retrieval_planner = None
    evidence_top_k = 5
    if pending and getattr(args, "corpus", None):
        from verifiqa.corpus_builder import load_corpus
        from verifiqa.retrieval.evidence_retriever import EvidenceRetriever
        from verifiqa.retrieval.planner import RetrievalPlanner
        corpus_chunks, corpus_stats = load_corpus(args.corpus)
        _print(args, f"Corpus: {corpus_stats['loaded_chunks']} chunks from {args.corpus}")
        evidence_top_k = args.top_k or 10
        first_stage_k = args.first_stage_k or max(50, evidence_top_k * 5)
        retriever_model = args.retriever_model or "BAAI/bge-large-en-v1.5"
        retriever_kwargs = {
            "model_name": retriever_model,
            "rerank": not args.no_rerank,
            "first_stage_k": first_stage_k,
            "embedding_cache_dir": args.corpus,
            "retrieval_mode": args.retrieval_mode,
        }
        if args.reranker_model:
            retriever_kwargs["reranker_model"] = args.reranker_model
        _print(args, f"Building retriever ({retriever_model}, rerank={not args.no_rerank}, "
                     f"top_k={evidence_top_k}, first_stage_k={first_stage_k})...")
        retriever = EvidenceRetriever(corpus_chunks, **retriever_kwargs)
        retrieval_planner = RetrievalPlanner(agent.llm)

    file_mode = "a" if args.resume else "w"
    with results_path.open(file_mode, encoding="utf-8") as fh:
        for i, ex in enumerate(examples):
            if args.ids and ex.financebench_id not in args.ids:
                continue
            if ex.financebench_id in done_ids:
                continue
            if retriever is not None:
                plan = retrieval_planner.plan(ex.question)
                evidence_chunks = retriever.retrieve_with_plan(
                    ex.question, plan, doc_name=ex.doc_name or "",
                    top_k=evidence_top_k, financebench_id="",
                )
                evidence_text = _oracle_text_from_chunks(evidence_chunks)
            else:
                evidence_chunks = _oracle_chunks(ex)
                evidence_text = _oracle_text_from_chunks(evidence_chunks)
            if not evidence_text and retriever is None:
                _print(args, f"[{i+1}/{len(examples)}] {ex.financebench_id} — SKIP (no oracle evidence)")
                continue

            generation_chunks, unit_context = _with_report_unit_context(
                evidence_chunks, evidence_text, ex.doc_name,
                getattr(args, "xbrl_artifacts", None),
            )
            generation_evidence = _oracle_text_from_chunks(generation_chunks)

            _print(args, f"[{i+1}/{len(examples)}] {ex.financebench_id} — {ex.question[:80]}")
            t_ex = time.time()
            claim_source = _requested_claim_source(args)
            try:
                claim_answer, claim_source = _claim_answer(
                    ex,
                    args,
                    answer_generator=answer_generator,
                    evidence_chunks=generation_chunks,
                    evidence_text=evidence_text,
                )
            except Exception as exc:
                result = AgentResult(
                    question=ex.question,
                    answer="",
                    status="ABSTAIN",
                    failure_reason=f"answer_generation_failed:{type(exc).__name__}:{exc}",
                )
            else:
                source_pdf = _source_pdf_for_example(ex, args.data) if retriever is None else None
                run_options = {"doc_name": ex.doc_name}
                if source_pdf is not None:
                    run_options["source_pdf_path"] = source_pdf
                result = agent.run(ex.question, claim_answer, evidence_text, **run_options)

            repair_info = None
            if getattr(args, "repair", False) and result.status == "VIOLATED":
                result, repair_info = _repair_once(
                    agent, answer_generator, ex, args,
                    evidence_chunks=generation_chunks, evidence_text=evidence_text,
                    prev_answer=claim_answer, prev_result=result,
                )
            elapsed = time.time() - t_ex

            counts[result.status] = counts.get(result.status, 0) + 1
            row = _result_to_dict(result, ex, generation_evidence, elapsed, claim_source)
            if unit_context is not None:
                # Baselines receive the same complete evidence as generation.
                # Preserve the original excerpt separately for PDF-bound replays.
                row["source_evidence_text"] = evidence_text
                row["source_report_unit_context"] = unit_context
            if run_metadata is not None:
                row["llm_run"] = run_metadata
            if repair_info:
                row["repair"] = repair_info
            _annotate_verified_correctness(row, scorer=scorer, scoring_view=scoring_view)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()

            # Persist the individual SMT next to the results file, one .smt2 per
            # question, so each verification can be inspected/re-run standalone.
            smt_text = result.smtlib or ""
            (smt_dir / f"{ex.financebench_id}_{result.status}.smt2").write_text(
                smt_text or f"; no SMT generated (status={result.status}"
                            f"{', ' + result.failure_reason if result.failure_reason else ''})\n",
                encoding="utf-8",
            )

            _print(args, f"  → {result.status}  metric={result.metric}  {elapsed:.1f}s"
                   + (f"  [{result.failure_reason}]" if result.failure_reason else ""))

    summary = _summarize_rows(rows, elapsed_s=round(time.time() - t0, 1),
                              scorer=scorer, scoring_view=scoring_view)
    if run_metadata is not None:
        summary["llm_run"] = run_metadata
    if scorer is not None:
        summary["scoring"]["result_sha256"] = hashlib.sha256(results_path.read_bytes()).hexdigest()
        from verifiqa.eval.financebench_scoring import FinanceBenchScorer
        if isinstance(scorer, FinanceBenchScorer):
            summary["authority_audit"] = scorer.authority_audit(results_path)
    summary_path.write_text(json.dumps(summary, indent=2))

    total = summary["total"]
    verified = summary["verified"]
    violated = summary["violated"]
    abstain = summary["abstain"]
    precision = summary["verification_precision"]

    print(f"\n{'─'*50}")
    print(f"  Total:      {total}")
    print(f"  VERIFIED:   {verified}")
    print(f"  VIOLATED:   {violated}")
    print(f"  ABSTAIN:    {abstain}")
    if scorer is not None:
        label = "source_reviewed (post hoc)" if scoring_view == "source_reviewed" else "benchmark"
        print(f"  Scoring:    {label}")
    if "scored_decisions" in summary:
        graded_verified = summary["verified_correct"] + summary["verified_incorrect"]
        print(f"  FA:         {summary['verified_incorrect']}  (confirmed among graded accepts)")
        print(f"  Unscored:   {summary['unscored_verified']} verified, "
              f"{summary['unscored_rejected']} violated")
        print(f"  Precision:  {_format_rate(precision)}  "
              f"({summary['verified_correct']}/{graded_verified} graded verified)")
        print(f"  Accuracy:   {_format_rate(summary['decision_accuracy'])}  "
              f"({summary['verified_correct'] + summary['violated_correct']}/"
              f"{summary['scored_decisions']} graded decisions)")
        print(f"  Bounds:     {_format_rate(summary['precision_lower_bound'])}–"
              f"{_format_rate(summary['precision_upper_bound'])} "
              "(all accepts; unresolved truth, not a confidence interval)")
        audit = summary.get("authority_audit", {})
        if audit.get("known_grounding_defects") is not None:
            print(f"  Grounding:  {audit['known_grounding_defects']} audited defect(s); "
                  "separate from answer correctness")
    else:
        print(f"  Precision:  {_format_rate(precision)}  (correct verified / verified)")
        print(f"  Accuracy:   {_format_rate(summary['decision_accuracy'])}  (correct SAT/UNSAT decisions)")
    print(f"  Results:    {results_path}")
    print(f"  Summary:    {summary_path}")


def _repair_core_diagnosis(result) -> tuple[list[str], str] | None:
    """Re-solve the VIOLATED result's stored SMT to get a minimized unsat core and
    the direction of the mismatch. Isolated: uses its own Z3Runner, never touches
    the agent's runner or config. Returns None (and callers fall back to the plain
    message) if anything about this doesn't work out -- diagnosis is a best-effort
    enhancement, not a requirement for --repair to function."""
    from verifiqa.verification.z3_runner import (
        Z3Runner, minimize_unsat_core, interpret_core_direction,
    )
    smtlib = getattr(result, "smtlib", "") or ""
    if not smtlib.strip():
        return None
    try:
        runner = Z3Runner(debug_unsat_cores=True)
        solved = runner.run(smtlib, label="repair_diagnosis")
        if solved.solver_status != "UNSAT" or not solved.unsat_core:
            return None
        minimized = minimize_unsat_core(runner, smtlib, solved.unsat_core)
        direction = interpret_core_direction(solved.unsat_core)
        return minimized, direction
    except Exception:
        return None


def _repair_feedback(prev_answer: str, result) -> str:
    """Diagnostic-only feedback for a rejected answer. Never reveals the correct value,
    so the LLM must genuinely re-derive rather than transcribe a dictated answer.

    Enhanced with a minimized unsat core and mismatch direction when available (see
    _repair_core_diagnosis); falls back to the plain fact-list message otherwise.
    This only runs inside --repair's own code path and never affects a run made
    without --repair.
    """
    metric = (getattr(result, "metric", "") or "the requested quantity").replace("_", " ")
    lines = [
        f"Your previous answer was: {prev_answer!r}.",
        f"It was checked against the filing's own reported figures for {metric} and found "
        "INCONSISTENT: it does not satisfy the filing's stated arithmetic.",
    ]

    facts_dict = getattr(result, "facts", None) or {}
    diagnosis = _repair_core_diagnosis(result)
    xbrl_bound = (getattr(result, "diagnostics", None) or {}).get(
        "xbrl_role_fact_binding", {}
    ).get("status") == "ok"

    if diagnosis is not None:
        minimized, direction = diagnosis
        if direction == "too_low":
            lines.append("Your claimed value is TOO LOW relative to what the filing's figures imply.")
        elif direction == "too_high":
            lines.append("Your claimed value is TOO HIGH relative to what the filing's figures imply.")
        # facts actually implicated by the (minimized) core, if any matched by name
        implicated = [
            name for name in facts_dict
            if f"evidence_{name}" in minimized
        ] or list(facts_dict)
    else:
        implicated = list(facts_dict)

    if xbrl_bound and facts_dict:
        # Operands are bound to the filing's own company facts, not freely extracted --
        # re-reading the evidence won't change them. State the authoritative values and
        # formula directly and ask for a recomputation, rather than asking to "re-check"
        # values that are already correct by construction.
        operand_lines = [f"  {name} = {fact.value}" for name, fact in facts_dict.items()]
        lines.append(
            "These operand values are bound to the filing's own reported figures "
            "(not extracted freely) and are not in question:\n" + "\n".join(operand_lines)
        )
        formula = getattr(result, "formula", "") or ""
        if formula:
            lines.append(f"Formula: {formula}")
        lines.append(
            "Recompute the requested value from the formula and these operand values, "
            "and state the result. Do not change the operand values."
        )
    elif implicated:
        lines.append(
            "One or more of the values used may be wrong (wrong line, wrong period, or a "
            "misread number). Re-examine these in particular: " + ", ".join(implicated) + "."
        )
        lines.append("Recompute the answer step by step from the evidence, double-checking every value "
                     "you read. Do not simply repeat the previous answer.")
    else:
        lines.append("Recompute the answer step by step from the evidence, double-checking every value "
                     "you read. Do not simply repeat the previous answer.")
    return "\n".join(lines)


def _repair_once(agent, answer_generator, example, args, *,
                 evidence_chunks, evidence_text, prev_answer, prev_result):
    """Re-ask the LLM once with diagnostic feedback, then re-verify. Adopt the new
    result only if it VERIFIES; otherwise keep the original VIOLATED. Acceptance still
    requires passing the verifier, so soundness (no false accepts) is preserved."""
    if answer_generator is None:
        return prev_result, {"attempted": False, "reason": "no_answer_generator"}
    feedback = _repair_feedback(prev_answer, prev_result)
    chunks = evidence_chunks if evidence_chunks is not None else _oracle_chunks(example)
    try:
        new_answer = answer_generator.generate(
            example.question, chunks,
            formula_context=_xbrl_answer_formula_context(example, args, evidence_text),
            feedback=feedback,
        )
    except Exception as exc:
        return prev_result, {"attempted": True, "regenerated": False,
                             "reason": f"regeneration_failed:{type(exc).__name__}"}
    source_pdf = (_source_pdf_for_example(example, getattr(args, "data", None))
                  if not getattr(args, "corpus", None) else None)
    run_options = {"doc_name": example.doc_name}
    if source_pdf is not None:
        run_options["source_pdf_path"] = source_pdf
    result2 = agent.run(example.question, new_answer, evidence_text, **run_options)
    info = {"attempted": True, "regenerated": True,
            "previous_answer": prev_answer, "repaired_answer": new_answer,
            "repair_status": result2.status, "repaired": result2.status == "VERIFIED"}
    return (result2 if result2.status == "VERIFIED" else prev_result), info


def _source_pdf_for_example(example, data_path: Path | None) -> Path | None:
    """Resolve an explicitly supplied local source path, without corpus search.

    Handoff packages keep the dataset one directory below their documents.
    Ordinary datasets without source-PDF metadata retain the text-only path.
    """
    raw = getattr(example, "raw", None) or {}
    relative = raw.get("source_pdf")
    if not relative or data_path is None:
        return None
    relative = Path(str(relative))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".pdf":
        return None
    parent = Path(data_path).resolve().parent
    matches = set()
    for root in (parent, parent.parent):
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            matches.add(candidate)
    return next(iter(matches)) if len(matches) == 1 else None


def _cmd_ask(args) -> None:
    evidence = args.evidence
    if evidence.startswith("@"):
        evidence = Path(evidence[1:]).read_text(encoding="utf-8")

    agent = _make_agent(args)

    print(f"Question: {args.question}")
    print(f"Answer:   {args.answer}")
    print(f"Evidence: {len(evidence)} chars")
    print()

    result = agent.run(args.question, args.answer, evidence)

    print(f"Status:         {result.status}")
    print(f"Metric:         {result.metric}")
    print(f"Formula:        {result.formula}")
    print(f"Formula source: {result.formula_source}")
    print(f"Claimed value:  {result.claimed_value}")
    if result.failure_reason:
        print(f"Failure:        {result.failure_reason}")
    if result.facts:
        print("\nExtracted facts:")
        for name, fact in result.facts.items():
            print(f"  {name} = {fact.value} {fact.unit}  [{fact.row_label}]  period={fact.period}")
    if result.smtlib:
        print(f"\nSMT ({len(result.smtlib)} chars, solver={result.solver_status})")


def _oracle_chunks(example) -> list[EvidenceChunk]:
    return chunks_from_examples([example])


def _oracle_text_from_chunks(chunks: list[EvidenceChunk]) -> str:
    return "\n\n".join(chunk.text for chunk in chunks if chunk.text)


def _with_report_unit_context(chunks, evidence_text, doc_name, artifacts_dir):
    """Supply the same source-backed display declaration for every candidate."""
    if not doc_name or artifacts_dir is None:
        return chunks, None
    from verifiqa.agent.source_report_units import recover_report_unit
    witness = recover_report_unit(evidence_text, doc_name, artifacts_dir)
    if witness is None:
        return chunks, None
    text = ("Filing unit declaration for the financial-statement notes:\n"
            + witness["quote"] + "\nSource: " + witness["source_url"])
    context = EvidenceChunk(
        chunk_id=doc_name + "__report_unit", doc_name=doc_name, page=None,
        text=text, source_type="filing_unit_declaration",
    )
    return [*chunks, context], witness


def _requested_claim_source(args) -> str:
    claim_source = getattr(args, "claim_source", None)
    if claim_source:
        return str(claim_source)
    legacy_finqa_source = getattr(args, "finqa_claim_source", None)
    if legacy_finqa_source == "display_answer":
        return "finqa_display_answer"
    if legacy_finqa_source == "program_answer":
        return "finqa_program_answer"
    return "generated_answer"


def _claim_answer(
    example,
    args,
    *,
    answer_generator: AnswerGenerator | None = None,
    evidence_chunks: list[EvidenceChunk] | None = None,
    evidence_text: str = "",
) -> tuple[str, str]:
    claim_source = _requested_claim_source(args)
    if claim_source == "generated_answer":
        if answer_generator is None:
            raise ValueError("missing_answer_generator")
        chunks = evidence_chunks if evidence_chunks is not None else _oracle_chunks(example)
        return answer_generator.generate(
            example.question,
            chunks,
            formula_context=_xbrl_answer_formula_context(example, args, evidence_text),
        ), "generated_answer"

    if claim_source == "example_answer":
        return str(getattr(example, "answer", "")), "example_answer"

    raw = getattr(example, "raw", {}) or {}
    qa = raw.get("qa") if isinstance(raw, dict) else None
    qa = qa if isinstance(qa, dict) else {}
    if claim_source == "finqa_display_answer":
        answer = qa.get("answer")
        if answer is not None:
            return str(answer), claim_source
    elif claim_source == "finqa_program_answer":
        answer = qa.get("exe_ans")
        if answer is not None:
            return str(answer), claim_source
    return str(getattr(example, "answer", "")), "example_answer"


def _xbrl_answer_formula_context(example, args, evidence_text: str) -> str:
    artifacts_dir = getattr(args, "xbrl_artifacts", None)
    doc_name = getattr(example, "doc_name", "") or ""
    if not artifacts_dir or not doc_name or "[redacted]" not in (evidence_text or ""):
        return ""
    seed = ClaimSpec(
        metric="",
        formula="",
        roles=[],
        claim_unit="USD millions",
        tolerance=1.0,
        formula_source="no_formula",
    )
    claim_spec, diagnostics = _maybe_authorize_xbrl_linkbase_claim(
        seed,
        getattr(example, "question", "") or "",
        evidence_text,
        doc_name,
        artifacts_dir,
    )
    if claim_spec.formula_source != "xbrl_linkbase":
        return ""

    lines = [
        "Use this deterministic XBRL calculation-linkbase formula.",
        "Use the first/leftmost numeric column in the evidence table as the current filing period.",
        f"Parent concept: {diagnostics.get('xbrl_claimspec_parent', claim_spec.metric)}",
        f"Formula: {claim_spec.metric} = {claim_spec.formula}",
        "Operands:",
    ]
    for role in claim_spec.roles:
        aliases = ", ".join(role.aliases[:4]) if role.aliases else role.name
        lines.append(f"- {role.name}: row label aliases [{aliases}]; period={role.period}")
    return "\n".join(lines)


def _result_to_dict(
    result: AgentResult,
    example,
    evidence_text: str,
    elapsed: float,
    claim_source: str = "example_answer",
) -> dict:
    raw = getattr(example, "raw", {}) or {}
    qa = raw.get("qa") if isinstance(raw, dict) else None
    qa = qa if isinstance(qa, dict) else {}
    raw_answer = qa.get("answer")
    if raw_answer is None and isinstance(raw, dict):
        raw_answer = raw.get("answer")
    if raw_answer is None:
        raw_answer = getattr(example, "answer", None)
    computed_value, computed_error = _computed_value(result)
    row = {
        "id": example.financebench_id,
        "question": result.question,
        "claim_source": claim_source,
        "claim_input_answer": result.answer,
        "answer": result.answer,
        "raw_answer": raw_answer,
        "gold_scale": (
            raw.get("scale", raw.get("gold_scale", qa.get("scale", "")))
            if isinstance(raw, dict) else ""
        ),
        "exe_ans": qa.get("exe_ans"),
        "program": qa.get("program"),
        "program_re": qa.get("program_re"),
        "status": result.status,
        "metric": result.metric,
        "formula": result.formula,
        "formula_source": result.formula_source,
        "claimed_value": result.claimed_value,
        "computed_value": computed_value,
        "computed_error": computed_error,
        "solver_status": result.solver_status,
        "failure_reason": result.failure_reason,
        "facts": {
            name: {
                "value": f.value,
                "unit": f.unit,
                "period": f.period,
                "row_label": f.row_label,
                "source_quote": f.source_quote,
                "fact_type": f.fact_type,
                "concept": f.concept,
                "source_value": f.source_value,
                "source_unit": f.source_unit,
                "source_section": f.source_section,
                "source_column": f.source_column,
                "source_table_id": f.source_table_id,
                "xbrl_context_id": f.xbrl_context_id,
                "source_ref": f.source_ref,
            }
            for name, f in result.facts.items()
        },
        "source_quotes": {
            name: f.source_quote for name, f in result.facts.items()
        },
        "smtlib": result.smtlib,
        "diagnostics": result.diagnostics or {},
        "evidence": getattr(example, "evidence", None),
        "evidence_text": evidence_text,
        "elapsed_s": round(elapsed, 2),
    }
    normalization = row["diagnostics"].get("claim_normalization")
    if normalization is not None:
        # The saved number is the prepared solver input, whose scale may differ
        # from the classifier's display unit (including an unresolved "other").
        # Prefer the runtime's explicit representation over inferring it from
        # operand units. Bind it to this exact number so stale retry metadata
        # cannot silently change the quantity evaluated by the scorer.
        if (not isinstance(normalization, dict)
                or normalization.get("claimed_value") != result.claimed_value
                or not isinstance(normalization.get("unit"), str)
                or not normalization["unit"].strip()):
            raise ValueError("Claim normalization metadata does not match the exported claim")
        row["claimed_value_unit"] = normalization["unit"]
    quantity = claim_quantity(row)
    row["claimed_value_unit"] = quantity["unit"] if quantity["valid"] else ""
    return row


def _computed_value(result: AgentResult) -> tuple[float | None, str]:
    if not result.formula or not result.facts:
        return None, ""
    try:
        values = {name: fact.value for name, fact in result.facts.items()}
        return float(evaluate_formula(result.formula, values)), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _run_scoring_context(args):
    """Resolve evaluator-only references without passing them to the agent."""
    from verifiqa.eval.tatdqa_scoring import DEFAULT_SCORING_VIEW, TatdqaScoringContext

    data_path = getattr(args, "data", None)
    selected = getattr(args, "scoring_view", None)
    if data_path is None:
        if selected == "source_reviewed":
            raise ValueError("source_reviewed scoring requires a supported evaluation dataset")
        return None, selected
    data_path = Path(data_path).resolve()
    repo_root = getattr(args, "scoring_root", None)
    if repo_root is None:
        candidates = (Path.cwd(), *data_path.parents, Path(__file__).resolve().parents[3])
        repo_root = next((path for path in candidates
                          if (path / "notebooks/tatdqa_scoring_adjudications.json").is_file()),
                         Path(__file__).resolve().parents[3])
    repo_root = Path(repo_root).resolve()
    financebench_review = repo_root / "data/financebench_scoring_review.json"
    if financebench_review.is_file():
        review = json.loads(financebench_review.read_text())
        canonical_path = (repo_root / review["dataset"]["path"]).resolve()
        if (data_path == canonical_path or
                hashlib.sha256(data_path.read_bytes()).hexdigest() == review["dataset"]["sha256"]):
            from verifiqa.eval.financebench_scoring import FinanceBenchScorer
            return FinanceBenchScorer.from_root(repo_root), selected or "source_reviewed"
    scorer = TatdqaScoringContext.from_dataset(data_path, repo_root)
    if scorer is None:
        with data_path.open() as handle:
            records = (json.loads(line) for line in handle if line.strip())
            is_tadqa = any(str(row.get("financebench_id") or row.get("id") or "").startswith("tatdqa_")
                           for row in records)
        if is_tadqa or selected == "source_reviewed":
            raise ValueError("Evaluation metadata is missing; use the canonical dataset and --scoring-root")
        return None, selected
    return scorer, selected or DEFAULT_SCORING_VIEW


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _annotate_verified_correctness(row: dict, *, scorer=None, scoring_view=None) -> None:
    if scorer is not None:
        from verifiqa.eval.financebench_scoring import FinanceBenchScorer
        if isinstance(scorer, FinanceBenchScorer):
            row.update(scorer.annotate_row(row, view=scoring_view or "source_reviewed"))
            return
    answer = str(row.get("answer") or "")
    gold = _gold_answer(row)
    row["gold_answer"] = gold
    if scorer is None:
        row["answer_matches_gold"] = _claim_matches_gold_answer(row, answer, gold)
    else:
        from verifiqa.eval.tatdqa_scoring import DEFAULT_SCORING_VIEW, SCORING_VIEWS
        view = scoring_view or DEFAULT_SCORING_VIEW
        if view not in SCORING_VIEWS:
            raise ValueError(f"Unknown scoring view: {view}")
        evaluated = scorer.score_row(row)
        row["answer_matches_gold"] = evaluated[f"{view}_candidate_correct"]
        row["evaluation"] = {
            "primary_view": view,
            "views": {name: {"candidate_correct": evaluated[f"{name}_candidate_correct"],
                             "outcome": evaluated[f"{name}_outcome"]} for name in SCORING_VIEWS},
            "gold_raw_answer": evaluated["gold_raw_answer"],
            "gold_scale": evaluated["gold_scale"],
            "normalization_reason": evaluated["normalization_reason"],
            "source_review_classification": evaluated["source_review_classification"],
        }

    actual_solver_status = _solver_status(row)
    row["actual_solver_status"] = actual_solver_status
    row["expected_solver_status"] = _expected_solver_status(row)
    row["verifier_decision"] = _solver_decision(actual_solver_status)
    if row["answer_matches_gold"] is None:
        reason = "no_gold_answer" if gold in (None, "") else "no_numeric_claim"
        row["solver_correct"] = None
        row["solver_correct_reason"] = reason
        row["decision_correct"] = None
        row["decision_correct_reason"] = reason
        row["verified_correct"] = None
        return

    if actual_solver_status not in {"SAT", "UNSAT"}:
        row["solver_correct"] = None
        row["solver_correct_reason"] = "no_solver_verdict"
        row["decision_correct"] = None
        row["decision_correct_reason"] = "no_solver_verdict"
        row["verified_correct"] = None
        return

    correct = actual_solver_status == row["expected_solver_status"]
    row["solver_correct"] = correct
    row["solver_correct_reason"] = (
        "solver_status_matches_gold_expectation"
        if correct
        else "solver_status_differs_from_gold_expectation"
    )
    row["decision_correct"] = correct
    row["decision_correct_reason"] = row["solver_correct_reason"]
    row["verified_correct"] = correct if actual_solver_status == "SAT" else None


def _gold_answer(row: dict):
    for field in ("gold_answer", "raw_answer", "reference_answer", "gold"):
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def _claim_matches_gold_answer(row: dict, answer: str, gold) -> bool | None:
    return candidate_matches_gold(
        row,
        gold,
        gold_scale=row.get("gold_scale") or "",
        reviewed_conversion=row.get("gold_unit_conversion"),
    )


def _expected_solver_status(row: dict) -> str | None:
    if row.get("answer_matches_gold") is True:
        return "SAT"
    if row.get("answer_matches_gold") is False:
        return "UNSAT"
    return None
def _save_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

def _solver_status(row: dict) -> str:
    row_status = str(row.get("status") or "").upper()
    if row_status in {"ABSTAIN", "UNVERIFIED_FORMULA"}:
        # SAT/UNSAT of an unauthorized formula is a diagnostic, not a verdict.
        return "ABSTAIN"
    status = str(row.get("solver_status") or "").upper()
    if status in {"SAT", "UNSAT"}:
        return status
    if row_status == "VERIFIED":
        return "SAT"
    if row_status == "VIOLATED":
        return "UNSAT"
    return status or row_status or "UNKNOWN"


def _solver_decision(status: str) -> str:
    if status == "SAT":
        return "accept"
    if status == "UNSAT":
        return "reject"
    return "abstain"


def _summarize_rows(rows: list[dict], elapsed_s: float, *, scorer=None, scoring_view=None) -> dict:
    total = len(rows)
    counts: dict[str, int] = {}
    abstain_reasons: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "")
        if status == "UNVERIFIED_FORMULA":
            status = "ABSTAIN"
            category = "unverified_formula"
        else:
            category = (row.get("diagnostics") or {}).get("abstention_category")
        counts[status] = counts.get(status, 0) + 1
        if status == "ABSTAIN":
            reason = category or str(row.get("failure_reason") or "unspecified").split(":", 1)[0]
            abstain_reasons[reason] = abstain_reasons.get(reason, 0) + 1

    if scorer is not None:
        from verifiqa.eval.financebench_scoring import FinanceBenchScorer
        if isinstance(scorer, FinanceBenchScorer):
            view = scoring_view or "source_reviewed"
            summary = scorer.summarize_rows(rows, elapsed_s=elapsed_s, view=view)
            summary["abstain_reasons"] = abstain_reasons
            for row in rows:
                _annotate_verified_correctness(row, scorer=scorer, scoring_view=view)
            return summary
        from verifiqa.eval.tatdqa_scoring import DEFAULT_SCORING_VIEW, summary_for_view
        view = scoring_view or DEFAULT_SCORING_VIEW
        # Resume files may contain historical grading flags. Recompute them
        # from frozen claims; existing JSONL rows are not rewritten.
        for row in rows:
            _annotate_verified_correctness(row, scorer=scorer, scoring_view=view)
        evaluated = {**scorer.score_rows(rows), "abstention_reasons": abstain_reasons,
                     "provenance": scorer.provenance}
        return summary_for_view(evaluated, primary_view=view, elapsed_s=elapsed_s)

    gold_rows = [row for row in rows if _gold_answer(row) not in (None, "")]
    sat_rows = [
        row for row in rows
        if _solver_status(row) == "SAT" and row.get("expected_solver_status") is not None
    ]
    unsat_rows = [
        row for row in rows
        if _solver_status(row) == "UNSAT" and row.get("expected_solver_status") is not None
    ]
    solver_rows = sat_rows + unsat_rows

    sat_correct = sum(1 for row in sat_rows if row.get("expected_solver_status") == "SAT")
    sat_incorrect = sum(1 for row in sat_rows if row.get("expected_solver_status") != "SAT")
    unsat_correct = sum(1 for row in unsat_rows if row.get("expected_solver_status") == "UNSAT")
    unsat_incorrect = sum(1 for row in unsat_rows if row.get("expected_solver_status") != "UNSAT")
    correct_solver_verdicts = sum(1 for row in solver_rows if row.get("solver_correct") is True)
    incorrect_solver_verdicts = sum(1 for row in solver_rows if row.get("solver_correct") is False)
    accuracy = _safe_div(correct_solver_verdicts, len(solver_rows))

    return {
        "total": total,
        "gold_evaluable": len(gold_rows),
        "all_verifications": len(solver_rows),
        "correct_verifications": correct_solver_verdicts,
        "incorrect_verifications": incorrect_solver_verdicts,
        "solver_verdicts": len(solver_rows),
        "correct_solver_verdicts": correct_solver_verdicts,
        "incorrect_solver_verdicts": incorrect_solver_verdicts,
        "expected_sat": sum(1 for row in gold_rows if row.get("expected_solver_status") == "SAT"),
        "expected_unsat": sum(1 for row in gold_rows if row.get("expected_solver_status") == "UNSAT"),
        "actual_sat": len(sat_rows),
        "actual_unsat": len(unsat_rows),
        "verified": counts.get("VERIFIED", 0),
        "violated": counts.get("VIOLATED", 0),
        "abstain": counts.get("ABSTAIN", 0),
        "abstain_reasons": abstain_reasons,
        "verified_correct": sat_correct,
        "verified_incorrect": sat_incorrect,
        "violated_correct": unsat_correct,
        "violated_incorrect": unsat_incorrect,
        "correct_decisions": correct_solver_verdicts,
        "incorrect_decisions": incorrect_solver_verdicts,
        "accuracy": round(accuracy, 4),
        "solver_accuracy": round(accuracy, 4),
        "coverage": round(_safe_div(len(solver_rows), len(gold_rows)), 4),
        "decision_accuracy": round(accuracy, 4),
        "overall_accuracy": round(_safe_div(correct_solver_verdicts, len(gold_rows)), 4),
        "sat_precision": round(_safe_div(sat_correct, len(sat_rows)), 4),
        "unsat_precision": round(_safe_div(unsat_correct, len(unsat_rows)), 4),
        "verification_precision": round(_safe_div(sat_correct, len(sat_rows)), 4),
        "violation_precision": round(_safe_div(unsat_correct, len(unsat_rows)), 4),
        "elapsed_s": elapsed_s,
    }


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _print(args, msg: str) -> None:
    if getattr(args, "verbose", False):
        print(msg, flush=True)


def _noop(*a, **kw):
    pass


if __name__ == "__main__":
    main()
