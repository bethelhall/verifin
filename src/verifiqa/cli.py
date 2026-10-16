from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Auto-load .env from project root if ANTHROPIC_API_KEY is not already set
# cli.py lives at src/verifiqa/cli.py  →  3 parents up = project root
_env_file = Path(__file__).parent.parent.parent / ".env"
if not os.environ.get("ANTHROPIC_API_KEY") and _env_file.exists():
    _env_contents = _env_file.read_text().strip()
    for _line in _env_contents.splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#"):
            continue
        if "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
        elif _line.startswith("sk-ant-"):
            # bare key (no variable name) — assume it's the Anthropic key
            os.environ.setdefault("ANTHROPIC_API_KEY", _line)
    del _env_file, _env_contents, _line

from verifiqa.baselines.vanilla_rag import VanillaRagBaseline
from verifiqa.baselines.reflection import ReflectionBaseline
from verifiqa.baselines.llm_judge import LlmJudgeBaseline
from verifiqa.dataset import load_financebench
from verifiqa.eval.report import write_report
from verifiqa.formalization.formalizer import Formalizer
from verifiqa.financebench_sample import DownloadError, download_financebench_sample, summarize_sample
from verifiqa.generation.answer_generator import AnswerGenerator, _format_evidence as _format_answer_evidence
from verifiqa.generation.llm_client import LoggedLlmClient, make_llm_client
from verifiqa.corpus_builder import build_corpus, evidence_page_supplements_from_records, load_corpus
from verifiqa.pipeline import CounterexampleRavPipeline, PipelineConfig
from verifiqa.policy import PolicySemanticChecker
from verifiqa.policy.source_builder import write_policy_source_report
from verifiqa.reconciler import ReconcilerRunner
from verifiqa.question_filter import is_numerical_example
from verifiqa.retrieval.evidence_retriever import EvidenceRetriever
from verifiqa.retrieval.planner import RetrievalPlanner
from verifiqa.retrieval.table_rows import extract_structured_rows
from verifiqa.types import RavResult, RetrievalPlan, to_jsonable, ABSTENTION_MESSAGE
from verifiqa.verification.smt_generator import SmtGenerator
from verifiqa.verification.z3_runner import Z3Runner

_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_FINQA_SPLIT_CHOICES = [
    "train",
    "dev",
    "test",
    "train_xbrl",
    "dev_xbrl",
    "test_xbrl",
    "train_xbrl_validated",
    "dev_xbrl_validated",
    "test_xbrl_validated",
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="verifiqa")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run = sub.add_parser("run", help="Run the RAV pipeline on FinanceBench")
    run.add_argument("--data", type=Path, default=None, help="FinanceBench data directory")
    run.add_argument("--finqa-dir", type=Path, default=None,
                     help="FinQA dataset directory (use instead of --data for FinQA oracle runs)")
    run.add_argument("--finqa-split", default="dev", choices=_FINQA_SPLIT_CHOICES,
                     help="FinQA split to use (default: dev)")
    run.add_argument("--financebench-dir", type=Path, default=None, help=argparse.SUPPRESS)
    run.add_argument("--out", type=Path, required=True, help="Output directory")
    run.add_argument("--llm-mode", choices=["claude", "anthropic", "vllm"], default="claude")
    run.add_argument("--model", default=_DEFAULT_MODEL, help="Claude model name")
    run.add_argument("--vllm-base-url", default="http://localhost:8000/v1",
                     help="OpenAI-compatible chat completions base URL for --llm-mode vllm")
    run.add_argument("--limit", type=int, default=0, help="Cap number of questions (0 = all)")
    run.add_argument("--no-llm-terminal-log", action="store_true",
                     help="Do not print full LLM prompts/responses to the terminal")
    run.add_argument("--baseline", choices=["vanilla_rag", "reflection", "llm_judge"], default=None,
                     help="Run a baseline instead of the full pipeline")
    run.add_argument("--top-k", "--evidence-top-k", dest="top_k", type=int, default=0,
                     help="Number of retrieved evidence chunks (default: 10 with --corpus, otherwise 5)")
    run.add_argument("--first-stage-k", type=int, default=0,
                     help="Number of dense candidates before final selection/reranking")
    run.add_argument("--retriever-model", default="",
                     help="SentenceTransformer embedding model for evidence retrieval")
    run.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="dense",
                     help="First-stage retrieval scoring: dense or hybrid BM25+dense RRF")
    run.add_argument("--hybrid-rrf-k", type=int, default=60,
                     help="RRF k constant for --retrieval-mode hybrid")
    run.add_argument("--reranker-model", default="",
                     help="CrossEncoder reranker model for corpus mode")
    run.add_argument("--no-rerank", action="store_true",
                     help="Disable cross-encoder reranking in corpus mode")
    run.add_argument("--rerank-planned-queries", action="store_true",
                     help="Experiment: apply cross-encoder reranking to planned fact/query retrieval")
    run.add_argument("--adaptive-retrieval", action="store_true",
                     help="Experiment: on certificate grounding failure, retrieve again for the missing/ungrounded fact")
    run.add_argument("--policy-semantic-check", action="store_true",
                     help="Enable taxonomy-backed metric policy/semantic checks before SMT")
    run.add_argument("--no-certificate-grounding", action="store_true",
                     help="Ablation: skip evidence-grounding validation before SMT")
    run.add_argument("--no-finqa-program", action="store_true",
                     help="Deployment-realistic regime: do NOT use the FinQA gold "
                          "reasoning program as formula authority (it is a dataset "
                          "annotation/oracle); authorize via the standard registry only")
    run.add_argument("--no-generic-operations", action="store_true",
                     help="Disable the generic-operation authority (period change / "
                          "percentage change) used for FinQA arithmetic when no named "
                          "registry policy applies")
    run.add_argument("--verification-mode", choices=["current", "reconciler-shadow"], default="current",
                     help="Verification path: current pipeline, or opt-in Reconciler->Spec->SMT shadow artifacts")
    run.add_argument("--numeric-only-verification", action="store_true",
                     help="Disable unit-kind matching/authority checks; verify numeric values and formulas only")
    run.add_argument("--corpus", type=Path, default=None,
                     help="Pre-built filing corpus directory (from build-corpus). Enables real retrieval.")
    run.add_argument("--oracle-retrieval", action="store_true",
                     help="Restrict retrieval to each question's own evidence passages (oracle mode)")
    run.add_argument("--include-non-numerical", "--include-non-numeric", action="store_true",
                     help="Disable the default direct-numerical-question filter")
    run.add_argument("--resume", action="store_true",
                     help="Skip questions already in results.jsonl and append remaining results")
    run.add_argument("--xbrl-calculations", type=Path, default=None,
                     help=("Cached filing XBRL artifacts with manifest.json for calculation-linkbase SMT constraints "
                           "(default: artifacts/finqa_xbrl for --finqa-dir, otherwise artifacts/calculation_linkbases)"))
    run.add_argument("--no-xbrl-linkbase-smt", action="store_true",
                     help="Disable deterministic XBRL calculation-linkbase constraints in generated SMT")
    run.add_argument("--xbrl-llm-binding", action="store_true",
                     help=("Deprecated compatibility flag; registry/linkbase XBRL binding is used when "
                           "--xbrl-calculations is enabled"))
    run.add_argument("--debug-unsat-cores", action="store_true",
                     help="Request Z3 unsat cores in saved solver diagnostics")
    run.add_argument("--api-key-env", default="ANTHROPIC_API_KEY", help=argparse.SUPPRESS)

    # --- retrieve ---
    ret = sub.add_parser("retrieve", help="Run retrieval only and write evidence-quality metrics")
    ret.add_argument("--data", type=Path, required=True, help="FinanceBench data directory")
    ret.add_argument("--corpus", type=Path, required=True,
                     help="Pre-built filing corpus directory (from build-corpus)")
    ret.add_argument("--out", type=Path, required=True, help="Output directory")
    ret.add_argument("--limit", type=int, default=0, help="Cap number of questions (0 = all)")
    ret.add_argument("--top-k", "--evidence-top-k", dest="top_k", type=int, default=10,
                     help="Number of retrieved evidence chunks")
    ret.add_argument("--first-stage-k", type=int, default=100,
                     help="Number of dense candidates before final selection/reranking")
    ret.add_argument("--retriever-model", default="BAAI/bge-large-en-v1.5",
                     help="SentenceTransformer embedding model for evidence retrieval")
    ret.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="dense",
                     help="First-stage retrieval scoring: dense or hybrid BM25+dense RRF")
    ret.add_argument("--hybrid-rrf-k", type=int, default=60,
                     help="RRF k constant for --retrieval-mode hybrid")
    ret.add_argument("--reranker-model", default="",
                     help="CrossEncoder reranker model")
    ret.add_argument("--no-rerank", action="store_true",
                     help="Disable cross-encoder reranking")
    ret.add_argument("--rerank-planned-queries", action="store_true",
                     help="Experiment: apply cross-encoder reranking to planned fact/query retrieval")
    ret.add_argument("--adaptive-retrieval", action="store_true",
                     help="Experiment: expand retrieval for planned facts not covered by structured evidence")
    ret.add_argument("--include-non-numerical", "--include-non-numeric", action="store_true",
                     help="Disable the default direct-numerical-question filter")

    # --- evidence ---
    evq = sub.add_parser("evidence", help="Retrieve evidence for one question without calling an LLM")
    evq.add_argument("--question", required=True, help="Question to retrieve evidence for")
    evq.add_argument("--company", default="", help="Optional company hint")
    evq.add_argument("--doc-name", default="", help="Optional filing/document name hint")
    evq.add_argument("--corpus", type=Path, required=True,
                     help="Pre-built filing corpus directory (from build-corpus)")
    evq.add_argument("--top-k", "--evidence-top-k", dest="top_k", type=int, default=10,
                     help="Number of retrieved evidence chunks")
    evq.add_argument("--first-stage-k", type=int, default=100,
                     help="Number of dense candidates before final selection/reranking")
    evq.add_argument("--retriever-model", default="BAAI/bge-large-en-v1.5",
                     help="SentenceTransformer embedding model for evidence retrieval")
    evq.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="dense",
                     help="First-stage retrieval scoring: dense or hybrid BM25+dense RRF")
    evq.add_argument("--hybrid-rrf-k", type=int, default=60,
                     help="RRF k constant for --retrieval-mode hybrid")
    evq.add_argument("--reranker-model", default="",
                     help="CrossEncoder reranker model")
    evq.add_argument("--no-rerank", action="store_true",
                     help="Disable cross-encoder reranking")
    evq.add_argument("--rerank-planned-queries", action="store_true",
                     help="Experiment: apply cross-encoder reranking to planned fact/query retrieval")
    evq.add_argument("--adaptive-retrieval", action="store_true",
                     help="Experiment: expand retrieval for planned facts not covered by structured evidence")
    evq.add_argument("--format", choices=["json", "text"], default="json",
                     help="Output JSON or a prompt-like text evidence block")
    evq.add_argument("--out", type=Path, default=None,
                     help="Optional output file for the evidence record")

    # --- eval ---
    ev = sub.add_parser("eval", help="Summarise a run directory")
    ev.add_argument("--run", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)

    # --- build-corpus ---
    bc = sub.add_parser("build-corpus", help="Download filings and build a chunked retrieval corpus")
    bc.add_argument("--data", type=Path, required=True, help="FinanceBench data directory (for doc links)")
    bc.add_argument("--examples", type=Path, required=True, help="Filtered examples JSONL (to select which docs)")
    bc.add_argument("--out", type=Path, required=True, help="Output corpus directory")
    bc.add_argument("--cached-only", action="store_true",
                    help="Rebuild chunks only from PDFs already present in the corpus pdfs/ directory")
    bc.add_argument("--download-timeout", type=int, default=180,
                    help="PDF download timeout in seconds")
    bc.add_argument("--download-retries", type=int, default=3,
                    help="PDF download retry count")
    bc.add_argument("--include-benchmark-evidence-pages", action="store_true",
                    help="Add FinanceBench full evidence pages as labeled supplemental corpus chunks")

    # --- build-policy-data ---
    bpd = sub.add_parser("build-policy-data", help="Fetch/validate taxonomy-backed policy data")
    bpd.add_argument("--data-dir", type=Path, default=Path("src/verifiqa/policy/data"),
                     help="Policy data directory")
    bpd.add_argument("--cache-dir", type=Path, default=Path("data/taxonomies"),
                     help="Directory for downloaded taxonomy packages")
    bpd.add_argument("--out", type=Path, default=None,
                     help="Output policy source report JSON")
    bpd.add_argument("--offline", action="store_true",
                     help="Use already cached taxonomy packages; do not download")
    bpd.add_argument("--timeout", type=int, default=120,
                     help="Taxonomy package download timeout in seconds")

    # --- filter-dataset ---
    fd = sub.add_parser("filter-dataset", help="Pre-filter FinanceBench to direct numerical questions")
    fd.add_argument("--data", type=Path, required=True, help="FinanceBench data directory or JSONL")
    fd.add_argument("--out", type=Path, required=True, help="Output JSONL path")

    # --- download ---
    dl = sub.add_parser("download-sample", help="Download the FinanceBench open-source sample")
    dl.add_argument("--out", type=Path, default=Path("data/financebench_sample"))
    dl.add_argument("--force", action="store_true")

    # --- experiment ---
    exp = sub.add_parser("experiment", help="Run paper baseline/verifier experiment methods")
    exp.add_argument("--data", type=Path, default=None, help="FinanceBench data directory")
    exp.add_argument("--finqa-dir", type=Path, default=None,
                     help="FinQA dataset directory (use instead of --data)")
    exp.add_argument("--finqa-split", default="dev", choices=_FINQA_SPLIT_CHOICES)
    exp.add_argument("--corpus", type=Path, default=None,
                     help="Pre-built filing corpus directory for RAG/long-context methods")
    exp.add_argument("--out", type=Path, required=True, help="Output directory")
    exp.add_argument("--methods", default="all",
                     help="Comma-separated methods or all: rag_cot,rag_pot,long_context_cot,"
                          "oracle_cot,oracle_pot,llm_judge,faithfulness,verifiqa_strict,verifiqa_full")
    exp.add_argument("--llm-mode", choices=["claude", "anthropic", "vllm"], default="claude")
    exp.add_argument("--model", default=_DEFAULT_MODEL)
    exp.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    exp.add_argument("--api-key-env", default="ANTHROPIC_API_KEY", help=argparse.SUPPRESS)
    exp.add_argument("--limit", type=int, default=0, help="Cap number of questions (0 = all)")
    exp.add_argument("--include-non-numerical", "--include-non-numeric", action="store_true")
    exp.add_argument("--top-k", "--evidence-top-k", dest="top_k", type=int, default=10)
    exp.add_argument("--first-stage-k", type=int, default=100)
    exp.add_argument("--retriever-model", default="BAAI/bge-large-en-v1.5")
    exp.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="dense")
    exp.add_argument("--hybrid-rrf-k", type=int, default=60)
    exp.add_argument("--reranker-model", default="")
    exp.add_argument("--no-rerank", action="store_true")
    exp.add_argument("--policy-semantic-check", action="store_true")
    exp.add_argument("--no-certificate-grounding", action="store_true",
                     help="Ablation: skip evidence-grounding validation before SMT")
    exp.add_argument("--long-context-max-chars", type=int, default=200_000)
    exp.add_argument("--no-llm-terminal-log", action="store_true")

    # --- summarize ---
    sm = sub.add_parser("summarize", help="Summarise an experiment results.jsonl")
    sm.add_argument("--run", type=Path, required=True, help="Experiment run directory or results.jsonl")
    sm.add_argument("--out", type=Path, required=True, help="Output report directory")

    args = parser.parse_args(argv)
    if args.command == "run":
        _cmd_run(args)
    elif args.command == "retrieve":
        _cmd_retrieve(args)
    elif args.command == "evidence":
        _cmd_evidence(args)
    elif args.command == "eval":
        _cmd_eval(args)
    elif args.command == "build-corpus":
        _cmd_build_corpus(args)
    elif args.command == "build-policy-data":
        _cmd_build_policy_data(args)
    elif args.command == "filter-dataset":
        _cmd_filter_dataset(args)
    elif args.command == "download-sample":
        _cmd_download_sample(args)
    elif args.command == "experiment":
        _cmd_experiment(args)
    elif args.command == "summarize":
        _cmd_summarize(args)


def _resolve_xbrl_calculation_dir(args, is_finqa: bool) -> Optional[Path]:
    if getattr(args, "no_xbrl_linkbase_smt", False):
        return None
    explicit = getattr(args, "xbrl_calculations", None)
    if explicit is not None:
        return explicit
    if is_finqa:
        return Path("artifacts/finqa_xbrl")
    return Path("artifacts/calculation_linkbases")


def _cmd_run(args) -> None:
    numeric_only = not args.include_non_numerical

    # --- FinQA oracle mode ---
    finqa_dir = getattr(args, "finqa_dir", None)
    if finqa_dir is not None:
        from verifiqa.dataset_finqa import load_finqa
        print(f"Loading FinQA {args.finqa_split} split from {finqa_dir}...", file=sys.stderr)
        examples = load_finqa(finqa_dir, split=args.finqa_split)
        loaded_count = len(examples)
        if numeric_only:
            from verifiqa.dataset_finqa import _is_numeric
            examples = [e for e in examples if _is_numeric(e.answer)]
            print(f"Filtered to {len(examples)} numeric examples from {loaded_count}.", file=sys.stderr)
        else:
            print(f"Loaded {loaded_count} FinQA examples.", file=sys.stderr)
        if args.limit:
            examples = examples[:args.limit]
        # FinQA is always oracle — force the flag
        args.oracle_retrieval = True
        oracle_chunks = []  # not used in oracle mode
    else:
        # --- FinanceBench mode (default) ---
        data_dir = args.data or args.financebench_dir
        if data_dir is None:
            raise SystemExit("Provide --data /path/to/financebench  or  --finqa-dir /path/to/finqa")
        print("Loading data...", file=sys.stderr)
        examples, oracle_chunks = load_financebench(data_dir)
        loaded_count = len(examples)
        if numeric_only:
            examples = [e for e in examples if is_numerical_example(e)]
            print(f"Filtered to {len(examples)} direct numerical examples from {loaded_count}.", file=sys.stderr)
        else:
            print(f"Loaded {loaded_count} examples.", file=sys.stderr)
        if args.limit:
            examples = examples[:args.limit]

    xbrl_calculation_dir = _resolve_xbrl_calculation_dir(args, is_finqa=finqa_dir is not None)

    if args.corpus:
        print(f"Loading corpus from {args.corpus}...", file=sys.stderr)
        chunks, corpus_stats = load_corpus(args.corpus)
        print(
            f"Corpus: {corpus_stats['stored_records']} stored records -> "
            f"{corpus_stats['loaded_chunks']} retrieval chunks "
            f"({', '.join(f'{k}={v}' for k, v in corpus_stats['source_types'].items())}).",
            file=sys.stderr,
        )
    else:
        chunks = oracle_chunks
        corpus_stats = {}

    evidence_top_k = args.top_k or (10 if args.corpus else 5)
    first_stage_k = args.first_stage_k or (max(50, evidence_top_k * 5) if args.corpus else max(15, evidence_top_k))
    corpus_coverage = _corpus_coverage(examples, chunks) if args.corpus else {}
    if corpus_coverage:
        print(
            f"Corpus document coverage: {corpus_coverage['n_available_examples']}/"
            f"{corpus_coverage['n_examples']} examples, "
            f"{corpus_coverage['n_missing_docs']} missing docs.",
            file=sys.stderr,
        )
        if corpus_coverage["missing_docs"]:
            preview = ", ".join(corpus_coverage["missing_docs"][:8])
            print(f"Missing docs: {preview}", file=sys.stderr)

    retriever_model = args.retriever_model or (
        "BAAI/bge-large-en-v1.5" if args.corpus else "sentence-transformers/all-MiniLM-L6-v2"
    )
    use_rerank = bool(args.corpus) and not args.no_rerank
    if chunks:
        print(
            f"Building retriever ({retriever_model}, rerank={use_rerank}, "
            f"mode={getattr(args, 'retrieval_mode', 'dense')}, "
            f"top_k={evidence_top_k}, first_stage_k={first_stage_k})...",
            file=sys.stderr,
        )
        retriever_kwargs = {
            "model_name": retriever_model,
            "rerank": use_rerank,
            "first_stage_k": first_stage_k,
            "embedding_cache_dir": args.corpus if args.corpus else None,
            "rerank_planned_queries": getattr(args, "rerank_planned_queries", False),
            "retrieval_mode": getattr(args, "retrieval_mode", "dense"),
            "hybrid_rrf_k": getattr(args, "hybrid_rrf_k", 60),
        }
        if args.reranker_model:
            retriever_kwargs["reranker_model"] = args.reranker_model
        evidence_retriever = EvidenceRetriever(chunks, **retriever_kwargs)
    elif args.oracle_retrieval:
        print("Oracle retrieval: using evidence embedded in each example; no retriever corpus needed.", file=sys.stderr)
        evidence_retriever = _NoopEvidenceRetriever()
    else:
        raise SystemExit("No evidence chunks available. Provide --corpus or use --oracle-retrieval with embedded evidence.")

    progress = ProgressPrinter(len(examples))
    args.out.mkdir(parents=True, exist_ok=True)
    question_evidence_path = args.out / "question_evidence.jsonl"
    if not getattr(args, "resume", False):
        question_evidence_path.write_text("", encoding="utf-8")
    evidence_callback = _make_question_evidence_callback(question_evidence_path, evidence_top_k)

    llm = make_llm_client(mode=args.llm_mode, config={
        "base_url": args.vllm_base_url if args.llm_mode == "vllm" else "https://api.anthropic.com/v1",
        "model": args.model,
        "api_key_env": args.api_key_env,
        "max_tokens": 2048,
    })
    llm = LoggedLlmClient(
        llm,
        terminal=not args.no_llm_terminal_log,
        jsonl_path=args.out / "llm_calls.jsonl",
    )

    if args.baseline:
        runner = _make_baseline(args.baseline, evidence_retriever, llm,
                                evidence_callback=evidence_callback,
                                oracle_retrieval=args.oracle_retrieval)
    else:
        policy_checker = PolicySemanticChecker()
        verification_mode = getattr(args, "verification_mode", "current")
        reconciler_runner = None
        if verification_mode == "reconciler-shadow":
            reconciler_runner = ReconcilerRunner(
                llm,
                registry=policy_checker.registry,
                z3_runner=Z3Runner(
                    save_dir=args.out / "reconciler_smt",
                    debug_unsat_cores=getattr(args, "debug_unsat_cores", False),
                ),
            )
        runner = CounterexampleRavPipeline(
            evidence_retriever=evidence_retriever,
            answer_generator=AnswerGenerator(llm),
            formalizer=Formalizer(llm),
            retrieval_planner=None if args.oracle_retrieval else RetrievalPlanner(llm),
            smt_generator=SmtGenerator(llm),
            z3_runner=Z3Runner(
                save_dir=args.out / "smt",
                debug_unsat_cores=getattr(args, "debug_unsat_cores", False),
            ),
            policy_checker=policy_checker,
            reconciler_runner=reconciler_runner,
            config=PipelineConfig(
                evidence_top_k=evidence_top_k,
                oracle_retrieval=args.oracle_retrieval,
                numeric_only=numeric_only,
                adaptive_retrieval=getattr(args, "adaptive_retrieval", False),
                policy_semantic_check=getattr(args, "policy_semantic_check", False),
                certificate_grounding_check=not getattr(args, "no_certificate_grounding", False),
                use_finqa_program=not getattr(args, "no_finqa_program", False),
                use_generic_operations=not getattr(args, "no_generic_operations", False),
                xbrl_calculation_dir=xbrl_calculation_dir,
                xbrl_llm_binding=getattr(args, "xbrl_llm_binding", False),
                verification_mode=verification_mode,
                numeric_only_verification=getattr(args, "numeric_only_verification", False),
            ),
            progress_callback=progress.stage,
            evidence_callback=evidence_callback,
        )

    results = []
    interrupted = False
    results_path = args.out / "results.jsonl"

    # Resume: load already-completed results and skip those examples
    done_ids: set[str] = set()
    if getattr(args, "resume", False) and results_path.exists():
        with results_path.open("r", encoding="utf-8") as rh:
            for line in rh:
                if line.strip():
                    try:
                        row = json.loads(line)
                        fid = row.get("financebench_id", "")
                        if fid:
                            result_obj = RavResult(
                                financebench_id=fid,
                                question=row.get("question", ""),
                                answer=row.get("answer", ""),
                                final_status=row.get("final_status", "ABSTAIN"),
                                first_pass_solver_status=row.get("first_pass_solver_status", "INVALID"),
                                gold_answer=row.get("gold_answer"),
                                verified=row.get("verified", False),
                                abstained=row.get("abstained", True),
                                retrieved_chunk_ids=row.get("retrieved_chunk_ids", []),
                                metric=row.get("metric", ""),
                            )
                            results.append(result_obj)
                            done_ids.add(fid)
                    except Exception:
                        pass
        if done_ids:
            print(f"Resuming: {len(done_ids)} already completed, {len(examples) - len(done_ids)} remaining.",
                  file=sys.stderr)

    write_mode = "a" if done_ids else "w"
    with results_path.open(write_mode, encoding="utf-8") as handle:
        try:
            for index, example in enumerate(examples, start=1):
                if example.financebench_id in done_ids:
                    progress.done(index, "SKIP")
                    continue
                progress.start(index, example.financebench_id or example.question)
                try:
                    result = runner.run_example(example)
                except Exception as exc:
                    result = RavResult(
                        financebench_id=example.financebench_id,
                        question=example.question,
                        answer=ABSTENTION_MESSAGE,
                        final_status="ERROR",
                        first_pass_solver_status="INVALID",
                        gold_answer=example.answer,
                        abstained=True,
                    )
                    print(f"\nerror on {example.financebench_id or index}: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
                results.append(result)
                handle.write(json.dumps(to_jsonable(result), sort_keys=True) + "\n")
                handle.flush()
                _write_verifier_certificate(args.out, result)
                progress.done(index, result.final_status)
        except KeyboardInterrupt:
            interrupted = True
            sys.stderr.write("\nInterrupted; writing partial run artifacts.\n")
            sys.stderr.flush()

    progress.finish()
    _write_smt_question_map(results_path, args.out)
    try:
        from verifiqa.eval.run_summary import write_run_summary
        write_run_summary(args.out)
    except Exception as exc:  # summary is best-effort; never fail a run over it
        sys.stderr.write(f"summary.csv export failed: {type(exc).__name__}: {exc}\n")
    (args.out / "config.json").write_text(
        json.dumps({"model": args.model, "n_loaded": loaded_count, "n_examples": len(examples),
                    "n_completed": len(results), "interrupted": interrupted,
                    "pipeline": "schema_guided_rag_answer_formalize_z3", "baseline": args.baseline,
                    "oracle_retrieval": args.oracle_retrieval,
                    "numeric_only": numeric_only,
                    "evidence_top_k": evidence_top_k,
                    "first_stage_k": first_stage_k,
                    "retriever_model": retriever_model,
                    "retrieval_mode": getattr(args, "retrieval_mode", "dense"),
                    "hybrid_rrf_k": getattr(args, "hybrid_rrf_k", 60),
                    "reranker_model": args.reranker_model or "",
                    "rerank": use_rerank,
                    "rerank_planned_queries": getattr(args, "rerank_planned_queries", False),
                    "adaptive_retrieval": getattr(args, "adaptive_retrieval", False),
                    "policy_semantic_check": getattr(args, "policy_semantic_check", False),
                    "certificate_grounding_check": not getattr(args, "no_certificate_grounding", False),
                    **({"verification_mode": getattr(args, "verification_mode", "current")}
                       if getattr(args, "verification_mode", "current") != "current" else {}),
                    **({"numeric_only_verification": True}
                       if getattr(args, "numeric_only_verification", False) else {}),
                    "xbrl_linkbase_smt": not getattr(args, "no_xbrl_linkbase_smt", False),
                    "xbrl_calculations": str(xbrl_calculation_dir or ""),
                    "debug_unsat_cores": getattr(args, "debug_unsat_cores", False),
                    "question_evidence_path": str(question_evidence_path),
                    "corpus_stats": corpus_stats,
                    "corpus_coverage": corpus_coverage}, indent=2),
        encoding="utf-8",
    )
    if corpus_coverage:
        (args.out / "corpus_coverage.json").write_text(
            json.dumps(corpus_coverage, indent=2),
            encoding="utf-8",
        )

    if args.corpus and not args.baseline:
        from verifiqa.eval.retrieval_metrics import compute_retrieval_metrics
        ret_metrics = compute_retrieval_metrics(results, examples, k=evidence_top_k)
        (args.out / "retrieval_metrics.json").write_text(
            json.dumps(ret_metrics, indent=2), encoding="utf-8"
        )
        if ret_metrics.get("n", 0) > 0:
            print(
                f"Retrieval Hit@{ret_metrics['k']}: {ret_metrics['hit_rate_at_k']:.1%}  "
                f"Recall@{ret_metrics['k']}: {ret_metrics['mean_recall_at_k']:.1%}  "
                f"MRR@{ret_metrics['k']}: {ret_metrics['mrr_at_k']:.3f}  "
                f"(n={ret_metrics['n']})",
                file=sys.stderr,
            )

    print(json.dumps({"results": len(results), "out": str(args.out), "interrupted": interrupted}, indent=2))


def _write_verifier_certificate(out_dir: Path, result: RavResult) -> None:
    certificate = getattr(result, "verifier_certificate", None) or {}
    if not certificate:
        return
    cert_dir = out_dir / "verifier_certificates"
    cert_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_artifact_stem(result.financebench_id or result.question or "example")
    path = cert_dir / f"{stem}.json"
    path.write_text(json.dumps(to_jsonable(certificate), indent=2, sort_keys=True), encoding="utf-8")


def _write_smt_question_map(results_path: Path, out_dir: Path) -> None:
    if not results_path.exists():
        return

    rows = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    llm_answers = _answer_generation_answers_by_question(out_dir / "llm_calls.jsonl")
    mapping = [
        _smt_question_map_row(index, row, llm_answers)
        for index, row in enumerate(rows, start=1)
    ]
    (out_dir / "smt_question_map.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    csv_path = out_dir / "smt_question_map.csv"
    fieldnames = [
        "question_index",
        "financebench_id",
        "final_status",
        "first_pass_solver_status",
        "metric",
        "gold_answer",
        "llm_answer",
        "final_answer",
        "smt_path",
        "revised_smt_path",
        "question",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in mapping:
            writer.writerow({name: _csv_cell(row.get(name, "")) for name in fieldnames})


def _csv_cell(value) -> str:
    text = str(value or "")
    return " ".join(text.replace("\r", "\n").split())


def _smt_question_map_row(index: int, row: dict, llm_answers: Optional[dict[str, str]] = None) -> dict:
    smt = ((row.get("verifier_certificate") or {}).get("smt") or {})
    smt_path = smt.get("smt_path") or ""
    revised_smt_path = smt.get("revised_smt_path") or ""
    question = row.get("question", "")
    final_answer = row.get("final_answer") or row.get("answer", "")
    llm_answer = ""
    if llm_answers is not None:
        llm_answer = llm_answers.get(question, "")
    if not llm_answer:
        llm_answer = row.get("draft_answer", "")
    smt_files = []
    if smt_path:
        smt_files.append({
            "kind": "first",
            "path": smt_path,
            "solver_status": smt.get("first_pass_solver_status") or row.get("first_pass_solver_status", ""),
        })
    if revised_smt_path:
        smt_files.append({
            "kind": "revised",
            "path": revised_smt_path,
            "solver_status": smt.get("revised_solver_status", ""),
        })
    return {
        "question_index": index,
        "financebench_id": row.get("financebench_id", ""),
        "question": question,
        "gold_answer": row.get("gold_answer", ""),
        "llm_answer": llm_answer,
        "final_answer": final_answer,
        "final_status": row.get("final_status", ""),
        "first_pass_solver_status": row.get("first_pass_solver_status", ""),
        "metric": row.get("metric", ""),
        "smt_path": smt_path,
        "revised_smt_path": revised_smt_path,
        "smt_files": smt_files,
        "has_smt": bool(smt_path),
        "has_revised_smt": bool(revised_smt_path),
        "abstained_before_smt": bool(row.get("abstained")) and not smt_path,
    }


def _answer_generation_answers_by_question(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    answers: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                call = json.loads(line)
            except json.JSONDecodeError:
                continue
            if call.get("stage") != "answer_generation":
                continue
            messages = call.get("messages") or []
            content = messages[0].get("content", "") if messages else ""
            question = _question_from_answer_prompt(content)
            answer = _answer_from_llm_response(call.get("response", ""))
            if question and answer:
                answers[question] = answer
    return answers


def _question_from_answer_prompt(content: str) -> str:
    marker = "Question:\n"
    if marker not in content:
        return ""
    tail = content.split(marker, 1)[1]
    for end_marker in ("\n\nRetrieval plan:", "\n\nRetrieved filing evidence:"):
        if end_marker in tail:
            return tail.split(end_marker, 1)[0].strip()
    return tail.strip()


def _answer_from_llm_response(response: str) -> str:
    text = (response or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return text
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            malformed_answer = _answer_from_malformed_json_response(text)
            return malformed_answer or text
    if isinstance(data, dict):
        return str(data.get("answer", "")).strip()
    return text


def _answer_from_malformed_json_response(response: str) -> str:
    match = re.search(r'"answer"\s*:\s*"(.*)"\s*}\s*$', response, flags=re.DOTALL)
    if not match:
        return ""
    value = match.group(1)
    value = value.replace('\\n', '\n').replace('\\"', '"')
    return " ".join(value.split())


def _safe_artifact_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return (stem.strip("._") or "example")[:120]


def _cmd_retrieve(args) -> None:
    print("Loading data...", file=sys.stderr)
    examples, _ = load_financebench(args.data)
    loaded_count = len(examples)
    numeric_only = not args.include_non_numerical
    if numeric_only:
        examples = [e for e in examples if is_numerical_example(e)]
        print(f"Filtered to {len(examples)} direct numerical examples from {loaded_count}.", file=sys.stderr)
    else:
        print(f"Loaded {loaded_count} examples.", file=sys.stderr)
    if args.limit:
        examples = examples[:args.limit]

    print(f"Loading corpus from {args.corpus}...", file=sys.stderr)
    chunks, corpus_stats = load_corpus(args.corpus)
    corpus_coverage = _corpus_coverage(examples, chunks)
    print(
        f"Corpus: {corpus_stats['stored_records']} stored records -> "
        f"{corpus_stats['loaded_chunks']} retrieval chunks "
        f"({', '.join(f'{k}={v}' for k, v in corpus_stats['source_types'].items())}).",
        file=sys.stderr,
    )
    print(
        f"Corpus document coverage: {corpus_coverage['n_available_examples']}/"
        f"{corpus_coverage['n_examples']} examples, "
        f"{corpus_coverage['n_missing_docs']} missing docs.",
        file=sys.stderr,
    )

    use_rerank = not args.no_rerank
    print(
        f"Building retriever ({args.retriever_model}, rerank={use_rerank}, "
        f"mode={args.retrieval_mode}, top_k={args.top_k}, "
        f"first_stage_k={args.first_stage_k})...",
        file=sys.stderr,
    )
    retriever_kwargs = {
        "model_name": args.retriever_model,
        "rerank": use_rerank,
        "first_stage_k": args.first_stage_k,
        "embedding_cache_dir": args.corpus,
        "rerank_planned_queries": args.rerank_planned_queries,
        "retrieval_mode": args.retrieval_mode,
        "hybrid_rrf_k": args.hybrid_rrf_k,
    }
    if args.reranker_model:
        retriever_kwargs["reranker_model"] = args.reranker_model
    retriever = EvidenceRetriever(chunks, **retriever_kwargs)
    planner = RetrievalPlanner(None)
    progress = ProgressPrinter(len(examples))

    args.out.mkdir(parents=True, exist_ok=True)
    retrieval_path = args.out / "retrieval_results.jsonl"
    results: list[RavResult] = []
    with retrieval_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            progress.start(index, example.financebench_id or example.question)
            adaptive_missing_facts = []
            if example.doc_name and not retriever.has_document(example.doc_name):
                plan = planner.plan(example.question)
                retrieved = []
                status = f"MISSING_DOCUMENT:{example.doc_name}"
            else:
                plan = planner.plan(example.question)
                retrieved = retriever.retrieve_with_plan(
                    example.question,
                    plan,
                    company=example.company,
                    doc_name=example.doc_name,
                    top_k=args.top_k,
                )
                if args.adaptive_retrieval and retrieved:
                    adaptive_missing_facts = _missing_plan_facts_from_evidence(plan, retrieved)
                    if adaptive_missing_facts:
                        retrieved = retriever.expand_for_facts(
                            example.question,
                            plan,
                            adaptive_missing_facts,
                            retrieved,
                            company=example.company,
                            doc_name=example.doc_name,
                            top_k=args.top_k,
                        )
                status = "RETRIEVED" if retrieved else "NO_EVIDENCE"
                if adaptive_missing_facts and retrieved:
                    status = "ADAPTIVE_RETRIEVED"
            chunk_ids = [chunk.chunk_id for chunk in retrieved]
            result = RavResult(
                financebench_id=example.financebench_id,
                question=example.question,
                answer="",
                final_status=status,
                first_pass_solver_status="RETRIEVAL_ONLY",
                gold_answer=example.answer,
                verified=False,
                abstained=False,
                retrieved_chunk_ids=chunk_ids,
                retrieval_plan=plan,
                metric=plan.metric,
            )
            results.append(result)
            handle.write(json.dumps({
                "financebench_id": example.financebench_id,
                "question": example.question,
                "gold_answer": example.answer,
                "company": example.company,
                "doc_name": example.doc_name,
                "status": status,
                "adaptive_missing_facts": [fact.name for fact in adaptive_missing_facts],
                "retrieval_plan": to_jsonable(plan),
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_chunks": [_retrieved_chunk_record(chunk) for chunk in retrieved],
            }, sort_keys=True) + "\n")
            progress.done(index, status)
    progress.finish()

    from verifiqa.eval.retrieval_metrics import compute_retrieval_metrics

    metrics = compute_retrieval_metrics(results, examples, k=args.top_k)
    (args.out / "retrieval_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out / "config.json").write_text(
        json.dumps({
            "pipeline": "retrieval_only",
            "n_loaded": loaded_count,
            "n_examples": len(examples),
            "numeric_only": numeric_only,
            "evidence_top_k": args.top_k,
            "first_stage_k": args.first_stage_k,
            "retriever_model": args.retriever_model,
            "retrieval_mode": args.retrieval_mode,
            "hybrid_rrf_k": args.hybrid_rrf_k,
            "reranker_model": args.reranker_model or "",
            "rerank": use_rerank,
            "rerank_planned_queries": args.rerank_planned_queries,
            "adaptive_retrieval": args.adaptive_retrieval,
            "corpus_stats": corpus_stats,
            "corpus_coverage": corpus_coverage,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if metrics.get("n", 0) > 0:
        print(
            f"Retrieval Hit@{metrics['k']}: {metrics['hit_rate_at_k']:.1%}  "
            f"Recall@{metrics['k']}: {metrics['mean_recall_at_k']:.1%}  "
            f"MRR@{metrics['k']}: {metrics['mrr_at_k']:.3f}  "
            f"(n={metrics['n']})",
            file=sys.stderr,
        )
    print(json.dumps({"results": len(results), "out": str(args.out)}, indent=2))


def _cmd_evidence(args) -> None:
    print(f"Loading corpus from {args.corpus}...", file=sys.stderr)
    chunks, corpus_stats = load_corpus(args.corpus)
    use_rerank = not args.no_rerank
    print(
        f"Building retriever ({args.retriever_model}, rerank={use_rerank}, "
        f"mode={args.retrieval_mode}, top_k={args.top_k}, "
        f"first_stage_k={args.first_stage_k})...",
        file=sys.stderr,
    )
    retriever_kwargs = {
        "model_name": args.retriever_model,
        "rerank": use_rerank,
        "first_stage_k": args.first_stage_k,
        "embedding_cache_dir": args.corpus,
        "rerank_planned_queries": args.rerank_planned_queries,
        "retrieval_mode": args.retrieval_mode,
        "hybrid_rrf_k": args.hybrid_rrf_k,
    }
    if args.reranker_model:
        retriever_kwargs["reranker_model"] = args.reranker_model
    retriever = EvidenceRetriever(chunks, **retriever_kwargs)
    plan = RetrievalPlanner(None).plan(args.question)

    status = "RETRIEVED"
    adaptive_missing_facts = []
    if args.doc_name and not retriever.has_document(args.doc_name):
        retrieved = []
        status = f"MISSING_DOCUMENT:{args.doc_name}"
    else:
        retrieved = retriever.retrieve_with_plan(
            args.question,
            plan,
            company=args.company,
            doc_name=args.doc_name,
            top_k=args.top_k,
        )
        if args.adaptive_retrieval and retrieved:
            adaptive_missing_facts = _missing_plan_facts_from_evidence(plan, retrieved)
            if adaptive_missing_facts:
                retrieved = retriever.expand_for_facts(
                    args.question,
                    plan,
                    adaptive_missing_facts,
                    retrieved,
                    company=args.company,
                    doc_name=args.doc_name,
                    top_k=args.top_k,
                )
        if not retrieved:
            status = "NO_EVIDENCE"
        elif adaptive_missing_facts:
            status = "ADAPTIVE_RETRIEVED"

    row = {
        "question": args.question,
        "company": args.company,
        "doc_name": args.doc_name,
        "status": status,
        "evidence_top_k": args.top_k,
        "first_stage_k": args.first_stage_k,
        "retriever_model": args.retriever_model,
        "retrieval_mode": args.retrieval_mode,
        "hybrid_rrf_k": args.hybrid_rrf_k,
        "reranker_model": args.reranker_model or "",
        "rerank": use_rerank,
        "rerank_planned_queries": args.rerank_planned_queries,
        "adaptive_retrieval": args.adaptive_retrieval,
        "adaptive_missing_facts": [fact.name for fact in adaptive_missing_facts],
        "corpus_stats": corpus_stats,
        "retrieval_plan": to_jsonable(plan),
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved],
        "evidence": [
            _question_evidence_chunk_record(rank, chunk)
            for rank, chunk in enumerate(retrieved, start=1)
        ],
        "answer_prompt_evidence": _format_answer_evidence(retrieved),
    }

    output = _format_evidence_command_output(row, args.format)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(json.dumps({"out": str(args.out), "status": status, "n_evidence": len(retrieved)}, indent=2))
    else:
        print(output)


def _missing_plan_facts_from_evidence(plan, chunks) -> list:
    if plan is None or not plan.facts:
        return []
    missing = []
    for fact in plan.facts:
        if not _evidence_covers_fact(fact, chunks):
            missing.append(fact)
    return missing


def _evidence_covers_fact(fact, chunks) -> bool:
    marker = f'"required_fact": "{fact.name}"'
    fact_plan = RetrievalPlan(facts=[fact])
    for chunk in chunks:
        if marker in chunk.text:
            return True
        try:
            if extract_structured_rows(chunk, fact_plan):
                return True
        except Exception:
            pass
    return False


def _retrieved_chunk_record(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_name": chunk.doc_name,
        "page": chunk.page,
        "source_type": chunk.source_type,
        "text_preview": _shorten(chunk.text, 1200),
    }


def _make_question_evidence_callback(path: Path, evidence_top_k: int):
    def callback(example, retrieval_plan, chunks, status: str) -> None:
        row = {
            "financebench_id": example.financebench_id,
            "question": example.question,
            "company": example.company,
            "doc_name": example.doc_name,
            "gold_answer": example.answer,
            "status": status,
            "evidence_top_k": evidence_top_k,
            "retrieval_plan": to_jsonable(retrieval_plan),
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "evidence": [
                _question_evidence_chunk_record(rank, chunk)
                for rank, chunk in enumerate(chunks, start=1)
            ],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return callback


def _question_evidence_chunk_record(rank: int, chunk) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "doc_name": chunk.doc_name,
        "page": chunk.page,
        "source_type": chunk.source_type,
        "company": chunk.company,
        "financebench_id": chunk.financebench_id,
        "text": chunk.text,
    }


class _NoopEvidenceRetriever:
    """Placeholder retriever for oracle runs whose evidence lives on each example."""

    def has_document(self, _doc_name: str) -> bool:
        return True

    def retrieve(self, *_args, **_kwargs) -> list:
        return []

    def retrieve_with_plan(self, *_args, **_kwargs) -> list:
        return []

    def expand_for_facts(self, _question, _plan, _target_facts, current_chunks, **_kwargs) -> list:
        return list(current_chunks)


def _format_evidence_command_output(row: dict, output_format: str) -> str:
    if output_format == "text":
        plan = json.dumps(row["retrieval_plan"], indent=2)
        return (
            f"Question:\n{row['question']}\n\n"
            f"Retrieval plan:\n{plan}\n\n"
            f"Evidence passed to answer LLM:\n{row['answer_prompt_evidence']}"
        )
    return json.dumps(row, indent=2, sort_keys=True)


def _cmd_build_corpus(args) -> None:
    from verifiqa.dataset import find_financebench_jsonl, load_financebench_jsonl
    print("Loading document links...", file=sys.stderr)
    doc_info_path = args.data / "data" / "financebench_document_information.jsonl"
    if not doc_info_path.exists():
        raise SystemExit(f"Document info not found: {doc_info_path}")
    doc_links = {}
    with doc_info_path.open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                doc_links[d["doc_name"]] = d["doc_link"]

    with args.examples.open() as f:
        examples = [json.loads(l) for l in f if l.strip()]
    doc_names = sorted(set(e["doc_name"] for e in examples if e.get("doc_name")))
    action = "Rebuilding cached corpus for" if args.cached_only else "Downloading/building"
    print(f"{action} {len(doc_names)} documents...", file=sys.stderr)
    supplemental_chunks = (
        evidence_page_supplements_from_records(examples)
        if args.include_benchmark_evidence_pages
        else []
    )
    if supplemental_chunks:
        print(
            f"Prepared {len(supplemental_chunks)} benchmark evidence page supplements.",
            file=sys.stderr,
        )

    chunks = build_corpus(
        doc_names=doc_names,
        doc_links=doc_links,
        out_dir=args.out,
        progress=lambda msg: print(msg, file=sys.stderr),
        cached_only=args.cached_only,
        download_timeout=args.download_timeout,
        download_retries=args.download_retries,
        supplemental_chunks=supplemental_chunks,
    )
    print(json.dumps({"n_docs": len(doc_names), "n_chunks": len(chunks), "out": str(args.out)}))


def _cmd_build_policy_data(args) -> None:
    report = write_policy_source_report(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        out=args.out,
        fetch=not args.offline,
        timeout=args.timeout,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _cmd_filter_dataset(args) -> None:
    print("Loading data...", file=sys.stderr)
    examples, _ = load_financebench(args.data)
    filtered = [e for e in examples if is_numerical_example(e)]
    print(f"Filtered to {len(filtered)} direct numerical examples from {len(examples)}.", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for example in filtered:
            handle.write(json.dumps(example.raw, ensure_ascii=False) + "\n")
    print(json.dumps({"n_total": len(examples), "n_filtered": len(filtered), "out": str(args.out)}))


def _corpus_coverage(examples, chunks) -> dict:
    available_docs = {chunk.doc_name for chunk in chunks if chunk.doc_name}
    required_docs = sorted({example.doc_name for example in examples if example.doc_name})
    missing_docs = [doc for doc in required_docs if doc not in available_docs]
    missing_doc_set = set(missing_docs)
    missing_examples = [
        example.financebench_id or example.question
        for example in examples
        if example.doc_name in missing_doc_set
    ]
    return {
        "n_examples": len(examples),
        "n_required_docs": len(required_docs),
        "n_available_docs": len(required_docs) - len(missing_docs),
        "n_missing_docs": len(missing_docs),
        "n_available_examples": len(examples) - len(missing_examples),
        "n_missing_examples": len(missing_examples),
        "missing_docs": missing_docs,
        "missing_examples": missing_examples,
    }


def _cmd_eval(args) -> None:
    results = []
    with (args.run / "results.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            results.append(RavResult(
                financebench_id=r.get("financebench_id", ""),
                question=r.get("question", ""),
                answer=r.get("answer", ""),
                final_status=r.get("final_status", ""),
                first_pass_solver_status=r.get("first_pass_solver_status", ""),
                gold_answer=r.get("gold_answer", ""),
                verified=bool(r.get("verified", False)),
                abstained=bool(r.get("abstained", False)),
                rule_id=r.get("rule_id", ""),
                metric=r.get("metric", ""),
            ))
    path = write_report(results, args.out)
    print(path)


def _cmd_download_sample(args) -> None:
    try:
        statuses = download_financebench_sample(
            args.out, force=args.force,
            progress=lambda msg: print(msg, file=sys.stderr),
        )
    except DownloadError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"files": statuses, "summary": summarize_sample(args.out)}, indent=2))


def _cmd_experiment(args) -> None:
    from verifiqa.experiments.registry import parse_methods
    from verifiqa.experiments.runner import run_experiment

    try:
        specs = parse_methods(args.methods)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    numeric_only = not args.include_non_numerical
    if args.finqa_dir is not None:
        from verifiqa.dataset_finqa import _is_numeric, load_finqa
        print(f"Loading FinQA {args.finqa_split} split from {args.finqa_dir}...", file=sys.stderr)
        examples = load_finqa(args.finqa_dir, split=args.finqa_split)
        loaded_count = len(examples)
        if numeric_only:
            examples = [example for example in examples if _is_numeric(example.answer)]
            print(f"Filtered to {len(examples)} numeric examples from {loaded_count}.", file=sys.stderr)
    else:
        if args.data is None:
            raise SystemExit("Provide --data /path/to/financebench or --finqa-dir /path/to/finqa")
        print("Loading data...", file=sys.stderr)
        examples, _ = load_financebench(args.data)
        loaded_count = len(examples)
        if numeric_only:
            examples = [example for example in examples if is_numerical_example(example)]
            print(f"Filtered to {len(examples)} direct numerical examples from {loaded_count}.", file=sys.stderr)
    if args.limit:
        examples = examples[:args.limit]

    evidence_modes = {spec.evidence_mode for spec in specs}
    chunks = []
    retriever = None
    corpus_stats = {}
    corpus_coverage = {}
    if evidence_modes & {"rag", "long_context"}:
        if args.corpus is None:
            raise SystemExit("RAG and long-context experiment methods require --corpus")
        print(f"Loading corpus from {args.corpus}...", file=sys.stderr)
        chunks, corpus_stats = load_corpus(args.corpus)
        corpus_coverage = _corpus_coverage(examples, chunks)
        print(
            f"Corpus: {corpus_stats['stored_records']} stored records -> "
            f"{corpus_stats['loaded_chunks']} retrieval chunks.",
            file=sys.stderr,
        )
    if "rag" in evidence_modes:
        use_rerank = not args.no_rerank
        print(
            f"Building retriever ({args.retriever_model}, rerank={use_rerank}, "
            f"mode={args.retrieval_mode}, top_k={args.top_k}, "
            f"first_stage_k={args.first_stage_k})...",
            file=sys.stderr,
        )
        retriever_kwargs = {
            "model_name": args.retriever_model,
            "rerank": use_rerank,
            "first_stage_k": args.first_stage_k,
            "embedding_cache_dir": args.corpus,
            "retrieval_mode": args.retrieval_mode,
            "hybrid_rrf_k": args.hybrid_rrf_k,
        }
        if args.reranker_model:
            retriever_kwargs["reranker_model"] = args.reranker_model
        retriever = EvidenceRetriever(chunks, **retriever_kwargs)

    llm = make_llm_client(mode=args.llm_mode, config={
        "base_url": args.vllm_base_url if args.llm_mode == "vllm" else "https://api.anthropic.com/v1",
        "model": args.model,
        "api_key_env": args.api_key_env,
        "max_tokens": 2048,
    })
    args.out.mkdir(parents=True, exist_ok=True)
    llm = LoggedLlmClient(
        llm,
        terminal=not args.no_llm_terminal_log,
        jsonl_path=args.out / "llm_calls.jsonl",
    )

    results = run_experiment(
        examples=examples,
        method_names=args.methods,
        llm_client=llm,
        out_dir=args.out,
        retriever=retriever,
        corpus_chunks=chunks,
        evidence_top_k=args.top_k,
        long_context_max_chars=args.long_context_max_chars,
        policy_semantic_check=args.policy_semantic_check,
        certificate_grounding_check=not args.no_certificate_grounding,
        config={
            "model": args.model,
            "llm_mode": args.llm_mode,
            "n_loaded": loaded_count,
            "numeric_only": numeric_only,
            "corpus": str(args.corpus) if args.corpus else "",
            "retriever_model": args.retriever_model,
            "retrieval_mode": args.retrieval_mode,
            "hybrid_rrf_k": args.hybrid_rrf_k,
            "reranker_model": args.reranker_model or "",
            "rerank": not args.no_rerank,
            "first_stage_k": args.first_stage_k,
            "policy_semantic_check": args.policy_semantic_check,
            "certificate_grounding_check": not args.no_certificate_grounding,
            "corpus_stats": corpus_stats,
            "corpus_coverage": corpus_coverage,
        },
    )
    print(json.dumps({"results": len(results), "out": str(args.out)}, indent=2))


def _cmd_summarize(args) -> None:
    from verifiqa.experiments.runner import write_summary

    path = write_summary(args.run, args.out)
    print(path)


def _make_baseline(name, evidence_retriever, llm, evidence_callback=None, oracle_retrieval: bool = False):
    if name == "vanilla_rag":
        return VanillaRagBaseline(
            evidence_retriever=evidence_retriever,
            llm_client=llm,
            evidence_callback=evidence_callback,
            oracle_retrieval=oracle_retrieval,
        )
    if name == "reflection":
        return ReflectionBaseline(
            evidence_retriever=evidence_retriever,
            llm_client=llm,
            evidence_callback=evidence_callback,
            oracle_retrieval=oracle_retrieval,
        )
    if name == "llm_judge":
        return LlmJudgeBaseline(
            evidence_retriever=evidence_retriever,
            llm_client=llm,
            evidence_callback=evidence_callback,
            oracle_retrieval=oracle_retrieval,
        )
    raise ValueError(f"Unknown baseline: {name}")


class ProgressPrinter:
    def __init__(self, total: int):
        self.total = max(total, 1)
        self.current = 0
        self.current_id = ""
        self.start_time = time.time()
        self.stage_name = "starting"

    def start(self, index: int, example_id: str) -> None:
        self.current = index
        self.current_id = _shorten(example_id, 42)
        self.stage_name = "starting"
        self._render()

    def stage(self, stage: str, details: dict) -> None:
        pairs = [f"{k}={v}" for k, v in sorted(details.items())]
        self.stage_name = stage + (" " + " ".join(pairs) if pairs else "")
        self._render()

    def done(self, index: int, status: str) -> None:
        self.stage_name = f"done status={status}"
        self._render()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def finish(self) -> None:
        elapsed = time.time() - self.start_time
        sys.stderr.write(f"completed {self.current}/{self.total} in {elapsed:.1f}s\n")
        sys.stderr.flush()

    def _render(self) -> None:
        width = 24
        filled = int(width * self.current / self.total)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.time() - self.start_time
        sys.stderr.write(
            f"\r\033[2K[{bar}] {self.current}/{self.total} "
            f"{self.current_id} | {self.stage_name} | {elapsed:.1f}s"
        )
        sys.stderr.flush()


def _shorten(value: str, limit: int) -> str:
    value = value.replace("\n", " ")
    return value if len(value) <= limit else value[:limit - 3] + "..."


if __name__ == "__main__":
    main()
