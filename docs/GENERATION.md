# Generate new answers and verifier decisions

This package includes the current answer-generation and verification source,
the selected saved results, and the code used to score and visualize them.
Rebuilding figures from the saved results is an offline operation. Generating
new answers requires model access and, for TADQA and XBRLFiling, the full source
document bundles described below.

The source snapshot is **not an authenticated historical checkout for all 18
runs**. Most older result files do not record a source commit or complete model
configuration. New stochastic generation can also differ from the saved answers.
Use the frozen `results/new_submissions` inputs to reproduce the reported table
and figures; use `results/reruns` for new experiments.

## Environment

Run these commands from this package's root. Python 3.12 is the environment used
for the local submission checks.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[paper]'
```

The package declares `z3-solver`, `PyMuPDF`, NumPy, and sentence-transformers as
core dependencies. The `paper` extra adds notebook, plotting, and statistical
dependencies. The included model clients use HTTP through Python's standard
library. Local model commands additionally require a running Ollama service;
provider commands require the appropriate environment credential.

Credentials and model weights are not included. Set `OPENAI_API_KEY` for the
OpenAI client or `ANTHROPIC_API_KEY` for the Anthropic client in your own
environment. The source also supports loading a local root `.env`; do not
commit that file.

## Source corpora needed for generation

| Dataset | Question input | Additional runtime inputs |
| --- | --- | --- |
| FinanceBench, 67 questions | `data/numerical_questions.jsonl` | Supplied evidence text is in the JSONL. The documented run omits the optional unvalidated FinanceBench XBRL cache. |
| TADQA, 403 questions | `data/tatdqa_403_question_handoff_v1/dataset/financebench_xbrl_augmented.jsonl` | Restore the complete handoff's `documents/` and `xbrl/` directories, including its manifest and linked artifacts. Run inside the handoff directory because document paths are relative to it. |
| XBRLFiling, 600 questions | `data/multicompany_provable/data/provable_calcrequired.jsonl` | Restore `data/multicompany_provable/xbrl_artifacts/`, including the manifest and all referenced filings/linkbases. |

The package includes the inputs required for offline scoring and figure replay.
It does not include the full generation corpora: the original TADQA handoff is
about 622 MiB, and the XBRLFiling XBRL artifacts alone are about 639 MiB. A subset
of document files included to substantiate scoring reviews does not replace
these complete bundles. Obtain the original bundles and preserve their relative
paths before running those datasets. The copied data provenance and README
files identify the source assets where available.

Evaluator-only review annotations support grading; they are not model inputs.
The current CLI evaluates a completed claim after verification. It does not use
the reviewed gold target as answer-generation feedback.

## Supported launchers

Each launcher selects the model explicitly, uses generated answers and
deterministic SMT rendering, and resumes completed records in its output
directory. The package copies write new runs under `results/reruns` by default.
`RUN_OUTPUT_DIR` selects a different output directory; use an absolute path.
For a smoke test, use a fresh directory and append `--limit 1`.

| Launcher | Model configuration in the local source |
| --- | --- |
| `scripts/run_tatdqa_qwen25_7b.sh` | `qwen2.5:7b`, alias `qwen2.5-7b-tatdqa:latest`, 32,768-token context |
| `scripts/run_tatdqa_llama31_8b.sh` | `llama3.1:8b`, alias `llama3.1-8b-tatdqa:latest`, 32,768-token context |
| `scripts/run_tatdqa_qwen30b.sh` | `qwen3:30b-a3b-instruct-2507-q4_K_M`, alias `qwen3-30b-tatdqa:latest`, 32,768-token context |
| `scripts/run_tatdqa_fin_o1_8b.sh` | Qwen3-based `TheFinAI/Fin-o1-8B`, verified Q4_K_M conversion, 32,768-token context |
| `scripts/run_tatdqa_gpt55.sh` | Explicit `gpt-5.5`, OpenAI Responses transport, low reasoning effort, 8,192 output-token budget |
| `scripts/run_financebench_gpt55.sh` | Same explicit GPT-5.5 API configuration, FinanceBench text evidence |

These descriptions record the checked-in implementation; they do not guarantee
current model availability from an external provider.

```bash
# Start Ollama separately, then choose a local TADQA model.
bash scripts/run_tatdqa_qwen30b.sh
bash scripts/run_tatdqa_qwen25_7b.sh
bash scripts/run_tatdqa_llama31_8b.sh
bash scripts/run_tatdqa_fin_o1_8b.sh

# With OPENAI_API_KEY already configured:
bash scripts/run_tatdqa_gpt55.sh
bash scripts/run_financebench_gpt55.sh

# A separate smoke-test result, leaving both saved and full rerun results intact:
RUN_OUTPUT_DIR="$PWD/results/reruns/tatdqa_qwen3_smoke" \
  bash scripts/run_tatdqa_qwen30b.sh --limit 1
```

The local launchers pull missing base models and create their configured
aliases. Fin-o1 additionally runs `scripts/setup_fin_o1_8b.py`, which verifies
the GGUF checksum, architecture, parameter count, context, and a structured
readiness response. Its pinned conversion settings are in
`configs/fin_o1_8b_model.json`. Weights remain in Ollama's model storage.

The launchers perform a small readiness request before running benchmark
questions. Their `--help` output is available without generating answers.
Do not resume a directory after changing the model, prompt, verifier source, or
generation settings; use a new directory for the changed experiment.

## Direct CLI examples

The shared entry point is `python -m verifiqa.agent.cli run`. The following
commands describe new runs with this source snapshot. They do not claim to
recreate an undocumented historical environment.

Haiku on TADQA, with `ANTHROPIC_API_KEY` configured and the full handoff restored:

```bash
package_root="$PWD"
cd data/tatdqa_403_question_handoff_v1
PYTHONPATH="$package_root/src" "$package_root/.venv/bin/python" \
  -m verifiqa.agent.cli run \
  --data dataset/financebench_xbrl_augmented.jsonl \
  --xbrl-artifacts xbrl \
  --out "$package_root/results/reruns/tatdqa_haiku_4_5" \
  --source-plan-cache "$package_root/results/reruns/tatdqa_haiku_4_5/source_plans" \
  --llm-mode claude --model claude-haiku-4-5-20251001 \
  --claim-source generated_answer --deterministic-smt --workers 1 \
  --scoring-view source_reviewed --scoring-root "$package_root" \
  --resume --verbose
```

A local model on FinanceBench, from the package root, after creating the
specified Ollama alias:

```bash
PYTHONPATH=src .venv/bin/python -m verifiqa.agent.cli run \
  --data data/numerical_questions.jsonl \
  --out results/reruns/financebench_qwen3_30b \
  --source-plan-cache results/reruns/financebench_qwen3_30b/source_plans \
  --llm-mode vllm --model qwen3-30b-tatdqa:latest \
  --vllm-url http://127.0.0.1:11434/v1 \
  --claim-source generated_answer --deterministic-smt --workers 1 \
  --scoring-view source_reviewed --scoring-root "$PWD" --resume --verbose
```

Haiku on XBRLFiling, from the package root with the full XBRL artifacts restored:

```bash
PYTHONPATH=src .venv/bin/python -m verifiqa.agent.cli run \
  --data data/multicompany_provable/data/provable_calcrequired.jsonl \
  --xbrl-artifacts data/multicompany_provable/xbrl_artifacts \
  --out results/reruns/xbrlfiling_haiku_4_5 \
  --source-plan-cache results/reruns/xbrlfiling_haiku_4_5/source_plans \
  --llm-mode claude --model claude-haiku-4-5-20251001 \
  --claim-source generated_answer --deterministic-smt --workers 1 \
  --resume --verbose
```

For a different model, change both provider/model arguments and output paths.
The `vllm` mode accepts an OpenAI-compatible endpoint, including the Ollama URL
shown above. The `openai` mode accepts explicit
`--openai-reasoning-effort low --openai-max-output-tokens 8192`. The historical
XBRLFiling GPT-5.5 source directory references Codex, while the newer GPT-5.5
FinanceBench and TADQA manifests explicitly record the API transport. Treat an
API rerun of XBRLFiling as a new configuration.

## Historical provenance limits and repairs

All 18 selected logs identify the claim source as `generated_answer`.
The consolidation manifest records their original result directories and raw
result checksums. Only the selected GPT-5.5 FinanceBench and TADQA runs contain
a `run_manifest.json` and per-record `llm_run` metadata. The local TADQA runs
retain Ollama model descriptions, and the new TADQA Fin-o1 run additionally
retains checksum-verified model setup metadata.

The older Fin-o1 XBRLFiling and FinanceBench checkpoint identity is unresolved.
Their directory names do not establish a parameter count. The verified new
TADQA `TheFinAI/Fin-o1-8B` model must not be presented as proof of those older
runs' identity.

Historical logs also reflect different repair configurations. Saved `repair`
records occur in 294 Qwen3 XBRLFiling rows, 156 Fin-o1 XBRLFiling rows, 40 GPT-5.5
XBRLFiling rows, and 23 Fin-o1 FinanceBench rows. The current CLI supports
`--repair`, but these counts do not recover every original run flag or source
revision. Do not infer an identical generation configuration from the shared
scoring table.

The selected TADQA Haiku and Fin-o1 result sets include documented targeted
replacements. Their `result_replacements.json` files and associated saved audit
material preserve that history. The source-reviewed TADQA grading is post hoc;
the reported confidence intervals do not account for selection of reruns or
reviewed targets. Preserve these qualifications when reporting the results.
