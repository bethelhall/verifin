# Code map for the submission

The package separates the saved experiment records from new model runs. The
authoritative inputs for the included figures and tables are the 18 directories
selected by `scripts/submission_runs.py` under `results/new_submissions/`.

## From a question to a saved verifier decision

| Stage | Source | Role |
| --- | --- | --- |
| Run orchestration | `src/verifiqa/agent/cli.py` | Loads questions, chooses the model client, generates claims, invokes verification, saves JSONL/SMT outputs, and applies evaluation scoring. |
| Answer generation | `src/verifiqa/generation/answer_generator.py` | Contains the generation prompt and parses its structured answer response. |
| Model transports | `src/verifiqa/generation/llm_client.py` | Implements Anthropic, OpenAI Responses, OpenAI-compatible local endpoints, and Codex CLI clients. |
| Question/evidence loading | `src/verifiqa/dataset.py` | Loads FinanceBench-shaped question records and their supplied evidence. All three selected datasets use this interface. |
| Verification orchestration | `src/verifiqa/agent/agent.py` | Builds the claim, grounds facts, authorizes a formula, and obtains the solver verdict or an abstention. |
| Claim/fact processing | `src/verifiqa/agent/claim_classifier.py`, `fact_extractor.py`, `table_fact_extractor.py`, and `source_*.py` | Interprets the requested quantity, selects source evidence, and binds facts to typed rows, periods, and units. |
| Formula authority | `src/verifiqa/policy/`, `src/verifiqa/formulas/`, `src/verifiqa/verification/generic_ops.py` | Loads policies and evaluates permitted metric/operation formulas. The JSON files in `policy/data/` are runtime resources. |
| XBRL evidence | `src/verifiqa/verification/xbrl_linkbase.py` and related `xbrl_*.py` modules | Reads filing artifacts and calculation relationships and checks scope/period bindings. |
| SMT and solver | `src/verifiqa/verification/smt_generator.py`, `z3_runner.py` | Renders the verification problem and runs Z3. The selected launchers request deterministic SMT rendering. |
| Model setup | `scripts/setup_fin_o1_8b.py`, `configs/*tatdqa.Modelfile`, `configs/fin_o1_8b_model.json` | Configures local models; the Fin-o1 setup validates the pinned GGUF checksum and model identity. |

The complete `src/verifiqa/` tree is included because these stages share
imports, policies, source parsers, and data resources. Copying only the CLI and
prompt files would not produce a runnable implementation.

## From saved results to reported numbers

| Stage | Source | Role |
| --- | --- | --- |
| Selected run catalog | `scripts/submission_runs.py` | Maps six model keys and three datasets to the consolidated result directories. |
| Historical paper policy | `scripts/paper_scoring.py` and `publication_assets_source/results/` policy/adjudication inputs | Computes the canonical historical XBRLFiling and FinanceBench outcome definitions. |
| Shared claim comparison | `src/verifiqa/eval/candidate_scoring.py` | Compares the final claimed quantity under the selected unit/tolerance policy. |
| TADQA evaluator | `src/verifiqa/eval/tatdqa_scoring.py`, `scripts/score_tatdqa.py`, `scripts/reviewer_tatdqa.py` | Validates records, applies the declared evaluation view, and produces per-question outcomes and aggregate statistics. |
| FinanceBench evaluator | `src/verifiqa/eval/financebench_scoring.py` | Applies the reviewed FinanceBench targets used for the new GPT-5.5 run, retaining unscored decisions explicitly. |
| Three-dataset table | `scripts/model_dataset_table.py` | Builds the six-model table and exact pointwise 95% confidence intervals, and exports CSV/Overleaf/provenance. |
| Main table notebook | `notebooks/tatdqa_results.ipynb` | Displays the three-dataset table and Overleaf code, plus the six-model TADQA results and plot. |
| Detailed TADQA notebook | `notebooks/tatdqa_result_metrics.ipynb` | Provides per-record evaluation and additional statistical exports. |
| Heatmap and bars | `results/new_submissions/six_model_accuracy_heatmap.ipynb` | Contains the active plotting code for both supplied XBRLFiling visualizations and their data/provenance exports. |
| Existing figure implementation | `scripts/plot_six_model_accuracy_heatmap.py` | Retains the standalone heatmap implementation; the active submission notebook binds the selected consolidated inputs explicitly. |

The heatmap measures generated-answer accuracy. The grouped bars measure true
rejects divided by the number of questions in each statement family. The main
table's accuracy measures verifier decisions on scored, non-abstained cases.
These are distinct denominators; the underlying per-question counts reconcile.

## Included generation/configuration source allowlist

Alongside the full `src/verifiqa/` source and resource tree and `pyproject.toml`,
the identified current generation helpers are:

```text
scripts/run_tatdqa_qwen25_7b.sh
scripts/run_tatdqa_llama31_8b.sh
scripts/run_tatdqa_qwen30b.sh
scripts/run_tatdqa_fin_o1_8b.sh
scripts/run_tatdqa_gpt55.sh
scripts/run_financebench_gpt55.sh
scripts/setup_fin_o1_8b.py
configs/qwen2_5_7b_tatdqa.Modelfile
configs/llama3_1_8b_tatdqa.Modelfile
configs/qwen3_30b_tatdqa.Modelfile
configs/fin_o1_8b_tatdqa.Modelfile
configs/fin_o1_8b_model.json
```

The package also includes the scorer/plot/table helpers and their declared data
dependencies. It excludes model weights, virtual environments, bytecode caches,
and credentials. Package copies of the six launchers direct new generations to
`results/reruns/` and support `RUN_OUTPUT_DIR`; this packaging change preserves
the frozen reported-result directories.

The legacy `scripts/run_model_sweep.py` is not the authoritative specification
for the selected 18 runs: its model list and default settings differ from the
selected submission configurations. The direct agent CLI is the shared
generation entry point. See [GENERATION.md](GENERATION.md) for supported
commands, full-corpus requirements, and the limits of historical provenance.
