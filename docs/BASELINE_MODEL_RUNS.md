# Four one-word baseline judges

All **eight runs are complete**: four judges across FinanceBench and XBRLFiling,
with **2,668 saved records**. The output audit found **2,655 exact one-word
replies**, all on the first attempt, and **13 abstentions for empty candidate
answers**. There were no malformed replies, retries, or provider errors.

Open [the baseline notebook](../notebooks/baseline_results.ipynb), then go to
**Four model judge comparisons** for the completed results. All 44 previously
unscored answers now have explicit correctness reviews, so every comparison
reports precision, accuracy, and coverage. See [Scoring and reviews](#scoring-and-reviews)
for the grading policy and its limits.

## Run or resume a judge

Run these commands from the repository root. Each command evaluates
FinanceBench (67 questions) and XBRLFiling (600 questions), using that model's
own saved answers under `results/new_submissions/`.

```bash
# Qwen2.5-7B
.venv/bin/python scripts/run_model_baselines.py --model qwen2.5-7b

# Qwen3-30B
.venv/bin/python scripts/run_model_baselines.py --model qwen3-30b

# Llama-3.1-8B
.venv/bin/python scripts/run_model_baselines.py --model llama-3.1-8b

# GPT-5.5
.venv/bin/python scripts/run_model_baselines.py --model gpt-5.5
```

Run the local models sequentially on the shared GPU. Their existing Ollama
aliases are `qwen2.5-7b-tatdqa:latest`, `qwen3-30b-tatdqa:latest`, and
`llama3.1-8b-tatdqa:latest`, at `http://127.0.0.1:11434/v1`.
GPT-5.5 uses the OpenAI Responses client and reads `OPENAI_API_KEY` from the
environment or the root `.env`.

## Judge output

Every judge receives this instruction in its system and user prompts:

> Output exactly CORRECT or INCORRECT. One uppercase word only. No explanation.
> No reasoning text. Nothing else.

The parser accepts only those two exact words after removing surrounding
whitespace. It rejects explanations, punctuation, quotes, Markdown, multiple
words, or a verdict embedded inside a longer response. One format-only retry
is allowed; if both replies are invalid, the row is recorded as `ABSTAIN`
with `invalid_judge_output`. Raw replies remain available for inspection.

The complete candidate answer is presented alongside the question and source
evidence. Gold answers and VeriFin verdicts are not sent to the judge. Numeric
answer parsing does not block a judge call. Empty answers or missing evidence
remain explicit abstentions. This rule changes judge handling only; the saved
candidate answers and verifier decisions are preserved.

## Outputs and resuming

Each model writes its own `fb_judge/` and `xbrl_judge/` directories beneath:

```text
results/new_submissions/baselines/
├── qwen2.5_7b/
├── qwen3_30b/
├── llama3.1_8b/
├── gpt5.5/
├── run_status.csv
├── protocol_audit.csv
├── comparison.csv
├── question_outcomes.csv
├── completed_answer_reviews.csv
├── comparison_before_full_answer_review.csv
├── unscored_before_full_answer_review.csv
├── unscored_review_queue.csv
├── comparison_provenance.json
└── figures/
    └── four_model_baseline_comparison.{png,pdf,svg}
```

Each dataset directory contains `results.jsonl` and a `run_manifest.json`
binding the candidate file, dataset, code, prompts, model settings, and
`full_answer_one_word_v1` protocol. Commands resume matching completed records.
A provider error is saved and stops the run; fix access and use a new
`--out-root` instead of silently including API failures as completed experiments.
Changing inputs, code, or settings requires a new output root.

Useful options:

```bash
# Validate both candidate pools without model calls or result writes.
.venv/bin/python scripts/run_model_baselines.py --model gpt-5.5 --preflight

# Run just one dataset.
.venv/bin/python scripts/run_model_baselines.py --model qwen3-30b --dataset financebench

# One-question smoke test in a separate output location.
.venv/bin/python scripts/run_model_baselines.py --model qwen2.5-7b \
  --dataset financebench --limit 1 --out-root results/baseline_smoke
```

The local output budget is 32 tokens. GPT-5.5 uses an 8,192-token total Responses
budget with low internal reasoning effort; its visible reply must still be
exactly one word. Output length alone does not enforce a word count—the strict
response parser does. These effective settings are recorded in the manifest.

## Notebook comparison

Open [notebooks/baseline_results.ipynb](../notebooks/baseline_results.ipynb)
and choose **Run All** using the analysis environment. This rebuilds the
comparison offline from saved files, with no model calls or API key. It checks
question IDs, candidate text, execution manifests, input hashes, and every
saved verdict before producing the tables. Each judge is compared with VeriFin
on its own model's saved candidate pool. These runs cover FinanceBench and
XBRLFiling; they do not include TADQA.

The notebook exports the following beneath
[`results/new_submissions/baselines/`](../results/new_submissions/baselines/):

| File | Contents |
| --- | --- |
| [run_status.csv](../results/new_submissions/baselines/run_status.csv) | Completion status for all eight runs |
| [protocol_audit.csv](../results/new_submissions/baselines/protocol_audit.csv) | Verdict counts, malformed replies, retries, provider errors, and missing candidates |
| [comparison.csv](../results/new_submissions/baselines/comparison.csv) | Judge versus VeriFin counts and metrics after the full-answer reviews; no unscored decisions |
| [question_outcomes.csv](../results/new_submissions/baselines/question_outcomes.csv) | Per-question decisions and scoring outcomes, linked to source rows |
| [completed_answer_reviews.csv](../results/new_submissions/baselines/completed_answer_reviews.csv) | The 44 completed candidate reviews, with correctness labels and rationales |
| [comparison_before_full_answer_review.csv](../results/new_submissions/baselines/comparison_before_full_answer_review.csv) | Original numeric-only comparison before the reviews |
| [unscored_before_full_answer_review.csv](../results/new_submissions/baselines/unscored_before_full_answer_review.csv) | The original 44-row grading queue, preserved for inspection |
| [unscored_review_queue.csv](../results/new_submissions/baselines/unscored_review_queue.csv) | Current grading queue, now empty |
| [comparison_provenance.json](../results/new_submissions/baselines/comparison_provenance.json) | Input hashes, scoring policy and review extension, execution protocols, and reference bindings |

The comparison figure is available as
[PNG](../results/new_submissions/baselines/figures/four_model_baseline_comparison.png),
[PDF](../results/new_submissions/baselines/figures/four_model_baseline_comparison.pdf),
and [SVG](../results/new_submissions/baselines/figures/four_model_baseline_comparison.svg).
After exporting the CSVs, regenerate the figure offline with:

```bash
.venv/bin/python scripts/plot_model_baselines.py
```

## Scoring and reviews

The original numeric scorer could not grade **44 decided full-answer candidates**:
31 from FinanceBench and 13 from XBRLFiling. All 44 now have source-bound reviews
in [baseline_full_answer_reviews.json](../notebooks/baseline_full_answer_reviews.json):
**25 correct and 19 incorrect**. These are **post hoc, AI-assisted reviews, not
independent human annotations**. Selection included every previously unscored
decision, without filtering on whether the judge accepted or rejected it.

The notebook explicitly enables `review_full_answers=True`. Its
`full_answer_correctness_review_v1` extension supplements the original paper
numeric policy only for those 44 candidates, using the same correctness label
for the judge and VeriFin. Other candidates retain the original scoring rule.
Reviews consider the complete answer, question, canonical reference, supplied
evidence, and existing adjudications. Fully specified arithmetic expressions
can be evaluated; redaction placeholders and symbolic formulas without a
numerical answer are incomplete. No missing number is supplied from the gold.
All source answers, judge replies, verifier decisions, and abstentions remain
unchanged. The before-review exports preserve the original counts and queue.

Precision is TA / (TA + FA); decision accuracy is (TA + TR) / (TA + FA + TR + FR).
Coverage uses all saved accept/reject decisions divided by the full dataset N.
The reviewed comparison has no unscored decisions and reports all three metrics.

**Canonical reference caveats remain explicit.** For example,
`financebench_id_01328` asks for restructuring costs directly outlined in the
income statement and permits zero if absent, while its reference uses the
411 million disclosed in a footnote. The review retains that benchmark target
and documents the scope conflict; marking the zero candidate incorrect means
it differs from the retained target, not that the literal reading is disproved.
The review document also records the other reference-interpretation caveats.

The main model table uses its separately documented reviewed metrics; its
GPT-5.5 FinanceBench results are not interchangeable with this baseline scoring
extension. For 67 GPT-5.5 FinanceBench VeriFin records, the source stores
the reference in `raw_answer` instead of `gold_answer`. The loader requires an
exact match with the canonical dataset and binds the reference in memory;
the provenance records all 67 bindings and the saved files remain unchanged.

The existing Haiku comparison remains an explicitly historical scalar-parser
run. Its protocol differs from the new full-answer runs, so it is not silently
treated as a matched new experiment. Its historical significance tests and
multiple-comparison correction are not reused for these four judges. No new
significance claims are made for the completed runs.
