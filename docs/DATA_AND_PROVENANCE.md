# Data and provenance

## Included analysis inputs

All 18 current prediction files are copied byte-for-byte into
`results/new_submissions/`, with 600, 67, and 403 unique question records per
model for XBRLFiling, FinanceBench, and TADQA respectively. The selected run
catalog is `scripts/submission_runs.py`; it has no fallback to older runs.

The package also contains the question/gold files, scoring policies,
adjudications, normalization inputs, and reviewed source assets required by the
four included notebooks. Scoring policies and manual adjudications are in
[`data/scoring/`](../data/scoring/); analysis dependencies are listed in
[`requirements-analysis.txt`](../requirements-analysis.txt).
The source subset is not the complete document collection needed for fresh
TADQA or XBRLFiling generation; see [GENERATION.md](GENERATION.md).

## Provenance records

- `results/new_submissions/manifest.json` is the original consolidation
  manifest. Its raw-result hashes are verified during every reproduction.
  Its original copy counts describe the earlier full run-directory copies,
  rather than the current file layout.
- `paper/provenance/` and `results/new_submissions/figures/*provenance.json`
  trace regenerated tables and plots to their local inputs and code.
- `tests/fixtures/expected_tables/` contains the audited numeric snapshots used to
  check offline reproduction, including the confidence-interval endpoints.
- `results/new_submissions/baselines/comparison_provenance.json` records the
  baseline scoring policy, source hashes, execution protocols, and completed
  full-answer reviews. The review document remains in
  `notebooks/baseline_full_answer_reviews.json`.

Saved predictions retain historical relative path strings; these strings are
not used to select current runs. Historical command metadata uses
`<PROJECT_ROOT>` in place of identifying local prefixes. Historical audit
records may also cite the original locations of scoring files; the current
scoring inputs are in `data/scoring/`. Model weights,
the original environments, ordinary run logs, per-question SMT dumps, and
source-plan caches are omitted from the package unless part of retained rerun
evidence. Saved predictions remain fully available for scoring and inspection.

## Targeted replacement history

The Haiku and Fin-o1 TADQA runs include targeted result replacements requested
after inspecting outcomes. Their `result_replacements.json` records and saved
audit assets are retained. Do not treat these as untouched prospective runs.

| Historical reference | Packaged evidence |
| --- | --- |
| `results/new_submission/tatdqa_fin_o1_8b/reruns/…` (singular folder) | `results/new_submissions/tatdqa_fin_o1_8b/reruns/…` |
| `artifacts/tatdqa_fair_value_fix/before_replacement/` | Same relative path in this package |
| `results/tatdqa_haiku_4_5_fair_value_retry_applied/results.jsonl` | Same relative path in this package; audit evidence only |

Historical audit counts can precede later scoring changes. Preserve those
records as history; use the executed current table for current counts. A small
archived FinanceBench snapshot also supports evaluator regression fixtures and
is not an additional selected model run.

## Interpretation

The table retains the recorded historical policy for XBRLFiling and the older
FinanceBench runs. TADQA and the new GPT-5.5 FinanceBench run use the documented
post hoc source-reviewed views. Two GPT-5.5 FinanceBench decisions are ungraded:
they contribute to coverage but not scored precision/accuracy denominators.
The source-reviewed targets and generation candidates remain separate.

Confidence intervals in the main table are pointwise exact 95% binomial
intervals conditional on the saved run. They do not account for source-review
selection, targeted reruns, repeated runs, filing clusters, or multiple
comparisons. The detailed TADQA notebook provides additional declared interval
calculations. Historical Fin-o1 model identity remains unresolved outside the
new TADQA run.
