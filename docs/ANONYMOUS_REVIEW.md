# Anonymous review artifact

This repository contains the implementation, frozen evaluation results,
reproducible notebooks, and figures for VeriFin. The main evaluation covers
six models across three datasets; the additional baseline comparison covers
four judges on FinanceBench and XBRLFiling.

## What reviewers can reproduce

Follow the [README](../README.md) to install the analysis dependencies and
rebuild the main tables, confidence intervals, heatmap, and statement-family
figures without API access or model inference. The
[baseline notebook](../notebooks/baseline_results.ipynb) also runs offline;
its [guide](BASELINE_MODEL_RUNS.md) describes the scoring and review records.
The notebooks retain their tables and embedded figures for preview on GitHub.

The [validation record](../reproducibility/validation.json) distinguishes
earlier experiment validation from checks of this anonymous copy. The
[file inventory](../reproducibility/package_inventory.json) records the
delivered files and their SHA-256 hashes.

## Anonymization

Author and lab identifiers, identifying local paths, and the original source
commit pointer have been removed from presentation and packaging metadata.
The [anonymization record](../reproducibility/anonymization.json) documents
the checks and changed files.
Historical command arguments use `<PROJECT_ROOT>` for the withheld local
prefix. Those arguments record the experiment; the placeholder is not a
literal executable path. Privacy tests use synthetic identifiers.

The repository starts with fresh anonymous Git history. Original repository
history, credentials, environment files, model weights, and local caches are
excluded. Public company names, filing evidence, upstream dataset and model
attributions, and third-party license notices are retained.

Scientific dataset records, generated answers, saved judge and verifier
decisions, and grading annotations are unchanged from the validated source
package. No model was rerun to prepare this copy. Changed documentation and
presentation files are identified separately from their historical source
hashes in the packaging provenance.

## Interpretation

The README prompt figure presents selected instructions and response schemas
as JSON-style blocks; it is a structured summary of current templates, not an exact
prompt archive for every historical run. Deterministic planning and table
lookup can bypass the corresponding prompts. Formal constraints and Z3
determine the verdict in deterministic-SMT mode. The
[figure script](../scripts/render_prompt_overview.py) checks supporting source
excerpts before rendering the structured summaries. The accompanying JSON
contains the same summaries in copyable form.

The baseline notebook includes 44 documented, post hoc, AI-assisted
full-answer correctness reviews. These are not independent human labels.
Reference interpretation caveats and the before-review comparison remain
available. The historical Haiku comparison, new four-judge comparison, and
reviewed main model table retain their respective scoring definitions.

XBRLFiling's original dataset contributions use CC BY-NC 4.0. Other data
and model notices retain their own terms; this review package does not assign
a new software license. See [license status](../LICENSE_STATUS.md).
