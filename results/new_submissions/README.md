# Frozen selected results

The 18 runs in this folder are the six selected models across XBRLFiling,
FinanceBench, and TADQA. `manifest.json` records the original locations and
unchanged raw-result hashes. `scripts/submission_runs.py` at the package root
selects these inputs for analysis.

Open [the plotting notebook](six_model_accuracy_heatmap.ipynb) for both the
heatmap and statement-family bars. [Figure exports](figures/) include editable
vector formats, PNG previews, source CSVs, and provenance.

The [main notebook](../../../notebooks/tatdqa_results.ipynb) contains the
three-dataset table and Overleaf code. Rebuild all outputs with
`python scripts/reproduce_submission.py` from the package root.

Historical `summary.json` values retain their original evaluation semantics;
the executed notebooks and main table are the authoritative current reported
metrics. Audit/replacement records preserve historical paths and counts.
New model launchers write to `results/reruns/`, outside these frozen inputs.
