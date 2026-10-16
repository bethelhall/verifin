# Reproduction records

- [`expected/`](expected/) contains the audited full-precision CSV snapshots
  checked by `scripts/reproduce_submission.py`.
- [`validation.json`](validation.json) records the completed standalone
  notebook execution, regression tests, artifact comparisons, and scope limits.
- [`package_inventory.json`](package_inventory.json) identifies every delivered
  file by SHA-256 and size, excluding itself and temporary runtime caches.
- [`copied_sources.json`](copied_sources.json) maps copied files to the original
  working-tree paths and hashes. A source Git HEAD is contextual metadata, not
  proof that every packaged file matched that commit.
- [`analysis_dependencies.json`](analysis_dependencies.json) enumerates the
  notebook/scorer inputs and reviewed source assets.

All 18 raw prediction files are also bound by the original consolidation
manifest at `results/new_submissions/manifest.json`. The package includes three
small portability changes: launcher default outputs use `results/reruns`,
launcher help reflects the frozen paper selection, and the alternate standalone
heatmap script prefers its own root and creates its export directory.

Re-execution intentionally refreshes notebooks and derived provenance, so those
files may differ from the delivered inventory afterward. Raw prediction hashes
must remain unchanged. No new model inference was performed during packaging.
