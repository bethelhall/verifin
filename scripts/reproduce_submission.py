#!/usr/bin/env python3
"""Rebuild the submission figures and tables from frozen local results.

This entrypoint performs no model inference and makes no provider API calls.
Run it from any directory using the Python environment with the analysis
dependencies installed. Raw predictions are checked before and after execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DATASET_SIZES = {"xbrlfiling": 600, "financebench": 67, "tatdqa": 403}
NOTEBOOKS = (
    "results/new_submissions/six_model_accuracy_heatmap.ipynb",
    "notebooks/tatdqa_results.ipynb",
    "notebooks/tatdqa_result_metrics.ipynb",
)
CSV_OUTPUTS = (
    ("paper/tables/models_three_datasets.csv", ("Dataset", "Model")),
    ("results/new_submissions/figures/category_model_performance_1col_compact_data.csv",
     ("model", "metric")),
    ("results/new_submissions/figures/xbrl_error_profile_by_statement_family_data.csv",
     ("model", "family")),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes package: {relative}")
    return path


def check_inputs(root: Path = ROOT) -> dict[str, str]:
    """Verify the frozen 18-run manifest, row counts, unique IDs and catalog."""
    manifest = json.loads((root / "results/new_submissions/manifest.json").read_text())
    runs = manifest["runs"]
    if len(runs) != 18:
        raise ValueError(f"Expected 18 selected runs; found {len(runs)}")
    # Load the local catalog without resolving an installed or parent checkout.
    import runpy
    catalog = runpy.run_path(str(root / "scripts/submission_runs.py"))["RUNS"]
    selections = {(dataset, model): directory
                  for dataset, models in catalog.items() for model, directory in models.items()}
    if len(selections) != 18 or set(catalog) != set(DATASET_SIZES):
        raise ValueError("Run catalog does not contain the selected six models across three datasets")
    hashes: dict[str, str] = {}
    identities = {}
    seen = set()
    total = 0
    for run in runs:
        key = (run["dataset"], run["model_key"])
        if key in seen or key not in selections:
            raise ValueError(f"Duplicate or unknown manifest selection: {key}")
        seen.add(key)
        expected_relative = selections[key] + "/results.jsonl"
        relative = run["results"]
        if relative != expected_relative or not relative.startswith("results/new_submissions/"):
            raise ValueError(f"Result does not match the consolidated run catalog: {relative}")
        path = contained(root, relative)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != run["results_sha256"]:
            raise ValueError(f"Frozen result hash mismatch: {relative}")
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        n = DATASET_SIZES[run["dataset"]]
        ids = [row.get("id", row.get("financebench_id")) for row in records]
        if (len(records) != n or len(set(ids)) != n
                or any(not isinstance(value, str) or not value for value in ids)
                or run["rows"] != n or run["unique_ids"] != n):
            raise ValueError(f"Expected {n} unique question records: {relative}")
        prior = identities.setdefault(run["dataset"], set(ids))
        if set(ids) != prior:
            raise ValueError(f"Question IDs differ between selected models: {relative}")
        hashes[relative] = digest
        total += n
    if seen != set(selections) or total != 6420:
        raise ValueError("The package must contain exactly 18 runs and 6,420 result records")
    print("Verified 18 frozen runs, 6,420 records, and matching per-dataset question IDs.", flush=True)
    return hashes


def compare_csv(actual: Path, expected: Path, keys: tuple[str, ...]) -> int:
    """Compare all exported values against the audited full-precision snapshot."""
    def read(path):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        indexed = {tuple(row[key] for key in keys): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"Duplicate comparison keys in {path}")
        return fields, indexed
    fields, rows = read(actual)
    expected_fields, expected_rows = read(expected)
    if fields != expected_fields or set(rows) != set(expected_rows):
        raise ValueError(f"CSV columns or row identities changed: {actual}")
    for key, row in rows.items():
        for field in fields:
            value, reference = row[field], expected_rows[key][field]
            if value == reference:
                continue
            try:
                numeric, expected_numeric = float(value), float(reference)
            except (ValueError, TypeError):
                raise ValueError(f"CSV mismatch: {actual.name}, {key}, {field}: {value!r} != {reference!r}") from None
            if not math.isclose(numeric, expected_numeric, rel_tol=1e-11, abs_tol=1e-11):
                raise ValueError(f"CSV numeric mismatch: {actual.name}, {key}, {field}: {value} != {reference}")
    print(f"Matched {len(rows)} rows and every value: {actual.name}", flush=True)
    return len(rows)


def execute_notebook(root: Path, relative: str, timeout: int) -> None:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    path = contained(root, relative)
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    # Use this interpreter, including its installed analysis dependencies.
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, timeout=timeout, allow_errors=False,
                            resources={"metadata": {"path": str(root)}}, km=manager)
    print(f"Executing {relative}", flush=True)
    try:
        client.execute(cwd=str(root))
    finally:
        # An externally supplied manager is not owned/cleaned up by nbclient.
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    # A failed notebook never overwrites the checked-in rendered notebook.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", dir=path.parent,
                                     encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        nbformat.write(notebook, handle)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {relative}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate frozen input hashes and populations without executing notebooks")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Maximum seconds per notebook cell (default: 900)")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    before = check_inputs()
    if args.check_only:
        print("Input validation passed. Notebooks were not executed.")
        return 0
    # Make all plotting and import paths local to this portable package.
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "verifin-submission-matplotlib"))
    os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "src")))
    try:
        for relative in NOTEBOOKS:
            execute_notebook(ROOT, relative, args.timeout)
        for relative, keys in CSV_OUTPUTS:
            actual = ROOT / relative
            expected = ROOT / "tests/fixtures/expected_tables" / actual.name
            compare_csv(actual, expected, keys)
    finally:
        after = {relative: sha256(ROOT / relative) for relative in before}
        if after != before:
            changed = [relative for relative in before if before[relative] != after[relative]]
            raise RuntimeError(f"Raw result logs changed during reproduction: {changed}")
    print("Reproduction passed: all three notebooks executed, table/CIs and plot data match, raw results unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
