"""Recompute the six-model, three-dataset paper table from saved predictions.

This module only scores and exports existing artifacts. Historical scoring and
post hoc source review for TADQA and the new GPT-5.5 FinanceBench run remain
distinguished.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from scipy.stats import beta

from scripts import paper_scoring, reviewer_tatdqa, submission_runs


POLICY_PATH = "data/scoring/scoring_policy.json"
ADJUDICATIONS_PATH = "data/scoring/manual_adjudications.json"
FINANCEBENCH_GPT55_RUN = submission_runs.RUNS["financebench"]["gpt-5.5"]
DATASETS = (("xbrlfiling", "XBRLFiling", 600),
            ("financebench", "FinanceBench", 67), ("tatdqa", "TADQA", 403))
# The Fin-o1 row keeps its historical identity qualifier; only TADQA identifies 8B.
MODEL_SPECS = (
    ("qwen2.5-7b", "Qwen2.5-7B", "Qwen2.5 7B"),
    ("llama-3.1-8b", "Llama-3.1-8B", "Llama 3.1 8B"),
    ("qwen3-30b", "Qwen3-30B", "Qwen3 30B"),
    ("fino1-identity-pending", "Fin-o1", "Fin-o1 8B"),
    ("claude-haiku-4.5", "Claude Haiku 4.5", "Claude Haiku 4.5"),
    ("gpt-5.5", "GPT-5.5", "GPT-5.5"),
)
MODELS = tuple(spec[1] for spec in MODEL_SPECS)
METRICS = ("TA", "FA", "FR", "Abs", "Prec.", "Acc.", "Cov.")
PROPORTION_METRICS = ("Prec.", "Acc.", "Cov.")
CI_COLUMNS = tuple(f"{metric} CI {bound}" for metric in PROPORTION_METRICS
                   for bound in ("low", "high"))
GRADING_COLUMNS = ("U", "U verified", "U rejected", "Graded accepts", "Graded decisions")
CONFIDENCE_LEVEL = 0.95
COLUMNS = ("Dataset", "Model", "N", "State", "Progress", "TA", "FA", "TR",
           "FR", "Abs", "Prec.", "Acc.", "Cov.", *CI_COLUMNS, *GRADING_COLUMNS)
PREAMBLE = r"""% Add once in the Overleaf document preamble.
\usepackage{booktabs}
\usepackage{array}
\usepackage[table]{xcolor}
\usepackage{siunitx}
\usepackage{graphicx}
\definecolor{gainhl}{HTML}{EDF5EE}
\definecolor{gridln}{HTML}{B8C1C8}
"""
CAPTION = (r"\textsc{VeriFin} performance across six answer-generator models and three datasets. "
           r"TA/FA/FR/Abs are counts; Prec./Acc./Cov.\ are percentages. Brackets below "
           r"estimates show pointwise two-sided 95\% Clopper--Pearson confidence intervals, "
           r"using denominators $\mathrm{TA}+\mathrm{FA}$, $n-\mathrm{Abs}-U$, and $n$, respectively. "
           r"Intervals assume independent binomial observations conditional on the saved run; "
           r"they do not account for filing clusters, repeated runs, or multiple comparisons. True rejects "
           r"are omitted and recoverable as $n-(\mathrm{TA}+\mathrm{FA}+\mathrm{FR}+\mathrm{Abs}+U)$, "
           r"where $U$ counts ungraded decisions and is zero unless noted. "
           r"XBRLFiling and the other FinanceBench rows use historical scoring; "
           r"GPT-5.5 on FinanceBench and all TADQA rows use post hoc source-reviewed scoring. "
           r"TADQA logs include targeted reruns selected after outcome inspection; "
           r"the intervals do not account for this selection. "
           r"Dashes denote unavailable, incomplete, or undefined metrics. "
           r"Fin-o1 uses TheFinAI/Fin-o1-8B for TADQA; its historical "
           r"XBRLFiling/FinanceBench checkpoint identities remain unconfirmed.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_row(dataset: str, model: str, n: int, state: str = "Missing",
               progress: str | None = None) -> dict[str, Any]:
    return {"Dataset": dataset, "Model": model, "N": n, "State": state,
            "Progress": progress or f"0/{n}",
            **{key: None for key in (*METRICS, "TR", *CI_COLUMNS, *GRADING_COLUMNS)}}


def exact_binomial_interval(successes: int, trials: int) -> tuple[float | None, float | None]:
    """Return a two-sided 95% Clopper--Pearson interval in percentage points.

    A zero denominator is undefined, rather than a zero-rate observation.
    Boundary outcomes retain nonzero uncertainty, including zero false accepts.
    """
    if (not isinstance(successes, int) or not isinstance(trials, int)
            or not 0 <= successes <= trials):
        raise ValueError("Binomial counts must be integers with 0 <= successes <= trials")
    if trials == 0:
        return None, None
    tail = (1 - CONFIDENCE_LEVEL) / 2
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - tail, successes + 1, trials - successes))
    return 100 * low, 100 * high


def _complete_row(dataset: str, model: str, n: int, counts: dict[str, int], *,
                  unscored_verified: int = 0, unscored_rejected: int = 0) -> dict[str, Any]:
    unknown = (unscored_verified, unscored_rejected)
    if any(not isinstance(value, int) or value < 0 for value in unknown):
        raise ValueError(f"{dataset}/{model}: invalid ungraded decision counts")
    unscored = sum(unknown)
    if any(not isinstance(counts.get(key), int) or counts[key] < 0
           for key in paper_scoring.OUTCOMES) or sum(counts.values()) + unscored != n:
        raise ValueError(f"{dataset}/{model}: outcomes do not reconcile with N={n}")
    accepted = counts["TA"] + counts["FA"]
    decided = n - counts["AB"]
    graded_decisions = decided - unscored
    proportions = {"Prec.": (counts["TA"], accepted),
                   "Acc.": (counts["TA"] + counts["TR"], graded_decisions),
                   "Cov.": (decided, n)}
    intervals = {f"{metric} CI {bound}": value
                 for metric, (successes, trials) in proportions.items()
                 for bound, value in zip(("low", "high"), exact_binomial_interval(successes, trials))}
    return {"Dataset": dataset, "Model": model, "N": n, "State": "Complete",
            "Progress": f"{n}/{n}", **{key: counts[key] for key in ("TA", "FA", "TR", "FR")},
            "Abs": counts["AB"], "Prec.": 100 * counts["TA"] / accepted if accepted else None,
            "Acc.": 100 * (counts["TA"] + counts["TR"]) / graded_decisions if graded_decisions else None,
            "Cov.": 100 * decided / n, **intervals,
            "U": unscored, "U verified": unscored_verified, "U rejected": unscored_rejected,
            "Graded accepts": accepted, "Graded decisions": graded_decisions}


def _historical_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, sources, runs = [], {}, {}
    missing_inputs = [relative for relative in (POLICY_PATH, ADJUDICATIONS_PATH)
                      if not (root / relative).is_file()]
    policy = adjudications = None
    if not missing_inputs:
        policy = paper_scoring.load_policy(root / POLICY_PATH)
        adjudications = paper_scoring.load_manual_adjudications(root / ADJUDICATIONS_PATH)
        sources.update({relative: _sha256(root / relative)
                        for relative in (POLICY_PATH, ADJUDICATIONS_PATH)})
    for dataset, label, n in DATASETS[:2]:
        expected_ids = None
        if policy is not None:
            dataset_spec = policy["datasets"][dataset]
            if dataset_spec["denominator"] != n:
                raise ValueError(f"{label}: policy denominator differs from table header")
            relative = dataset_spec["source_path"]
            if (root / relative).is_file():
                sources[relative] = _sha256(root / relative)
                expected_ids = paper_scoring.validate_dataset_ids(
                    paper_scoring.load_jsonl(root / relative), dataset, dataset_spec)
        for key, model, _ in MODEL_SPECS:
            if dataset == "financebench" and key == "gpt-5.5":
                continue  # The fresh API run uses the shared source-reviewed scorer.
            row = _empty_row(label, model, n)
            entry: dict[str, Any] = {"model_key": key, "state": "Missing", "results": None}
            if policy is None or expected_ids is None:
                entry["reason"] = "Required scoring inputs or benchmark dataset are unavailable."
            else:
                selection = policy["canonical_runs"][dataset]["answer_models"].get(key, {})
                entry["historical_selection"] = selection
                directory = submission_runs.RUNS[dataset][key]
                relative = f"{directory}/results.jsonl"
                if not (root / relative).is_file():
                    entry["reason"] = "Consolidated saved prediction log is unavailable."
                    entry["results"] = {"path": relative, "sha256": None}
                else:
                    digest = _sha256(root / relative)
                    records = paper_scoring.load_jsonl(root / relative)
                    ids = [paper_scoring.record_id(record) for record in records]
                    if len(ids) != len(set(ids)) or set(ids) - expected_ids:
                        paper_scoring.validate_run_ids(records, expected_ids, dataset, key)
                    entry["results"] = {"path": relative, "sha256": digest}
                    sources[relative] = digest
                    if len(records) < n:
                        row = _empty_row(label, model, n, "Pending", f"{len(records)}/{n}")
                        entry["reason"] = "Incomplete run; no benchmark metrics calculated."
                    else:
                        paper_scoring.validate_run_ids(records, expected_ids, dataset, key)
                        if any("baseline" in record for record in records):
                            raise ValueError(f"{dataset}/{key}: baseline rows cannot supply model scores")
                        counts = paper_scoring.score_rows(records, dataset, policy, adjudications)
                        row = _complete_row(label, model, n, counts)
                        entry["counts"] = counts
            entry.update(state=row["State"], progress=row["Progress"])
            if key == "fino1-identity-pending":
                entry["identity_note"] = "Checkpoint identity and parameter count are unrecorded."
            rows.append(row)
            runs[f"{dataset}/{key}"] = entry
    for relative, digest in sources.items():
        if _sha256(root / relative) != digest:
            raise ValueError(f"Source changed during table scoring: {relative}; rerun the notebook")
    return rows, {"scoring_view": "historical_magnitude_any_token",
                  "policy": POLICY_PATH, "adjudications": ADJUDICATIONS_PATH,
                  "sources": sources, "runs": runs, "missing_inputs": missing_inputs}


def _financebench_gpt55_row(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score only the identified, complete API run; never infer true rejects."""
    from verifiqa.eval.financebench_scoring import FinanceBenchScorer

    relative = f"{FINANCEBENCH_GPT55_RUN}/results.jsonl"
    manifest_relative = f"{FINANCEBENCH_GPT55_RUN}/run_manifest.json"
    row = _empty_row("FinanceBench", "GPT-5.5", 67)
    entry: dict[str, Any] = {
        "scoring_view": "source_reviewed", "state": row["State"],
        "progress": row["Progress"], "results": {"path": relative, "sha256": None},
        "sources": {},
    }
    if not (root / relative).is_file():
        entry["reason"] = "Saved GPT-5.5 FinanceBench prediction log is unavailable."
        return row, entry

    sources = {relative: _sha256(root / relative)}
    records = paper_scoring.load_jsonl(root / relative)
    entry["results"]["sha256"] = sources[relative]
    if records:
        if not (root / manifest_relative).is_file():
            raise ValueError("FinanceBench/GPT-5.5: run manifest is missing")
        sources[manifest_relative] = _sha256(root / manifest_relative)
        manifest = json.loads((root / manifest_relative).read_text())
        metadata = manifest.get("llm_run", {})
        settings = manifest.get("settings", {})
        if (metadata.get("model") != "gpt-5.5" or metadata.get("mode") != "openai"
                or settings.get("model") != "gpt-5.5" or settings.get("mode") != "openai"
                or settings.get("claim_source") != "generated_answer"
                or settings.get("deterministic_smt") is not True
                or any(record.get("llm_run") != metadata for record in records)):
            raise ValueError("FinanceBench/GPT-5.5: model or run identity mismatch")
        if any("baseline" in record for record in records):
            raise ValueError("FinanceBench/GPT-5.5: baseline rows cannot supply model scores")
    scorer = FinanceBenchScorer.from_root(root)
    if len(scorer.dataset) != 67:
        raise ValueError("FinanceBench/GPT-5.5: dataset denominator differs from table header")
    sources.update({scorer.provenance["review_path"]: scorer.provenance["review_sha256"],
                    scorer.provenance["dataset_path"]: scorer.provenance["dataset_sha256"]})
    ids = [paper_scoring.record_id(record) for record in records]
    if len(ids) != len(set(ids)) or set(ids) - set(scorer.dataset):
        raise ValueError("FinanceBench/GPT-5.5: duplicate or unknown question IDs")
    # Validate canonical questions/references even while the run is incomplete.
    scorer.score_rows(records, view="source_reviewed")
    if len(records) < 67:
        row = _empty_row("FinanceBench", "GPT-5.5", 67, "Pending", f"{len(records)}/67")
        entry["reason"] = "Incomplete run; no benchmark metrics calculated."
    else:
        summary = scorer.summarize_rows(records, view="source_reviewed")
        entry["summary"] = summary
        counts = {key: summary[key] for key in paper_scoring.OUTCOMES}
        row = _complete_row("FinanceBench", "GPT-5.5", 67, counts,
                            unscored_verified=summary["unscored_verified"],
                            unscored_rejected=summary["unscored_rejected"])
        entry["counts"] = counts
        if summary["unscored_decisions"]:
            entry["reason"] = (f"{summary['unscored_decisions']} decision(s) lack a gradeable reference; "
                               "precision and accuracy condition on graded decisions, while coverage "
                               "includes every decision. Ungraded decisions remain U, not true rejects.")
    for source, digest in sources.items():
        if _sha256(root / source) != digest:
            raise ValueError(f"Source changed during table scoring: {source}; rerun the notebook")
    entry.update(state=row["State"], progress=row["Progress"], sources=sources,
                 scoring=scorer.provenance)
    return row, entry


def load_table(root: Path, *, tatdqa_report: tuple[list[dict[str, Any]], dict[str, Any]] | None = None
               ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return 18 long-form rows and provenance; never score incomplete runs.

    An existing ``reviewer_tatdqa.load_latest`` result may be passed to use the
    exact same run snapshot as other cells in a notebook.
    """
    root = Path(root).resolve()
    rows, historical = _historical_rows(root)
    financebench_row, financebench_provenance = _financebench_gpt55_row(root)
    rows.append(financebench_row)
    if tatdqa_report is None and (root / reviewer_tatdqa.INPUTS["dataset"]).is_file():
        tatdqa_report = reviewer_tatdqa.load_latest(root, allow_incomplete=True)
    selected_tadqa = {name for _, _, name in MODEL_SPECS if name is not None}
    tadqa_rows, tadqa_provenance = tatdqa_report or ([], {})
    if tadqa_provenance and tadqa_provenance.get("scoring_view") != "source_reviewed":
        raise ValueError("The table requires TADQA source-reviewed scoring")
    indexed = {}
    for row in tadqa_rows:
        if row["Model"] in selected_tadqa:
            if row["Model"] in indexed:
                raise ValueError(f"Duplicate TADQA model: {row['Model']}")
            indexed[row["Model"]] = row
    for _, model, tadqa_model in MODEL_SPECS:
        original = indexed.get(tadqa_model)
        if original is None:
            row = _empty_row("TADQA", model, reviewer_tatdqa.N)
        elif original["State"] == "Complete":
            if original["N"] != reviewer_tatdqa.N:
                raise ValueError(f"TADQA/{model}: incomplete benchmark denominator")
            row = _complete_row("TADQA", model, reviewer_tatdqa.N,
                                {key: original[key] for key in paper_scoring.OUTCOMES})
        else:
            row = _empty_row("TADQA", model, reviewer_tatdqa.N,
                             original["State"], original["Progress"])
        rows.append(row)
    selected_provenance = dict(tadqa_provenance)
    for field in ("runs", "result_replacements"):
        if field in selected_provenance:
            selected_provenance[field] = {model: value for model, value in selected_provenance[field].items()
                                         if model in selected_tadqa}
    model_order = {name: index for index, name in enumerate(MODELS)}
    dataset_order = {label: index for index, (_, label, _) in enumerate(DATASETS)}
    rows.sort(key=lambda row: (model_order[row["Model"]], dataset_order[row["Dataset"]]))
    provenance = {
        "schema_version": 3, "models": list(MODELS), "historical": historical,
        "submission_runs": submission_runs.RUNS,
        "tatdqa": selected_provenance, "financebench_gpt55": financebench_provenance,
        "qualification": "Historical XBRLFiling and other FinanceBench scoring; post hoc "
                         "source-reviewed GPT-5.5 FinanceBench and TADQA. Missing and pending runs "
                         "have no table metrics. Completed runs with ungraded decisions show graded "
                         "precision/accuracy and full decision coverage, with U disclosed. Fin-o1's TADQA run identifies "
                         "TheFinAI/Fin-o1-8B; its historical XBRLFiling/FinanceBench identities remain unconfirmed.",
        "model_identity_notes": {
            "Fin-o1": {"tatdqa_checkpoint": "TheFinAI/Fin-o1-8B",
                       "historical_checkpoint": None,
                       "same_checkpoint_across_datasets_confirmed": False},
        },
        "metric_definitions": {"Prec.": "100*TA/(TA+FA)", "Acc.": "100*(TA+TR)/(N-Abs-U)",
                               "Cov.": "100*(N-Abs)/N"},
        "ungraded_decisions": {
            "columns": list(GRADING_COLUMNS),
            "definition": "U = U verified + U rejected; complete-run counts sum to N only after adding U.",
            "rates": "Precision excludes ungraded accepts; accuracy excludes ungraded decisions; "
                     "coverage includes all decided questions.",
            "runs": [{key: row[key] for key in ("Dataset", "Model", *GRADING_COLUMNS)}
                     for row in rows if row.get("U")],
        },
        "confidence_intervals": {
            "method": "Clopper-Pearson exact binomial", "level": CONFIDENCE_LEVEL,
            "sidedness": "two-sided", "scope": "pointwise", "units": "percent",
            "numerators": {"Prec.": "TA", "Acc.": "TA+TR", "Cov.": "N-Abs"},
            "denominators": {"Prec.": "TA+FA", "Acc.": "N-Abs-U", "Cov.": "N"},
            "columns": list(CI_COLUMNS), "zero_denominator": "undefined; no interval",
            "assumptions": "Independent binomial observations conditional on each saved run; "
                           "no filing-cluster correction, repeated-run uncertainty, or "
                           "multiple-comparison adjustment. These intervals do not quantify "
                           "uncertainty in scoring or post hoc adjudications. Precision and accuracy "
                           "intervals condition on gradeable outcomes; ungraded outcomes are excluded.",
        },
        "scope": "Saved predictions only; no model inference or changes to source logs.",
    }
    return rows, provenance


def grading_notes(rows: list[dict[str, Any]]) -> list[str]:
    """Explain completed runs whose conditional metrics omit unresolved grades."""
    notes = []
    for row in rows:
        if row["State"] != "Complete" or not row.get("U"):
            continue
        note = (f"{row['Dataset']}/{row['Model']}: U={row['U']} ungraded decisions "
                f"({row['U verified']} accepted, {row['U rejected']} rejected). "
                f"Precision uses {row['Graded accepts']} graded accepts; accuracy uses "
                f"{row['Graded decisions']} graded decisions; coverage uses all {row['N']} questions.")
        if row["U verified"]:
            all_accepts = row["Graded accepts"] + row["U verified"]
            low = 100 * row["TA"] / all_accepts
            high = 100 * (row["TA"] + row["U verified"]) / all_accepts
            note += (f" Overall precision is bounded by {low:.1f}%-{high:.1f}% depending on "
                     "ungraded accepts; these are identification bounds, not confidence intervals.")
        notes.append(note)
    return notes


def render_latex(rows: list[dict[str, Any]]) -> str:
    """Render the requested grouped booktabs/siunitx table, fitting text width."""
    indexed = {(row["Model"], row["Dataset"]): row for row in rows}
    expected = {(model, label) for model in MODELS for _, label, _ in DATASETS}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Expected exactly six models across three datasets")
    blocks = []
    for _, dataset, _ in DATASETS:
        formats = []
        for index, metric in enumerate(METRICS):
            decimals = 0 if index < 4 else 1
            values = [row[metric] for row in rows if row["Dataset"] == dataset
                      and row["State"] == "Complete" and row[metric] is not None]
            # Derive width after rounding, so e.g. 99.96 reserves room for 100.0.
            digits = max((len(f"{value:.{decimals}f}".split(".")[0]) for value in values), default=1)
            prefix = r">{\columncolor{gainhl}}" if metric == "FA" else ""
            formats.append(prefix + f"S[table-format={digits}.{decimals}]")
        blocks.append(" ".join(formats))
    lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
             r"\renewcommand{\arraystretch}{1.3}", r"\setlength{\aboverulesep}{0pt}",
             r"\setlength{\belowrulesep}{0pt}", r"\sisetup{detect-weight=true, mode=text}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}l",
             blocks[0], r"!{\color{gridln}\vrule width 0.6pt}", blocks[1],
             r"!{\color{gridln}\vrule width 0.6pt}", blocks[2] + r"@{}}", r"\toprule"]
    lines.append("& " + " & ".join(r"\multicolumn{7}{c}{\textbf{" + label + r"} ($n{=}"
                                  + str(n) + "$)}" for _, label, n in DATASETS) + r" \\")
    lines.extend([r"\cmidrule(lr){2-8}\cmidrule(lr){9-15}\cmidrule(l){16-22}",
                  r"\textbf{Claim model}"])
    header = r"& {TA} & {FA$\,\downarrow$} & {FR} & {Abs} & {Prec.} & {Acc.} & {Cov.}"
    lines.extend([header, header, header + r" \\", r"\midrule"])
    for model in MODELS:
        label = model.replace("Haiku ", "Haiku~")
        if any(row.get("U") for row in rows if row["Model"] == model and row["State"] == "Complete"):
            label += r"$^{\dagger}$"
        groups, interval_groups = [], []
        for _, dataset, n in DATASETS:
            row = indexed[model, dataset]
            if row["N"] != n:
                raise ValueError(f"{dataset}/{model}: table header denominator mismatch")
            if row["State"] != "Complete":
                values = ["{--}"] * len(METRICS)
            else:
                values = ["{--}" if row[key] is None else
                          f"{row[key]:.0f}" if index < 4 else f"{row[key]:.1f}"
                          for index, key in enumerate(METRICS)]
            groups.append(" & ".join(values))
            intervals = []
            for metric in PROPORTION_METRICS:
                low, high = (row[f"{metric} CI {bound}"] for bound in ("low", "high"))
                intervals.append(r"{\scriptsize " + f"[{low:.1f}, {high:.1f}]" + "}"
                                 if row["State"] == "Complete" and low is not None and high is not None
                                 else "{--}")
            interval_groups.append(" & ".join(["", "", "", "", *intervals]))
        lines.append(label + " & " + "\n                 & ".join(groups) + r" \\")
        lines.append(" & " + "\n                 & ".join(interval_groups) + r" \\")
    notes = " ".join(r"$^{\dagger}$ " + note.replace("%", r"\%") for note in grading_notes(rows))
    caption = CAPTION + (" " + notes if notes else "")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\caption{" + caption + "}",
                  r"\label{tab:models-three-datasets}", r"\end{table*}", ""])
    return "\n".join(lines)


def export(root: Path, rows: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, str]:
    """Export copyable Overleaf code, numeric table, and input provenance."""
    root = Path(root).resolve()
    latex = render_latex(rows)
    outputs = {"latex": "paper/tables/models_three_datasets.tex",
               "preamble": "paper/tables/models_three_datasets_preamble.tex",
               "table": "paper/tables/models_three_datasets.csv",
               "provenance": "paper/provenance/models_three_datasets.json"}
    for relative in outputs.values():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / outputs["latex"]).write_text(latex)
    (root / outputs["preamble"]).write_text(PREAMBLE)
    with (root / outputs["table"]).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    record = {**provenance, "outputs": {
        key: {"path": path, "sha256": _sha256(root / path)}
        for key, path in outputs.items() if key != "provenance"}}
    implementation = root / "scripts/model_dataset_table.py"
    if implementation.is_file():
        record["implementation"] = {"path": "scripts/model_dataset_table.py",
                                     "sha256": _sha256(implementation)}
    paths_implementation = root / "scripts/submission_runs.py"
    if paths_implementation.is_file():
        record["submission_paths_implementation"] = {
            "path": "scripts/submission_runs.py", "sha256": _sha256(paths_implementation)}
    (root / outputs["provenance"]).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return outputs
