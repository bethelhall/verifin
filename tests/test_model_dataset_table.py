"""The paper table must preserve model identities and complete denominators."""
import csv
import json
from pathlib import Path
import shutil

import pytest
from scipy.stats import binom

from scripts import model_dataset_table as table, submission_runs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture_root(tmp_path):
    for relative in (table.POLICY_PATH, table.ADJUDICATIONS_PATH):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    policy = json.loads((tmp_path / table.POLICY_PATH).read_text())
    for dataset in policy["datasets"].values():
        target = tmp_path / dataset["source_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / dataset["source_path"], target)
    dataset = policy["datasets"]["xbrlfiling"]
    questions = [json.loads(line) for line in (tmp_path / dataset["source_path"]).read_text().splitlines()]
    records = []
    for index, question in enumerate(questions):
        if index < 5:
            status, value = "VERIFIED", 10
        elif index < 7:
            status, value = "VERIFIED", 20
        elif index < 10:
            status, value = "VIOLATED", 10
        elif index < 14:
            status, value = "VIOLATED", 20
        else:
            status, value = "ABSTAIN", None
        records.append({"id": question[dataset["id_field"]], "gold_answer": "10",
                        "claimed_value": value, "status": status})
    relative = submission_runs.RUNS["xbrlfiling"]["qwen2.5-7b"]
    result_path = tmp_path / relative / "results.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return tmp_path, result_path, records


def tadqa_snapshot():
    rows = []
    for _, _, model in table.MODEL_SPECS:
        if model is not None:
            rows.append({"Model": model, "State": "Complete", "Progress": "403/403", "N": 403,
                         "TA": 2, "FA": 1, "TR": 3, "FR": 4, "AB": 393})
    rows.append({"Model": "GPT-6.1 Sol", "State": "Complete", "N": 403})
    return rows, {"scoring_view": "source_reviewed",
                  "runs": {row["Model"]: {"state": "Complete"} for row in rows}}


def test_counts_rates_and_missing_logs_preserve_full_denominators(fixture_root):
    root, path, _ = fixture_root
    before = path.read_bytes()
    rows, provenance = table.load_table(root, tatdqa_report=tadqa_snapshot())
    indexed = {(row["Model"], row["Dataset"]): row for row in rows}
    xbrl = indexed["Qwen2.5-7B", "XBRLFiling"]
    assert [xbrl[key] for key in ("N", "TA", "FA", "TR", "FR", "Abs")] == [600, 5, 2, 4, 3, 586]
    assert xbrl["Prec."] == pytest.approx(100 * 5 / 7)
    assert xbrl["Acc."] == pytest.approx(100 * 9 / 14)
    assert xbrl["Cov."] == pytest.approx(100 * 14 / 600)
    for metric, successes, trials in (("Prec.", 5, 7), ("Acc.", 9, 14), ("Cov.", 14, 600)):
        # Invert binomial tails independently of the beta-quantile implementation.
        assert binom.sf(successes - 1, trials, xbrl[f"{metric} CI low"] / 100) == pytest.approx(.025)
        assert binom.cdf(successes, trials, xbrl[f"{metric} CI high"] / 100) == pytest.approx(.025)
    missing = indexed["GPT-5.5", "FinanceBench"]
    assert missing["N"] == 67 and missing["State"] == "Missing"
    assert all(missing[key] is None for key in (*table.METRICS, *table.CI_COLUMNS))
    assert indexed["Fin-o1", "TADQA"]["State"] == "Complete"
    assert indexed["Fin-o1", "TADQA"]["TA"] == 2
    assert provenance["model_identity_notes"]["Fin-o1"] == {
        "tatdqa_checkpoint": "TheFinAI/Fin-o1-8B", "historical_checkpoint": None,
        "same_checkpoint_across_datasets_confirmed": False,
    }
    assert len(rows) == 18 and set(row["Model"] for row in rows) == set(table.MODELS)
    assert "GPT-6.1 Sol" not in provenance["tatdqa"]["runs"]
    assert path.read_bytes() == before


def test_valid_partial_run_has_no_partial_metrics(fixture_root):
    root, path, records = fixture_root
    path.write_text("".join(json.dumps(row) + "\n" for row in records[:7]))
    rows, _ = table.load_table(root)
    row = next(row for row in rows if row["Dataset"] == "XBRLFiling" and row["Model"] == "Qwen2.5-7B")
    assert row["State"] == "Pending" and row["Progress"] == "7/600"
    assert all(row[key] is None for key in (*table.METRICS, *table.CI_COLUMNS))


def test_consolidated_run_is_authoritative_without_legacy_fallback(fixture_root):
    root, path, records = fixture_root
    policy = json.loads((root / table.POLICY_PATH).read_text())
    selection = policy["canonical_runs"]["xbrlfiling"]["answer_models"]["qwen2.5-7b"]
    legacy_path = root / selection["source_path"] / "results.jsonl"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    # Distinct complete old results make choosing the wrong copy observable.
    legacy_path.write_text("".join(json.dumps({**row, "status": "ABSTAIN"}) + "\n"
                                   for row in records))
    rows, provenance = table.load_table(root)
    row = next(row for row in rows if row["Dataset"] == "XBRLFiling" and row["Model"] == "Qwen2.5-7B")
    assert row["TA"] == 5 and row["Abs"] == 586
    source = provenance["historical"]["runs"]["xbrlfiling/qwen2.5-7b"]
    assert source["results"]["path"] == str(path.relative_to(root))
    assert source["historical_selection"] == selection
    assert str(legacy_path.relative_to(root)) not in provenance["historical"]["sources"]

    path.unlink()
    rows, provenance = table.load_table(root)
    row = next(row for row in rows if row["Dataset"] == "XBRLFiling" and row["Model"] == "Qwen2.5-7B")
    assert row["State"] == "Missing"
    assert all(row[key] is None for key in (*table.METRICS, *table.CI_COLUMNS))
    assert provenance["historical"]["runs"]["xbrlfiling/qwen2.5-7b"]["results"]["path"] == str(path.relative_to(root))


def test_submission_registry_covers_exact_six_models_and_three_datasets():
    assert set(submission_runs.RUNS) == {dataset for dataset, _, _ in table.DATASETS}
    paths = []
    for dataset, models in submission_runs.RUNS.items():
        assert set(models) == {key for key, _, _ in table.MODEL_SPECS}
        assert all(path.startswith(f"results/new_submissions/{dataset}_") for path in models.values())
        paths.extend(models.values())
    assert len(paths) == len(set(paths)) == 18
    assert submission_runs.RUNS["tatdqa"]["fino1-identity-pending"].endswith("tatdqa_fin_o1_8b")
    assert "gpt-6.1-sol" not in submission_runs.RUNS["tatdqa"]


def test_duplicate_ids_are_not_disguised_as_pending(fixture_root):
    root, path, records = fixture_root
    path.write_text(json.dumps(records[0]) + "\n" + json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="denominator"):
        table.load_table(root)


def test_tadqa_wrong_population_is_rejected(tmp_path):
    rows, provenance = tadqa_snapshot()
    rows[0]["N"] = 402
    with pytest.raises(ValueError, match="denominator"):
        table.load_table(tmp_path, tatdqa_report=(rows, provenance))


def test_presentation_has_six_models_three_groups_and_copyable_overleaf(tmp_path):
    rows, provenance = table.load_table(tmp_path, tatdqa_report=tadqa_snapshot())
    latex = table.render_latex(rows)
    assert latex.count(r"\multicolumn{7}{c}") == 3
    assert latex.count(r"\columncolor{gainhl}") == 3
    assert latex.count(r"\vrule width 0.6pt") == 2
    assert r"\cmidrule(l){16-22}" in latex
    assert r"\resizebox{\textwidth}{!}" in latex
    assert "GPT-6.1" not in latex and "Fin-o1-8B & " not in latex
    assert "TheFinAI/Fin-o1-8B for TADQA" in latex
    assert "checkpoint identities remain unconfirmed" in latex
    assert "post hoc" in latex and "historical scoring" in latex
    assert r"95\% Clopper--Pearson" in latex and "pointwise two-sided" in latex
    assert "filing clusters, repeated runs, or multiple comparisons" in latex
    body = latex.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    lines = [line for line in body.split(r" \\") if line.strip()]
    assert len(lines) == 12  # Estimate and interval rows for each of six models.
    assert all(len(line.split("&")) == 22 for line in lines)
    assert r"{\scriptsize [9.4, 99.2]}" in latex  # Precision for 2/3 accepts.
    for model in table.MODELS:
        assert latex.count(model.replace("Haiku ", "Haiku~") + " & ") == 1
    # Pending states never print numeric values even if stale values were supplied.
    row = next(row for row in rows if row["Model"] == "GPT-5.5" and row["Dataset"] == "TADQA")
    row.update(State="Pending", Progress="120/403",
               **{key: None for key in (*table.METRICS, "TR", *table.CI_COLUMNS)})
    outputs = table.export(tmp_path, rows, provenance)
    assert (tmp_path / outputs["latex"]).read_text() == table.render_latex(rows)
    assert (tmp_path / outputs["preamble"]).read_text() == table.PREAMBLE
    with (tmp_path / outputs["table"]).open() as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 18
    pending = next(row for row in exported if row["Model"] == "GPT-5.5" and row["Dataset"] == "TADQA")
    assert pending["TA"] == "" and pending["Cov."] == ""
    assert all(pending[key] == "" for key in table.CI_COLUMNS)
    complete = next(row for row in exported if row["Model"] == "Qwen3-30B" and row["Dataset"] == "TADQA")
    original = next(row for row in rows if row["Model"] == "Qwen3-30B" and row["Dataset"] == "TADQA")
    assert all(float(complete[key]) == original[key] for key in table.CI_COLUMNS)
    saved_provenance = json.loads((tmp_path / outputs["provenance"]).read_text())
    assert saved_provenance["outputs"]["latex"]["sha256"]
    uncertainty = saved_provenance["confidence_intervals"]
    assert uncertainty["method"] == "Clopper-Pearson exact binomial"
    assert uncertainty["level"] == .95 and uncertainty["sidedness"] == "two-sided"
    assert uncertainty["scope"] == "pointwise"
    assert uncertainty["denominators"] == {"Prec.": "TA+FA", "Acc.": "N-Abs-U", "Cov.": "N"}
    assert "conditional on each saved run" in uncertainty["assumptions"]
    assert "post hoc adjudications" in uncertainty["assumptions"]


@pytest.mark.parametrize("state,progress", [("Missing", "0/403"), ("Pending", "1/403")])
def test_fino1_incomplete_tadqa_run_has_no_scores_or_borrowed_historical_identity(tmp_path, state, progress):
    snapshot, source = tadqa_snapshot()
    fin = next(row for row in snapshot if row["Model"] == "Fin-o1 8B")
    fin.update(State=state, Progress=progress)
    rows, provenance = table.load_table(tmp_path, tatdqa_report=(snapshot, source))
    row = next(row for row in rows if row["Dataset"] == "TADQA" and row["Model"] == "Fin-o1")
    assert row["State"] == state and row["Progress"] == progress
    assert all(row[key] is None for key in (*table.METRICS, *table.CI_COLUMNS))
    identity = provenance["model_identity_notes"]["Fin-o1"]
    assert identity["historical_checkpoint"] is None
    assert not identity["same_checkpoint_across_datasets_confirmed"]


def test_incomplete_table_or_misleading_header_is_rejected(tmp_path):
    rows, _ = table.load_table(tmp_path)
    with pytest.raises(ValueError, match="six models"):
        table.render_latex(rows[:-1])
    rows[0]["N"] = 599
    with pytest.raises(ValueError, match="denominator"):
        table.render_latex(rows)


@pytest.fixture
def financebench_run(tmp_path):
    for relative in ("data/numerical_questions.jsonl", "data/financebench_scoring_review.json"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    metadata = {"model": "gpt-5.5", "mode": "openai", "transport": "responses_api"}
    manifest = {"llm_run": metadata, "settings": {"model": "gpt-5.5", "mode": "openai",
                "claim_source": "generated_answer", "deterministic_smt": True}}
    questions = [json.loads(line) for line in (tmp_path / "data/numerical_questions.jsonl").read_text().splitlines()]
    records = [{"id": item["financebench_id"], "question": item["question"],
                "gold_answer": item["answer"], "status": "ABSTAIN", "claimed_value": None,
                "claimed_value_unit": "", "llm_run": metadata} for item in questions]
    directory = tmp_path / table.FINANCEBENCH_GPT55_RUN
    directory.mkdir(parents=True)
    (directory / "run_manifest.json").write_text(json.dumps(manifest))
    path = directory / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return tmp_path, path, records


@pytest.mark.parametrize("value,outcome", [(1577, "TA"), (-1577, "FA")])
def test_gpt55_financebench_uses_shared_source_review_not_historical_magnitudes(financebench_run, value, outcome):
    root, path, records = financebench_run
    records[0].update(status="VERIFIED", claimed_value=value, claimed_value_unit="USD millions",
                      answer_matches_gold=True)  # Stale score flags cannot override the candidate.
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    before = path.read_bytes()
    rows, provenance = table.load_table(root)
    row = next(row for row in rows if row["Dataset"] == "FinanceBench" and row["Model"] == "GPT-5.5")
    assert len(rows) == 18 and row["State"] == "Complete"
    assert row[outcome] == 1 and row["Abs"] == 66 and row["TR"] == 0
    assert row["Cov."] == pytest.approx(100 / 67)
    assert provenance["financebench_gpt55"]["scoring_view"] == "source_reviewed"
    assert "financebench/gpt-5.5" not in provenance["historical"]["runs"]
    assert provenance["financebench_gpt55"]["sources"][str(path.relative_to(root))]
    assert path.read_bytes() == before


def test_gpt55_financebench_partial_run_is_pending_without_metrics(financebench_run):
    root, path, records = financebench_run
    path.write_text(json.dumps(records[0]) + "\n")
    row, source = table._financebench_gpt55_row(root)
    assert row["State"] == "Pending" and row["Progress"] == "1/67"
    assert all(row[key] is None for key in (*table.METRICS, "TR", *table.CI_COLUMNS))
    assert "summary" not in source


def test_gpt55_financebench_unscored_decision_is_not_counted_as_true_reject(financebench_run):
    root, path, records = financebench_run
    review = json.loads((root / "data/financebench_scoring_review.json").read_text())
    ambiguous = next(item for item in review["records"] if item["kind"] == "ambiguous_reference_scope")
    candidate = next(row for row in records if row["id"] == ambiguous["id"])
    target = ambiguous["plausible_targets"][0]
    candidate.update(status="VIOLATED", claimed_value=target["value"], claimed_value_unit=target["unit"])
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    row, source = table._financebench_gpt55_row(root)
    assert row["State"] == "Complete" and row["Progress"] == "67/67"
    assert row["U"] == row["U rejected"] == 1 and row["U verified"] == 0
    assert row["TR"] == 0 and row["Abs"] == 66
    assert row["Prec."] is None and row["Acc."] is None
    assert row["Cov."] == pytest.approx(100 / 67)
    assert source["summary"]["unscored_decisions"] == 1
    assert source["summary"]["TR"] == 0
    assert "not true rejects" in source["reason"]


def test_complete_run_with_ungraded_accept_and_reject_discloses_conditional_metrics(tmp_path):
    counts = {"TA": 34, "FA": 0, "TR": 5, "FR": 6, "AB": 20}
    row = table._complete_row("FinanceBench", "GPT-5.5", 67, counts,
                              unscored_verified=1, unscored_rejected=1)
    assert row["N"] == sum(counts.values()) + row["U"] == 67
    assert row["Graded accepts"] == 34 and row["Graded decisions"] == 45
    assert row["Prec."] == 100 and row["Acc."] == pytest.approx(100 * 39 / 45)
    assert row["Cov."] == pytest.approx(100 * 47 / 67)
    for metric, successes, trials in (("Prec.", 34, 34), ("Acc.", 39, 45), ("Cov.", 47, 67)):
        assert (row[f"{metric} CI low"], row[f"{metric} CI high"]) == pytest.approx(
            table.exact_binomial_interval(successes, trials))
    assert row["N"] - sum(row[key] for key in ("TA", "FA", "FR", "Abs", "U")) == row["TR"]
    rows, _ = table.load_table(tmp_path)
    index = next(i for i, item in enumerate(rows)
                 if item["Model"] == "GPT-5.5" and item["Dataset"] == "FinanceBench")
    rows[index] = row
    latex = table.render_latex(rows)
    assert r"GPT-5.5$^{\dagger}$" in latex
    assert "34 & 0 & 6 & 20 & 100.0 & 86.7 & 70.1" in latex
    assert "U=2 ungraded decisions (1 accepted, 1 rejected)" in latex
    assert "34 graded accepts" in latex and "45 graded decisions" in latex
    assert r"97.1\%-100.0\%" in latex
    assert "identification bounds, not confidence intervals" in latex
    # A complete run cannot silently omit ungraded decisions from its population.
    with pytest.raises(ValueError, match="reconcile"):
        table._complete_row("FinanceBench", "GPT-5.5", 67, counts)


@pytest.mark.parametrize("defect", ["missing_manifest", "wrong_manifest_model", "mixed_row_model", "duplicate", "unknown"])
def test_gpt55_financebench_rejects_wrong_identity_and_population(financebench_run, defect):
    root, path, records = financebench_run
    manifest_path = path.parent / "run_manifest.json"
    if defect == "missing_manifest":
        manifest_path.unlink()
    elif defect == "wrong_manifest_model":
        manifest = json.loads(manifest_path.read_text())
        manifest["settings"]["model"] = "gpt-6.1-sol"
        manifest_path.write_text(json.dumps(manifest))
    elif defect == "mixed_row_model":
        records[0]["llm_run"] = {**records[0]["llm_run"], "model": "gpt-6.1-sol"}
    elif defect == "duplicate":
        records = [records[0], records[0]]
    elif defect == "unknown":
        records[0]["id"] = "financebench_unknown"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="manifest|identity|question IDs"):
        table._financebench_gpt55_row(root)


@pytest.mark.parametrize("successes,trials,expected", [
    (10, 10, (69.15028921812392, 100.0)),
    (560, 560, (99.34343638593862, 100.0)),
    (0, 10, (0.0, 30.84971078187608)),
    (0, 0, (None, None)),
    (1, 1, (2.5, 100.0)),
    (0, 1, (0.0, 97.5)),
])
def test_exact_intervals_include_boundary_uncertainty(successes, trials, expected):
    assert table.exact_binomial_interval(successes, trials) == pytest.approx(expected)


@pytest.mark.parametrize("successes,trials", [(-1, 10), (11, 10), (0, -1), (1.5, 10), (1, 10.5)])
def test_exact_intervals_reject_invalid_counts(successes, trials):
    with pytest.raises(ValueError, match="Binomial counts"):
        table.exact_binomial_interval(successes, trials)


def test_all_abstentions_have_undefined_precision_accuracy_but_coverage_interval():
    row = table._complete_row("FinanceBench", "GPT-5.5", 67,
                              {"TA": 0, "FA": 0, "TR": 0, "FR": 0, "AB": 67})
    assert row["Prec."] is None and row["Acc."] is None
    assert all(row[f"{metric} CI {bound}"] is None
               for metric in ("Prec.", "Acc.") for bound in ("low", "high"))
    assert row["Cov."] == 0 and row["Cov. CI low"] == 0
    assert row["Cov. CI high"] == pytest.approx(100 * (1 - .025 ** (1 / 67)))


def test_latex_keeps_estimates_and_formats_intervals_without_false_perfection(tmp_path):
    rows, _ = table.load_table(tmp_path)
    perfect = table._complete_row("FinanceBench", "Qwen2.5-7B", 67,
                                 {"TA": 10, "FA": 0, "TR": 23, "FR": 0, "AB": 34})
    index = next(i for i, row in enumerate(rows)
                 if row["Dataset"] == "FinanceBench" and row["Model"] == "Qwen2.5-7B")
    rows[index] = perfect
    latex = table.render_latex(rows)
    assert "10 & 0 & 0 & 34 & 100.0 & 100.0 & 49.3" in latex
    assert r"{\scriptsize [69.2, 100.0]}" in latex
    # Stale intervals must not leak into the LaTeX from a pending row.
    rows[index] = {**perfect, "State": "Pending", "Progress": "10/67"}
    assert r"{\scriptsize [69.2, 100.0]}" not in table.render_latex(rows)
