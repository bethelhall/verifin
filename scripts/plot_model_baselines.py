"""Plot the four completed judge comparisons directly from their scored table.

No model calls or rescoring occur here. Missing metrics remain missing and are
labelled ``Ungraded`` rather than being rendered as zero-valued observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "results/new_submissions/baselines/comparison.csv"
DEFAULT_OUTPUT = ROOT / "results/new_submissions/baselines/figures"
OUTPUT_STEM = "four_model_baseline_comparison"
MODEL_LABELS = {
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen3-30b": "Qwen3-30B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "gpt-5.5": "GPT-5.5",
}
DATASETS = ("FinanceBench", "XBRLFiling")
METHODS = ("LLM Judge", "VeriFin")
METRICS = (
    ("Precision (%)", "Precision"),
    ("Decision accuracy (%)", "Decision accuracy"),
    ("Coverage (%)", "Coverage"),
)
COLORS = {"LLM Judge": "#3D6FA6", "VeriFin": "#D5A243"}


def _validated_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate the complete comparison grain without replacing any metric."""
    keys = ["Candidate model", "Dataset", "Method"]
    required = {*keys, "Unscored", *(name for name, _ in METRICS)}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Comparison table is missing columns: {sorted(missing)}")
    if table.duplicated(keys).any():
        raise ValueError("Duplicate model/dataset/method comparison rows")
    expected = {(model, dataset, method) for model in MODEL_LABELS
                for dataset in DATASETS for method in METHODS}
    actual = set(table[keys].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError("Plot requires exactly four models × two datasets × two methods")
    frame = table.copy(deep=True)
    for column, _ in METRICS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
            raise ValueError(f"{column} must contain percentages between 0 and 100")
    if frame["Coverage (%)"].isna().any():
        raise ValueError("Coverage must be available for every complete run")
    unscored = pd.to_numeric(frame["Unscored"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(unscored).all() or (unscored < 0).any()
            or (unscored != np.floor(unscored)).any()):
        raise ValueError("Unscored must contain nonnegative integer counts")
    frame["Unscored"] = unscored.astype(int)
    missing_metrics = frame[[name for name, _ in METRICS[:2]]].isna().any(axis=1)
    if (missing_metrics & frame["Unscored"].eq(0)).any():
        raise ValueError("Ungraded metrics require an explicit unscored count")
    if (frame["Unscored"].gt(0) & frame["Decision accuracy (%)"].notna()).any():
        raise ValueError("Unresolved decisions cannot have a full-run accuracy estimate")
    if "Reference excluded" in frame and frame["Reference excluded"].ne(0).any():
        raise ValueError("This figure requires the full runs without reference exclusions")
    if "N" in frame:
        for dataset, group in frame.groupby("Dataset"):
            population = pd.to_numeric(group["N"], errors="raise")
            if population.nunique() != 1 or population.isna().any():
                raise ValueError(f"{dataset}: inconsistent dataset denominator")
            value = float(population.iloc[0])
            if not np.isfinite(value) or value <= 0 or value != int(value):
                raise ValueError(f"{dataset}: invalid dataset denominator")
    return frame.set_index(keys)


def plot_comparison(table: pd.DataFrame, output_dir: str | Path) -> plt.Figure:
    """Render the table as six panels, export PNG/PDF/SVG, and return the figure.

    The input table is unchanged. Each nonmissing metric produces one bar whose
    width is the supplied percentage. Missing metrics produce text only.
    """
    frame = _validated_table(table)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "text.color": "#24313A", "axes.labelcolor": "#24313A",
        "xtick.color": "#4F5962", "ytick.color": "#24313A",
        "axes.edgecolor": "#B9C2CA", "axes.linewidth": 0.7,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "hatch.linewidth": 0.35,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(14.4, 9.1), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.135, right=0.975, bottom=0.205,
                            top=0.825, wspace=0.16, hspace=0.64)
        fig.suptitle("Four-model baseline comparison", x=0.055, y=0.985,
                     ha="left", fontsize=20, fontweight="bold")
        review_enabled = "Full-answer reviewed" in frame.columns
        scoring_label = ("Original numeric scoring + documented full-answer review"
                         if review_enabled else "Original paper scoring · matched candidate pools")
        fig.text(0.055, 0.945, scoring_label,
                 fontsize=11.5, color="#4F5962")
        fig.legend(handles=[
            Patch(facecolor=COLORS["LLM Judge"], label="LLM Judge"),
            Patch(facecolor=COLORS["VeriFin"], edgecolor="#775B25",
                  hatch="//", linewidth=0.4, label="VeriFin"),
        ], loc="upper right", bbox_to_anchor=(0.978, 0.925), ncol=2,
            frameon=False, handlelength=2.2, columnspacing=1.7)

        positions = np.arange(len(MODEL_LABELS), dtype=float)
        for row_index, dataset in enumerate(DATASETS):
            for column_index, (metric, metric_label) in enumerate(METRICS):
                ax = axes[row_index, column_index]
                ax.set_title(metric_label, loc="left", pad=11,
                             fontweight="bold", fontsize=12)
                ax.set_axisbelow(True)
                ax.xaxis.grid(True, color="#E3E8ED", linewidth=0.7)
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
                ax.tick_params(axis="x", labelbottom=True, length=0, pad=7)
                ax.tick_params(axis="y", length=0, pad=9,
                               labelleft=column_index == 0)
                ax.set_yticks(positions, list(MODEL_LABELS.values()))
                ax.set_ylim(len(MODEL_LABELS) - 0.5, -0.5)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                for model_index, model in enumerate(MODEL_LABELS):
                    for method_index, method in enumerate(METHODS):
                        y = positions[model_index] + (-0.17 if method_index == 0 else 0.17)
                        value = frame.loc[(model, dataset, method), metric]
                        identifier = "|".join((dataset, model, method, metric))
                        if pd.isna(value):
                            label = ax.text(2, y, "Ungraded", va="center", ha="left",
                                            color="#69747D", fontstyle="italic", fontsize=9.5)
                            label.set_gid(identifier + "|ungraded")
                            continue
                        value = float(value)
                        bars = ax.barh(y, value, height=0.275, color=COLORS[method],
                                       edgecolor="#775B25" if method == "VeriFin" else "none",
                                       linewidth=0.35, hatch="//" if method == "VeriFin" else None)
                        bars[0].set_gid(identifier)
                        inside = value >= 20
                        label_color = "white" if inside and method == "LLM Judge" else "#24313A"
                        ax.text(value - 2 if inside else value + 1.5, y,
                                f"{value:.1f}%", ha="right" if inside else "left",
                                va="center", fontsize=9.3, color=label_color)
            left_axis = axes[row_index, 0].get_position()
            population = (int(frame.xs(dataset, level="Dataset")["N"].iloc[0])
                          if "N" in frame else None)
            dataset_label = dataset + (f" · n = {population}" if population is not None else "")
            fig.text(0.055, left_axis.y1 + 0.065, dataset_label,
                     fontweight="bold", fontsize=12.5)

        unresolved = int(frame["Unscored"].sum())
        fig.text(0.055, 0.115,
                 "Precision = TA / (TA + FA).  Decision accuracy = (TA + TR) / (TA + FA + TR + FR).",
                 fontsize=10, color="#4F5962")
        fig.text(0.055, 0.088,
                 "Coverage = (VERIFIED + VIOLATED) / N; the denominator includes all questions.",
                 fontsize=10, color="#4F5962")
        reviewed = (int(frame.xs("LLM Judge", level="Method")["Full-answer reviewed"].sum())
                    if review_enabled else 0)
        grading_note = (f"{reviewed} candidate answers have documented correctness reviews; {unresolved} decisions remain ungraded."
                        if review_enabled else
                        f"Ungraded: {unresolved} decisions await correctness grading; affected metrics are withheld.")
        fig.text(0.055, 0.061, grading_note,
                 fontsize=10, color="#4F5962")
        fig.text(0.055, 0.034, "Point estimates only; no significance claims.",
                 fontsize=9.5, color="#69747D")
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"{OUTPUT_STEM}.{extension}",
                        dpi=220, bbox_inches="tight", facecolor="white")
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    table = pd.read_csv(args.table)
    figure = plot_comparison(table, args.output_dir)
    plt.close(figure)
    for extension in ("png", "pdf", "svg"):
        print(args.output_dir / f"{OUTPUT_STEM}.{extension}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
