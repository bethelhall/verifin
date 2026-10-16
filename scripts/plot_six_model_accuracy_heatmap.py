"""Plot six-model XBRLBench answer accuracy before VeriFin is applied.

The figure is intentionally separate from VeriFin verdict plots.  It measures
the quality of the numerical candidates produced by each answer model, using
the paper's magnitude-based correctness rule:

    abs(|candidate| - |gold|) <= max($1 million, 1% of |gold|)

Missing or nonnumerical candidates count as incorrect.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle

try:
    from scripts import submission_runs
except ImportError:  # Direct execution: ``python scripts/plot_six_model_accuracy_heatmap.py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import submission_runs


N_QUESTIONS = 600

# This order matches the final six-model paper figure and keeps Llama 3.1
# before Qwen2.5, as requested.
MODEL_SPECS = (
    (
        "GPT-5.5",
        "GPT-5.5\n",
        f"{submission_runs.RUNS['xbrlfiling']['gpt-5.5']}/results.jsonl",
    ),
    (
        "Claude Haiku 4.5",
        "Claude\nHaiku 4.5",
        f"{submission_runs.RUNS['xbrlfiling']['claude-haiku-4.5']}/results.jsonl",
    ),
    # The supplied artifact directory is named ``mc-fino-30b-latest`` but
    # contains no model-identity metadata. Keep the display size-neutral until
    # the immutable checkpoint identifier is recovered.
    (
        "Fin-o1",
        "Fin-o1",
        f"{submission_runs.RUNS['xbrlfiling']['fino1-identity-pending']}/results.jsonl",
    ),
    (
        "Qwen3-30B",
        "Qwen3\n30B",
        f"{submission_runs.RUNS['xbrlfiling']['qwen3-30b']}/results.jsonl",
    ),
    (
        "Llama-3.1-8B",
        "Llama 3.1\n8B",
        f"{submission_runs.RUNS['xbrlfiling']['llama-3.1-8b']}/results.jsonl",
    ),
    (
        "Qwen2.5-7B",
        "Qwen2.5\n7B",
        f"{submission_runs.RUNS['xbrlfiling']['qwen2.5-7b']}/results.jsonl",
    ),
)

FAMILY_ORDER = ("Income statement", "Balance sheet", "Cash flow")
METRIC_SPECS = (
    ("gross profit", "Gross profit", "Income statement"),
    ("operating income", "Operating income", "Income statement"),
    ("operating expenses", "Operating expenses", "Income statement"),
    (
        "income before income taxes",
        "Income before taxes",
        "Income statement",
    ),
    ("net income", "Net income", "Income statement"),
    ("total current assets", "Current assets", "Balance sheet"),
    ("total non-current assets", "Non-current assets", "Balance sheet"),
    ("total current liabilities", "Current liabilities", "Balance sheet"),
    (
        "total non-current liabilities",
        "Non-current liabilities",
        "Balance sheet",
    ),
    ("total liabilities", "Total liabilities", "Balance sheet"),
    ("total shareholders", "Shareholders' equity", "Balance sheet"),
    ("net property", "Net PP&E", "Balance sheet"),
    ("net cash from operating", "Cash from operations", "Cash flow"),
    ("net cash from investing", "Cash from investing", "Cash flow"),
    ("net cash from financing", "Cash from financing", "Cash flow"),
)

METRIC_ORDER = tuple(label for _, label, _ in METRIC_SPECS)
METRIC_FAMILY = {label: family for _, label, family in METRIC_SPECS}
MODEL_ORDER = tuple(name for name, _, _ in MODEL_SPECS)
MODEL_TICK_LABELS = tuple(label for _, label, _ in MODEL_SPECS)

NUMBER_PATTERN = re.compile(r"-?\d[\d,]*\.?\d*")

INK = "#20202A"
MUTED = "#666674"
SEPARATOR = "#7A7584"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "candidate_accuracy_purple",
    ("#F7F5FA", "#DDD5E8", "#AA91C3", "#765394", "#4B2D68"),
)
SINGLE_COLUMN_CMAP = ListedColormap(
    ("#F7F5FA", "#E1D9E9", "#BEADD0", "#765394", "#4B2D68"),
    name="candidate_accuracy_purple_print",
)
SINGLE_COLUMN_NORM = BoundaryNorm(
    (0.0, 20.0, 40.0, 60.0, 80.0, 100.0001),
    SINGLE_COLUMN_CMAP.N,
)

FIG_WIDTH_IN = 7.08
FIG_HEIGHT_IN = 4.55
SINGLE_COLUMN_WIDTH_IN = 3.33
SINGLE_COLUMN_HEIGHT_IN = 4.20
TRANSPOSED_WIDTH_IN = 3.33
TRANSPOSED_HEIGHT_IN = 4.55


def read_jsonl(path: Path) -> list[dict]:
    """Read a UTF-8 JSONL file, skipping blank lines."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def result_paths(root: Path) -> tuple[tuple[str, str, Path], ...]:
    return tuple(
        (name, tick_label, root / relative_path)
        for name, tick_label, relative_path in MODEL_SPECS
    )


def resolve_root() -> Path:
    """Prefer this script's bundle when locating the selected result files."""

    cwd = Path.cwd().resolve()
    candidates = (
        Path(__file__).resolve().parents[1],
        cwd,
        *cwd.parents,
        cwd / "verifiqa",
        Path("/content/verifiqa"),
        Path("/content/drive/MyDrive/verifiqa"),
    )
    for candidate in candidates:
        if all(path.is_file() for _, _, path in result_paths(candidate)):
            return candidate
    raise FileNotFoundError(
        "Could not locate all six result files. Run from the verifiqa "
        "repository or call main(ROOT) from the notebook."
    )


def metric_for_question(question: str) -> tuple[str, str]:
    """Map a question to the first matching metric and statement family."""

    lowered = str(question).lower()
    for search_term, metric, family in METRIC_SPECS:
        if search_term in lowered:
            return metric, family
    raise ValueError(f"Unmapped XBRLBench question: {question}")


def parse_numbers(text: object) -> list[float]:
    values: list[float] = []
    for token in NUMBER_PATTERN.findall(str(text or "")):
        try:
            values.append(abs(float(token.replace(",", ""))))
        except ValueError:
            continue
    return values


def is_correct(claimed_value: object, gold_answer: object) -> bool:
    """Apply the paper's magnitude-based numerical correctness rule."""

    try:
        claimed = abs(float(claimed_value))
    except (TypeError, ValueError):
        return False

    return any(
        abs(claimed - gold) <= max(1.0, 0.01 * gold)
        for gold in parse_numbers(gold_answer)
    )


def load_accuracy_data(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load, validate, and score all six result artifacts.

    Returns
    -------
    accuracy_pct:
        Metric-by-model accuracy percentages.
    correct_counts:
        Metric-by-model counts of correct candidates.
    denominators:
        Number of questions in each metric row.
    """

    runs: dict[str, dict[str, dict]] = {}
    reference_ids: set[str] | None = None
    reference_metric_by_id: dict[str, str] | None = None

    for model, _, path in result_paths(root):
        if not path.is_file():
            raise FileNotFoundError(path)

        rows = read_jsonl(path)
        if len(rows) != N_QUESTIONS:
            raise ValueError(
                f"{path}: expected {N_QUESTIONS} rows; found {len(rows)}."
            )

        rows_by_id = {row.get("id"): row for row in rows}
        if None in rows_by_id or len(rows_by_id) != N_QUESTIONS:
            raise ValueError(f"{path}: result IDs are missing or duplicated.")

        ids = set(rows_by_id)
        metric_by_id = {
            row_id: metric_for_question(row.get("question", ""))[0]
            for row_id, row in rows_by_id.items()
        }
        if reference_ids is None:
            reference_ids = ids
            reference_metric_by_id = metric_by_id
        else:
            if ids != reference_ids:
                raise ValueError(f"{path}: IDs do not match the reference run.")
            if metric_by_id != reference_metric_by_id:
                raise ValueError(
                    f"{path}: question-to-metric mapping differs from the "
                    "reference run."
                )

        missing_gold = [
            row_id
            for row_id, row in rows_by_id.items()
            if not row.get("gold_answer")
        ]
        if missing_gold:
            raise ValueError(
                f"{path}: {len(missing_gold)} rows lack gold_answer."
            )
        runs[model] = rows_by_id

    assert reference_metric_by_id is not None

    denominators = (
        pd.Series(reference_metric_by_id, name="metric")
        .value_counts()
        .reindex(METRIC_ORDER)
        .astype(int)
    )
    if denominators.isna().any() or int(denominators.sum()) != N_QUESTIONS:
        raise AssertionError("Metric denominators do not cover all questions.")

    correct_counts = pd.DataFrame(
        0,
        index=pd.Index(METRIC_ORDER, name="Metric"),
        columns=pd.Index(MODEL_ORDER, name="Answer model"),
        dtype=int,
    )

    for model in MODEL_ORDER:
        for row_id, row in runs[model].items():
            metric = reference_metric_by_id[row_id]
            correct_counts.loc[metric, model] += int(
                is_correct(row.get("claimed_value"), row["gold_answer"])
            )

    accuracy_pct = correct_counts.div(denominators, axis=0) * 100.0

    if not ((correct_counts >= 0).all().all()):
        raise AssertionError("Negative correct count.")
    for metric in METRIC_ORDER:
        if (correct_counts.loc[metric] > denominators.loc[metric]).any():
            raise AssertionError(f"Correct count exceeds n for {metric}.")

    return accuracy_pct, correct_counts, denominators


def _format_pct(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{value:.0f}"
    return f"{value:.1f}"


def source_data_frame(
    accuracy_pct: pd.DataFrame,
    correct_counts: pd.DataFrame,
    denominators: pd.Series,
) -> pd.DataFrame:
    """Return long-form source data for auditing the plotted values."""

    records: list[dict] = []
    for model, _, relative_path in MODEL_SPECS:
        records.append(
            {
                "family": "Overall",
                "metric": "Overall",
                "model": model,
                "correct": int(correct_counts[model].sum()),
                "n": N_QUESTIONS,
                "accuracy_pct": 100.0
                * float(correct_counts[model].sum())
                / N_QUESTIONS,
                "result_path": relative_path,
            }
        )
        for metric in METRIC_ORDER:
            records.append(
                {
                    "family": METRIC_FAMILY[metric],
                    "metric": metric,
                    "model": model,
                    "correct": int(correct_counts.loc[metric, model]),
                    "n": int(denominators.loc[metric]),
                    "accuracy_pct": float(accuracy_pct.loc[metric, model]),
                    "result_path": relative_path,
                }
            )
    return pd.DataFrame.from_records(records)


def make_figure(
    accuracy_pct: pd.DataFrame,
    correct_counts: pd.DataFrame,
    denominators: pd.Series,
    output_stem: Path,
) -> plt.Figure:
    """Create the standalone, full-width six-model heatmap."""

    overall = pd.DataFrame(
        [
            100.0
            * correct_counts.sum(axis=0).to_numpy(dtype=float)
            / N_QUESTIONS
        ],
        index=["Overall"],
        columns=MODEL_ORDER,
    )
    plotted = pd.concat(
        [overall, accuracy_pct.reindex(index=METRIC_ORDER, columns=MODEL_ORDER)]
    )
    matrix = plotted.to_numpy(dtype=float)

    row_labels = [f"Overall  ($n={N_QUESTIONS}$)"] + [
        f"{metric}  ($n={int(denominators.loc[metric])}$)"
        for metric in METRIC_ORDER
    ]

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica Neue",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 7.4,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    ):
        fig, ax = plt.subplots(
            figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
            facecolor="white",
        )
        fig.subplots_adjust(left=0.305, right=0.995, top=0.835, bottom=0.025)

        image = ax.imshow(
            matrix,
            cmap=HEATMAP_CMAP,
            vmin=0.0,
            vmax=100.0,
            interpolation="nearest",
            aspect="auto",
        )

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    _format_pct(value),
                    ha="center",
                    va="center",
                    color="white" if value >= 66.0 else INK,
                    fontsize=7.15 if row == 0 else 6.75,
                    fontweight="bold" if row == 0 else "semibold",
                )

        ax.set_xticks(
            np.arange(len(MODEL_ORDER)),
            MODEL_TICK_LABELS,
            fontsize=7.25,
            color=INK,
        )
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=5.5)

        ax.set_yticks(
            np.arange(len(row_labels)),
            row_labels,
            fontsize=7.05,
            color=INK,
        )
        ax.tick_params(axis="y", length=0, pad=5.0)
        ax.get_yticklabels()[0].set_fontweight("bold")

        ax.set_title(
            "Generated-answer accuracy before VeriFin (%)",
            loc="left",
            pad=35,
            fontsize=8.8,
            fontweight="semibold",
            color=INK,
        )

        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1.0), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.15)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Overall row and statement-family boundaries.
        for boundary, linewidth in ((0.5, 1.45), (5.5, 1.0), (12.5, 1.0)):
            ax.axhline(
                boundary,
                color=SEPARATOR,
                linewidth=linewidth,
                alpha=0.92,
                zorder=5,
            )

        family_positions = {
            "Income statement": 3.0,
            "Balance sheet": 9.0,
            "Cash flow": 14.0,
        }
        for family in FAMILY_ORDER:
            ax.text(
                -0.38,
                family_positions[family],
                family.upper().replace(" ", "\n"),
                transform=ax.get_yaxis_transform(),
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold",
                linespacing=0.9,
                color=MUTED,
                clip_on=False,
            )

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                matrix.shape[1],
                matrix.shape[0],
                fill=False,
                edgecolor="#D0CBD6",
                linewidth=0.7,
                clip_on=False,
            )
        )
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(matrix.shape[0] - 0.5, -0.5)

        # Retain the image reference for notebook and static-backend consumers.
        fig._verifin_heatmap_image = image  # type: ignore[attr-defined]

        output_stem = Path(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        # Keep the exported canvas at the ACM full-text width (7.08 in).
        # Every label is laid out inside that canvas, so tight bounding-box
        # expansion is unnecessary and would make the PDF slightly too wide.
        export_options = {"facecolor": "white"}
        fig.savefig(output_stem.with_suffix(".pdf"), **export_options)
        fig.savefig(output_stem.with_suffix(".svg"), **export_options)
        fig.savefig(
            output_stem.with_suffix(".png"),
            dpi=600,
            **export_options,
        )
        return fig


def make_single_column_figure(
    accuracy_pct: pd.DataFrame,
    correct_counts: pd.DataFrame,
    denominators: pd.Series,
    output_stem: Path,
) -> plt.Figure:
    """Create a vector-first heatmap sized for one ACM/ICAIF column.

    This is a distinct layout rather than a scaled copy of the full-width
    figure: labels are shortened, statement-family labels are rotated into the
    outer gutter, and every cell remains directly annotated at final size.
    """

    overall = pd.DataFrame(
        [
            100.0
            * correct_counts.sum(axis=0).to_numpy(dtype=float)
            / N_QUESTIONS
        ],
        index=["Overall"],
        columns=MODEL_ORDER,
    )
    plotted = pd.concat(
        [overall, accuracy_pct.reindex(index=METRIC_ORDER, columns=MODEL_ORDER)]
    )
    matrix = plotted.to_numpy(dtype=float)

    compact_row_labels = [
        "Overall",
        "Gross profit",
        "Operating inc.",
        "Operating exp.",
        "Income before tax",
        "Net income",
        "Current assets",
        "Noncurrent assets",
        "Current liabilities",
        "Noncurrent liab.",
        "Total liabilities",
        "Shareholders' eq.",
        "Net PP&E",
        "Cash: operations",
        "Cash: investing",
        "Cash: financing",
    ]
    if len(compact_row_labels) != matrix.shape[0]:
        raise AssertionError("Single-column row-label count is incorrect.")

    compact_model_labels = (
        "GPT\n5.5",
        "Claude\nH4.5",
        "Fin-o1",
        "Qwen3\n30B",
        "Llama\n3.1-8B",
        "Qwen\n2.5-7B",
    )
    row_denominators = np.concatenate(
        ([N_QUESTIONS], denominators.reindex(METRIC_ORDER).to_numpy(dtype=int))
    )

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica Neue",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 6.0,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
            "svg.fonttype": "none",
            "figure.dpi": 200,
            "savefig.dpi": 1200,
            "text.antialiased": True,
        }
    ):
        fig, ax = plt.subplots(
            figsize=(SINGLE_COLUMN_WIDTH_IN, SINGLE_COLUMN_HEIGHT_IN),
            facecolor="white",
        )
        fig.subplots_adjust(
            left=0.405,
            right=0.995,
            top=0.850,
            bottom=0.022,
        )

        # pcolormesh keeps all 96 heatmap cells as vector paths in PDF/SVG.
        x_edges = np.arange(matrix.shape[1] + 1, dtype=float) - 0.5
        y_edges = np.arange(matrix.shape[0] + 1, dtype=float) - 0.5
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            matrix,
            cmap=SINGLE_COLUMN_CMAP,
            norm=SINGLE_COLUMN_NORM,
            shading="flat",
            edgecolors="white",
            linewidth=0.52,
            antialiased=True,
            rasterized=False,
        )

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    _format_pct(value),
                    ha="center",
                    va="center",
                    color="white" if value >= 60.0 else INK,
                    fontsize=6.15 if row == 0 else 5.85,
                    fontweight="bold" if row == 0 else "semibold",
                    clip_on=True,
                    zorder=4,
                )

        ax.set_xticks(
            np.arange(len(MODEL_ORDER)),
            compact_model_labels,
            fontsize=5.85,
            color=INK,
            linespacing=0.92,
        )
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=3.2)

        ax.set_yticks(np.arange(len(compact_row_labels)))
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

        # Separate aligned label and n columns save width and keep the small
        # n=4 categories visibly distinct from the larger metric groups.
        for row, (label, n_value) in enumerate(
            zip(compact_row_labels, row_denominators)
        ):
            ax.text(
                -0.105,
                row,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=6.05,
                fontweight="bold" if row == 0 else "normal",
                color=INK,
                clip_on=False,
            )
            ax.text(
                -0.020,
                row,
                f"{int(n_value)}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=5.55,
                fontweight="bold" if row == 0 else "normal",
                color=MUTED,
                clip_on=False,
            )
        ax.text(
            -0.020,
            -0.88,
            "$n$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.5,
            color=MUTED,
            clip_on=False,
        )

        ax.set_title(
            "Answer accuracy before VeriFin (%)",
            loc="left",
            pad=26,
            fontsize=7.3,
            fontweight="semibold",
            color=INK,
        )

        # Stronger rules separate Overall and the three statement families.
        for boundary, linewidth in ((0.5, 0.95), (5.5, 0.72), (12.5, 0.72)):
            ax.axhline(
                boundary,
                color=SEPARATOR,
                linewidth=linewidth,
                alpha=0.95,
                zorder=5,
            )

        family_positions = {
            "Income statement": 3.0,
            "Balance sheet": 9.0,
            "Cash flow": 14.0,
        }
        family_labels = {
            "Income statement": "INCOME STMT",
            "Balance sheet": "BALANCE SHEET",
            "Cash flow": "CASH FLOW",
        }
        for family in FAMILY_ORDER:
            ax.text(
                -0.645,
                family_positions[family],
                family_labels[family],
                transform=ax.get_yaxis_transform(),
                rotation=90,
                ha="center",
                va="center",
                fontsize=5.25,
                fontweight="bold",
                color=MUTED,
                clip_on=False,
            )

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                matrix.shape[1],
                matrix.shape[0],
                fill=False,
                edgecolor="#C9C3D0",
                linewidth=0.5,
                clip_on=False,
                zorder=6,
            )
        )
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(matrix.shape[0] - 0.5, -0.5)

        # Retain the mesh reference for notebook and static-backend consumers.
        fig._verifin_heatmap_mesh = mesh  # type: ignore[attr-defined]

        output_stem = Path(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        figure_metadata = {
            "Title": "Answer accuracy before VeriFin",
            "Subject": "Six-model XBRLBench answer-accuracy heatmap",
        }
        fig.savefig(
            output_stem.with_suffix(".pdf"),
            facecolor="white",
            metadata=figure_metadata,
        )
        fig.savefig(
            output_stem.with_suffix(".svg"),
            facecolor="white",
        )
        fig.savefig(
            output_stem.with_suffix(".png"),
            dpi=1200,
            facecolor="white",
        )
        return fig


def make_transposed_single_column_figure(
    accuracy_pct: pd.DataFrame,
    correct_counts: pd.DataFrame,
    denominators: pd.Series,
    output_stem: Path,
) -> plt.Figure:
    """Create a stacked transposed heatmap for one ACM/ICAIF column.

    Models are rows.  Metrics are columns split into statement-family panels;
    this preserves readable cell widths that a literal 6-by-15 single panel
    cannot provide at 3.33 inches.
    """

    overall_pct = (
        100.0
        * correct_counts.sum(axis=0).reindex(MODEL_ORDER)
        / N_QUESTIONS
    )
    compact_model_names = (
        "GPT-5.5",
        "Claude H4.5",
        "Fin-o1",
        "Qwen3-30B",
        "Llama3.1-8B",
        "Qwen2.5-7B",
    )
    compact_model_labels = tuple(
        f"{label} · {_format_pct(float(overall_pct.loc[model]))}"
        for label, model in zip(compact_model_names, MODEL_ORDER)
    )

    panel_specs = (
        {
            "family": "INCOME STATEMENT",
            "metrics": METRIC_ORDER[:5],
            "headers": (
                "Gross\nprofit",
                "Op.\nincome",
                "Op.\nexpense",
                "Pretax\nincome",
                "Net\nincome",
            ),
            "bottom": 0.685,
        },
        {
            "family": "BALANCE SHEET",
            "metrics": METRIC_ORDER[5:12],
            "headers": (
                "Current\nassets",
                "Noncurr.\nassets",
                "Current\nliab.",
                "Noncurr.\nliab.",
                "Total\nliab.",
                "Shareh.\nequity",
                "Net\nPP&E",
            ),
            "bottom": 0.365,
        },
        {
            "family": "CASH FLOW",
            "metrics": METRIC_ORDER[12:],
            "headers": (
                "Oper.",
                "Invest.",
                "Financ.",
            ),
            "bottom": 0.045,
        },
    )

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica Neue",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 5.8,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
            "svg.fonttype": "none",
            "figure.dpi": 200,
            "savefig.dpi": 1200,
            "text.antialiased": True,
        }
    ):
        fig = plt.figure(
            figsize=(TRANSPOSED_WIDTH_IN, TRANSPOSED_HEIGHT_IN),
            facecolor="white",
        )
        fig.text(
            0.310,
            0.993,
            "Answer accuracy before VeriFin (%)",
            ha="left",
            va="top",
            fontsize=7.3,
            fontweight="semibold",
            color=INK,
        )

        axes: list[plt.Axes] = []
        for spec in panel_specs:
            metrics = tuple(spec["metrics"])
            panel_width = 0.0975 * len(metrics)
            matrix = (
                accuracy_pct.reindex(index=metrics, columns=MODEL_ORDER)
                .transpose()
                .to_numpy(dtype=float)
            )
            ax = fig.add_axes(
                [0.310, float(spec["bottom"]), panel_width, 0.205],
                facecolor="white",
            )
            axes.append(ax)

            x_edges = np.arange(matrix.shape[1] + 1, dtype=float) - 0.5
            y_edges = np.arange(matrix.shape[0] + 1, dtype=float) - 0.5
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                matrix,
                cmap=SINGLE_COLUMN_CMAP,
                norm=SINGLE_COLUMN_NORM,
                shading="flat",
                edgecolors="white",
                linewidth=0.48,
                antialiased=True,
                rasterized=False,
            )

            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    value = matrix[row, column]
                    ax.text(
                        column,
                        row,
                        _format_pct(value),
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        fontweight="semibold",
                        color="white" if value >= 60.0 else INK,
                        clip_on=True,
                        zorder=4,
                    )

            ax.set_xticks(np.arange(len(metrics)))
            ax.set_xticklabels([])
            ax.xaxis.tick_top()
            ax.tick_params(axis="x", length=0)
            ax.set_yticks(np.arange(len(MODEL_ORDER)))
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

            for column, (metric, header) in enumerate(
                zip(metrics, tuple(spec["headers"]))
            ):
                ax.text(
                    column,
                    1.145,
                    header,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=5.35,
                    linespacing=0.88,
                    color=INK,
                    clip_on=False,
                )
                ax.text(
                    column,
                    1.025,
                    f"({int(denominators.loc[metric])})",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=5.0,
                    color=MUTED,
                    clip_on=False,
                )

            for row, model_label in enumerate(compact_model_labels):
                ax.text(
                    -0.020,
                    row,
                    model_label,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=5.6,
                    color=INK,
                    clip_on=False,
                )

            fig.text(
                0.010,
                float(spec["bottom"]) + 0.260,
                str(spec["family"]),
                ha="left",
                va="bottom",
                fontsize=6.35,
                fontweight="bold",
                color=MUTED,
            )

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.add_patch(
                Rectangle(
                    (-0.5, -0.5),
                    matrix.shape[1],
                    matrix.shape[0],
                    fill=False,
                    edgecolor="#C9C3D0",
                    linewidth=0.5,
                    clip_on=False,
                    zorder=6,
                )
            )
            ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
            ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
            ax._verifin_heatmap_mesh = mesh  # type: ignore[attr-defined]

        fig._verifin_heatmap_axes = axes  # type: ignore[attr-defined]

        output_stem = Path(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        figure_metadata = {
            "Title": "Answer accuracy before VeriFin",
            "Subject": (
                "Transposed six-model XBRLBench answer-accuracy heatmap"
            ),
        }
        fig.savefig(
            output_stem.with_suffix(".pdf"),
            facecolor="white",
            metadata=figure_metadata,
        )
        fig.savefig(
            output_stem.with_suffix(".svg"),
            facecolor="white",
        )
        fig.savefig(
            output_stem.with_suffix(".png"),
            dpi=1200,
            facecolor="white",
        )
        return fig


def main(
    root: Path,
    *,
    show: bool = True,
    layout: str = "single_column_transposed",
) -> tuple[pd.DataFrame, plt.Figure]:
    """Score the six runs, export source data and the selected figure."""

    root = Path(root).expanduser().resolve()
    accuracy_pct, correct_counts, denominators = load_accuracy_data(root)
    source_data = source_data_frame(
        accuracy_pct,
        correct_counts,
        denominators,
    )

    if layout == "single_column_transposed":
        output_stem = (
            root / "figures/category_model_performance_transposed"
        )
        figure_builder = make_transposed_single_column_figure
    elif layout == "single_column":
        output_stem = (
            root / "figures/category_model_performance_single_column"
        )
        figure_builder = make_single_column_figure
    elif layout == "full_width":
        output_stem = root / "figures/category_model_performance"
        figure_builder = make_figure
    else:
        raise ValueError(
            "layout must be 'single_column_transposed', 'single_column', "
            "or 'full_width'."
        )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(
        root / "figures/category_model_performance_data.csv",
        index=False,
        float_format="%.6f",
    )
    figure = figure_builder(
        accuracy_pct,
        correct_counts,
        denominators,
        output_stem,
    )

    overall = (
        source_data[source_data["metric"] == "Overall"]
        .set_index("model")["accuracy_pct"]
        .reindex(MODEL_ORDER)
    )
    print("Overall generated-answer accuracy before VeriFin:")
    for model, value in overall.items():
        print(f"  {model}: {value:.1f}%")
    print(f"Saved {output_stem.with_suffix('.pdf')}")
    print(f"Saved {output_stem.with_suffix('.svg')}")
    print(f"Saved {output_stem.with_suffix('.png')}")
    print(
        "Saved "
        f"{root / 'figures/category_model_performance_data.csv'}"
    )

    if show:
        plt.show()
    return source_data, figure


if __name__ == "__main__":
    main(resolve_root())
