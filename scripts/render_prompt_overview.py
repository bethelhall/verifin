"""Render a compact, source-checked overview of three main VeriFin prompts."""
from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/prompt_overview"
CARDS = (
    {
        "title": "Generate an answer",
        "input": "Question + filing evidence",
        "quote": "Answer the financial question using only the retrieved filing evidence below.",
        "rule": "Do not invent missing numbers.",
        "output": "JSON answer with value, unit, and scale",
        "source": "src/verifiqa/generation/answer_generator.py",
        "symbol": "_PROMPT",
        "color": "#2D6196",
        "tint": "#F1F6FB",
    },
    {
        "title": "Plan verification",
        "input": "Question + evidence + formula policy",
        "quote": "Classify a financial question to set up formal arithmetic verification.",
        "rule": "You must not use, infer from, or anticipate any proposed answer.",
        "output": "Metric, formula, operand roles, and unit",
        "source": "src/verifiqa/agent/claim_classifier.py",
        "symbol": "_DECOMPOSE_PROMPT",
        "color": "#23766F",
        "tint": "#F0F8F6",
    },
    {
        "title": "Ground the facts",
        "input": "Required operand + period + evidence",
        "quote": "Extract the value for ONE financial line item from the evidence text below.",
        "rule": "Do not invent values not present in the evidence.",
        "output": "Source value, reported unit, and quotation",
        "source": "src/verifiqa/agent/fact_extractor.py",
        "symbol": "_PROMPT",
        "color": "#77528C",
        "tint": "#F7F3FA",
    },
)


def validate_excerpts() -> None:
    """Read prompt constants without importing or running model code."""
    for card in CARDS:
        tree = ast.parse((ROOT / card["source"]).read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == card["symbol"]
                                  for target in node.targets))
        prompt = " ".join(ast.literal_eval(assignment.value).split())
        for field in ("quote", "rule"):
            if " ".join(card[field].split()) not in prompt:
                raise ValueError(f"Displayed excerpt no longer matches {card['source']}: {field}")


def main() -> None:
    validate_excerpts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "pdf.fonttype": 42, "text.color": "#203345"}):
        fig = plt.figure(figsize=(13.2, 6.25), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, 1320), ylim=(625, 0))
        ax.axis("off")
        ax.text(48, 48, "Main prompts", fontsize=27, weight="bold", va="center")
        ax.text(48, 84, "Three selected instructions from the VeriFin pipeline",
                fontsize=14, color="#5C6E7D", va="center")
        for index, card in enumerate(CARDS):
            x = 48 + index * 416
            ax.add_patch(FancyBboxPatch((x, 126), 392, 366,
                         boxstyle="round,pad=0,rounding_size=15",
                         facecolor=card["tint"], edgecolor="#DCE4EB", linewidth=0.8))
            ax.plot((x + 24, x + 68), (151, 151), color=card["color"],
                    linewidth=4, solid_capstyle="round")
            ax.text(x + 24, 183, card["title"], fontsize=18.5,
                    weight="bold", va="center", color=card["color"])
            ax.text(x + 24, 215, card["input"], fontsize=10.8,
                    color="#5C6E7D", va="center")
            ax.text(x + 24, 252, textwrap.fill('“' + card["quote"] + '”', width=35),
                    fontsize=15.3, va="top", linespacing=1.45)
            ax.text(x + 24, 356, textwrap.fill(card["rule"], width=43),
                    fontsize=11.4, va="top", linespacing=1.4, color="#526575")
            ax.plot((x + 24, x + 368), (421, 421), color="#D8E1E8", linewidth=0.8)
            ax.text(x + 24, 442, "OUTPUT", fontsize=8.8, weight="bold",
                    color=card["color"], va="center")
            ax.text(x + 24, 465, card["output"], fontsize=10.7, va="center")
        ax.add_patch(FancyBboxPatch((48, 520), 1224, 57,
                     boxstyle="round,pad=0,rounding_size=11",
                     facecolor="#203B50", edgecolor="none"))
        ax.text(72, 549, "Structured claim + grounded facts  →  deterministic formal checks",
                fontsize=16.2, color="white", va="center")
        ax.text(1248, 549, "SMT / Z3", fontsize=12.5, weight="bold",
                color="#C6E3EB", va="center", ha="right")
        ax.text(48, 604, "Abridged excerpts from current code; full templates are linked in the README.",
                fontsize=10.5, color="#6D7D89", va="center")
        fig.savefig(OUTPUT.with_suffix(".png"), dpi=200, facecolor="white")
        fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
        plt.close(fig)
    print("Saved docs/figures/prompt_overview.png and .svg; all excerpts match their source templates.")


if __name__ == "__main__":
    main()
