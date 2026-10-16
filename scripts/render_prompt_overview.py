"""Render a minimal overview of three main VeriFin prompt instructions."""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/prompt_overview"
CARDS = (
    {
        "title": "Answer generation",
        "instruction": "Answer using only the evidence.\nDo not invent missing numbers.",
        "quote": "Answer the financial question using only the retrieved filing evidence below.",
        "rule": "Do not invent missing numbers.",
        "output": "Answer · unit · scale",
        "source": "src/verifiqa/generation/answer_generator.py",
        "symbol": "_PROMPT",
    },
    {
        "title": "Verification planning",
        "instruction": "Plan the calculation independently\nof the proposed answer.",
        "quote": "Classify a financial question to set up formal arithmetic verification.",
        "rule": "You must not use, infer from, or anticipate any proposed answer.",
        "output": "Formula · operands",
        "source": "src/verifiqa/agent/claim_classifier.py",
        "symbol": "_DECOMPOSE_PROMPT",
    },
    {
        "title": "Fact grounding",
        "instruction": "Extract source values and units.\nQuote the supporting evidence.",
        "quote": "Extract the value for ONE financial line item from the evidence text below.",
        "rule": "Do not invent values not present in the evidence.",
        "output": "Value · unit · source quote",
        "source": "src/verifiqa/agent/fact_extractor.py",
        "symbol": "_PROMPT",
    },
)


def validate_excerpts() -> None:
    """Check source anchors; displayed instructions are condensed paraphrases."""
    for card in CARDS:
        tree = ast.parse((ROOT / card["source"]).read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == card["symbol"]
                                  for target in node.targets))
        prompt = " ".join(ast.literal_eval(assignment.value).split())
        for field in ("quote", "rule"):
            if " ".join(card[field].split()) not in prompt:
                raise ValueError(f"Source anchor no longer matches {card['source']}: {field}")


def main() -> None:
    validate_excerpts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "pdf.fonttype": 42, "text.color": "#203345"}):
        fig = plt.figure(figsize=(12, 2.8), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set(xlim=(0, 1200), ylim=(280, 0))
        ax.axis("off")
        ax.text(24, 29, "Main prompts", fontsize=19, weight="bold", va="center")
        ax.text(1176, 29, "Condensed instructions", fontsize=10,
                color="#64748B", va="center", ha="right")
        for index, card in enumerate(CARDS):
            x = 24 + index * 390
            ax.add_patch(FancyBboxPatch((x, 64), 372, 192,
                         boxstyle="round,pad=0,rounding_size=9",
                         facecolor="#F8FAFC", edgecolor="#DFE5EC", linewidth=0.7))
            ax.text(x + 22, 100, card["title"], fontsize=14,
                    weight="bold", va="center", color="#244B6B")
            ax.text(x + 22, 132, card["instruction"], fontsize=12,
                    va="top", linespacing=1.55)
            ax.plot((x + 22, x + 350), (201, 201), color="#DFE5EC", linewidth=0.7)
            ax.text(x + 22, 228, card["output"], fontsize=10.5,
                    color="#64748B", va="center")
        fig.savefig(OUTPUT.with_suffix(".png"), dpi=200, facecolor="white")
        fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
        plt.close(fig)
    print("Saved docs/figures/prompt_overview.png and .svg; source anchors verified for the condensed instructions.")


if __name__ == "__main__":
    main()
