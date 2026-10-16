"""Flatten a run's results.jsonl into a one-row-per-query summary.csv.

Each row carries the LLM's original answer, whether the verifier caught it, the
corrected/authoritative value, the final answer, and gold — so a run can be eyeballed
or pivoted without digging into the nested JSON.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Optional

from verifiqa.eval.metrics import numeric_answer_accuracy
from verifiqa.units import numbers_in_text

COLUMNS = [
    "financebench_id",
    "metric",
    "grounding",
    "final_status",
    "failure_reason",
    "abstention_category",
    "caught",                  # verifier flagged the original claim (first-pass UNSAT)
    "first_pass_solver_status",
    "repair_rounds",
    "llm_original",            # the LLM's original claimed value (pre-repair)
    "llm_original_correct",    # original claim vs gold
    "corrected",               # authoritative/recomputed value (when repaired)
    "final_answer",            # final answer string the pipeline returned
    "final_correct",           # final answer vs gold
    "verdict_vs_gold",         # true_accept|false_accept|true_reject|false_reject|abstain
    "gold_answer",
    "question",
]


def write_run_summary(run_dir: Path | str) -> Optional[Path]:
    """Write ``<run_dir>/summary.csv`` from ``results.jsonl``. Returns the path or None."""
    run_dir = Path(run_dir)
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return None

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_path = run_dir / "summary.csv"
    summary_rows = [_summary_row(row) for row in rows]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for summary_row in summary_rows:
            writer.writerow(summary_row)
    _write_verdict_vs_gold(run_dir, summary_rows)
    return out_path


def _write_verdict_vs_gold(run_dir: Path, summary_rows: list[dict[str, Any]]) -> Path:
    """Aggregate the per-row verdict_vs_gold labels into a confusion matrix + accuracy."""
    counts = {
        "true_accept": 0,    # VERIFIED  and verified claim matches gold
        "false_accept": 0,   # VERIFIED  but claim is wrong  (verifier soundness miss)
        "true_reject": 0,    # VIOLATED  and the rejected claim was wrong
        "false_reject": 0,   # VIOLATED  but the claim actually matched gold
        "ungradeable": 0,    # decided, but gold is prose / ambiguous — can't score
        "abstain": 0,        # ABSTAIN / historical UNVERIFIED_FORMULA / ERROR
    }
    for summary_row in summary_rows:
        counts[summary_row["verdict_vs_gold"]] += 1
    decided = counts["true_accept"] + counts["false_accept"] + counts["true_reject"] + counts["false_reject"]
    correct = counts["true_accept"] + counts["true_reject"]
    report = {
        "counts": counts,
        "n": len(summary_rows),
        "decided": decided,
        "verdict_accuracy": (correct / decided) if decided else None,
        "false_accept_rate": (counts["false_accept"] / decided) if decided else None,
        "false_reject_rate": (counts["false_reject"] / decided) if decided else None,
    }
    out_path = run_dir / "verdict_vs_gold.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    ir = row.get("verification_ir") or {}
    claim = row.get("claim") or {}
    gold = row.get("gold_answer", "")
    fa = (row.get("verification_checks") or {}).get("formula_authority") or {}

    llm_original = claim.get("claimed_value")
    if llm_original is None:
        llm_original = ir.get("claimed_value")
    claim_unit = claim.get("unit") or ir.get("claim_unit") or ""

    corrected = fa.get("computed_value")
    if corrected is None:
        corrected = row.get("revised_claim")

    final_answer = row.get("answer", "")
    final_status = row.get("final_status", "")
    failure_reason = row.get("failure_reason", "")
    abstention_category = (row.get("diagnostics") or {}).get("abstention_category", "")
    if final_status == "UNVERIFIED_FORMULA":
        final_status = "ABSTAIN"
        failure_reason = failure_reason or "formula_unverified"
        abstention_category = abstention_category or "unverified_formula"
    accepted = final_status in {"VERIFIED", "REPAIRED_VERIFIED"}

    final_correct = (
        numeric_answer_accuracy(str(final_answer), str(gold))
        if accepted and final_answer != ""
        else None
    )
    llm_original_correct = _correct(llm_original, claim_unit, gold)

    question = row.get("question", "")
    # Grade an accept against both the verified claim value and the numbers in the
    # answer text. The claim covers cases where the answer states the sign/value in
    # words (e.g. "decreased by 1 percentage point" -> claim -1.0); the answer text
    # covers cases where the claim field holds an intermediate (common in FinQA).
    accept_values: list[float] = []
    if accepted:
        claim_value = _as_float(llm_original)
        if claim_value is not None:
            accept_values.append(claim_value)
        accept_values.extend(numbers_in_text(str(final_answer)))
    verdict = _verdict_vs_gold(
        final_status,
        accepted,
        accept_values=accept_values,
        reject_value=_as_float(llm_original),
        gold_targets=_gold_targets(gold, question),
        decimals=_question_decimals(question),
        computed_value=_as_float(fa.get("computed_value")),
        gold_decimals=_gold_decimals(gold),
    )

    return {
        "financebench_id":          row.get("financebench_id", ""),
        "metric":                   row.get("metric", ""),
        "grounding":                row.get("grounding", ""),
        "final_status":             final_status,
        "failure_reason":           failure_reason,
        "abstention_category":       abstention_category,
        "caught":                   row.get("first_pass_solver_status") == "UNSAT",
        "first_pass_solver_status": row.get("first_pass_solver_status", ""),
        "repair_rounds":            row.get("repair_rounds", 0),
        "llm_original":             llm_original,
        "llm_original_correct":     llm_original_correct,
        "corrected":                corrected,
        "final_answer":             _one_line(final_answer),
        "final_correct":            final_correct,
        "verdict_vs_gold":          verdict,
        "gold_answer":              gold,
        "question":                 _one_line(row.get("question", "")),
    }


_SCALE_WORDS = {
    "mn", "m", "bn", "b", "k", "million", "millions", "billion", "billions",
    "thousand", "thousands", "usd", "fy", "approximately", "approx", "about",
}
_ROUND_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
# Relative tolerance for grading when the question mandates no rounding precision.
# Matches the repo's own numeric_answer_accuracy default (5%) so the verdict
# metric grades consistently with the rest of the codebase, and absorbs coarse
# gold rounding (e.g. gold "$0.40" vs a precise filed 0.389 -> 2.75% gap) while
# still flagging materially wrong answers (the smallest real error observed was
# ~5.8%, on fixed-asset turnover).
_NO_PRECISION_REL_TOL = 0.05


def _question_decimals(question: str) -> Optional[int]:
    """The rounding precision the question mandates, if any (e.g. 'round to two
    decimal places' -> 2). Used to grade at the precision the answer was asked for
    rather than a blanket relative tolerance."""
    text = " ".join((question or "").lower().split())
    match = re.search(r"round(?:ed)?(?: your answer)? to (\d+)\s+decimal", text)
    if match:
        return int(match.group(1))
    match = re.search(r"round(?:ed)?(?: your answer)? to (\w+)\s+decimal", text)
    if match and match.group(1) in _ROUND_WORDS:
        return _ROUND_WORDS[match.group(1)]
    return None


def _clean_gold_number(gold: Any) -> Optional[float]:
    """Parse gold into a single number only when it is a clean numeric answer
    (e.g. '30.8%', '$1,577.00', '$2,018mn', '-0.02', '0'). Returns None when gold
    is prose or carries more than one distinct number, so the verdict can be
    marked ungradeable instead of guessing from text."""
    text = str(gold or "").strip()
    if not text:
        return None
    content_words = [
        word for word in re.findall(r"[A-Za-z]{2,}", text)
        if word.lower() not in _SCALE_WORDS
    ]
    if len(content_words) > 1:
        return None  # prose answer — don't fabricate a numeric verdict
    # Includes scientific notation (e.g. gold "1e-05" for a per-share price), which
    # otherwise parses as two numbers ("1" and "05") and is dropped as ambiguous.
    matches = re.findall(r"-?\$?\(?-?[\d,]+\.?\d*(?:[eE][-+]?\d+)?\)?%?", text)
    values = []
    for raw in matches:
        negative = "(" in raw
        token = raw.replace("$", "").replace(",", "").replace("(", "").replace(")", "")
        token = token[:-1] if token.endswith("%") else token
        try:
            value = float(token)
        except ValueError:
            continue
        values.append(-abs(value) if negative else value)
    distinct = {round(v, 6) for v in values}
    if len(distinct) != 1:
        return None  # zero or multiple numbers — ambiguous
    return values[0]


_CHANGE_HINTS = (
    "change", "changed", "increase", "decrease", "decline", "drop", "dropped",
    "grow", "growth", "difference", "vs", "versus", "compared",
)
_TOTAL_HINTS = (
    "total", "combined", "sum", "altogether", "how much", "expect to pay",
    "expected to pay", "pay for", "in total",
)


def _tagged_numbers(text: str) -> list[float]:
    """Numbers in prose that carry a value marker ($, %, or a scale word like
    'mn'/'million'/'billion'). Year tokens (FY2023, bare 2021) have no such marker
    and are therefore excluded — which is what lets '$2,018mn in FY 2023' resolve
    to 2018 rather than tripping over the year."""
    values: list[float] = []
    patterns = (
        r"\$\s*(-?[\d,]+\.?\d*)",                                              # $2,018
        r"(-?[\d,]+\.?\d*)\s*(?:mn|m|bn|b|k|million|billion|thousand)\b",      # 2,018mn
        r"(-?[\d,]+\.?\d*)\s*%",                                               # 21.6%
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                value = float(match.group(1).replace(",", ""))
            except ValueError:
                continue
            if all(abs(value - seen) > 1e-9 for seen in values):
                values.append(value)
    return values


def _gold_targets(gold: Any, question: str = "") -> set[float]:
    """Acceptable numeric target(s) for grading.

    Clean numeric gold -> {value}. For prose gold, two conservative patterns are
    handled so the answer becomes gradeable instead of ungradeable:
      * a single unit-tagged value (years ignored) -> {value}      ("$2,018mn ... FY2023")
      * a 'from X to Y' phrasing on a change/difference question -> {Y-X, X-Y}
        (sign-agnostic, since change conventions vary)              ("from 24.6% to 21.6%")
    Genuinely ambiguous prose returns an empty set -> stays ungradeable.
    """
    clean = _clean_gold_number(gold)
    if clean is not None:
        return {clean}
    text = str(gold or "")
    tagged = _tagged_numbers(text)
    # Change/difference question with exactly two unit-tagged values ("from 24.6%
    # ... to 21.6%") -> the delta. Tagged-only avoids latching onto year tokens;
    # sign-agnostic since change conventions vary.
    if len(tagged) == 2 and "to" in text.lower() and any(
        hint in question.lower() for hint in _CHANGE_HINTS
    ):
        delta = round(tagged[1] - tagged[0], 6)
        return {delta, -delta}
    # Total/combined question whose gold lists two same-unit component values
    # ("pension $1097M, health $862M" for "how much ... pay for retirees") -> the
    # answer is their sum; accept the components and the total.
    if len(tagged) == 2 and any(hint in question.lower() for hint in _TOTAL_HINTS):
        return {tagged[0], tagged[1], round(tagged[0] + tagged[1], 6)}
    if len(tagged) == 1:
        return {tagged[0]}
    # Textual zero: "flat", "zero", "nil", "unchanged" with no other number means 0
    # for a numeric metric (e.g. "Real Growth was flat", "coverage ratio is zero").
    if not tagged and _is_textual_zero(text):
        return {0.0}
    return set()


_TEXTUAL_ZERO = ("flat", "zero", "nil", "unchanged", "no change", "none")


def _is_textual_zero(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in _TEXTUAL_ZERO)


def _gold_decimals(gold: Any) -> Optional[int]:
    """Number of decimal places gold is displayed at (e.g. '16.5%' -> 1, '$0.40'
    -> 2). Used to grade at gold's own precision when the question mandates none."""
    match = re.search(r"\d+\.(\d+)", str(gold or ""))
    return len(match.group(1)) if match else None


def _gold_match(values: list[float], gold_targets: set[float], decimals: Optional[int]) -> Optional[bool]:
    """True if any candidate value matches any gold target. None when there's
    nothing to compare (no value or no numeric gold)."""
    if not values or not gold_targets:
        return None
    for value in values:
        for gold_number in gold_targets:
            # Accept percent<->ratio equivalence (e.g. 93.5 vs gold 0.935): same
            # value, only the ratio/percent presentation differs. Matches the
            # repo's numeric_answer_accuracy (allow_percent_ratio).
            for candidate in (gold_number, gold_number * 100.0, gold_number / 100.0):
                if decimals is not None:
                    if round(value, decimals) == round(candidate, decimals):
                        return True
                    continue
                # No mandated precision: gold is often a coarse rounding of the
                # precise filed value (e.g. "$0.40" for a filed $389M -> 0.389B).
                # Use a FinanceBench-style relative band so a correct-but-more-
                # precise answer still matches, while materially wrong ones don't.
                denom = max(abs(value), abs(candidate), 1e-9)
                if abs(value - candidate) / denom <= _NO_PRECISION_REL_TOL:
                    return True
    return False


def _verdict_vs_gold(
    final_status: str,
    accepted: bool,
    accept_values: list[float],
    reject_value: Optional[float],
    gold_targets: set[float],
    decimals: Optional[int],
    computed_value: Optional[float] = None,
    gold_decimals: Optional[int] = None,
) -> str:
    """Classify the verifier's verdict against gold.

    Accept (VERIFIED/REPAIRED_VERIFIED): correct when the verified *final answer*
    matches a gold target (the answer is what was certified and shown to the user).

    Reject (VIOLATED): correct when the rejected *claim* did NOT match the truth.
    The truth anchor is the policy-**computed value** when available (the grounded,
    formula-derived answer the verifier actually checked against) — falling back to
    gold otherwise — and the comparison uses the question's precision, else gold's
    own displayed precision. This avoids the loose relative band labelling a
    genuinely-wrong-but-close rejected claim (e.g. claim 16.8 vs computed 16.52,
    gold 16.5%) as a false reject.

    Gold that cannot be reduced to a numeric target is "ungradeable"; everything
    else is no-verdict ("abstain").
    """
    if not accepted and final_status != "VIOLATED":
        return "abstain"
    if accepted:
        match = _gold_match(accept_values, gold_targets, decimals)
        if match is None:
            return "ungradeable"
        return "true_accept" if match else "false_accept"
    reject_targets = {computed_value} if computed_value is not None else gold_targets
    reject_decimals = decimals if decimals is not None else gold_decimals
    match = _gold_match(
        [reject_value] if reject_value is not None else [], reject_targets, reject_decimals
    )
    if match is None:
        return "ungradeable"
    return "false_reject" if match else "true_reject"


def _correct(value: Any, unit: str, gold: Any) -> Optional[bool]:
    if value is None:
        return None
    # numeric_answer_accuracy normalises a "%" answer against a ratio gold, but a bare
    # claimed_value carries no unit — attach "%" when the claim is a percent.
    text = f"{value}%" if str(unit or "").strip().lower() in ("percent", "percents", "%") else str(value)
    return numeric_answer_accuracy(text, str(gold))


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split())
