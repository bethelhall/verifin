from __future__ import annotations

import re
from typing import Iterable

from verifiqa.types import RavResult


def numeric_answer_accuracy(predicted: str, gold: str, rel_tolerance: float = 0.05) -> bool:
    if _is_non_answer(predicted):
        return False
    targets = _gold_targets(gold)
    candidates = _all_numbers(predicted)
    if not targets:
        if not candidates:
            return predicted.strip().lower() == gold.strip().lower()
        return False
    if not candidates:
        return False
    return any(_numbers_close(candidate, target, rel_tolerance) for candidate in candidates for target in targets)


def summarize_results(results: Iterable[RavResult]) -> dict:
    rows = list(results)
    total = len(rows)
    if total == 0:
        return {}

    # Per-question answer correctness (gold-answer numeric match)
    correctness = {
        row.financebench_id: numeric_answer_accuracy(row.answer, row.gold_answer)
        for row in rows
        if row.gold_answer
    }
    correct = list(correctness.values())

    # Outcome buckets
    verified = [row for row in rows if row.verified]                              # VERIFIED + REPAIRED_VERIFIED
    violated = [row for row in rows if row.final_status == "VIOLATED"]
    abstained = [row for row in rows if row.abstained]
    repaired = [row for row in rows if row.final_status == "REPAIRED_VERIFIED"]

    # Verification precision / false-positive rate (on verified rows only)
    verified_correct = sum(1 for row in verified if correctness.get(row.financebench_id, False))
    verified_incorrect = sum(
        1 for row in verified
        if row.financebench_id in correctness and not correctness[row.financebench_id]
    )

    # Violation precision: among VIOLATED rows, how often was the gold answer
    # actually wrong (i.e., the counterexample correctly caught a bad answer)?
    violated_gold_wrong = sum(
        1 for row in violated
        if row.financebench_id in correctness and not correctness[row.financebench_id]
    )

    # Selective accuracy: accuracy on rows where the pipeline gave an affirmative
    # answer (VERIFIED / REPAIRED_VERIFIED). VIOLATED and ABSTAIN are excluded —
    # they are refusals, not positive claims.
    verified_correct_vals = [
        numeric_answer_accuracy(row.answer, row.gold_answer)
        for row in verified
        if row.gold_answer
    ]

    return {
        "n": total,
        "qa_accuracy": _safe_mean(correct),
        # Coverage / outcome rates
        "verified_coverage": _safe_div(len(verified), total),
        "violation_rate": _safe_div(len(violated), total),
        "abstention_rate": _safe_div(len(abstained), total),
        "repair_rate": _safe_div(len(repaired), total),
        "coverage": _safe_div(len(verified) + len(violated), total),  # definitive verdicts
        # Quality of verdicts
        "verification_precision": _safe_div(verified_correct, len(verified)),
        "false_positive_rate": _safe_div(verified_incorrect, len(verified)),
        "violation_precision": _safe_div(violated_gold_wrong, len(violated)),
        # Accuracy on affirmative answers only
        "selective_accuracy": _safe_mean(verified_correct_vals),
        # SMT / solver diagnostics
        "smt_validity_rate": _safe_div(
            sum(1 for row in rows if row.first_pass_solver_status != "INVALID"), total
        ),
        "sat_first_pass_rate": _safe_div(
            sum(1 for row in rows if row.first_pass_solver_status == "SAT"), total
        ),
        "error_rate": _safe_div(sum(1 for row in rows if row.final_status == "ERROR"), total),
        "by_status": _by_status(rows, correctness),
        "by_metric": _by_metric(rows, correctness),
        "by_first_pass_status": _by_first_pass_status(rows),
    }


def _all_numbers(text: str) -> list:
    """Return numeric answer candidates with percent and money-scale variants."""
    values = []
    if _has_textual_zero(text):
        values.append(0.0)
    for token in _number_tokens(text, drop_years=False):
        values.extend(_candidate_values(token))
    return _dedupe(values)


def _gold_targets(text: str) -> list:
    values = []
    if _has_textual_zero(text):
        values.append(0.0)
    tokens = _number_tokens(text, drop_years=True)
    for token in tokens:
        values.extend(_candidate_values(token))

    percent_tokens = [token for token in tokens if token["percentish"]]
    for left in percent_tokens:
        for right in percent_tokens:
            if left is right:
                continue
            diff = left["value"] - right["value"]
            values.extend([diff, diff / 100.0])
    return _dedupe(values)


_DECREASE_WORDS = re.compile(
    r"\b(decreas|declin|fell|fall|drop|reduc|lost|loss|lower|shrink|contract)\w*\b"
)


def _number_tokens(text: str, *, drop_years: bool) -> list[dict]:
    tokens = []
    number_re = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?P<sign>[-+]?)\$?"
        r"(?P<number>(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
        r"(?P<suffix>mn|mm|bn|m|b)?"
        r"(?P<percent>%?)"
        r"(?![A-Za-z0-9,])",
        re.IGNORECASE,
    )
    for match in number_re.finditer(text or ""):
        raw = match.group(0)
        sign = match.group("sign") or ""
        value = float(f"{sign}{match.group('number')}".replace(",", ""))
        before = (text[max(0, match.start() - 24):match.start()] or "").lower()
        after = (text[match.end():match.end() + 32] or "").lower()
        if drop_years and _looks_like_year(raw, value, before, after):
            continue
        compact_scale = _compact_money_scale(match.group("suffix"), raw, before, after)
        tokens.append({
            "raw": raw,
            "value": value,
            "percentish": bool(match.group("percent")) or "percent" in after or "percentage" in after,
            "money_scale": compact_scale or _nearby_money_scale(before, after),
            "negated": value > 0 and bool(_DECREASE_WORDS.search(before)),
        })
    return tokens


def _candidate_values(token: dict) -> list[float]:
    value = token["value"]
    values = [value]
    if token.get("negated"):
        values.append(-value)
    if token["percentish"]:
        values.append(value / 100.0)
        if token.get("negated"):
            values.append(-value / 100.0)
    scale = token.get("money_scale")
    if scale:
        values.append(value * scale)
    return values


def _nearby_money_scale(before: str, after: str) -> float | None:
    context = f"{before} {after}"
    if "billion" in context:
        return 1_000_000_000.0
    if "million" in context:
        return 1_000_000.0
    if "thousand" in context:
        return 1_000.0
    return None


def _compact_money_scale(suffix: str | None, raw: str, before: str, after: str) -> float | None:
    if not suffix:
        return None
    suffix = suffix.lower()
    if suffix in {"mn", "mm"}:
        return 1_000_000.0
    if suffix == "bn":
        return 1_000_000_000.0

    context = f"{before} {raw} {after}"
    if "$" in raw or re.search(r"\b(usd|dollar|dollars|revenue|sales|income|ebitda|ebit|cash|debt|assets|liabilities|equity)\b", context):
        if suffix == "m":
            return 1_000_000.0
        if suffix == "b":
            return 1_000_000_000.0
    return None


def _looks_like_year(raw: str, value: float, before: str, after: str) -> bool:
    if raw.startswith("$") or raw.endswith("%") or re.search(r"(mn|mm|bn|[mb])%?$", raw, re.IGNORECASE) or not float(value).is_integer():
        return False
    year = int(value)
    if not 1900 <= year <= 2100:
        return False
    context = f"{before} {after}"
    return bool(re.search(r"\b(fy|fiscal|year|q[1-4]|ended|ending|as of|between|from|to|vs|versus)\b", context))


def _is_non_answer(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "could not be verified",
            "not verified",
            "cannot be verified",
            "z3 found a counterexample",
        )
    )


def _has_textual_zero(text: str) -> bool:
    return bool(re.search(r"\b(zero|flat|no change)\b", (text or "").lower()))


def _numbers_close(candidate: float, target: float, rel_tolerance: float) -> bool:
    if target == 0.0:
        return candidate == 0.0
    denom = max(abs(candidate), abs(target))
    if abs(candidate - target) / denom <= rel_tolerance:
        return True
    # Sign-convention tolerance: datasets sometimes express a decline as a positive
    # magnitude. Accept when absolute values are close but signs differ.
    if candidate * target < 0:
        return abs(abs(candidate) - abs(target)) / denom <= rel_tolerance
    return False


def _dedupe(values: list[float]) -> list[float]:
    out = []
    seen = set()
    for value in values:
        key = round(float(value), 12)
        if key not in seen:
            seen.add(key)
            out.append(float(value))
    return out


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _safe_mean(values) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _by_status(rows, correctness) -> dict:
    statuses = sorted({row.final_status or "UNKNOWN" for row in rows})
    output = {}
    for status in statuses:
        group = [row for row in rows if (row.final_status or "UNKNOWN") == status]
        correct = sum(
            1 for row in group
            if row.gold_answer and numeric_answer_accuracy(row.answer, row.gold_answer)
        )
        has_gold = sum(1 for row in group if row.gold_answer)
        output[status] = {
            "n": len(group),
            "answer_accuracy": _safe_div(correct, has_gold),
        }
    return output


def _by_metric(rows, correctness) -> dict:
    metrics = sorted({row.metric or "unknown" for row in rows})
    output = {}
    for metric in metrics:
        group = [row for row in rows if (row.metric or "unknown") == metric]
        verified = [row for row in group if row.verified]
        violated = [row for row in group if row.final_status == "VIOLATED"]
        verified_correct = sum(1 for row in verified if correctness.get(row.financebench_id, False))
        verified_incorrect = sum(
            1
            for row in verified
            if row.financebench_id in correctness and not correctness[row.financebench_id]
        )
        output[metric] = {
            "n": len(group),
            "verified_coverage": _safe_div(len(verified), len(group)),
            "violation_rate": _safe_div(len(violated), len(group)),
            "verification_precision": _safe_div(verified_correct, len(verified)),
            "false_positive_rate": _safe_div(verified_incorrect, len(verified)),
            "abstention_rate": _safe_div(sum(1 for row in group if row.abstained), len(group)),
        }
    return output


def _by_first_pass_status(rows) -> dict:
    statuses = sorted({row.first_pass_solver_status or "EMPTY" for row in rows})
    return {
        status: {
            "n": sum(1 for row in rows if (row.first_pass_solver_status or "EMPTY") == status),
            "abstained": sum(
                1
                for row in rows
                if (row.first_pass_solver_status or "EMPTY") == status and row.abstained
            ),
        }
        for status in statuses
    }
