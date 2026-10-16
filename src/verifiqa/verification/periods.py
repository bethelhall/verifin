"""Value-independent ordering of explicitly reported financial periods."""
from __future__ import annotations

import re
from typing import Any

_MONTHS = {
    name: index for index, name in enumerate(
        ("january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"), 1
    )
}

_OPENING = re.compile(r"\b(?:beginning|opening|start)\b")
_CLOSING = re.compile(r"\b(?:ending|closing|end)\b")
_QUARTER = re.compile(r"\bq([1-4])\b|\b([1-4])(?:st|nd|rd|th)?\s+quarter\b|\b(first|second|third|fourth)\s+quarter\b")


def _normalise(text: str) -> str:
    return re.sub(r"[_]+", " ", (text or "").lower())


def _years(text: str) -> set[str]:
    return set(re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text))


def _calendar_dates(text: str) -> set[tuple[int, int, int]]:
    dates = {tuple(int(v) for v in match) for match in re.findall(
        r"((?:19|20)\d{2})-(\d{2})-(\d{2})", text)}
    for month_name, month in _MONTHS.items():
        for day, year in re.findall(r"\b" + month_name + r"\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s+((?:19|20)\d{2})\b", text):
            dates.add((int(year), month, int(day)))
    return dates


def _calendar_months(text: str) -> set[int]:
    # A quarter's encoded ordering month is not a calendar month: fiscal Q1
    # can end outside March. Count only written months and complete dates.
    return {month for name, month in _MONTHS.items() if re.search(r"\b" + name + r"\b", text)} | {
        month for _year, month, _day in _calendar_dates(text)
    }


def _without_dated_fiscal_annotation(text: str) -> str:
    # FY2020 and August 31, 2019 can label the same reported fiscal quarter.
    # A complete calendar date in this very field is authoritative; do not
    # reinterpret an unmarked contradictory year or infer the fiscal calendar.
    if _calendar_dates(text):
        return re.sub(r"\b(?:fy\s*|fiscal\s+year\s*)((?:19|20)\d{2})\b", " ", text)
    return text


def _measure_period_text(text: str) -> str:
    # An instrument's maturity year is not its observation period. This keeps
    # source-column authority for, e.g., notes due 2021 measured at 2019 year-end.
    return re.sub(r"\bdue\s+(?:19|20)\d{2}\b", "due", _normalise(text))


def _boundaries(text: str) -> set[int]:
    return ({-1} if _OPENING.search(text) else set()) | ({13} if _CLOSING.search(text) else set())


def _quarters(text: str) -> set[int]:
    return {int(match.group(1) or match.group(2)) if match.group(1) or match.group(2)
            else ("first", "second", "third", "fourth").index(match.group(3)) + 1
            for match in _QUARTER.finditer(text)}


def quarter_number(text: str) -> int | None:
    quarters = _quarters(_normalise(text))
    return next(iter(quarters)) if len(quarters) == 1 else None


def period_key(text: str) -> tuple[int, int, int] | None:
    text = _without_dated_fiscal_annotation(_normalise(text))
    years = _years(text)
    if len(years) != 1:
        return None
    year = int(next(iter(years)))
    dates = _calendar_dates(text)
    months = _calendar_months(text)
    if len(months) > 1:
        return None
    if dates:
        return next(iter(dates)) if len(dates) == 1 else None
    quarters = _quarters(text)
    if quarters:
        return (year, next(iter(quarters)) * 3, 0) if len(quarters) == 1 else None
    if len(months) == 1:
        month = next(iter(months))
        month_name = next(name for name, index in _MONTHS.items() if index == month)
        day = re.search(r"\b" + month_name + r"\s+(\d{1,2})\b", text)
        return year, month, int(day.group(1)) if day else 0
    boundaries = _boundaries(text)
    if len(boundaries) > 1:
        return None
    if boundaries:
        return year, next(iter(boundaries)), 0
    return year, 0, 0


def fact_period_key(name: str, fact: Any) -> tuple[int, int, int] | None:
    """Combine the reported year with its row or role's period qualifier.

    Source quotes may contain a whole table, including unrelated years and
    balances. Only the selected column, row label and role participate here.
    Opening/closing keys remain fiscal-period markers, not guessed dates.
    """
    key, _conflict = _fact_period_details(name, fact)
    return key


def fact_period_conflict(name: str, fact: Any) -> bool:
    """Distinguish contradictory metadata from genuinely absent metadata."""
    _key, conflict = _fact_period_details(name, fact)
    return conflict


def _fact_period_details(name: str, fact: Any) -> tuple[tuple[int, int, int] | None, bool]:
    named = _measure_period_text(name)
    # start/end often name the first/last comparative year of a flow, such as
    # revenue_start and revenue_end. Only explicit balance roles can supply
    # these bare qualifiers; the selected source row can qualify any measure.
    if not re.search(r"\b(?:balance|reserves?|accruals?)\b", named):
        named = re.sub(r"\b(?:start|end)\b", " ", named)
    texts = [
        _normalise(str(getattr(fact, "period", "") or "")),
        _normalise(str(getattr(fact, "column", "") or "")),
        named,
        _measure_period_text(str(getattr(fact, "row_label", "") or "")),
    ]
    texts = [_without_dated_fiscal_annotation(text) for text in texts]
    years = set().union(*(_years(text) for text in texts))
    if len(years) > 1:
        return None, True
    if not years:
        return None, len(set().union(*(_boundaries(text) for text in texts))) > 1
    year = next(iter(years))
    dates = set().union(*(_calendar_dates(text) for text in texts))
    months = set().union(*(_calendar_months(text) for text in texts))
    quarters = set().union(*(_quarters(text) for text in texts))
    if len(dates) > 1 or len(months) > 1 or (not dates and len(quarters) > 1):
        return None, True
    # Supply the one explicit year to yearless row/role qualifiers. More precise
    # dates, months and quarters retain precedence over annual opening/closing.
    keys = [period_key(f"{text} {year}") for text in texts]
    precise = [key for key in keys if key and key[1] not in (0, -1, 13)]
    if precise:
        # Two exact dates or two explicitly stated quarters cannot disagree.
        dated = {key for key in precise if key[2]}
        quartered = {key for text, key in zip(texts, keys) if key and _quarters(text)}
        if len(dated) > 1 or (not dated and len(quartered) > 1):
            return None, True
        if dated:
            return next(iter(dated)), False
        return precise[0], False
    boundaries = set().union(*(_boundaries(text) for text in texts))
    if len(boundaries) > 1:
        return None, True
    return (int(year), next(iter(boundaries)) if boundaries else 0, 0), False
