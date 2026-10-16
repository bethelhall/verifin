"""Read explicitly typed useful-life cells beside monetary carrying values.

This reader requires a printed two-column header and a dated table introduction.
It never infers a life from a money column or fills an omitted cell.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable

from verifiqa.agent.types import RoleSpec

_MONTH = r"January|February|March|April|May|June|July|August|September|October|November|December"
_NUMBER = r"\$?\s*[-+]?\d[\d,]*(?:\.\d+)?"
_LIFE_HEADER = re.compile(
    r"\bweighted\s+average\s+(?:estimated\s+)?(?:useful\s+)?life\s*\(\s*in\s+years\s*\)",
    re.I,
)
_DATED_INTRO = re.compile(
    r"\b(?:as\s+of|(?:during\s+the\s+)?year\s+ended)\s+(?:" + _MONTH
    + r")\s+\d{1,2},?\s+((?:19|20)\d{2})\b", re.I,
)


def useful_life_cells(
    role: RoleSpec, evidence_text: str, *, question: str = "",
    row_matches: Callable[[RoleSpec, str], bool],
) -> list[dict]:
    """Return source cells only when the role requests the explicit years column."""
    role_words = " ".join([role.name, *role.aliases]).replace("_", " ")
    asks_life = bool(re.search(r"\b(?:life|lives)\b", role_words, re.I))
    if not asks_life:
        asks_life = (bool(re.search(r"\b(?:life|lives)\b", question, re.I))
                     and not re.search(r"\bcarrying\s+value\b", role_words, re.I))
    years = re.findall(r"(?:19|20)\d{2}", role.period)
    if not asks_life or len(set(years)) != 1:
        return []
    # An annual acquisition table cannot satisfy a quarterly or explicitly
    # dated role merely because its year happens to match.
    if re.search(r"\bq[1-4]\b|quarter|\d{4}-\d{2}-\d{2}|" + _MONTH, role.period, re.I):
        return []
    lines = evidence_text.splitlines()
    cells = []
    for index, line in enumerate(lines):
        if not re.fullmatch(r"\s*Gross\s+carrying\s+value\s*", line, re.I):
            continue
        # A wrapped header may span several lines, but must contain exactly
        # these two typed columns and no body values or additional columns.
        header_end = None
        header_text = ""
        for end in range(index + 1, min(index + 6, len(lines))):
            candidate = " ".join(text.strip() for text in lines[index + 1:end + 1])
            if _LIFE_HEADER.fullmatch(candidate):
                header_end, header_text = end, candidate
                break
        if header_end is None:
            continue
        intro = " ".join(lines[max(0, index - 8):index])
        dated_years = set(_DATED_INTRO.findall(intro))
        if dated_years != {years[0]}:
            continue
        section = next((text.strip() for text in reversed(lines[max(0, index - 8):index])
                        if re.match(r"\s*(?:Estimated\s+)?Useful\s+Lives?\b", text, re.I)
                        and not re.search(r"\d|[.!?:]", text)), "")
        if not section:
            section = "Gross carrying value / " + header_text
        table_id = "typed:" + str(index) + ":" + hashlib.sha256(
            (intro + "\n" + header_text).encode()
        ).hexdigest()[:16]
        for row in lines[header_end + 1:]:
            if not row.strip():
                break
            if re.fullmatch(r"[A-Za-z][A-Za-z\s/-]*:\s*", row):
                continue  # A printed row group adds no numeric column.
            match = re.fullmatch(
                r"\s*(?P<label>[A-Za-z][A-Za-z\s/&'’(),.-]*?)\s+(?P<money>"
                + _NUMBER + r")\s+(?P<life>[-+]?\d+(?:\.\d+)?)\s*", row,
            )
            if match is None:
                # A one-cell monetary subtotal supplies no useful-life value.
                subtotal = re.fullmatch(
                    r"\s*[A-Za-z][A-Za-z\s/&'’(),.-]*?\s+" + _NUMBER + r"\s*", row,
                )
                if subtotal:
                    continue
                break
            label = match.group("label").strip()
            if not row_matches(role, label):
                continue
            raw = match.group("life")
            value = float(raw)
            if value < 0:
                continue
            cells.append({
                "row_label": label, "col_header": header_text,
                "raw": raw, "value": value, "source_quote": row,
                "unit": "years", "section": section, "table_id": table_id,
            })
    return cells
