"""Bounded table plans from an explicitly supplied, matching source PDF.

The PDF path is document provenance supplied by the caller. This module never
searches a corpus, receives benchmark IDs, or uses answers to select a table.
Geometry supplies missing hierarchy and header spans; all chosen values must
also occur in the supplied retrieval text.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re


_MONTH = r"January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = r"(?:" + _MONTH + r")\s+\d{1,2},?\s+(?:19|20)\d{2}"


def source_pdf_plan(question, evidence, source_pdf_path):
    """Resolve one supported geometry-dependent plan, otherwise return None."""
    loaded = _matching_pdf(evidence, source_pdf_path)
    if loaded is None:
        return None
    document, digest = loaded
    try:
        candidates = []
        for page_number, page in enumerate(document):
            lines = _lines(page)
            group = _purchase_group_plan(question, evidence, lines, digest, page_number)
            if group is not None:
                candidates.append(group)
            currency = _annual_currency_plan(question, evidence, lines, digest, page_number)
            if currency is not None:
                candidates.append(currency)
        if len(candidates) == 1:
            candidates[0].spec.source_document_sha256 = digest
            return candidates[0]
        return None
    finally:
        document.close()


def source_pdf_group_matches(ir, question, evidence, source_pdf_path=""):
    """Rebuild a PDF group's or native-currency mean's entire population."""
    purchase = bool(re.search(r"\bpurchas(?:e|ing)\s+obligations\b", question or "", re.I))
    expected = source_pdf_plan(question, evidence, source_pdf_path)
    if expected is None:
        return not purchase and not any(getattr(fact, "source_table_id", "").startswith("source_pdf:")
                                        for fact in ir.facts.values())
    if expected.spec.operation not in {"sum", "average"} or len(expected.facts) != len(ir.facts):
        return False
    from verifiqa.verification.operation_bindings import _source_population_member_matches
    from verifiqa.verification.periods import fact_period_key, period_key
    remaining = list(ir.facts.items())
    for source in expected.facts.values():
        matches = [(name, fact) for name, fact in remaining
                   if getattr(fact, "source_table_id", "") == source.source_table_id
                   and fact_period_key(name, fact) == period_key(source.period)
                   and _source_population_member_matches(fact, source)]
        if len(matches) != 1:
            return False
        remaining.remove(matches[0])
    return not remaining


def _matching_pdf(evidence, source_pdf_path):
    if not evidence or not source_pdf_path:
        return None
    try:
        import pymupdf
        path = Path(source_pdf_path)
        if path.suffix.lower() != ".pdf" or not path.is_file():
            return None
        raw = path.read_bytes()
        document = pymupdf.open(stream=raw, filetype="pdf")
        if not document or document.is_encrypted:
            document.close()
            return None
        pdf_text = "\n".join(page.get_text() for page in document)
        # These excerpts may flatten wrapped table labels or reorder header
        # cells. Require the entire excerpt's textual/numeric content, rather
        # than a nearby common label or a value-only coincidence.
        if _content(pdf_text) != _content(evidence):
            document.close()
            return None
        return document, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, RuntimeError, ImportError):
        return None


def _content(text):
    words = Counter(re.findall(r"[a-z0-9]+", text.lower()))
    signs = Counter(re.findall(r"\(\s*-?\d[\d,]*(?:\.\d+)?\s*\)|(?<![\w])[-−]\s*\d[\d,]*(?:\.\d+)?", text))
    return words, Counter(re.sub(r"\s+", "", item) for item in signs.elements())


def _lines(page):
    groups = []
    for word in sorted(page.get_text("words"), key=lambda item: (item[1], item[0])):
        matches = [line for line in groups[-3:] if abs(line["y"] - word[1]) <= 2.0]
        if matches:
            matches[-1]["words"].append(word)
        else:
            groups.append({"y": word[1], "words": [word]})
    for line in groups:
        line["words"].sort(key=lambda word: word[0])
        line["text"] = " ".join(word[4] for word in line["words"])
    return sorted(groups, key=lambda line: line["y"])


def _purchase_question(question):
    return bool(re.fullmatch(r"\s*What\s+(?:is|was|are)\s+(?:the\s+)?total\s+"
                             r"purchas(?:e|ing)\s+obligations(?:\s+due)?\s*\??\s*", question or "", re.I))


def _raw_cell(words, *, currency="USD"):
    from verifiqa.agent.source_question_planner import _number
    from verifiqa.verification.money_units import explicit_money_currencies
    cells = []
    for word in words:
        token = word[4]
        declared = explicit_money_currencies(token) if token != "$" else set()
        if declared and declared != {currency}:
            return None
        if token in {"$", "NT$", "US$", "USD", "TWD"}:
            continue
        cells.append(re.sub(r"^(?:NT|US)(?=\$)", "", token))
    if len(cells) != 1 or _number(cells[0]) is None:
        return None
    return cells[0], _number(cells[0])


def _source_row(evidence, label, value, *, first_column=True, column_index=0):
    """Find the same literal label and selected cell in retrieval text."""
    from verifiqa.agent.source_question_planner import _number
    escaped = r"\s+".join(re.escape(word) for word in label.split())
    pattern = re.compile(r"(?<![A-Za-z])" + escaped + r"\s+(?P<cells>[^\n]*)", re.I)
    matches = []
    for match in pattern.finditer(evidence):
        numeric = r"\$?\s*(?:\(\s*-?\d[\d,]*(?:\.\d+)?\s*\)|-?\d[\d,]*(?:\.\d+)?)"
        quote_end = match.end()
        cursor = quote_end
        # Some PDF text readers put each cell on its own line. Extend only
        # across literal numeric/currency lines, stopping at the next label.
        for continuation in evidence[cursor:].splitlines(keepends=True):
            if not continuation.strip():
                cursor += len(continuation)
                continue
            if not re.fullmatch(r"[\d,.()$%\s—–−-]+", continuation):
                break
            cursor += len(continuation)
            quote_end = cursor
        tail = evidence[match.start("cells"):quote_end]
        cells = list(re.finditer(numeric, tail))
        if len(cells) <= column_index:
            continue
        selected = cells[column_index]
        if (first_column and tail[:selected.start()].strip()
                or _number(selected.group()) != value):
            continue
        matches.append(evidence[match.start():quote_end].rstrip())
    return matches[0] if len(matches) == 1 else None


def _purchase_group_plan(question, evidence, lines, digest, page_number):
    if not _purchase_question(question):
        return None
    from datetime import date
    from verifiqa.agent.source_question_planner import _Table, _Row, _build, _scope_supported
    from verifiqa.agent.table_fact_extractor import _caption_money_unit
    from verifiqa.agent.source_typed_tables import _foreign_currency
    from verifiqa.verification.money_units import explicit_money_currencies
    from verifiqa.verification.periods import period_key
    candidates = []
    for index, heading in enumerate(lines):
        if not re.fullmatch(r"Purchase\s+obligations\s*:", heading["text"], re.I):
            continue
        group_x = min(word[0] for word in heading["words"])
        above = [line for line in lines[:index] if heading["y"] - line["y"] < 110]
        totals = [word for line in above for word in line["words"]
                  if word[4].lower() == "total" and word[0] > group_x + 80]
        if len(totals) != 1:
            continue
        total = totals[0]
        if not any(re.search(r"\bContractual\s+Obligations\b", line["text"], re.I)
                   for line in above if abs(line["y"] - total[1]) < 3):
            continue
        captions = [line["text"] for line in above
                    if re.search(r"Payments\s+Due\s+By\s+Period", line["text"], re.I)]
        if len(captions) != 1:
            continue
        unit = _caption_money_unit(captions[0])
        table_context = " ".join(line["text"] for line in above)
        if (not unit or _foreign_currency(table_context)
                or explicit_money_currencies(table_context) != {"USD"}):
            continue
        less = [word for line in above for word in line["words"]
                if word[4].lower() == "less" and word[0] > total[2] and abs(word[1] - total[1]) < 15]
        if len(less) != 1:
            continue
        total_center = (total[0] + total[2]) / 2
        next_center = (less[0][0] + less[0][2]) / 2
        half_width = (next_center - total_center) / 2
        if half_width <= 5:
            continue
        left, right = total_center - half_width, total_center + half_width
        prefix = " ".join(line["text"] for line in lines[:index] if total[1] - 100 < line["y"] < total[1])
        if _foreign_currency(prefix):
            continue
        dates = re.findall(r"\b(?:at|as\s+of)\s+(" + _DATE + r"),?\s*,?\s*were\s+as\s+follows", prefix, re.I)
        if len(set(dates)) != 1:
            continue
        reporting_date = dates[0]
        try:
            date(*period_key(reporting_date))
        except (TypeError, ValueError):
            continue
        table = _Table(["Total"], section="Contractual Obligations / " + heading["text"], unit=unit,
                       identity=f"source_pdf:{digest}:page{page_number}:group{index}", unit_declared=True)
        pending, indent, closed, invalid = [], None, False, False
        for line in lines[index + 1:]:
            label_words = [word for word in line["words"] if (word[0] + word[2]) / 2 < left and word[4] != "$"]
            if not label_words:
                invalid = True
                break
            row_x = min(word[0] for word in label_words)
            if row_x <= group_x + 3:
                closed = True
                break
            if indent is None:
                indent = row_x
            if abs(row_x - indent) > 2.0 or row_x < group_x + 5:
                invalid = True
                break
            label_part = " ".join(word[4] for word in label_words)
            selected = _raw_cell([word for word in line["words"] if left <= (word[0] + word[2]) / 2 < right])
            if selected is None:
                if any((word[0] + word[2]) / 2 >= left for word in line["words"]) or label_part.endswith(":"):
                    invalid = True
                    break
                pending.append(label_part)
                continue
            raw, value = selected
            if value < 0:
                invalid = True
                break
            label = " ".join(pending + [label_part])
            pending = []
            quote = _source_row(evidence, label, value)
            if quote is None or any(row.label.lower() == label.lower() for row in table.rows):
                invalid = True
                break
            row = _Row(label, [value], [raw], quote, unit=unit)
            if not _scope_supported(question, table, row):
                invalid = True
                break
            table.rows.append(row)
        if invalid or not closed or pending or len(table.rows) < 2:
            continue
        plan = _build(question, "sum", table, [(row, 0) for row in table.rows])
        if plan is not None:
            plan.spec.metric = "purchase_obligations_sum"
            plan.spec.period = reporting_date
            for role in plan.spec.roles:
                role.period = reporting_date
                fact = plan.facts[role.name]
                fact.period = reporting_date
                fact.source_column = "Total | Payments Due By Period"
            candidates.append(plan)
    return candidates[0] if len(candidates) == 1 else None


def _annual_currency_plan(question, evidence, lines, digest, page_number):
    from verifiqa.agent.source_question_planner import _Table, _Row, _build, _operation, _row_match, _scope_supported, _most_complete_rows
    from verifiqa.verification.money_units import explicit_money_currencies
    if _operation(question) != "average":
        return None
    candidates = []
    for index, line in enumerate(lines[:-3]):
        if not re.fullmatch(r"Years?\s+ended\s+(?:" + _MONTH + r")\s+\d{1,2},?", line["text"], re.I):
            continue
        year_line, currency_line, scale_line = lines[index + 1:index + 4]
        if not all(re.fullmatch(r"(?:19|20)\d{2}", word[4]) for word in year_line["words"]):
            continue
        years = year_line["words"]
        currencies = currency_line["words"]
        if (len(years) != 2 or years[0][4] == years[1][4] or len(currencies) != 3
                or any(word[4] not in {"NT$", "US$"} for word in currencies)):
            continue
        scale = re.fullmatch(r"\(in\s+(thousands|millions|billions)\)", scale_line["text"], re.I)
        if not scale:
            continue
        centers = [(word[0] + word[2]) / 2 for word in currencies]
        year_centers = [(word[0] + word[2]) / 2 for word in years]
        assignment = [min(range(2), key=lambda j: abs(center - year_centers[j])) for center in centers]
        if any(abs(center - year_centers[0]) == abs(center - year_centers[1]) for center in centers):
            continue
        if any(not [i for i, group in enumerate(assignment) if group == year] or abs(
                sum(centers[i] for i, group in enumerate(assignment) if group == year)
                / assignment.count(year) - year_centers[year]) > 3.0 for year in range(2)):
            continue
        observations = [(years[group][4], "TWD" if word[4] == "NT$" else "USD")
                        for group, word in zip(assignment, currencies)]
        requested = explicit_money_currencies(question)
        complete = [currency for currency in {item[1] for item in observations}
                    if len({year for year, code in observations if code == currency}) == 2
                    and sum(code == currency for _, code in observations) == 2
                    and (not requested or requested == {currency})]
        if len(complete) != 1:
            continue
        code = complete[0]
        indices = [i for i, (_, currency) in enumerate(observations) if currency == code]
        borders = [centers[0] - (centers[1] - centers[0]) / 2]
        borders += [(left + right) / 2 for left, right in zip(centers, centers[1:])]
        borders += [centers[-1] + (centers[-1] - centers[-2]) / 2]
        table = _Table(["Year ended " + re.sub(r"^Years?\s+ended\s+", "", line["text"], flags=re.I)
                        + " " + observations[i][0] for i in indices], section=line["text"] + " / " + currencies[indices[0]][4],
                       unit=code + " " + scale.group(1).lower(),
                       identity=f"source_pdf:{digest}:page{page_number}:currency{index}", unit_declared=True)
        closed, invalid = False, False
        for body in lines[index + 4:]:
            label_words = [word for word in body["words"] if (word[0] + word[2]) / 2 < borders[0]]
            label = " ".join(word[4] for word in label_words)
            if not label_words or re.match(r"\s*\(\d+\)", label):
                break
            cells = [_raw_cell([word for word in body["words"] if borders[i] <= (word[0] + word[2]) / 2 < borders[i + 1]],
                               currency=observations[i][1])
                     for i in range(3)]
            if any(cell is None for cell in cells):
                invalid = True
                break
            values = [cells[i][1] for i in indices]
            quote = _source_row(evidence, label, cells[0][1], column_index=0)
            # Retrieval text must retain every selected currency observation,
            # not merely another amount with the same row name.
            if quote is None or any(_source_row(evidence, label, cells[i][1], first_column=False, column_index=i) is None for i in indices):
                invalid = True
                break
            table.rows.append(_Row(label, values, [cells[i][0] for i in indices], quote, unit=table.unit))
            if label.lower() == "total":
                closed = True
                break
        if invalid or not closed:
            continue
        rows = _most_complete_rows([row for row in table.rows if _row_match(row, question) and _scope_supported(question, table, row)])
        if len(rows) != 1 or any(sum(other.label.lower() == row.label.lower() for other in table.rows) != 1 for row in rows):
            continue
        from verifiqa.agent.source_question_planner import _selected_periods
        selected = _selected_periods(question, table, "average")
        if selected is None or len(selected) != 2:
            continue
        plan = _build(question, "average", table, [(rows[0], i) for i in selected])
        if plan is not None:
            candidates.append(plan)
    return candidates[0] if len(candidates) == 1 else None
