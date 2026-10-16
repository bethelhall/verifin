"""Rank complete, independently delimited source populations.

No answer, proposed formula or benchmark annotation enters this planner. A
reported/reconciled total closes the PDF population; malformed/missing cells,
competing tables and a tie at the selection boundary remain unresolved.
"""
from __future__ import annotations

import re
from decimal import Decimal


def _request(question):
    matched = re.search(r'\b(?:top|largest|highest)\s+3\s+(.+?)(?:\s+add\s+up\b|\s+(?:in|for)\s+(?:(?:fiscal|fy)\s*)?(?:19|20)\d{2}\b|[?]|$)', question, re.I)
    years = set(re.findall(r'\b(?:19|20)\d{2}\b', question))
    if not matched or len(years) != 1 or re.search(r'\bquarter\b|\bq[1-4]\b', question, re.I):
        return None
    from .source_question_planner import _tokens
    population = {t for t in _tokens(matched.group(1)) if not t.isdigit()}
    return (next(iter(years)), population) if population else None


def _caption_matches(population, context):
    from .source_question_planner import _tokens
    return population <= set(_tokens(context))


def _number(raw):
    # Ranking cannot assume a blank/dash is zero: its missing value could be
    # larger than a selected item.
    from .source_question_planner import _number as parse
    import math
    value = parse(raw)
    return value if value is not None and math.isfinite(value) else None


def _decimal(raw):
    text = raw.replace('$', '').replace(',', '').strip()
    negative = text.startswith('(') and text.endswith(')')
    number = Decimal(text.strip('() '))
    return -abs(number) if negative else number


def _closed_rows(body, count):
    """Read complete labelled rows through one independently reconciled total."""
    from .table_fact_extractor import _PDF_CELL
    number = re.compile(r'(?<![A-Za-z])' + _PDF_CELL + r'(?![A-Za-z])')
    tokens = list(number.finditer(body))
    if not tokens:
        return None
    rows, cursor, index = [], 0, 0
    while index < len(tokens):
        start = index
        while index + 1 < len(tokens) and not body[tokens[index].end():tokens[index + 1].start()].strip():
            index += 1
        group = tokens[start:index + 1]
        label = body[cursor:group[0].start()].strip()
        label = re.sub(r'\s*\(\d{1,2}\)\s*$', '', label)
        # Numbers in row labels (dates, age/life subcolumns, note text) cannot
        # be guessed into the requested annual value columns.
        if len(group) not in {count, count * 2}:
            return None
        raws = [m.group().strip() for m in group]
        if any(_number(raw) is None for raw in raws):
            return None
        total_label = bool(re.fullmatch(r'(?:(?:grand\s+)?total|subtotal)(?:\s+.+)?', label, re.I))
        if re.match(r'\bsubtotal\b', label, re.I):
            return None
        if label and not total_label:
            if not re.search(r'[A-Za-z]', label) or re.search(r'\d|[.!?]', label):
                return None
            if label.lower().startswith(('less ', 'add ')):
                return None
            rows.append((label, raws[:count]))
        elif not total_label and not rows:
            return None
        closing = raws[count:] if len(raws) == count * 2 else (raws if not label or total_label else None)
        if closing is not None:
            if len(rows) < 3 or len({label.lower() for label, _ in rows}) != len(rows):
                return None
            if any(sum((_decimal(values[col]) for _, values in rows), Decimal(0)) != _decimal(closing[col])
                   for col in range(count)):
                return None
            tail = body[group[-1].end():].strip()
            if tail and not re.match(r'Less\s+accumulated\s+depreciation\b', tail, re.I):
                return None
            return rows
        cursor = group[-1].end()
        index += 1
    return None


def _pdf_populations(question, evidence, year, population):
    from .table_fact_extractor import _annual_pdf_header, _caption_money_unit
    from .source_question_planner import _Row, _Table, _identity
    lines = evidence.splitlines()
    tables = []
    for index, line in enumerate(lines):
        header = _annual_pdf_header(line)
        if not header:
            continue
        years, prefix, first_body = header
        if year not in years or len(set(years)) != len(years):
            continue
        # The closest printed definition supplies population identity, rather
        # than narrative from an earlier unrelated table.
        prior = lines[max(0, index - 5):index]
        definitions = [p.strip() for p in prior if re.search(
            r'\b(?:consist(?:s|ed)?|are\s+as\s+follows|composition|breakdown)\b', p, re.I)]
        context = definitions[-1] if definitions else ' '.join(prior[-2:])
        if not _caption_matches(population, context):
            continue
        unit = _caption_money_unit(context + ' ' + line)
        if not unit:
            continue
        body_lines = [first_body] if first_body else []
        # A complete population must close before a malformed row, another
        # dated grid or a paragraph. Never rank a parsed prefix of a table.
        for offset, following in enumerate(lines[index + 1:], index + 1):
            if _annual_pdf_header(following) or re.match(r'\s*(?:NOTE\s+\d|\d+\.\s+[A-Z])', following, re.I):
                break
            if not following.strip():
                if body_lines:
                    break
                continue
            body_lines.append(following)
            parsed = _closed_rows(' '.join(body_lines), len(years))
            if parsed is not None:
                after = next((rest.strip() for rest in lines[offset + 1:] if rest.strip()), '')
                # A subtotal cannot hide another component on the next line.
                # The full population ends at prose/a new table or a clearly
                # separated contra-asset calculation after its gross total.
                from .table_fact_extractor import _PDF_CELL
                cells = list(re.finditer(_PDF_CELL, after))
                if (after and not _annual_pdf_header(after) and not re.match(
                        r'Less\s+accumulated\s+depreciation\b', after, re.I)
                        and len(cells) >= len(years) and cells[-1].end() == len(after)):
                    body_lines.append(after)
                break
            if re.search(r'\.[ \t]+[A-Z]|\.$', following):
                break
        body = ' '.join(body_lines)
        parsed = _closed_rows(body, len(years))
        if parsed is None:
            continue
        columns = [((prefix[y] if isinstance(prefix, dict) else f'{prefix} {y}').strip()) for y in years]
        table = _Table(columns, section=context, unit=unit,
                       identity=_identity(line + '\n' + body, context))
        for label, raws in parsed:
            table.rows.append(_Row(label, [_number(raw) for raw in raws], raws,
                                   line + '\n' + body, unit=unit))
        tables.append(table)
    return tables


def _structured_populations(question, evidence, year, population):
    from .table_fact_extractor import _caption_money_unit, _pipe_tables_with_context
    from .source_question_planner import _Row, _Table, _identity
    tables = []
    for grid, title, raw_lines in _pipe_tables_with_context(evidence):
        if not grid:
            continue
        if all(row and not row[0] and not row[-1] for row in grid):
            grid = [row[1:-1] for row in grid]
        header = grid[0]
        if len(header) < 2 or not all(re.fullmatch(r'\s*(?:19|20)\d{2}\s*', value) for value in header[1:]):
            continue
        if year not in header[1:] or len(set(header[1:])) != len(header) - 1:
            continue
        context = title + ' ' + header[0]
        unit = _caption_money_unit(context)
        if not unit or not _caption_matches(population, context):
            continue
        table = _Table(header[1:], section=context, unit=unit,
                       identity=_identity('\n'.join(raw_lines), context))
        valid, closed = True, False
        for row in grid[1:]:
            if all(re.fullmatch(r'\s*:?-+:?\s*', cell or '') for cell in row):
                continue
            if len(row) != len(header) or closed:
                valid = False
                break
            values = [_number(raw) for raw in row[1:]]
            if not row[0].strip() or any(value is None for value in values):
                valid = False
                break
            if re.fullmatch(r'(?:total|subtotal)(?:\s+.+)?', row[0].strip(), re.I):
                if any(sum((_decimal(r.raws[col]) for r in table.rows), Decimal(0)) != _decimal(row[col + 1])
                       for col in range(len(table.columns))):
                    valid = False
                closed = True
                continue
            table.rows.append(_Row(row[0], values, row[1:], '\n'.join(raw_lines), unit=unit))
        if valid and closed and len(table.rows) >= 3 and len({r.label.lower() for r in table.rows}) == len(table.rows):
            tables.append(table)
    return tables


def rank_plan(question, evidence):
    """Return a complete source-grounded top-three sum plan or ``None``."""
    requested = _request(question)
    if requested is None:
        return None
    year, population = requested
    # A vague revenue question cannot silently prefer geographical revenue
    # over quarterly revenue when both populations are present in the excerpt.
    if 'revenue' in population and not re.search(r'\b(?:geographic|geographical|regions?|countr(?:y|ies))\b', question, re.I):
        quarter = re.search(r'^\s*Fiscal\s+' + year + r'\s*$([\s\S]{0,1800})', evidence, re.I | re.M)
        if quarter and re.search(r'First\s+Quarter.*Fourth\s+Quarter.*\bRevenues?\b', quarter.group(1), re.I | re.S):
            return None
    from .source_question_planner import _build, _signature
    candidates = {}
    for table in (_pdf_populations(question, evidence, year, population)
                  + _structured_populations(question, evidence, year, population)):
        indices = [index for index, column in enumerate(table.columns) if re.search(r'\b' + year + r'\b', column)]
        if len(indices) != 1:
            continue
        column = indices[0]
        ranked = sorted(table.rows, key=lambda row: row.values[column], reverse=True)
        if len(ranked) < 3 or (len(ranked) > 3 and ranked[2].values[column] == ranked[3].values[column]):
            continue
        plan = _build(question, 'sum', table, [(row, column) for row in ranked[:3]])
        if not plan:
            continue
        plan.spec.operation = 'selected_three_row_sum'
        plan.spec.metric += '_top_three'
        key = (_signature(plan.spec), tuple((f.row_label, f.value, f.source_table_id) for f in plan.facts.values()))
        candidates[key] = plan
    return next(iter(candidates.values())) if len(candidates) == 1 else None
