"""Typed percentage subrows and explicitly disclosed annual changes.

Only question text and literal source layout select these operands. A reported
percentage subrow inherits its immediately preceding financial row's measure,
while an annual change retains both source years rather than pretending to be a
balance for one of those years.
"""
from __future__ import annotations

import re


def reported_percent_tables(evidence):
    from verifiqa.agent.source_question_planner import _Table, _Row, _identity, _number
    from verifiqa.agent.table_fact_extractor import _PDF_CELL, _MONTH, _caption_money_unit
    lines, result = evidence.splitlines(), []
    heading = re.compile(r"\s*Years?\s+Ended\s+(?:" + _MONTH + r")\s+\d{1,2},?\s*,?\s*%\s*Change\s*", re.I)
    parent_row = re.compile(r"\s*(?P<label>[A-Za-z][A-Za-z ,/&'’()-]*?)\s+"
                            r"(?P<a>" + _PDF_CELL + r")\s+(?P<b>" + _PDF_CELL + r")\s+"
                            r"(?P<change>\(?[-+]?\d+(?:\.\d+)?\)?\s*%)\s*", re.I)
    percent_row = re.compile(r"\s*(?P<label>%\s+of\s+[A-Za-z][A-Za-z '&/-]*?)\s+"
                             r"(?P<a>\(?[-+]?\d+(?:\.\d+)?\)?%)\s+"
                             r"(?P<b>\(?[-+]?\d+(?:\.\d+)?\)?%)\s*", re.I)
    for index, line in enumerate(lines):
        if not heading.fullmatch(line) or index + 2 >= len(lines):
            continue
        years = re.fullmatch(r"\s*((?:19|20)\d{2})\s+((?:19|20)\d{2})\s+((?:19|20)\d{2})\s*", lines[index + 1])
        if not years or years.group(1) == years.group(2) or years.group(1) != years.group(3):
            continue
        # The printed scale line establishes the three-column financial grid;
        # the subrow's explicit percent signs establish its own output unit.
        if not _caption_money_unit(lines[index + 2]):
            continue
        prefix = re.sub(r"\s*%\s*Change\s*$", "", line, flags=re.I).strip()
        table = _Table([prefix + " " + year for year in years.groups()[:2]],
                       section=lines[index - 1].strip() if index else "", unit="percent")
        parent, raw_lines, invalid = "", [line, lines[index + 1], lines[index + 2]], False
        for row_line in lines[index + 3:]:
            if not row_line.strip():
                continue
            parent_match = parent_row.fullmatch(row_line)
            if parent_match:
                if any(_number(parent_match.group(key)) is None for key in ("a", "b", "change")):
                    invalid = True
                    break
                parent = parent_match.group("label").strip()
                raw_lines.append(row_line)
                continue
            percent_match = percent_row.fullmatch(row_line)
            if percent_match and parent:
                raws = [percent_match.group("a"), percent_match.group("b")]
                values = [_number(raw) for raw in raws]
                if any(value is None for value in values):
                    invalid = True
                    break
                table.rows.append(_Row(parent + " " + percent_match.group("label"), values, raws,
                                       row_line, parent, "percent"))
                raw_lines.append(row_line)
                # A second percent subrow needs another explicit parent.
                parent = ""
                continue
            # A malformed numeric/table row cannot silently close the table.
            # Narrative text closes it only after a complete percent subrow.
            if re.match(r"\s*%\s+of\b", row_line, re.I) or not table.rows:
                invalid = True
            break
        if table.rows and not invalid:
            table.identity = _identity("\n".join(raw_lines), table.section)
            result.append(table)
    return result


def disclosed_change_plan(question, evidence, operation):
    """Bind disclosed changes; summing contiguous changes telescopes endpoints."""
    if operation not in {"period_change", "difference_of_changes"}:
        return None
    from verifiqa.agent.source_question_planner import (
        _Plan, _Table, _Row, _identity, _number, _row_match, _scope_supported, _slug,
    )
    from verifiqa.agent.types import ClaimSpec, GroundedFact, RoleSpec
    from verifiqa.agent.table_fact_extractor import _annual_row_groups, _caption_money_unit
    from verifiqa.answer_spec import answer_spec_from_question
    lines, candidates = evidence.splitlines(), []
    pair = re.compile(r"((?:19|20)\d{2})\s+and\s+((?:19|20)\d{2})", re.I)
    for index, line in enumerate(lines[:-1]):
        if not re.fullmatch(r"\s*Change\s+between\s+(?:Fiscal\s+)?increase\s*\(decrease\)\s*", line, re.I):
            continue
        header = lines[index + 1]
        unit = _caption_money_unit(header)
        if not unit:
            continue
        captionless = re.sub(r"^\s*\(?in\s+(?:thousands|millions|billions)\)?\s*", "", header, flags=re.I)
        matches = list(pair.finditer(captionless))
        if len(matches) < 2 or pair.sub("", captionless).strip():
            continue
        links = [(int(m.group(2)), int(m.group(1))) for m in matches]
        if any(end - start != 1 for start, end in links) or len(set(links)) != len(links):
            continue
        requested = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", question)]
        if operation == "period_change":
            if len(requested) != 2 or requested[0] >= requested[1]:
                continue
            first, last = requested
            selected = [i for i, (start, end) in enumerate(links) if first <= start < end <= last]
            chain = sorted((links[i] for i in selected))
            if (not chain or chain[0][0] != first or chain[-1][1] != last
                    or any(left[1] != right[0] for left, right in zip(chain, chain[1:]))):
                continue
            selected.sort(key=lambda i: links[i])
        else:
            requested_pairs = [(int(m.group(2)), int(m.group(1))) for m in pair.finditer(question)]
            if len(requested_pairs) != 2 or any(link not in links for link in requested_pairs):
                continue
            selected = [links.index(link) for link in requested_pairs]
            if len(set(selected)) != 2:
                continue
        columns = [line.strip() + " " + m.group() for m in matches]
        table = _Table(columns, section=line.strip(), unit=unit,
                       identity=_identity(line + "\n" + header, "disclosed changes"))
        for row_line in lines[index + 2:]:
            parsed = _annual_row_groups(row_line, len(columns))
            if parsed is None:
                break
            for section, label, raws in parsed:
                table.rows.append(_Row(label, [_number(raw) for raw in raws], raws, row_line, section, unit))
        rows = [row for row in table.rows if _row_match(row, question) and _scope_supported(question, table, row)]
        if len(rows) != 1 or any(rows[0].values[i] is None for i in selected):
            continue
        row, roles, facts = rows[0], [], {}
        for column_index in selected:
            start, end = links[column_index]
            name = _slug(row.label) + f"_change_{end}_{start}"
            column = columns[column_index]
            roles.append(RoleSpec(name, [row.label], column))
            facts[name] = GroundedFact(name, row.values[column_index], unit, row.quote,
                                       period=column, row_label=row.label, fact_type="source_table_numeric",
                                       source_value=row.values[column_index], source_unit=unit,
                                       source_section=table.section, source_column=column, source_table_id=table.identity)
        names = list(facts)
        formula = " + ".join(names) if operation == "period_change" else names[0] + " - " + names[1]
        spec = ClaimSpec(_slug(row.label) + "_" + operation, formula, roles, unit,
                         answer_spec_from_question(question).tolerance, "generic_operation",
                         period=str(links[selected[-1]][1]), operation=operation,
                         planning_source="deterministic_source_table")
        candidates.append(_Plan(spec, facts))
    return candidates[0] if len(candidates) == 1 else None
