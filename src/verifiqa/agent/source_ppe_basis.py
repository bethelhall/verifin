"""Resolve PPE carrying amounts from the printed gross-to-net reconciliation.

An unqualified PPE total does not mean gross cost merely because a registry
alias says so. A note explicitly headed PPE, NET identifies its final carrying
amount. Unlabelled subtotals are usable only when the complete adjacent
reconciliation proves their role. No proposed answer is an input.
"""
from __future__ import annotations

from decimal import Decimal
import re


_PPE = r"property\s*(?:,?\s*plant\s*)?(?:,?\s+and\s+|\s*&\s*)equipment"
_MEASURE = re.compile(r"\b" + _PPE + r"\b", re.I)
_OPERATIONS = {"percent_change", "period_change", "average", "sum"}
_COMPONENT_QUESTION = re.compile(r"\b(?:depreciation|amortization|impairment|expense|disposal|purchases?|"
                                 r"additions?|sales?|equipment\s+under|leased)\b", re.I)
_HEADING = re.compile(r"\s*(?:(?:note\s*)?\d+[.)]?\s*)?(?P<measure>" + _PPE
                      + r"(?:\s*,?\s*(?:net|gross|at\s+cost))?)\s*", re.I)
_CELL = r"\$?\s*(?:\(\s*-?\d[\d,]*(?:\.\d+)?\s*\)|-?\d[\d,]*(?:\.\d+)?)"


def resolve_ppe_basis(question: str, evidence_text: str):
    """Return ``(applies, plan)``; an applicable unresolved basis must abstain.

    This deliberately handles only single-measure annual operations. Ratios
    requesting both net and gross retain their existing two-measure planner.
    """
    from verifiqa.agent.source_question_planner import _operation
    if (_operation(question) not in _OPERATIONS or len(_MEASURE.findall(question)) != 1
            or not _MEASURE.search(evidence_text)):
        return False, None
    # A net-only geographic/segment table already names its valuation basis;
    # it is not a gross-to-net reconciliation and retains its typed planner.
    competition = (re.search(_PPE + r"[^.\n]{0,30}(?:\bgross\b|\bat\s+cost\b)", evidence_text, re.I)
                   or re.search(r"\baccumulated\s+depreciation\b", evidence_text, re.I))
    if not competition:
        return False, None
    # Components/expenses are different measures, even when their names contain
    # PPE. Their existing planners remain responsible for selecting them.
    if _COMPONENT_QUESTION.search(question):
        return False, None
    net = bool(re.search(r"\bnet\b|\bcarrying\s+(?:amount|value)\b", question, re.I))
    gross = bool(re.search(r"\bgross\b|\bat\s+cost\b", question, re.I))
    if net and gross:
        return True, None
    requested = "net" if net else "gross" if gross else ""
    candidates = []
    lines = evidence_text.splitlines()
    headings = [(i, match) for i, line in enumerate(lines)
                if (match := _HEADING.fullmatch(line))]
    for position, (start, heading) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        # A later numbered note cannot donate cells or headings to this one.
        for i in range(start + 1, end):
            if re.match(r"\s*(?:NOTE\s+)?\d+[.)]\s+[A-Za-z]", lines[i], re.I):
                end = i
                break
        plan = _reconciliation(question, lines, start, end, heading.group("measure"), requested)
        if plan is not None:
            candidates.append(plan)
    # An explicitly requested basis does not need the unqualified-total
    # disambiguator when this particular reconciliation layout is unsupported.
    # Existing typed rows and XBRL binding can establish NET/GROSS directly;
    # ppe_source_basis_matches still checks that requested basis at authority.
    if requested and not candidates:
        return False, None
    # Competing source notes remain ambiguous, even when their numbers agree.
    return True, candidates[0] if len(candidates) == 1 else None


def _reconciliation(question, lines, start, end, heading, requested):
    from verifiqa.agent.source_question_planner import (
        _Row, _Table, _build, _identity, _number, _operation,
        _scope_supported, _selected_periods, _tokens,
    )
    from verifiqa.agent.table_fact_extractor import _annual_pdf_header, _caption_money_unit
    heading_net = bool(re.search(r"\bnet\b", heading, re.I))
    heading_gross = bool(re.search(r"\bgross\b|\bat\s+cost\b", heading, re.I))
    # Without an explicit requested basis, a source must identify the note's
    # overall measure. A cost caption introducing components is not a NET title.
    basis = requested or ("net" if heading_net else "gross" if heading_gross else "")
    if not basis:
        return None
    header_index, columns, unit = None, None, ""
    for i in range(start + 1, end):
        declared = _caption_money_unit(lines[i])
        if declared:
            unit = declared
        header = _annual_pdf_header(lines[i])
        if header is not None:
            years, prefix, body = header
            if body or len(years) != len(set(years)) or isinstance(prefix, dict):
                return None
            columns = [(str(prefix) + " " + year).strip() for year in years]
            header_index = i
            break
    if header_index is None or not unit:
        return None
    width = len(columns)
    values = _CELL + r"(?:\s+" + _CELL + r"){" + str(width - 1) + r"}"
    row_re = re.compile(r"(?P<label>.*?)(?P<values>(?<!\S)" + values + r")\s*", re.I)
    rows = []
    final_index = None
    for i in range(header_index + 1, end):
        line = lines[i].strip()
        if not line:
            continue
        match = row_re.fullmatch(line)
        if match is None:
            if not rows and _caption_money_unit(line) == unit:
                continue
            break
        label = match.group("label").strip()
        raws = [cell.group().strip() for cell in re.finditer(_CELL, match.group("values"))]
        if len(raws) != width:
            return None
        numbers = [_number(raw) for raw in raws]
        if any(value is None for value in numbers):
            return None
        rows.append((label, numbers, raws, lines[i]))
        final_index = i
    if len(rows) < 4:
        return None
    # Complete adjacent sequence: components, gross subtotal, accumulated
    # depreciation deduction, final net carrying amount. Do not infer a net
    # amount by choosing whichever number agrees with a candidate.
    gross_row, deduction, net_row = rows[-3:]
    components = rows[:-3]
    if not all(label and not re.search(r"\b(?:less|net|gross|accumulated)\b", label, re.I)
               for label, *_ in components):
        return None
    if gross_row[0] and not (_MEASURE.search(gross_row[0]) and re.search(
            r"\bgross\b|\bat\s+cost\b", gross_row[0], re.I)):
        return None
    if not re.match(r"\s*(?:Less\s*:?\s*)?Accumulated\s+(?:depreciation|amortization)\b",
                    deduction[0], re.I) or any(value > 0 for value in deduction[1]):
        return None
    if net_row[0] and not (_MEASURE.search(net_row[0]) and re.search(r"\bnet\b", net_row[0], re.I)):
        return None
    if not heading_net and not net_row[0]:
        return None  # An unlabeled footer needs the actual NET note heading.
    if not gross_row[0] and not any(_MEASURE.search(line) and re.search(r"\bat\s+cost\b", line, re.I)
                                  for line in lines[start:header_index]):
        return None
    for column in range(width):
        if sum(Decimal(str(row[1][column])) for row in components) != Decimal(str(gross_row[1][column])):
            return None
        if Decimal(str(gross_row[1][column])) + Decimal(str(deduction[1][column])) != Decimal(str(net_row[1][column])):
            return None
    selected = net_row if basis == "net" else gross_row
    label = selected[0] or (heading if basis == "net" else
                           next(_MEASURE.search(line).group() + ", at cost"
                                for line in lines[start:header_index]
                                if _MEASURE.search(line) and re.search(r"\bat\s+cost\b", line, re.I)))
    quote = "\n".join(lines[start:final_index + 1])
    row = _Row(label, selected[1], selected[2], quote, unit=unit)
    table = _Table(columns, [row], heading, unit, _identity(quote, heading), unit_declared=True)
    filler = {"what", "how", "wa", "was", "is", "are", "were", "doe", "did", "the", "change",
              "percentage", "percent", "increase", "decrease", "growth", "decline", "difference",
              "average", "mean", "sum", "total", "all", "both", "between", "during", "year", "ending",
              "ended", "end", "fiscal", "annual", "period", "value", "amount", "company", "s", "its",
              "net", "gros", "at", "cost", "carrying", "usd", "dollar", "thousand", "million", "billion"}
    from verifiqa.agent.table_fact_extractor import _MONTH
    filler.update(word.lower() for word in _MONTH.split("|"))
    requested_words = {word for word in _tokens(question) if not word.isdigit() and word not in filler}
    if not requested_words <= set(_tokens(heading + " " + label)):
        return None
    if not _scope_supported(question, table, row):
        return None
    operation = _operation(question)
    indices = _selected_periods(question, table, operation)
    if indices is None or len(indices) < 2:
        return None
    return _build(question, operation, table, [(row, i) for i in indices])


def ppe_source_basis_matches(ir, question, evidence_text):
    """Recheck the selected source measure for fallback and retry paths too."""
    # Explicit questions retain normal labelled-row/XBRL planning, but a broad
    # registry alias cannot substitute the opposite valuation basis. A QName
    # independently bound by the filing is valid even when its terse source
    # label (for example "Owned property and equipment") omits the word NET.
    from verifiqa.agent.source_question_planner import _operation
    if (_operation(question) in _OPERATIONS and len(_MEASURE.findall(question)) == 1
            and not _COMPONENT_QUESTION.search(question)):
        requested_net = bool(re.search(r"\bnet\b|\bcarrying\s+(?:amount|value)\b", question, re.I))
        requested_gross = bool(re.search(r"\bgross\b|\bat\s+cost\b", question, re.I))
        if requested_net != requested_gross:
            wanted = "net" if requested_net else "gross"
            for fact in ir.facts.values():
                bases = set()
                if re.search(r"\bnet\b", fact.row_label, re.I):
                    bases.add("net")
                if re.search(r"\bgross\b|\bat\s+cost\b", fact.row_label, re.I):
                    bases.add("gross")
                local = fact.concept.split(":")[-1]
                if local == "PropertyPlantAndEquipmentNet":
                    bases.add("net")
                if local == "PropertyPlantAndEquipmentGross":
                    bases.add("gross")
                if bases != {wanted}:
                    return False
    applies, plan = resolve_ppe_basis(question, evidence_text)
    if not applies:
        return True
    if plan is None or len(plan.facts) != len(ir.facts):
        return False
    from verifiqa.verification.operation_bindings import _source_population_member_matches
    unmatched = list(ir.facts.values())
    for source in plan.facts.values():
        matches = [fact for fact in unmatched if _source_population_member_matches(fact, source)
                   and fact.source_table_id == source.source_table_id
                   and fact.source_column == source.source_column]
        if len(matches) != 1:
            return False
        unmatched.remove(matches[0])
    return not unmatched
