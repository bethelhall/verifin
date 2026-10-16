"""Question-bound arithmetic beyond the five original operation families.

Never use the proposed answer or its formula to select operands. Ambiguous roles,
incomplete date sets, incompatible units, and zero divisors remain unresolved.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from verifiqa.formulas.evaluator import evaluate_formula
from verifiqa.types import VerificationIR
from .periods import fact_period_key


def compatible_units(ir: VerificationIR, names: list[str]) -> bool:
    units = {re.sub(r"\s+", " ", ir.facts[n].unit.lower()).strip().replace('millions', 'million')
             .replace('thousands', 'thousand').replace('billions', 'billion') for n in names if ir.facts[n].unit}
    return len(units) <= 1


_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
          r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_QUARTER = r"(?:\bq[1-4]\b|\b(?:first|second|third|fourth|[1-4](?:st|nd|rd|th)?)\s+quarter\b)"


def _measure_root(text: str) -> str:
    """Remove observation dates, retaining the identity of the measured item.

    A date in a row is as legitimate as a date in a column. In particular,
    ``Balance at January 1`` and ``Balance at December 31`` measure a balance;
    ``First Quarter`` alone still needs the role/column's measure (high or low).
    """
    text = (text or '').lower().replace('_', ' ')
    text = re.sub(r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b', ' ', text)
    text = re.sub(r'\b' + _MONTH + r'\b\.?\s*(?:\d{1,2}(?:st|nd|rd|th)?\b\s*,?)?', ' ', text)
    text = re.sub(_QUARTER, ' ', text)
    text = re.sub(r'\b(?:fy\s*)?(?:19|20)\d{2}\b|\bfy\d{2}\b', ' ', text)
    text = re.sub(r'\b(?:fiscal\s+years?|years?\s+ended|period\s+ended)\b', ' ', text)
    text = re.sub(r'\b(?:start|end|beginning|ending|opening|closing|earlier|later)\b', ' ', text)
    text = re.sub(r'\b(?:as\s+of|at|in|for)\s*$', ' ', text)
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z]+', ' ', text)).strip()


def _fact_measure(name, fact):
    label = fact.row_label or ''
    # A numeric cell mistakenly used as a row label is not a measure identity.
    if label and not re.search(r'[a-z]', label, re.I):
        return ''
    root = _measure_root(label)
    if root:
        return root
    root = _measure_root(name)
    generic = {'fact', 'value', 'amount', 'input', 'variable', 'first', 'second', 'source',
               'usd', 'dollars', 'year', 'fiscal year', 'quarter', 'period', 'date'}
    if len(root) >= 2 and root not in generic:
        return root
    columns = {root for root in (_measure_root(getattr(fact, 'source_column', '')),
                                 _measure_root(fact.column)) if root and root not in generic}
    if len(columns) == 1:
        return next(iter(columns))
    return ''


def same_measure(ir: VerificationIR, names: list[str]) -> bool:
    from verifiqa.policy.registry import DEFAULT_POLICY_REGISTRY
    from .xbrl_scope import requested_scopes
    if len({tuple(requested_scopes(ir.facts[n])) for n in names}) > 1:
        return False
    sections = {_measure_root(getattr(ir.facts[n], 'source_section', '')) for n in names}
    sections.discard('')
    if len(sections) > 1:
        return False
    roots = set()
    for name in names:
        fact = ir.facts[name]
        text = _fact_measure(name, fact)
        if not text:
            return False
        roots.add(text)
    # A broad inferred concept must not collapse distinct source quantities.
    qualifiers = ('basic', 'diluted', 'gross', 'net', 'high', 'low', 'current', 'noncurrent')
    for qualifier in qualifiers:
        if len({bool(re.search(r'\b' + qualifier + r'\b', root)) for root in roots}) > 1:
            return False
    if len(roots) == 1:
        return True
    concepts = set()
    for text in roots:
        # Only complete registered aliases establish synonymy here. A fuzzy
        # 'revenue' substring cannot equate two differently named businesses.
        exact = {concept.concept_id for concept in DEFAULT_POLICY_REGISTRY.concepts.values()
                 if any(_measure_root(alias) == text for alias in concept.aliases)}
        concepts.add(next(iter(exact)) if len(exact) == 1 else None)
    return None not in concepts and len(concepts) == 1


def question_scope_matches(ir: VerificationIR, question: str, operation: str) -> bool:
    """Check a named single-measure scope against actual source cell identities.

    A planner-created role called ``us_revenue`` cannot turn a Europe row into
    U.S. revenue. XBRL numeric facts have already passed dimensional binding;
    table/plain numeric facts must carry the scope in their source row/section.
    """
    if operation not in {'percent_change', 'period_change', 'average', 'sum',
                         'percentage_point_change', 'change_of_growth_rates',
                         'difference_of_changes'}:
        return True
    from .xbrl_scope import requested_scopes
    from verifiqa.types import VerificationFact
    requested = requested_scopes(VerificationFact(question, 0, ''))
    if len(requested) != 1:
        return True
    scope = requested[0]
    for fact in ir.facts.values():
        source = fact if fact.fact_type == 'xbrl_numeric' else VerificationFact(
            '', 0, '', row_label=fact.row_label,
            source_section=getattr(fact, 'source_section', ''))
        if scope not in requested_scopes(source):
            return False
    return True


def source_sign_basis_supported(ir, question, operation, evidence_text):
    """An unqualified contra-allowance change has two distinct sign bases.

    The printed negative deduction and the positive allowance balance can
    change with opposite signs. Do not select a basis by candidate agreement.
    Relative percentage changes cancel that common sign and stay supported.
    """
    if operation != 'period_change':
        return True
    measure = r'(?:provision|allowance)\s+for\s+doubtful\s+(?:receivables|accounts)'
    if not re.search(r'\b' + measure + r'\b', question, re.I):
        return True
    if re.search(r'\bless\s*:|\bsigned\b|\b(?:printed|shown)\s+(?:values?|amounts?|row)\b', question, re.I):
        return True
    source = evidence_text + '\n' + '\n'.join(fact.source_quote or '' for fact in ir.facts.values())
    return not bool(re.search(r'\bLess\s*:\s*' + measure + r'\s+\$?\s*(?:\(\s*\d|[-−]\s*\d)', source, re.I))


def question_population_matches(ir: VerificationIR, question: str, operation: str,
                                evidence_text: str) -> bool:
    """Authorize officer age/salary aggregates only for the complete population.

    This protects old/fallback LLM decompositions as well as deterministic
    plans: two grounded people are insufficient when the source names three.
    """
    if operation not in {'average', 'sum'}:
        return True
    ages = bool(re.search(r'\bage\b', question, re.I) and re.search(
        r'\b(?:executive\s+officers?|vice\s+presidents?)\b', question, re.I))
    salaries = bool(re.search(r'\bsalar(?:y|ies)\b', question, re.I) and re.search(
        r'\b(?:NEOs?|named\s+executive\s+officers?)\b', question, re.I))
    if not (ages or salaries):
        return True
    from verifiqa.agent.source_question_planner import plan_from_source, source_plan_facts
    spec = plan_from_source(question, evidence_text)
    if spec is None or spec.operation != operation:
        return False
    expected = source_plan_facts(spec, question, evidence_text)
    if expected is None or len(expected) != len(ir.facts):
        return False
    unmatched = list(ir.facts.values())
    for source in expected.values():
        matches = [fact for fact in unmatched if _source_population_member_matches(fact, source)]
        if len(matches) != 1:
            return False
        unmatched.remove(matches[0])
    return not unmatched


def undated_average_periods_match(ir: VerificationIR, question: str) -> bool:
    """Check measure and count for an explicitly annual undated mean.

    This is intentionally narrow: an annual/both-years/N-years request needs
    one distinct observation per year. Officer age and NEO salary populations
    have their own complete source-membership guard above.
    """
    people = (re.search(r'\bage\b', question, re.I) and re.search(
        r'\b(?:executive\s+officers?|vice\s+presidents?)\b', question, re.I)) or (
        re.search(r'\bsalar(?:y|ies)\b', question, re.I) and re.search(
            r'\b(?:NEOs?|named\s+executive\s+officers?)\b', question, re.I))
    if people:
        return True
    words = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
             'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
    count = re.search(r'\b([2-9]|10|two|three|four|five|six|seven|eight|nine|ten)\s+(?:fiscal\s+)?years?\b', question, re.I)
    both = bool(re.search(r'\bboth\s+(?:fiscal\s+)?years?\b', question, re.I))
    if not (count or both or re.search(r'\bannual\b', question, re.I)):
        return True
    names = list(ir.facts)
    if len(names) < 2 or not same_measure(ir, names):
        return False
    if count:
        raw = count.group(1).lower()
        requested = int(raw) if raw.isdigit() else words[raw]
        if len(names) != requested or (both and requested != 2):
            return False
    if both and len(names) != 2:
        return False
    years = []
    for name, fact in ir.facts.items():
        if re.search(_QUARTER, f'{name} {fact.row_label} {fact.period} {fact.column} {fact.source_column}'.replace('_', ' '), re.I):
            return False
        key = fact_period_key(name, fact)
        if key is None:
            return False
        years.append(key[0])
    return len(set(years)) == len(names)


def resolve_promotion_salary_change(ir, question, evidence_text):
    """Bind an explicitly disclosed salary change to its named promotion event.

    Before and after salaries can share one effective date. Ordinary fiscal
    ordering cannot establish those event states, and annual pay cannot replace
    either disclosed promotion salary.
    """
    from verifiqa.agent.source_typed_tables import promotion_salary_plans
    plans = promotion_salary_plans(question, evidence_text, 'percent_change')
    if len(plans) != 1 or len(ir.facts) != 2 or not compatible_units(ir, list(ir.facts)):
        return None
    roles = {}
    for source in plans[0].facts.values():
        matches = [name for name, fact in ir.facts.items()
                   if _source_population_member_matches(fact, source)]
        state = 'before' if re.search(r'\bsalary before promotion$', source.row_label, re.I) else (
            'after' if re.search(r'\bsalary after promotion$', source.row_label, re.I) else '')
        if len(matches) != 1 or not state or state in roles:
            return None
        roles[state] = matches[0]
    if set(roles) != {'before', 'after'} or roles['before'] == roles['after']:
        return None
    before, after = roles['before'], roles['after']
    if float(ir.facts[before].value) <= 0:
        return None
    return _make(ir, f'({after} - {before}) / {before} * 100',
                 start=before, end=after, event='promotion')


def composition_source_matches(ir, question, evidence_text):
    """A total-row contribution needs the actual, complete source column grid."""
    from verifiqa.agent.source_column_planner import column_layout_plans
    plans = column_layout_plans(question, evidence_text, 'proportion')
    if len(plans) != 1 or len(ir.facts) != len(plans[0].facts):
        return False
    unmatched = list(ir.facts.values())
    for source in plans[0].facts.values():
        matches = [fact for fact in unmatched if _source_population_member_matches(fact, source)
                   and fact.source_column == source.source_column
                   and fact.source_table_id == source.source_table_id]
        if len(matches) != 1:
            return False
        unmatched.remove(matches[0])
    return not unmatched


def _source_population_member_matches(fact, source):
    import math
    def words(text):
        return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()
    actual, expected = words(fact.row_label), words(source.row_label)
    if actual != expected:
        # A typed Age/Salary column may carry the measure outside the row name.
        measure = 'age' if source.unit == 'years' else 'salary'
        if not (words(getattr(fact, 'source_column', '')) == words(source.source_column)
                and words(actual + ' ' + measure) == expected):
            return False
    if not fact.source_quote or not source.source_quote:
        return False
    quote, expected_quote = words(fact.source_quote), words(source.source_quote)
    if not quote or not (quote in expected_quote or expected_quote in quote):
        return False
    source_years = set(re.findall(r'\b(?:19|20)\d{2}\b', source.period))
    from .periods import fact_period_conflict
    if fact_period_conflict(fact.name, fact):
        return False
    key = fact_period_key(fact.name, fact)
    if source_years and (not key or str(key[0]) not in source_years):
        return False
    column = getattr(fact, 'source_column', '') or fact.column
    if column and words(column) != words(source.source_column):
        return False
    if source.unit == 'years':
        if fact.unit.lower().strip() not in {'year', 'years'}:
            return False
        left, right = float(fact.value), float(source.value)
    else:
        from .money_units import money_currency
        currency = money_currency(source.unit)
        if currency is None or money_currency(fact.unit) != currency:
            return False
        left, right = (_money_base_value(fact.value, fact.unit, currency=currency),
                       _money_base_value(source.value, source.unit, currency=currency))
    if left is None or right is None or not all(math.isfinite(v) for v in (left, right)):
        return False
    if abs(left - right) > max(abs(right) * 1e-12, 1e-7):
        return False
    raw = getattr(fact, 'raw_value', None)
    raw_unit = getattr(fact, 'raw_unit', '')
    if raw is not None:
        raw_base = (float(raw) if source.unit == 'years' and raw_unit in {'year', 'years'}
                    else _money_base_value(raw, raw_unit, currency=currency if source.unit != 'years' else 'USD'))
        if raw_base is None or not math.isfinite(raw_base) or abs(raw_base - right) > max(abs(right) * 1e-12, 1e-7):
            return False
    return True


def _text(name, fact):
    return f"{name.replace('_', ' ')} {fact.row_label or ''}".lower()


def _match(ir, phrase, names=None):
    from .generic_ops import _best_fact_for_phrase
    entities = re.findall(r'\b(customer|supplier)\s+([a-z0-9]+)\b', phrase, re.I)
    if entities:
        # In "Customer A", A identifies a disclosed entity; it is not an
        # article to discard during ordinary financial-measure tokenization.
        if len(entities) != 1:
            return None
        kind, identifier = entities[0]
        pattern = r'\b' + re.escape(kind) + r'\s+' + re.escape(identifier) + r'\b'
        matches = [name for name in (names or list(ir.facts))
                   if re.search(pattern, ir.facts[name].row_label, re.I)]
        if len(matches) != 1:
            return None
        fact = ir.facts[matches[0]]
        if re.search(r'\b(?:revenues?|sales)\b', phrase, re.I):
            source = ' '.join((fact.row_label, getattr(fact, 'source_section', ''),
                               getattr(fact, 'source_column', '')))
            if not re.search(r'\b(?:revenues?|sales)\b', source, re.I):
                return None
        return matches[0]
    return _best_fact_for_phrase([(n, ir.facts[n]) for n in (names or list(ir.facts))], phrase)


def _unique(ir, pattern, names=None):
    matches = [n for n in (names or list(ir.facts)) if re.search(pattern, _text(n, ir.facts[n]), re.I)]
    return matches[0] if len(matches) == 1 else None


def _make(ir, formula, names=None, **roles):
    names = names or list(ir.facts)
    try:
        value = evaluate_formula(formula, {n: float(ir.facts[n].value) for n in names})
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    return dict(formula=formula, variables=sorted(names), computed=value, **roles)


def _difference_pairs(question):
    match = re.search(r"\bbetween\s+(.+?)(?:\?|$)", question, re.I)
    text = match.group(1) if match else re.sub(r"^.*?difference\s+(?:in|of)\s+", '', question, flags=re.I)
    # Try each boundary because financial row names themselves contain 'and'.
    for boundary in re.finditer(r"\s+(?:and|versus|against|vs\.?|over)\s+", text, re.I):
        yield text[:boundary.start()], text[boundary.end():]


def _difference(ir, question):
    if len(ir.facts) != 2 or not compatible_units(ir, list(ir.facts)):
        return None
    high, low = _unique(ir, r'\bhigh\b'), _unique(ir, r'\blow\b')
    if high and low and re.search(r'\bhigh\b.*\blow\b', question, re.I):
        return _make(ir, f'{high} - {low}', first_item=high, second_item=low)
    pairs = set()
    for first, second in _difference_pairs(question):
        a, b = _match(ir, first), _match(ir, second)
        if a and b and a != b:
            pairs.add((a, b))
    if len(pairs) != 1:
        return None
    a, b = next(iter(pairs))
    return _make(ir, f'{a} - {b}', first_item=a, second_item=b)


def _rate_value(name, fact):
    unit = (fact.unit or '').lower()
    if unit in {'percent', 'percentage', '%', 'percentage points'}:
        return name
    if unit in {'ratio', 'pure'}:
        return f'({name} * 100)'
    return None


def _point_change(ir, question):
    from .generic_ops import _facts_ordered_by_period, decline_magnitude_requested
    if not same_measure(ir, list(ir.facts)):
        return None
    ordered = _facts_ordered_by_period(ir)
    if not ordered:
        return None
    start, end = ordered
    a, b = _rate_value(start, ir.facts[start]), _rate_value(end, ir.facts[end])
    if not a or not b:
        return None
    formula = f'{a} - {b}' if decline_magnitude_requested(question) else f'{b} - {a}'
    return _make(ir, formula, start=start, end=end)


def _sum(ir, question):
    from .generic_ops import _year_span_from_question, _fact_years
    names = list(ir.facts)
    if len(names) < 2 or not compatible_units(ir, names):
        return None
    expected = _year_span_from_question(question)
    if expected is None:
        return None
    if expected:
        if not same_measure(ir, names):
            return None
        periods = [_annual_bucket(n, ir.facts[n]) for n in names]
        if any(p is None for p in periods):
            return None
        if any(len(p) > 1 for p in periods) and not _question_names_measures(ir, names, question):
            return None
        covered = [year for period in periods for year in period]
        if sorted(covered) != expected:
            return None
        names.sort(key=lambda n: _annual_bucket(n, ir.facts[n])[0])
    elif re.search(r'\bdue\s+within\s+\d+\s+years?\b', question, re.I):
        return _maturity_sum(ir, question)
    elif not re.search(r'\band\b|\b(?:total salary|annual salary|purchas(?:e|ing) obligations)\b', question, re.I):
        return None
    # A source total plus its children would double count the same population.
    if len(names) > 1 and any(re.search(r'\btotal\b', ir.facts[n].row_label, re.I) for n in names) and not expected:
        return None
    return _make(ir, ' + '.join(names), names)


def _annual_bucket(name, fact):
    """The actual reported annual bucket; never expand a role's guessed years."""
    from .periods import fact_period_conflict
    for text in (getattr(fact, 'source_column', ''), fact.column, fact.period):
        text = (text or '').strip().lower()
        if not text:
            continue
        interval = re.fullmatch(r'(?:fy\s*|fiscal\s+years?\s*)?((?:19|20)\d{2})\s*(?:-|–|—|to|through)\s*((?:19|20)\d{2})(?:\s+years?)?', text)
        if interval:
            first, last = map(int, interval.groups())
            return list(range(first, last + 1)) if 0 <= last - first <= 20 else None
        if re.fullmatch(r'(?:fy\s*|fiscal\s*(?:year\s*)?)?(?:19|20)\d{2}', text):
            if fact_period_conflict(name, fact):
                return None
            key = fact_period_key(name, fact)
            if key and key[1] not in (0, -1, 13):
                return None
            return [int(re.search(r'(?:19|20)\d{2}', text).group())]
        years = set(re.findall(r'\b(?:19|20)\d{2}\b', text))
        if len(years) == 1 and not fact_period_conflict(name, fact):
            year = int(next(iter(years)))
            # An annual flow explicitly says year ended. A reported stock
            # balance may instead use its complete as-of date in each year.
            annual = re.search(r'\b(?:fiscal\s+)?years?\s+(?:ended|ending)\b', text)
            measure = f'{fact.row_label} {getattr(fact, "source_section", "")}'.lower()
            stock = re.search(r'\b(?:balances?|accruals?|reserves?|assets?|liabilit(?:y|ies)|goodwill)\b', measure)
            key = fact_period_key(name, fact)
            explicit_quarter = re.search(_QUARTER, f'{name} {fact.period} {fact.row_label} {text}'.replace('_', ' '), re.I)
            if not explicit_quarter and (annual or (stock and key and key[2])):
                return [year]
        # A specific reported quarter/date or a change column is not annual.
        return None
    from .generic_ops import _fact_years
    years = _fact_years(name, fact)
    return years if len(years) == 1 and not fact_period_conflict(name, fact) else None


def _maturity_bucket(fact):
    # Contractual maturities are column identities, not reporting dates or
    # assistant-created role aliases. Missing source column authority abstains.
    text = str(getattr(fact, 'source_column', '') or fact.column).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    first = re.fullmatch(r'(?:less than|under)\s+(\d+)\s+years?', text)
    interval = re.fullmatch(r'(\d+)\s*(?:-|–|—|to)\s*(\d+)\s+years?', text)
    if first:
        return 0, int(first.group(1))
    if interval:
        return tuple(map(int, interval.groups()))
    return None


def _maturity_sum(ir, question):
    match = re.search(r'\bdue\s+within\s+(\d+)\s+years?\b', question, re.I)
    if not match or not same_measure(ir, list(ir.facts)):
        return None
    if not _question_names_measures(ir, list(ir.facts), question):
        return None
    horizon = int(match.group(1))
    buckets = []
    for name, fact in ir.facts.items():
        bucket = _maturity_bucket(fact)
        if not bucket or not 0 <= bucket[0] < bucket[1] <= horizon:
            return None
        buckets.append((*bucket, name))
    buckets.sort()
    if not buckets or buckets[0][0] != 0 or buckets[-1][1] != horizon:
        return None
    if any(left[1] != right[0] for left, right in zip(buckets, buckets[1:])):
        return None
    periods = {fact_period_key(n, f) for n, f in ir.facts.items()}
    if len(periods) != 1 or None in periods:
        return None
    tables = {f.source_table_id for f in ir.facts.values() if f.source_table_id}
    if len(tables) > 1:
        return None
    names = [name for _, _, name in buckets]
    return _make(ir, ' + '.join(names), names)


def _question_names_measures(ir, names, question):
    from .generic_ops import _content_tokens
    def tokens(text):
        return {word[:-1] if len(word) > 4 and word.endswith('s') else word
                for word in _content_tokens(_measure_root(text))}
    requested = tokens(question)
    return all(bool(tokens(ir.facts[name].row_label))
               and tokens(ir.facts[name].row_label) <= requested for name in names)


def _source_populations(evidence_text, names, ir, year):
    """Find complete source tables containing all selected labels at one year.

    Supports pipe tables and the simple year-header/row layout of PDF excerpts.
    Ranking occurs on source cells only. A table boundary or ambiguous column
    causes this conservative parser to leave the operation unresolved.
    """
    from verifiqa.agent.table_fact_extractor import _parse_tables, _parse_number
    labels = {re.sub(r'\s+', ' ', ir.facts[n].row_label.strip().lower()) for n in names}
    if '' in labels:
        return []
    tables = _parse_tables(evidence_text)
    # Reconstruct only contiguous, unambiguous numeric PDF rows.
    lines = (evidence_text or '').splitlines()
    for index, line in enumerate(lines):
        years = re.findall(r'\b(?:19|20)\d{2}\b', line)
        if str(year) not in years or len(years) != len(set(years)):
            continue
        if re.sub(r'\b(?:19|20)\d{2}\b|\s', '', line):
            continue
        table = [['', *years]]
        for row in lines[index + 1:]:
            match = re.fullmatch(r'(.+?)\s+((?:\$?\s*\(?-?\d[\d,.]*\)?%?|[—–-])(?:\s+(?:\$?\s*\(?-?\d[\d,.]*\)?%?|[—–-]))*)\s*', row.strip())
            if not match:
                break
            cells = re.findall(r'\(?-?\d[\d,.]*\)?%?|[—–-]', match.group(2))
            if len(cells) != len(years):
                break
            table.append([match.group(1), *cells])
        if len(table) > 1:
            tables.append(table)
    populations = []
    for table in tables:
        columns = [j for j, header in enumerate(table[0]) if re.search(r'\b' + str(year) + r'\b', header)]
        if len(columns) != 1:
            continue
        col = columns[0]
        population = {}
        for row in table[1:]:
            if len(row) <= col or not row[0].strip() or re.search(r'\b(?:total|subtotal)\b', row[0], re.I):
                continue
            number = _parse_number(row[col])
            if number is not None:
                population[re.sub(r'\s+', ' ', row[0].strip().lower())] = number
        if labels <= population.keys():
            populations.append(population)
    return populations


def _top_three(ir, question, evidence_text):
    if len(ir.facts) != 3 or not compatible_units(ir, list(ir.facts)):
        return None
    if not re.search(r'\b(?:top|largest|highest)\s+3\b', question, re.I):
        return None
    years = set(re.findall(r'\b(?:19|20)\d{2}\b', question))
    if len(years) != 1:
        return None
    from verifiqa.agent.source_rank_planner import rank_plan
    planned = rank_plan(question, evidence_text)
    if planned is not None:
        def label(text):
            return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()
        expected = {label(f.row_label): f for f in planned.facts.values()}
        actual = {label(f.row_label): (name, f) for name, f in ir.facts.items()}
        if len(expected) != 3 or len(actual) != 3 or set(expected) != set(actual):
            return None
        names = []
        year = int(next(iter(years)))
        for identity, source in expected.items():
            name, fact = actual[identity]
            key = fact_period_key(name, fact)
            if not key or key[0] != year or key[1] not in (0, -1, 13):
                return None
            left = _money_base_value(fact.value, fact.unit)
            right = _money_base_value(source.value, source.unit)
            if left is None or right is None or abs(left - right) > max(abs(right) * 1e-12, 1e-7):
                return None
            names.append(name)
        return _make(ir, ' + '.join(names), names)
    # The older structured-table path has an explicit physical table boundary.
    # PDF row prefixes without an independent closing total are insufficient.
    if '|' not in evidence_text:
        return None
    from verifiqa.agent.source_rank_planner import _request, _caption_matches
    from verifiqa.agent.table_fact_extractor import _pipe_tables_with_context
    requested = _request(question)
    if requested is None:
        return None
    matching_tables = [table for table, title, _lines in _pipe_tables_with_context(evidence_text)
                       if table and _caption_matches(requested[1], title + ' ' + table[0][0])]
    if len(matching_tables) != 1:
        return None
    populations = _source_populations(evidence_text, list(ir.facts), ir, next(iter(years)))
    if len(populations) != 1:
        return None
    ranked = sorted(populations[0], key=populations[0].get, reverse=True)
    if len(ranked) < 3 or (len(ranked) > 3 and populations[0][ranked[2]] == populations[0][ranked[3]]):
        return None
    labels = {re.sub(r'\s+', ' ', f.row_label.strip().lower()) for f in ir.facts.values()}
    if labels != set(ranked[:3]):
        return None
    return _make(ir, ' + '.join(ir.facts))


def _money_base_value(value, unit, *, currency='USD'):
    text = unit.lower()
    from .money_units import money_currency
    if money_currency(unit) != currency:
        return None
    scale = next((scale for word, scale in (('billion', 1e9), ('million', 1e6), ('thousand', 1e3))
                  if word in text), 1)
    try:
        return float(value) * scale
    except (ValueError, TypeError):
        return None


def _exclusion(ir, question):
    if len(ir.facts) != 2 or not compatible_units(ir, list(ir.facts)):
        return None
    match = re.search(r'\bexcluding\s+(.+?)(?:\?|$)', question, re.I)
    if match:
        phrase = re.split(r',\s*(?:what|how|which)\b', match.group(1), flags=re.I)[0]
        component = _match(ir, phrase)
        if not component:
            return None
        whole = next(n for n in ir.facts if n != component)
    else:
        whole = _unique(ir, r'\b(?:original|initial)\b')
        component = _unique(ir, r'\b(?:outstanding|remaining|unpaid)\b')
        if not whole or not component or whole == component:
            return None
    return _make(ir, f'{whole} - {component}', whole=whole, component=component)


def _groups(ir):
    groups = defaultdict(list)
    for name, fact in ir.facts.items():
        key = fact_period_key(name, fact)
        if not key:
            return None
        measure = re.sub(r'\b(?:19|20)\d{2}\b|\bq[1-4]\b|\b(?:start|end|earlier|later|beginning|ending)\b', '', name.replace('_', ' ').lower())
        measure = re.sub(r'\s+', ' ', measure).strip()
        groups[measure].append((key, name))
    for values in groups.values():
        values.sort()
        if len({key for key, _ in values}) != len(values):
            return None
    return groups


def _difference_of_changes(ir, question):
    if not compatible_units(ir, list(ir.facts)):
        return None
    if len(ir.facts) == 2:
        return _disclosed_change_difference(ir, question)
    groups = _groups(ir)
    if not groups:
        return None
    if len(groups) == 1 and len(ir.facts) == 3:
        ordered = next(iter(groups.values()))
        requested = set(re.findall(r'\b(?:19|20)\d{2}\b', question))
        if requested != {str(k[0]) for k, _ in ordered}:
            return None
        a, b, c = [n for _, n in ordered]
        return _make(ir, f'({c} - {b}) - ({b} - {a})')
    if len(groups) == 2 and all(len(v) == 2 for v in groups.values()):
        from .generic_ops import _best_fact_for_phrase
        representatives = [(g, ir.facts[v[0][1]]) for g, v in groups.items()]
        pairs = set()
        for first, second in _difference_pairs(question):
            first = re.sub(r'\b(?:19|20)\d{2}\b', '', first)
            second = re.sub(r'\b(?:19|20)\d{2}\b', '', second)
            ga = _best_fact_for_phrase(representatives, first)
            gb = _best_fact_for_phrase(representatives, second)
            if ga and gb and ga != gb:
                pairs.add((ga, gb))
        if len(pairs) == 1:
            ga, gb = next(iter(pairs))
            if [k for k, _ in groups[ga]] != [k for k, _ in groups[gb]]:
                return None
            a0, a1 = [n for _, n in groups[ga]]
            b0, b1 = [n for _, n in groups[gb]]
            return _make(ir, f'({a1} - {a0}) - ({b1} - {b0})')
    return None


def resolve_disclosed_period_change(ir, question, operation):
    """Telescope a complete chain of source-disclosed level changes.

    A 2019-vs-2018 change plus a 2018-vs-2017 change establishes the absolute
    2017-to-2019 change. It does not establish a percentage change without the
    original level. Every interval must be an actual typed source column.
    """
    if operation != 'period_change' or len(ir.facts) < 2:
        return None
    names = list(ir.facts)
    if not compatible_units(ir, names) or not same_measure(ir, names):
        return None
    if not _question_names_measures(ir, names, question):
        return None
    asked = sorted({int(year) for year in re.findall(r'\b(?:19|20)\d{2}\b', question)})
    if len(asked) != 2 or re.search(_QUARTER + r'|\b' + _MONTH + r'\b', question, re.I):
        return None
    reverse = re.search(r'\bfrom\s+((?:19|20)\d{2})\s+to\s+((?:19|20)\d{2})\b', question, re.I)
    if reverse and int(reverse.group(1)) > int(reverse.group(2)):
        return None
    pattern = r'\b((?:19|20)\d{2})\s*(?:and|versus|vs\.?)\s*((?:19|20)\d{2})\b'
    chain = []
    for name, fact in ir.facts.items():
        column = str(getattr(fact, 'source_column', '') or fact.column)
        if not column or not re.search(r'\bchange\b', column, re.I):
            return None
        pairs = re.findall(pattern, column, re.I)
        if len(pairs) != 1:
            return None
        end, start = map(int, pairs[0])
        if not asked[0] <= start < end <= asked[1]:
            return None
        chain.append((start, end, name))
    chain.sort()
    if chain[0][0] != asked[0] or chain[-1][1] != asked[1]:
        return None
    if any(left[1] != right[0] for left, right in zip(chain, chain[1:])):
        return None
    from .generic_ops import decline_magnitude_requested
    names = [name for _, _, name in chain]
    expression = ' + '.join(names)
    if decline_magnitude_requested(question):
        expression = f'-({expression})'
    return _make(ir, expression, names, disclosed_change_intervals=[(a, b) for a, b, _ in chain])


def _disclosed_change_difference(ir, question):
    """Subtract explicitly reported change columns, without differencing again."""
    from .generic_ops import _content_tokens
    if not same_measure(ir, list(ir.facts)):
        return None
    pair_pattern = r'\b((?:19|20)\d{2})\s*(?:and|versus|vs\.?|to)\s*((?:19|20)\d{2})\b'
    requested = [tuple(map(int, pair)) for pair in re.findall(pair_pattern, question, re.I)]
    if len(requested) != 2 or requested[0] == requested[1]:
        return None
    measure = re.search(r'\bfor\s+(.+?)(?:\?|$)', question, re.I)
    if not measure:
        return None
    wanted = _content_tokens(measure.group(1))
    columns = {}
    for name, fact in ir.facts.items():
        # Generic change headers are insufficient to identify payroll, travel,
        # etc. The source row itself must identify the requested measure.
        label_tokens = _content_tokens(fact.row_label)
        if not wanted or not wanted <= label_tokens:
            return None
        texts = (getattr(fact, 'source_column', ''), fact.column, fact.period)
        pairs = set()
        for text in texts:
            pairs.update(tuple(map(int, pair)) for pair in re.findall(pair_pattern, text or '', re.I))
        if len(pairs) != 1:
            return None
        pair = next(iter(pairs))
        if pair[0] <= pair[1] or pair in columns:
            return None
        change_text = f'{name} {fact.row_label} {getattr(fact, "source_column", "")} {fact.column}'
        if not re.search(r'\bchange\b', change_text.replace('_', ' '), re.I):
            return None
        columns[pair] = name
    if set(columns) != set(requested):
        return None
    first, second = [columns[pair] for pair in requested]
    return _make(ir, f'{first} - {second}', first_change=first, second_change=second)


def _share_of_change(ir, question):
    if len(ir.facts) != 4 or not compatible_units(ir, list(ir.facts)):
        return None
    groups = _groups(ir)
    if not groups or len(groups) != 2 or any(len(v) != 2 for v in groups.values()):
        return None
    totals = [g for g, vs in groups.items() if all(re.search(r'\btotal\b', _text(n, ir.facts[n])) for _, n in vs)]
    if len(totals) != 1:
        return None
    total = totals[0]
    component = next(g for g in groups if g != total)
    if [k for k, _ in groups[total]] != [k for k, _ in groups[component]]:
        return None
    t0, t1 = [n for _, n in groups[total]]
    c0, c1 = [n for _, n in groups[component]]
    return _make(ir, f'({c0} - {c1}) / ({t0} - {t1}) * 100')


def _growth_change(ir, question):
    if len(ir.facts) == 2:
        return _point_change(ir, question)
    groups = _groups(ir)
    if not groups or len(groups) != 1 or len(ir.facts) != 3 or not compatible_units(ir, list(ir.facts)):
        return None
    a, b, c = [n for _, n in next(iter(groups.values()))]
    return _make(ir, f'(({c} - {b}) / {b} - ({b} - {a}) / {a}) * 100')


def _ratio_change(ir, question):
    if len(ir.facts) != 4:
        return None
    groups = _groups(ir)
    if not groups or len(groups) != 2 or any(len(v) != 2 for v in groups.values()):
        return None
    # Bind numerator/denominator from the clause describing each date's ratio.
    match = re.search(r'(?:for|in)\s+(.+?)\s+(?:expressed\s+)?as\s+a\s+percentage\s+(?:to|of)\s+(.+?)(?:\?|$)', question, re.I)
    if not match:
        return None
    representatives = [(g, ir.facts[v[0][1]]) for g, v in groups.items()]
    from .generic_ops import _best_fact_for_phrase
    numerator = _best_fact_for_phrase(representatives, re.sub(r'\b(?:19|20)\d{2}\b', '', match.group(1)))
    denominator = _best_fact_for_phrase(representatives, re.sub(r'\b(?:19|20)\d{2}\b', '', match.group(2)))
    if not numerator or not denominator or numerator == denominator:
        return None
    if [k for k, _ in groups[numerator]] != [k for k, _ in groups[denominator]]:
        return None
    a0, a1 = [n for _, n in groups[numerator]]
    b0, b1 = [n for _, n in groups[denominator]]
    if not compatible_units(ir, [a0, b0, a1, b1]):
        return None
    start, end = f'({a0} / {b0})', f'({a1} / {b1})'
    if re.search(r'percentage points?', question, re.I):
        return _make(ir, f'({end} - {start}) * 100')
    if re.search(r'percentage change|relative|percent change', question, re.I):
        return _make(ir, f'({end} - {start}) / {start} * 100')
    return None  # 'percentage difference' alone does not fix the scale.


def compound_ratio_phrases(question):
    """Return the two explicit ratio sides for complete operand coverage checks."""
    from .generic_ops import _ratio_phrases
    phrases = _ratio_phrases(question)
    if not phrases:
        match = re.search(r'\b(?:proportion|percentage|percent)\s+of\s+(.+?)\s+over\s+(.+?)(?:\?|$)', question, re.I)
        if not match:
            match = re.search(r'^.*?\b(?:is|was|are|were)\s+(.+?)\s+(?:expressed\s+)?as\s+a\s+percent(?:age)?\s+(?:of|to)\s+(.+?)(?:\?|$)', question, re.I)
        if match:
            phrases = match.group(1), match.group(2)
    if phrases:
        # In "net value expressed as a percentage to gross value for 2019",
        # value is question scaffolding. Keep net/gross and actual valuation
        # labels such as fair value or net realizable value intact.
        phrases = tuple(re.sub(
            r'\b(net|gross)\s+value(?=\s+(?:for|in)\s+(?:fiscal\s+year\s+)?(?:19|20)\d{2}\s*$|\s*$)',
            r'\1', phrase.strip(), flags=re.I,
        ) for phrase in phrases)
        # A single trailing observation year qualifies both sides of this
        # ratio. Never let a role from another year fill its undated side.
        shared_period = re.search(
            r'\s+(?:for|in)\s+(?:fiscal\s+year\s+)?((?:19|20)\d{2})\s*$', phrases[1], re.I,
        )
        if shared_period and not re.search(r'\b(?:19|20)\d{2}\b', phrases[0]):
            phrases = (f'{phrases[0]} in {shared_period.group(1)}', phrases[1])
    return phrases


def _sum_ratio(ir, question):
    phrases = compound_ratio_phrases(question)
    if not phrases or len(ir.facts) < 3 or not compatible_units(ir, list(ir.facts)):
        return None
    numerator = _complete_phrase_parts(ir, phrases[0])
    denominator = _complete_phrase_parts(ir, phrases[1])
    if not numerator or not denominator or set(numerator) & set(denominator):
        return None
    if set(numerator + denominator) != set(ir.facts):
        return None
    keys = {fact_period_key(n, ir.facts[n]) for n in ir.facts}
    if len(keys) > 1:
        return None
    left, right = ' + '.join(numerator), ' + '.join(denominator)
    formula = f'({left}) / ({right})'
    if re.search(r'\bpercent(?:age)?\b|\bproportion\b', question, re.I):
        formula += ' * 100'
    return _make(ir, formula, numerator_parts=numerator, denominator_parts=denominator)


def _complete_phrase_parts(ir, phrase):
    """Cover the entire named side of a ratio, including every conjunction.

    Whole labels such as 'furniture and equipment' remain one quantity when
    directly matched. Otherwise enumerate conjunction boundaries; incomplete
    or ambiguous decompositions stay unresolved.
    """
    from .generic_ops import _content_tokens
    def identity_tokens(value):
        # "Total" is part of the requested row, not interchangeable with an
        # arbitrary component sharing its remaining words.
        tokens = _content_tokens(value)
        if re.search(r'\btotal\b', value, re.I):
            tokens.add('total')
        return tokens

    observation = re.search(r'\s+(?:for|in)\s+(?:fiscal\s+year\s+)?((?:19|20)\d{2})\s*$', phrase, re.I)
    text = phrase[:observation.start()].strip() if observation else phrase.strip()
    wanted_years = set(re.findall(r'\b(?:19|20)\d{2}\b', phrase))
    wanted = identity_tokens(text)
    matches = []
    for name, fact in ir.facts.items():
        years = set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', f'{name} {fact.row_label} {fact.period} {fact.column} {fact.source_column}'))
        if wanted_years and not wanted_years <= years:
            continue
        if observation:
            key = fact_period_key(name, fact)
            source_years = set(re.findall(r'\b(?:19|20)\d{2}\b', fact.source_column))
            observation_years = {observation.group(1)}
            # A debt maturity in the measure ("notes due 2020") is not its
            # observation year ("in 2019"). Only the latter binds the column.
            if (not key or observation_years != {str(key[0])}
                    or source_years and source_years != observation_years):
                continue
        if not wanted:
            if wanted_years:
                matches.append(name)
            continue
        label = identity_tokens(fact.row_label)
        role = identity_tokens(name.replace('_', ' '))
        if label and wanted == label:
            matches.append(name)
        elif not fact.row_label and wanted == role:
            matches.append(name)
    if len(matches) == 1:
        return matches
    if matches:
        return None
    # A phrase without a conjunction cannot be repaired by selecting the best
    # partial word overlap. This blocks a missing consultants operand, etc.
    boundaries = list(re.finditer(r'\s+and\s+|\s*,\s*', text, re.I))
    solutions = set()
    for boundary in boundaries:
        first = _complete_phrase_parts(ir, text[:boundary.start()])
        second = _complete_phrase_parts(ir, text[boundary.end():])
        if first and second and not set(first) & set(second):
            if not _content_tokens(text[boundary.end():]) and not same_measure(ir, first + second):
                continue
            solutions.add(tuple(first + second))
    return list(next(iter(solutions))) if len(solutions) == 1 else None


def _quantity_price(ir, question):
    if len(ir.facts) != 2:
        return None
    quantity = _unique(ir, r'\b(?:eligible quantity|option count|number of (?:eligible )?options|stock options outstanding)\b')
    price = _unique(ir, r'\b(?:exercise price|applicable price|price per)\b')
    if not quantity or not price or quantity == price:
        return None
    if re.search(r'\brestricted\b|\brsus?\b', ir.facts[quantity].row_label, re.I):
        return None
    if ir.facts[quantity].unit.lower() not in {'count', 'shares', 'options'}:
        return None
    if 'share' not in ir.facts[price].unit.lower() and 'option' not in ir.facts[price].unit.lower():
        return None
    return _make(ir, f'{quantity} * {price}', eligible_quantity=quantity, applicable_price=price)


def _netting(ir, question):
    if len(ir.facts) != 2 or not compatible_units(ir, list(ir.facts)):
        return None
    gain, loss = _unique(ir, r'\bgains?\b'), _unique(ir, r'\bloss(?:es)?\b')
    if not gain or not loss or gain == loss:
        return None
    # A parenthesized disclosure is a signed loss; a positive gross-loss amount
    # is its magnitude. This normalizes sign, never selects a role by value.
    operator = '+' if float(ir.facts[loss].value) < 0 else '-'
    return _make(ir, f'{gain} {operator} {loss}', gross_gains=gain, gross_loss_magnitude=loss)


def resolve_extended_operation(ir: VerificationIR, question: str, operation: str, *, evidence_text='') -> dict[str, Any] | None:
    if operation == 'exclusion_or_remaining_amount':
        from verifiqa.agent.source_exclusion_planner import exclusion_source_matches
        if not exclusion_source_matches(ir, question, evidence_text):
            return None
    handlers = {
        'line_item_difference': _difference,
        'percentage_point_change': _point_change,
        'sum': _sum,
        'exclusion_or_remaining_amount': _exclusion,
        'difference_of_changes': _difference_of_changes,
        'share_of_a_change': _share_of_change,
        'change_of_growth_rates': _growth_change,
        'change_of_ratios': _ratio_change,
        'sum_then_ratio': _sum_ratio,
        'quantity_times_price': _quantity_price,
        'net_gains_losses': _netting,
    }
    if operation == 'selected_three_row_sum':
        return _top_three(ir, question, evidence_text)
    handler = handlers.get(operation)
    return handler(ir, question) if handler else None
