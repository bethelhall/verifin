"""Filter filing contexts by explicit scope and period, without operand values."""
from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from .periods import fact_period_conflict, fact_period_key, quarter_number

_SCOPES = (
    (r'\bunited states\b|\busa\b|\bu s\b', ('united states', 'us', 'usa')),
    (r'\bunited kingdom\b', ('united kingdom', 'uk', 'gb')),
    (r'\bsouth africa\b', ('south africa', 'za')),
    (r'\bnorth america\b', ('north america',)),
    (r'\bsouth america\b', ('south america',)),
    (r'\basia pacific\b', ('asia pacific', 'apac')),
    (r'\brest of world\b', ('rest of world', 'rest of the world')),
    (r'\bcanada\b', ('canada', 'ca')),
    (r'\btaiwan\b', ('taiwan', 'tw')),
    (r'\bsingapore\b', ('singapore', 'sg')),
    (r'\beurope\b', ('europe',)),
    (r'\bforeign\b', ('foreign',)),
    (r'\bdomestic\b', ('domestic',)),
    (r'\bcustomer a\b', ('customer a',)),
    (r'\bcustomer b\b', ('customer b',)),
    (r'\bsupplier a\b', ('supplier a',)),
    (r'\bsupplier b\b', ('supplier b',)),
    (r'\bmunicipal bonds?\b', ('municipal',)),
    (r'\bcorporate bonds?\b', ('corporate bonds', 'corporate debt')),
    (r'\bpayments? segment\b', ('payment', 'payments')),
    (r'\bnet periodic (?:pension |defined benefit )?(?:benefit )?cost\b',
     ('net periodic benefit cost', 'net periodic pension cost', 'net periodic defined benefit cost')),
    (r'\b(?:projected |defined )?benefit obligations?\b|\bpbo\b',
     ('benefit obligation', 'benefit obligations')),
)


def _words(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def requested_scopes(fact):
    # Only an explicit table heading contributes section scope. A broad source
    # quotation may mention neighboring sections and cannot identify this cell.
    section = _words(getattr(fact, 'source_section', ''))
    if re.search(r'\btax(?:es|ation)?\b', section):
        # A consolidated tax reconciliation can refer to the U.S. statutory
        # rate without limiting every reconciled item to U.S. operations.
        # Remove only that jurisdiction reference from the heading. Explicit
        # country qualifiers in the fact name or row remain authoritative, as
        # do other geographic phrases in the same heading.
        section = re.sub(
            r'\b(?:u s|usa|united states)\s+(?:federal\s+)?statutory\s+'
            r'(?:(?:income\s+)?tax\s+)?rate\b',
            'statutory rate', section,
        )
    text = _words(f'{fact.name} {fact.row_label} {section}')
    return [aliases for pattern, aliases in _SCOPES if re.search(pattern, text)]


def unscoped_measure_texts(fact):
    out = []
    for text in (fact.row_label, fact.name):
        text = _words(text or '')
        for pattern, _ in _SCOPES:
            text = re.sub(pattern, ' ', text)
        out.append(text)
    return out


def filter_context_scope(candidates, fact):
    section = _words(getattr(fact, 'source_section', ''))
    segment = re.fullmatch(r'segment results (.+)', section)
    if segment:
        # The table heading selects the business segment before any numeric
        # value is inspected. A company prefix may precede its complete member
        # name, but partial word overlap cannot establish an identity.
        heading = segment.group(1)
        members = set()
        for _numeric, context in candidates:
            for dimension in context.dimensions:
                axis, separator, member = dimension.partition('=')
                if not separator or not re.search(r'\b(?:business )?segments?\b', _words(axis.split(':')[-1])):
                    continue
                label = re.sub(r'\s+member$', '', _words(member.split(':')[-1]))
                if label and (heading == label or heading.endswith(' ' + label)):
                    members.add((axis, member))
        if len(members) != 1:
            return []
        axis, member = next(iter(members))
        candidates = [(numeric, context) for numeric, context in candidates
                      if context.dimensions == (axis + '=' + member,)]
    scopes = requested_scopes(fact)
    if not scopes:
        unscoped = [candidate for candidate in candidates if not candidate[1].dimensions]
        return unscoped or candidates
    out = []
    for numeric, context in candidates:
        # Issuer-specific concepts can themselves carry the requested scope.
        concept_text = ' ' + _words(numeric.concept.split(':')[-1]) + ' '
        members = []
        for dimension in context.dimensions:
            axis, _, member = dimension.partition('=')
            axis = _words(axis.split(':')[-1])
            members.append((axis, ' ' + _words(member.split(':')[-1]) + ' '))
        def matches(aliases):
            def contains(text, alias):
                if re.search(r'\b(?:non|not|excluding|outside|other than)\s+(?:the\s+)?' + re.escape(alias) + r'\b', text):
                    return False
                return ' ' + alias + ' ' in text
            if any(contains(concept_text, alias) for alias in aliases):
                return True
            if any('customer' in a for a in aliases):
                pattern = r'customer|concentration'
            elif any('supplier' in a for a in aliases):
                pattern = r'supplier|vendor'
            elif any('municipal' in a or 'corporate' in a for a in aliases):
                pattern = r'securities|security|investment|instrument|bond|debt'
            elif aliases in [('payment', 'payments')]:
                pattern = r'segment|business'
            else:
                pattern = r'country|countries|geographical|geographic|geography|region|area|segment'
            return any(re.search(pattern, axis) and any(contains(text, alias) for alias in aliases)
                       for axis, text in members)
        if all(matches(aliases) for aliases in scopes):
            out.append((numeric, context))
    return out


def filter_context_period(candidates, fact, bundle):
    if fact_period_conflict(fact.name, fact):
        return []
    key = fact_period_key(fact.name, fact)
    if not key or key[1] == 0:
        return candidates
    year, month, day = key
    text = _words(f'{fact.name} {fact.period} {fact.column} {fact.row_label}')
    quarter = quarter_number(text)
    if day or (quarter and month != quarter * 3):
        quarter = None  # A more precise source date/month took precedence.
    full_year = []
    for context in bundle.contexts.values():
        if context.start_date and context.end_date:
            try:
                if 350 <= (date.fromisoformat(context.end_date) - date.fromisoformat(context.start_date)).days <= 380:
                    full_year.append(context)
            except ValueError:
                continue
    if quarter:
        endings = {(c.end_date[5:]) for c in full_year}
        if len(endings) != 1:
            return []  # Fiscal quarter alignment has no unique source definition.
        fiscal_month, fiscal_day = (int(v) for v in next(iter(endings)).split('-'))
        offset = fiscal_month - (4 - quarter) * 3
        end_year = year if offset > 0 else year - 1
        end_month = offset if offset > 0 else offset + 12
        last_day = calendar.monthrange(end_year, end_month)[1]
        target_day = min(fiscal_day, last_day) if fiscal_day < 28 else last_day
        target = date(end_year, end_month, target_day).isoformat()
        out = []
        for numeric, context in candidates:
            if context.instant == target:
                out.append((numeric, context))
            elif context.end_date == target and context.start_date:
                days = (date.fromisoformat(context.end_date) - date.fromisoformat(context.start_date)).days
                if 75 <= days <= 105:
                    out.append((numeric, context))
        return out
    if month in (-1, 13):
        annual = [c for c in full_year if c.end_date.startswith(str(year))]
        periods = {(c.start_date, c.end_date) for c in annual}
        if len(periods) != 1:
            return []
        start, end = next(iter(periods))
        target = (date.fromisoformat(start) - timedelta(days=1)).isoformat() if month == -1 else end
        return [(n, c) for n, c in candidates if c.instant == target]
    # Explicit month/date comparisons must not silently prefer a year-end value.
    out = [(n, c) for n, c in candidates if (c.instant or c.end_date).startswith(f'{year:04d}-{month:02d}')
           and (not day or (c.instant or c.end_date).endswith(f'-{day:02d}'))]
    return out
