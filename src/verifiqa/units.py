from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable, Optional

from verifiqa.types import CertificateFact, EvidenceChunk, VerificationCertificate


CANONICAL_MONEY_UNIT = "USD millions"
SOURCE_SCALE_TO_USD_MILLIONS = {
    "ones": 1 / 1_000_000,
    "thousands": 1 / 1_000,
    "millions": 1.0,
    "billions": 1_000.0,
}
SUPPORTED_SOURCE_SCALES = set(SOURCE_SCALE_TO_USD_MILLIONS)


class QuantityError(ValueError):
    pass


def normalize_certificate_quantities(
    certificate: VerificationCertificate,
    evidence_chunks: Iterable[EvidenceChunk],
    *,
    strict: bool = False,
) -> VerificationCertificate:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in evidence_chunks}
    facts = []
    for fact in certificate.facts:
        chunk = chunk_by_id.get(fact.chunk_id)
        chunk_text = chunk.text if chunk is not None else ""
        facts.append(normalize_fact_quantity(fact, chunk_text, strict=strict))
    return replace(certificate, facts=facts)


def normalize_fact_quantity(
    fact: CertificateFact,
    chunk_text: str = "",
    *,
    strict: bool = False,
) -> CertificateFact:
    if _is_absence_implies_zero_fact(fact):
        return _normalize_absence_zero_fact(fact, chunk_text, strict=strict)

    if not _is_money_unit(fact.unit):
        return _normalize_scalar_fact(fact, chunk_text, strict=strict)
    if _is_per_share_money_fact(fact):
        return _normalize_per_share_money_fact(fact, chunk_text, strict=strict)

    # Trust the structured fields the formalizer emits (raw_value, source_scale,
    # source_scale_quote, sign_convention) instead of re-deriving them from the
    # quote text. The deterministic layer's job here is to verify those claims
    # against the evidence (grounding + canonical-value arithmetic), not to guess
    # them.
    raw_value = fact.raw_value
    scale = _normalize_source_scale(fact.source_scale)
    scale_quote = fact.source_scale_quote

    if raw_value is None:
        if strict:
            raise QuantityError(f"raw_value_not_in_source_quote:{fact.name}")
        return fact

    if not _number_supported(raw_value, fact.source_quote):
        if strict:
            raise QuantityError(f"raw_value_not_in_source_quote:{fact.name}")
        return fact

    if not scale:
        if strict:
            raise QuantityError(f"missing_source_scale:{fact.name}")
        return replace(
            fact,
            raw_value=raw_value,
            raw_unit=fact.raw_unit or "USD",
        )

    if scale not in SUPPORTED_SOURCE_SCALES:
        if strict:
            raise QuantityError(f"unsupported_source_scale:{fact.name}:{scale}")
        return fact

    if scale != "ones" and strict:
        _validate_scale_grounding(fact, scale, scale_quote, chunk_text)

    canonical = raw_value * SOURCE_SCALE_TO_USD_MILLIONS[scale]
    use_magnitude = (fact.sign_convention or "").strip().lower() == "magnitude"
    if use_magnitude:
        canonical = abs(canonical)
    expected_value = abs(fact.value) if use_magnitude else fact.value
    if strict and not _close(canonical, expected_value):
        raise QuantityError(f"canonical_value_mismatch:{fact.name}")

    return replace(
        fact,
        raw_value=raw_value,
        raw_unit=fact.raw_unit or "USD",
        source_scale=scale,
        source_scale_quote=scale_quote,
        value=canonical,
        unit=CANONICAL_MONEY_UNIT,
    )


def numbers_in_text(text: str) -> list[float]:
    values = []
    pattern = r"\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?"
    for raw in re.findall(pattern, text):
        value = _parse_number_token(raw)
        if value is not None:
            values.append(value)
    return values


def number_supported(value: float, text: str, *, allow_percent_ratio: bool = True) -> bool:
    return _number_supported(value, text, allow_percent_ratio=allow_percent_ratio)


def _normalize_absence_zero_fact(
    fact: CertificateFact,
    chunk_text: str,
    *,
    strict: bool,
) -> CertificateFact:
    if not _close(float(fact.value), 0.0):
        if strict:
            raise QuantityError(f"absence_value_must_be_zero:{fact.name}")
        return fact
    if fact.raw_value is not None and not _close(float(fact.raw_value), 0.0):
        if strict:
            raise QuantityError(f"absence_raw_value_must_be_null_or_zero:{fact.name}")
        return fact
    if strict and not _absence_quote_supported(fact, chunk_text):
        raise QuantityError(f"absence_not_supported:{fact.name}")
    return replace(
        fact,
        raw_value=None,
        raw_unit=fact.raw_unit or "USD",
        source_scale="ones",
        source_scale_quote="",
        value=0.0,
        unit=CANONICAL_MONEY_UNIT if _is_money_unit(fact.unit) else fact.unit,
    )


def _normalize_per_share_money_fact(fact: CertificateFact, chunk_text: str, *, strict: bool) -> CertificateFact:
    raw_value = fact.raw_value
    if raw_value is None:
        raw_value = _infer_raw_value(numbers_in_text(fact.source_quote), fact.value, "")
    if raw_value is None:
        if strict:
            raise QuantityError(f"raw_value_not_in_source_quote:{fact.name}")
        return fact
    if strict and not _number_supported(raw_value, fact.source_quote):
        raise QuantityError(f"raw_value_not_in_source_quote:{fact.name}")
    if strict and not _close(raw_value, fact.value):
        raise QuantityError(f"canonical_value_mismatch:{fact.name}")
    return replace(
        fact,
        raw_value=raw_value,
        raw_unit=fact.raw_unit or "USD/share",
        source_scale=fact.source_scale or "ones",
        source_scale_quote=fact.source_scale_quote,
        value=raw_value,
        unit="USD/share",
    )


def _normalize_scalar_fact(fact: CertificateFact, chunk_text: str, *, strict: bool) -> CertificateFact:
    raw_value = fact.raw_value
    if raw_value is None:
        raw_value = _infer_raw_value(numbers_in_text(fact.source_quote), fact.value, "")
        if raw_value is None and _zero_supported_by_blank_table_cell(fact, chunk_text):
            raw_value = 0.0
    if raw_value is None:
        if strict:
            raise QuantityError(f"raw_value_not_in_source_quote:{fact.name}")
        return fact
    if strict and not _number_supported(raw_value, fact.source_quote):
        if _zero_supported_by_blank_table_cell(fact, chunk_text):
            raw_value = 0.0
        else:
            raise QuantityError(f"raw_value_not_in_source_quote:{fact.name}")
    if strict and not _close(raw_value, fact.value):
        raise QuantityError(f"canonical_value_mismatch:{fact.name}")
    return replace(
        fact,
        raw_value=raw_value,
        raw_unit=fact.raw_unit or fact.unit,
        source_scale=fact.source_scale or "ones",
        value=raw_value,
    )


def _zero_supported_by_blank_table_cell(fact: CertificateFact, chunk_text: str) -> bool:
    if not _close(float(fact.value), 0.0):
        return False
    if fact.raw_value is not None and not _close(float(fact.raw_value), 0.0):
        return False
    unit_text = (fact.unit or fact.raw_unit or "").lower()
    if not ("percent" in unit_text or "%" in unit_text or "ratio" in unit_text):
        return False
    if any(_close(number, 0.0) for number in numbers_in_text(fact.source_quote)):
        return True
    if not fact.row_label or not fact.column or not fact.source_quote or not chunk_text:
        return False
    quote_tokens = set(_significant_tokens(fact.source_quote))
    row_tokens = set(_significant_tokens(fact.row_label))
    column_tokens = set(_significant_tokens(fact.column))
    chunk_tokens = set(_significant_tokens(chunk_text))
    if not quote_tokens or not row_tokens or not column_tokens:
        return False
    if not row_tokens.issubset(quote_tokens | chunk_tokens):
        return False
    if not column_tokens.issubset(chunk_tokens):
        return False
    row_text = f"{fact.row_label} {fact.source_quote}".lower()
    if not any(marker in row_text for marker in ("%", "percent", "growth", "margin", "rate")):
        return False
    # SEC table extraction often drops visually blank zero cells.  Accept this
    # only for scalar percentage/ratio rows with explicit row and column
    # grounding; monetary facts still require literal numeric support.
    return True


def _absence_quote_supported(fact: CertificateFact, chunk_text: str) -> bool:
    quote = _normalized_words(fact.source_quote)
    context = _normalized_words(f"{fact.source_quote} {chunk_text}")
    if not quote or not context:
        return False
    if _has_absence_language(quote):
        if not _scope_metric_supported(fact, context):
            return False
        period = (fact.absence_scope or {}).get("period") or fact.period
        if period and not _period_supported(period, context):
            return False
        return True
    return _statement_omission_supported(fact, chunk_text) or _chunk_omission_supported(
        fact, chunk_text
    )


def _chunk_omission_supported(fact: CertificateFact, chunk_text: str = "") -> bool:
    """Ground an absence claim against the chunk when the witness is a paraphrase.

    The model may assert an omission in its own words ("the statement does not
    contain a line item for X") instead of pasting the statement excerpt. In that
    case verify directly that the chunk is the relevant statement scope, covers
    the period, and genuinely omits the requested row label.
    """
    if not chunk_text:
        return False
    chunk = _normalized_words(chunk_text)
    scope = _normalized_words(
        f"{chunk_text} {(fact.absence_scope or {}).get('statement', '')}"
    )
    statement_terms = (
        " statement ",
        " statements ",
        " operations ",
        " income ",
        " earnings ",
        " balance sheet ",
        " financial position ",
        " cash flow ",
        " cash flows ",
    )
    if not any(term in scope for term in statement_terms):
        return False
    line_item_terms = (
        " revenue ",
        " cost ",
        " sales ",
        " expense ",
        " expenses ",
        " income ",
        " loss ",
        " margin ",
        " assets ",
        " liabilities ",
        " cash ",
        " debt ",
        " tax ",
        " interest ",
        " goodwill ",
        " impairment ",
    )
    if sum(1 for term in line_item_terms if term in chunk) < 3:
        return False
    target_tokens = _absence_target_tokens(fact)
    if not target_tokens:
        return False
    if target_tokens & set(_significant_tokens(chunk_text)):
        return False
    period = (fact.absence_scope or {}).get("period") or fact.period or fact.column
    if period and not _period_supported(period, _normalized_words(chunk_text)):
        return False
    return True


def _statement_omission_supported(fact: CertificateFact, chunk_text: str = "") -> bool:
    quote_text = fact.source_quote or ""
    quote = _normalized_words(quote_text)
    scope = _normalized_words(
        f"{quote_text} {(fact.absence_scope or {}).get('statement', '')}"
    )
    if not any(
        term in scope
        for term in (
            " statement ",
            " statements ",
            " operations ",
            " income ",
            " earnings ",
            " balance sheet ",
            " financial position ",
            " cash flow ",
            " cash flows ",
        )
    ):
        return False

    line_item_terms = (
        " revenue ",
        " cost ",
        " sales ",
        " expense ",
        " expenses ",
        " income ",
        " loss ",
        " margin ",
        " assets ",
        " liabilities ",
        " cash ",
        " debt ",
        " tax ",
        " interest ",
        " goodwill ",
        " impairment ",
    )
    if sum(1 for term in line_item_terms if term in quote) < 3:
        return False

    target_tokens = _absence_target_tokens(fact)
    if not target_tokens:
        return False
    quote_tokens = set(_significant_tokens(quote_text))
    if target_tokens & quote_tokens:
        return False

    period = (fact.absence_scope or {}).get("period") or fact.period or fact.column
    if period:
        period_context = _normalized_words(f"{quote_text} {fact.column} {chunk_text}")
        if not _period_supported(period, period_context):
            return False
    return True


def _absence_target_tokens(fact: CertificateFact) -> set[str]:
    pieces = []
    if not _looks_like_statement_label(fact.row_label):
        pieces.append(fact.row_label)
    pieces.extend([
        (fact.absence_scope or {}).get("metric", ""),
        fact.name.replace("_", " "),
    ])
    generic = {
        "cost",
        "costs",
        "expense",
        "expenses",
        "charge",
        "charges",
        "amount",
        "quantity",
        "line",
        "item",
    }
    return {
        token
        for piece in pieces
        for token in _significant_tokens(piece)
        if token not in generic and len(token) > 2
    }




def _looks_like_statement_label(value: str) -> bool:
    text = _normalized_words(value)
    return any(
        term in text
        for term in (
            " statement ",
            " statements ",
            " operations ",
            " income statement ",
            " balance sheet ",
            " cash flow ",
            " cash flows ",
        )
    )


def _has_absence_language(text: str) -> bool:
    return any(pattern in text for pattern in (
        " no such ",
        " no ",
        " none ",
        " not explicitly ",
        " not outlined ",
        " not incurred ",
        " did not incur ",
        " there were no ",
        " there was no ",
    ))


def _scope_metric_supported(fact: CertificateFact, context: str) -> bool:
    pieces = [
        fact.name.replace("_", " "),
        fact.row_label,
        (fact.absence_scope or {}).get("metric", ""),
    ]
    for piece in pieces:
        tokens = [
            token for token in _significant_tokens(piece)
            if token not in {"cost", "costs", "expense", "expenses", "amount", "quantity"}
        ]
        if tokens and all(token in context for token in tokens):
            return True
    return False


def _period_supported(period: str, context: str) -> bool:
    years = re.findall(r"(?:19|20)\d{2}", period)
    if years:
        return all(year in context for year in years)
    tokens = _significant_tokens(period)
    return bool(tokens) and all(token in context for token in tokens)


def _is_absence_implies_zero_fact(fact: CertificateFact) -> bool:
    return (fact.fact_type or "").strip() == "absence_implies_zero"


def _normalized_words(value: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip() + " "


def _infer_raw_value(numbers: list[float], canonical_value: float, scale: str) -> Optional[float]:
    if not numbers:
        return None
    if scale in SOURCE_SCALE_TO_USD_MILLIONS:
        multiplier = SOURCE_SCALE_TO_USD_MILLIONS[scale]
        for number in numbers:
            if _close(number * multiplier, canonical_value):
                return number
            if _close(-number * multiplier, canonical_value):
                return -number
    for number in numbers:
        if _close(number, canonical_value):
            return number
        if _close(-number, canonical_value):
            return -number
    if len(numbers) == 1:
        return numbers[0]
    return None


def _validate_scale_grounding(
    fact: CertificateFact,
    scale: str,
    scale_quote: str,
    chunk_text: str,
) -> None:
    if scale_quote:
        quote_scale, _ = _scale_from_text(scale_quote)
        if quote_scale and quote_scale != scale:
            raise QuantityError(f"source_scale_quote_mismatch:{fact.name}")
        if _scale_quote_supported(scale_quote, fact.source_quote) or _scale_quote_supported(scale_quote, chunk_text):
            return
        quote_scale, _ = _scale_from_text(fact.source_quote)
        if quote_scale == scale:
            return
        try:
            chunk_scale, _ = _scale_from_chunk(chunk_text)
        except QuantityError as exc:
            raise QuantityError(f"ambiguous_source_scale:{fact.name}") from exc
        if chunk_scale == scale:
            return
        raise QuantityError(f"source_scale_quote_not_in_chunk:{fact.name}")

    quote_scale, _ = _scale_from_text(fact.source_quote)
    if quote_scale == scale:
        return
    chunk_scale, _ = _scale_from_chunk(chunk_text)
    if chunk_scale == scale:
        return
    raise QuantityError(f"missing_source_scale:{fact.name}")


def _scale_from_chunk(text: str) -> tuple[str, str]:
    mentions = _scale_mentions(text)
    if not mentions:
        return "", ""
    scales = {scale for scale, _ in mentions}
    if len(scales) > 1:
        raise QuantityError("ambiguous_source_scale")
    return mentions[0]


def _scale_from_text(text: str) -> tuple[str, str]:
    mentions = _scale_mentions(text)
    return mentions[0] if mentions else ("", "")


def _scale_mentions(text: str) -> list[tuple[str, str]]:
    found = []
    normalized = text or ""
    compact = _compact_text(normalized)
    compact_patterns = [
        ("thousands", "inthousands"),
        ("millions", "inmillions"),
        ("billions", "inbillions"),
        ("thousands", "dollarsinthousands"),
        ("millions", "dollarsinmillions"),
        ("billions", "dollarsinbillions"),
        ("thousands", "usdthousands"),
        ("millions", "usd millions".replace(" ", "")),
        ("billions", "usdbillions"),
    ]
    for scale, marker in compact_patterns:
        if marker in compact:
            found.append((scale, _display_scale_marker(marker)))

    patterns = [
        ("thousands", r"\bin\s+thousands\b|\bthousands\s+of\s+(?:dollars|usd)\b|\bdollars\s+in\s+thousands\b|\$\s*in\s*thousands\b|\bthousand\b|\bthousands\b"),
        ("millions", r"\bin\s+millions\b|\bmillions\s+of\s+(?:dollars|usd)\b|\bdollars\s+in\s+millions\b|\busd\s+millions?\b|\$\s*in\s*millions\b|\bmillion\b|\bmillions\b"),
        ("billions", r"\bin\s+billions\b|\bbillions\s+of\s+(?:dollars|usd)\b|\bdollars\s+in\s+billions\b|\busd\s+billions?\b|\$\s*in\s*billions\b|\bbillion\b|\bbillions\b"),
    ]
    for scale, pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            found.append((scale, match.group(0)))
    return found


def _normalize_source_scale(scale: str) -> str:
    value = (scale or "").strip().lower().replace("_", " ")
    aliases = {
        "one": "ones",
        "ones": "ones",
        "unit": "ones",
        "units": "ones",
        "raw": "ones",
        "raw units": "ones",
        "actual": "ones",
        "thousand": "thousands",
        "thousands": "thousands",
        "million": "millions",
        "millions": "millions",
        "billion": "billions",
        "billions": "billions",
    }
    return aliases.get(value, "")


def _is_money_unit(unit: str) -> bool:
    text = (unit or "").lower()
    return "usd" in text or "$" in text or "dollar" in text


def _is_per_share_money_fact(fact: CertificateFact) -> bool:
    text = f"{fact.name} {fact.row_label} {fact.source_quote} {fact.raw_unit} {fact.unit}".lower()
    if any(marker in text for marker in ("per share", "per-share", "$/share", "usd/share", "earnings per share", "book value per share")):
        return True
    return bool(re.search(r"\b(tbvps|eps)\b", text))


def _scale_quote_supported(quote: str, text: str) -> bool:
    if not quote or not text:
        return False
    compact_quote = _compact_text(quote)
    compact_text = _compact_text(text)
    if compact_quote and compact_quote in compact_text:
        return True
    quote_tokens = set(_significant_tokens(quote))
    text_tokens = set(_significant_tokens(text))
    return bool(quote_tokens) and quote_tokens.issubset(text_tokens)


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _display_scale_marker(marker: str) -> str:
    if "thousands" in marker:
        return "in thousands"
    if "billions" in marker:
        return "in billions"
    return "in millions"


def _significant_tokens(value: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return [token for token in cleaned.split() if token]


def _number_supported(value: float, text: str, *, allow_percent_ratio: bool = True) -> bool:
    for number in numbers_in_text(text):
        if _close(number, value) or _close(-number, value):
            return True
        if allow_percent_ratio and (_close(number / 100.0, value) or _close(-number / 100.0, value)):
            return True
    return False


def _parse_number_token(raw: str) -> Optional[float]:
    token = raw.strip()
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()").replace("$", "").replace(",", "").strip()
    if token.endswith("%"):
        token = token[:-1].strip()
    if not token:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return -value if negative else value


def _close(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= max(1e-8, scale * 1e-8)
