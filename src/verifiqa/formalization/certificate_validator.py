from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from verifiqa.formulas.evaluator import formula_variables
from verifiqa.types import EvidenceChunk, RetrievalFact, RetrievalPlan, VerificationCertificate
from verifiqa.units import QuantityError, normalize_fact_quantity, number_supported


@dataclass
class CertificateValidationResult:
    valid: bool
    reason: str = ""


class CertificateValidator:
    """Validate that a certificate is grounded in retrieved evidence text."""

    def validate(
        self,
        certificate: VerificationCertificate,
        evidence_chunks: Iterable[EvidenceChunk],
        answer: str,
        question: str = "",
        retrieval_plan: RetrievalPlan | None = None,
        require_units: bool = True,
    ) -> CertificateValidationResult:
        try:
            self._validate(certificate, list(evidence_chunks), answer, question, retrieval_plan, require_units=require_units)
            return CertificateValidationResult(valid=True)
        except ValueError as exc:
            return CertificateValidationResult(valid=False, reason=str(exc))

    def _validate(
        self,
        certificate: VerificationCertificate,
        evidence_chunks: list[EvidenceChunk],
        answer: str,
        question: str = "",
        retrieval_plan: RetrievalPlan | None = None,
        require_units: bool = True,
    ) -> None:
        if require_units and not certificate.claim.unit:
            raise ValueError("claim_missing_unit")
        if not _number_supported(certificate.claim.claimed_value, answer):
            reported = certificate.claim.reported_value
            if reported is None or not _number_supported(reported, answer):
                raise ValueError("claimed_value_not_supported_by_answer")

        chunk_by_id = {chunk.chunk_id: chunk for chunk in evidence_chunks}
        fact_names = {fact.name for fact in certificate.facts}
        variables = formula_variables(certificate.formula)
        if fact_names != variables:
            missing = sorted(variables - fact_names)
            if missing:
                raise ValueError(f"formula_variable_missing_fact:{missing[0]}")
        facts_to_validate = [fact for fact in certificate.facts if fact.name in variables]

        for fact in facts_to_validate:
            if require_units and not fact.unit:
                raise ValueError(f"fact_missing_unit:{fact.name}")
            if not fact.source_quote:
                raise ValueError(f"fact_missing_source_quote:{fact.name}")
            if not fact.chunk_id:
                raise ValueError(f"fact_missing_chunk_id:{fact.name}")
            chunk = chunk_by_id.get(fact.chunk_id)
            if chunk is None:
                raise ValueError(f"unknown_chunk_id:{fact.name}:{fact.chunk_id}")

            if fact.fact_type == "absence_implies_zero":
                # An absence witness is intentionally a meta-description of an
                # omission ("the statement does not contain a line item for X"),
                # so it will not appear verbatim in the chunk. Skip the literal
                # containment check and require filing-grounded absence evidence
                # instead: an explicit filing absence phrase, or a statement/table
                # excerpt whose row list omits the requested line.
                if question and not _question_allows_zero_if_absent(question):
                    raise ValueError(f"absence_zero_not_allowed_by_question:{fact.name}")
                if _source_quote_is_retrieval_failure(fact.source_quote or ""):
                    raise ValueError(
                        f"absence_zero_retrieval_failure_not_filing_absence:{fact.name}"
                    )
                if not (
                    _source_quote_has_absence_language(fact.source_quote or "")
                    or _source_quote_supports_statement_omission(fact, chunk.text)
                    or _chunk_confirms_statement_omission(fact, chunk.text)
                ):
                    raise ValueError(
                        f"absence_zero_no_absence_language_in_quote:{fact.name}"
                    )
            elif not _value_grounded_in_chunk(fact, chunk.text):
                # Ground the fact against the independent source directly: the
                # reported number must appear in the real chunk text. This binds
                # the value to the document rather than string-matching the LLM's
                # self-reported source_quote (which fails on legitimate table
                # formatting drift and derived values).
                raise ValueError(f"fact_value_not_in_chunk:{fact.name}")

            try:
                normalize_fact_quantity(fact, chunk.text, strict=True)
            except QuantityError as exc:
                raise ValueError(str(exc)) from exc
            statement_error = _statement_context_error(fact, chunk.text, retrieval_plan)
            if statement_error:
                raise ValueError(statement_error)


def certificate_facts(certificate: VerificationCertificate) -> dict:
    return {fact.name: fact.value for fact in certificate.facts}


def certificate_provenance(certificate: VerificationCertificate) -> dict:
    return {
        fact.name: {
            "chunk_id": fact.chunk_id,
            "source_quote": fact.source_quote,
            "unit": fact.unit,
            "fact_type": fact.fact_type,
            "raw_value": fact.raw_value,
            "raw_unit": fact.raw_unit,
            "source_scale": fact.source_scale,
            "source_scale_quote": fact.source_scale_quote,
            "absence_scope": fact.absence_scope,
            "canonical_value": fact.value,
            "period": fact.period,
            "row_label": fact.row_label,
            "column": fact.column,
        }
        for fact in certificate.facts
    }


def _quote_supported(quote: str, text: str) -> bool:
    # PDF table extraction often collapses row labels and adjacent numeric
    # columns, e.g. "charges411247289".  Validate quote support by requiring
    # the alphabetic evidence words and the numeric value fragments to be
    # present, rather than relying on whitespace-sensitive token equality.
    if not quote or not text:
        return False
    if _compact_text(quote) in _compact_text(text):
        return True
    quote_words = set(_word_tokens(quote))
    text_words = set(_word_tokens(text))
    if quote_words and not quote_words.issubset(text_words):
        return False
    quote_numbers = _number_strings(quote)
    text_numbers = _number_strings(text)
    if quote_numbers:
        for number in quote_numbers:
            if not any(number == candidate or number in candidate for candidate in text_numbers):
                return False
    return bool(quote_words or quote_numbers)


def _question_allows_zero_if_absent(question: str) -> bool:
    text = _normalize_text(question)
    if "state 0" in text or "state zero" in text:
        return True
    if "if" in text and ("not explicitly" in text or "not outlined" in text or "not present" in text):
        return "0" in text or "zero" in text
    return False


# ---------------------------------------------------------------------------
# Helpers for absence_implies_zero source_quote validation
# ---------------------------------------------------------------------------

# Phrases that signal a retrieval-system meta-statement, not filing language.
_RETRIEVAL_FAILURE_PHRASES = (
    "was retrieved",
    "were retrieved",
    "not retrieved",
    "data from the",
    "data from this",
    "no data was",
    "no data were",
    "no information was",
    "no information were",
    "evidence does not",
    "evidence did not",
    "evidence contains",
    "not found in",
    "could not be found",
    "retrieved from",
    "is not available in",
    "are not available in",
    "was not available",
    "were not available",
    "could not retrieve",
    "unable to retrieve",
    "the passage",
    "the chunk",
    "from the evidence",
    "in the evidence",
    "the retrieved",
)

# Phrases found in genuine filings indicating a line item is absent/nil.
_GENUINE_ABSENCE_PHRASES = (
    "none",
    "nil",
    "n/a",
    "not applicable",
    "no such",
    "no restructuring",
    "no impairment",
    "no goodwill",
    "not reported",
    "not disclosed",
    "not incurred",
    "not recognized",
    "were no ",
    "was no ",
    "are no ",
    "is no ",
    "had no ",
    "have no ",
    "has no ",
    "no capital",
    "no charges",
    "no costs",
    "no expenses",
    "no revenue",
    "no income",
    "no loss",
    "no gain",
    "no debt",
    "no assets",
    "no liabilities",
    "not present",
    "not outstanding",
    "not applicable",
)


def _source_quote_supports_statement_omission(fact, chunk_text: str = "") -> bool:
    """Validate absence by omission from a statement/table excerpt.

    This supports FinanceBench-style questions that explicitly say to state 0
    when a line item is not outlined. The witness is the filed statement excerpt
    itself: it must look like a real statement/table scope, and the requested
    row label must be absent from that excerpt.
    """
    quote = fact.source_quote or ""
    normalized = _normalize_text(quote)
    if not _looks_like_statement_or_table_scope(normalized, fact):
        return False

    target = _absence_target_tokens(fact)
    if not target:
        return False
    quote_tokens = set(_word_tokens(quote))
    if target & quote_tokens:
        return False

    period = (fact.absence_scope or {}).get("period") or fact.period or fact.column
    years = re.findall(r"(?:19|20)\d{2}", period or "")
    if years:
        period_context = _normalize_text(f"{quote} {fact.column}")
        if not any(year in period_context for year in years):
            return False
        if not any(year in normalized for year in years):
            column_quote = fact.column or years[0]
            if chunk_text and not _quote_supported(column_quote, chunk_text):
                return False

    return True


def _chunk_confirms_statement_omission(fact, chunk_text: str) -> bool:
    """Ground an absence_implies_zero claim against the evidence chunk itself.

    When the witness quote is the model's own paraphrase of an omission
    ("the statement does not contain a line item for X") rather than a verbatim
    statement excerpt, trust the chunk instead of the paraphrase: confirm the
    chunk is the relevant statement scope, is for the right period, and that the
    requested row label is genuinely absent from it.
    """
    if not chunk_text:
        return False
    normalized = _normalize_text(chunk_text)
    if not _looks_like_statement_or_table_scope(normalized, fact):
        return False
    period = (fact.absence_scope or {}).get("period") or fact.period or fact.column
    years = re.findall(r"(?:19|20)\d{2}", period or "")
    if years and not any(year in normalized for year in years):
        return False
    target = _absence_target_tokens(fact)
    if not target:
        return False
    return not (target & set(_word_tokens(chunk_text)))


def _looks_like_statement_or_table_scope(text: str, fact) -> bool:
    scope = _normalize_text(
        " ".join(
            [
                text,
                (fact.absence_scope or {}).get("statement", ""),
            ]
        )
    )
    statement_terms = (
        "statement",
        "statements",
        "operations",
        "income",
        "earnings",
        "balance sheet",
        "financial position",
        "cash flow",
        "cash flows",
    )
    if not any(term in scope for term in statement_terms):
        return False

    line_item_terms = (
        "revenue",
        "cost",
        "sales",
        "expense",
        "expenses",
        "income",
        "loss",
        "margin",
        "assets",
        "liabilities",
        "cash",
        "debt",
        "tax",
        "interest",
        "goodwill",
        "impairment",
    )
    return sum(1 for term in line_item_terms if term in text) >= 3


def _absence_target_tokens(fact) -> set[str]:
    pieces = []
    if not _looks_like_statement_label(fact.row_label):
        pieces.append(fact.row_label)
    pieces.extend([
        (fact.absence_scope or {}).get("metric", ""),
        fact.name,
    ])
    target_text = " ".join(value for value in pieces if value)
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
    return {token for token in _word_tokens(target_text) if token not in generic and len(token) > 2}




def _looks_like_statement_label(value: str) -> bool:
    text = _normalize_text(value or "")
    return any(
        term in text
        for term in (
            "statement",
            "statements",
            "operations",
            "income statement",
            "balance sheet",
            "cash flow",
            "cash flows",
        )
    )


def _source_quote_is_retrieval_failure(source_quote: str) -> bool:
    """Return True if the source_quote looks like an LLM retrieval-failure meta-statement."""
    text = source_quote.lower()
    return any(phrase in text for phrase in _RETRIEVAL_FAILURE_PHRASES)


def _source_quote_has_absence_language(source_quote: str) -> bool:
    """Return True if the source_quote contains genuine filing absence language."""
    text = source_quote.lower()
    return any(phrase in text for phrase in _GENUINE_ABSENCE_PHRASES)


def _significant_tokens(value: str) -> list[str]:
    # Collapse runs of non-alphanumeric chars to spaces, then split on whitespace.
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return [t for t in cleaned.split() if t]


def _word_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z]+", (value or "").lower())


def _number_strings(value: str) -> list[str]:
    return [
        re.sub(r"[^0-9]", "", match)
        for match in re.findall(r"\(?[-+]?\$?\d[\d,]*(?:\.\d+)?%?\)?", value or "")
        if re.sub(r"[^0-9]", "", match)
    ]


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _number_supported(expected: float, text: str) -> bool:
    return number_supported(expected, text, allow_percent_ratio=True)


def _value_grounded_in_chunk(fact, chunk_text: str) -> bool:
    """The fact's reported number must be present in the chunk's real text.

    This is the independent-source grounding that replaces the legacy
    source_quote substring match. We check the as-reported value (raw_value)
    and the canonical value against the digit strings found in the chunk, so
    formatting differences (commas, currency symbols, whitespace) do not cause
    spurious rejections, but a number absent from the document still fails.
    """
    if not chunk_text:
        return False
    text_numbers = _number_strings(chunk_text)
    if not text_numbers:
        return False
    candidates = [v for v in (fact.raw_value, fact.value) if v is not None]
    for value in candidates:
        for digits in _number_strings(_format_number_for_match(value)):
            if any(digits == token or digits in token for token in text_numbers):
                return True
    return False


def _format_number_for_match(value: float) -> str:
    if value == int(value):
        return str(int(abs(value)))
    return repr(abs(value))


def _close(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return abs(left - right) <= max(1e-9, scale * 1e-9)


def _statement_context_error(fact, chunk_text: str, retrieval_plan: RetrievalPlan | None) -> str:
    if fact.fact_type == "absence_implies_zero":
        return ""
    plan_fact = _matching_plan_fact(fact.name, retrieval_plan)
    if plan_fact is None:
        return ""
    statement = _statement_kind(plan_fact.statement)
    if not statement:
        return ""
    context = _normalize_text(chunk_text[:600])
    if _looks_like_notes_page(context):
        return f"fact_source_statement_mismatch:{fact.name}:notes_page_for_{statement}"
    return ""


def _matching_plan_fact(name: str, retrieval_plan: RetrievalPlan | None) -> RetrievalFact | None:
    if retrieval_plan is None:
        return None
    target = _base_fact_name(name)
    for fact in retrieval_plan.facts:
        if _base_fact_name(fact.name) == target:
            return fact
    return None


def _base_fact_name(name: str) -> str:
    base = re.sub(r"(^|_)(?:fy)?(?:19|20)\d{2}(?=$|_)", "_", (name or "").lower())
    return re.sub(r"_+", "_", base).strip("_")


def _statement_kind(statement: str) -> str:
    text = (statement or "").lower()
    if any(term in text for term in ("income", "operations", "p&l", "profit")):
        return "income_statement"
    if any(term in text for term in ("balance", "financial position")):
        return "balance_sheet"
    if "cash" in text and "flow" in text:
        return "cash_flow_statement"
    return ""


def _looks_like_notes_page(context: str) -> bool:
    return (
        "notes to consolidated financial statements" in context
        or "notes to financial statements" in context
        or "notes to consolidated financial statement" in context
    )
