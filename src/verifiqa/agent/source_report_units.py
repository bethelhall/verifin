"""Recover a missing notes caption from a pinned copy of the same SEC report.

No candidate, reference answer, question identifier, or numeric-value comparison
participates in the recovery. Unsupported layouts and conflicting units abstain.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from verifiqa.verification.money_units import explicit_money_currencies


_NOTES = re.compile(r"\bNOTES\s+TO\s+(?:THE\s+)?CONSOLIDATED\s+FINANCIAL\s+STATEMENTS\b", re.I)
_CAPTION = re.compile(
    r"\(\s*(?:(?:Amounts|Figures|Dollars)\s+)?In\s+"
    r"(?:(?:USD|U\.?\s*S\.?\s+dollars?)\s+)?(?P<scale>Thousands|Millions|Billions)"
    r"(?:\s*,\s*Except\s+Per\s+Share\s+(?:Amounts|Data))?\s*\)", re.I,
)
_LOCAL_SCALE = re.compile(
    r"\b(?:in|amounts\s+in|figures\s+in|dollars\s+in)\s+(?:[A-Z]{3}\s+)?"
    r"(?:thousands|millions|billions)\b|\b(?:USD|CAD|EUR|GBP|AUD|JPY)\s+(?:thousands|millions|billions)\b", re.I,
)
_EXCEPTIONS = re.compile(r"\b(?:unless\s+otherwise|except\s+(?!per\s+share)|per\s+share|percentage|percent|share\s+count)\b", re.I)
_NOTE_HEADING = re.compile(r"(?im)^\s*NOTE\s+(?P<number>\d+)\s*[.:-]\s*(?P<title>[^\n]{2,100})\s*$")


class _ReportText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "ix:hidden", "ix:header"}:
            self.hidden += 1
        if not self.hidden and tag.lower() in {"div", "p", "tr", "br", "h1", "h2", "h3", "hr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "ix:hidden", "ix:header"} and self.hidden:
            self.hidden -= 1
        if not self.hidden and tag.lower() in {"div", "p", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self):
        return "\n".join(re.sub(r"[\t \xa0]+", " ", line).strip()
                         for line in "".join(self.parts).splitlines() if line.strip())


def _words(text, *, numeric=False):
    return " ".join(re.findall(r"[a-z0-9]+" if numeric else r"[a-z]+",
                               unicodedata.normalize("NFKC", text).lower()))


def _anchors(evidence):
    anchors = []
    for sentence in re.split(r"[.!?]\s+|\n+", evidence):
        text = _words(sentence)
        if len(text.split()) < 12 or len(text) < 80:
            continue
        # Numeric tokens are deliberately absent. Separate long sentences
        # must each be located in the same named note, not elsewhere in a filing.
        if not any(text in old or old in text for old in anchors):
            anchors.append(text)
    return anchors


def _entry(manifest, doc_name):
    matches = []
    docs = manifest.get("docs") or {}
    if isinstance(docs, dict) and isinstance(docs.get(doc_name), dict):
        matches.append(docs[doc_name])
    matches.extend(entry for entry in manifest.get("results", [])
                   if isinstance(entry, dict) and entry.get("doc_name") == doc_name)
    unique = {json.dumps(item, sort_keys=True): item for item in matches}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _pinned_report(item, entry, package):
    if not isinstance(item, dict) or item.get("status") != "saved":
        return None
    raw_path = Path(str(item.get("path") or ""))
    digest = str(item.get("sha256") or "").lower()
    if raw_path.is_absolute() or ".." in raw_path.parts or not re.fullmatch(r"[a-f0-9]{64}", digest):
        return None
    path = (package / raw_path).resolve()
    try:
        path.relative_to(package)
    except ValueError:
        return None
    parsed = urlparse(str(item.get("url") or ""))
    accession = re.sub(r"\D", "", str(entry.get("accession") or entry.get("filing_accession") or ""))
    cik = str(entry.get("cik") or "").lstrip("0")
    expected = f"/Archives/edgar/data/{cik}/{accession}/"
    if (not cik or len(accession) != 18 or parsed.scheme != "https" or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        return None
    is_pdf = path.suffix.lower() == ".pdf"
    if is_pdf:
        if (item.get("source_type") != "issuer_annual_report"
                or item.get("doc_name") != entry.get("doc_name")
                or re.sub(r"\D", "", str(item.get("filing_accession") or "")) != accession):
            return None
    elif (path.suffix.lower() not in {".htm", ".html"}
          or parsed.netloc.lower() not in {"www.sec.gov", "sec.gov"}
          or not parsed.path.startswith(expected) or Path(parsed.path).name != path.name):
        return None
    try:
        if path.stat().st_size > 30_000_000:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if hashlib.sha256(raw).hexdigest() != digest:
        return None
    if is_pdf:
        try:
            import fitz
            with fitz.open(stream=raw, filetype="pdf") as document:
                if document.needs_pass or len(document) > 500:
                    return None
                return "\n".join(page.get_text() for page in document)
        except (ImportError, RuntimeError, ValueError):
            return None
    parser = _ReportText()
    try:
        parser.feed(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError):
        return None
    return parser.text()


def recover_report_unit(evidence_text: str, doc_name: str, artifacts_dir: Path) -> dict | None:
    """Recover one notes-only monetary unit with its exact filing provenance."""
    if not evidence_text or not doc_name or not artifacts_dir:
        return None
    if _LOCAL_SCALE.search(evidence_text) or _EXCEPTIONS.search(evidence_text):
        return None
    currencies = explicit_money_currencies(evidence_text)
    if currencies != {"USD"}:
        return None
    headings = list(_NOTE_HEADING.finditer(evidence_text))
    if len(headings) != 1:
        return None
    wanted = _words(headings[0].group(), numeric=True)
    anchors = _anchors(evidence_text)
    if len(anchors) < 2:
        return None
    artifacts = Path(artifacts_dir).resolve()
    try:
        manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    entry = _entry(manifest, doc_name)
    if not entry or entry.get("doc_name") != doc_name:
        return None
    recovered = []
    for item in entry.get("report_documents") or []:
        text = _pinned_report(item, entry, artifacts.parent)
        if text is None:
            continue
        source_notes = list(_NOTE_HEADING.finditer(text))
        for index, note in enumerate(source_notes):
            if _words(note.group(), numeric=True) != wanted:
                continue
            stop = source_notes[index + 1].start() if index + 1 < len(source_notes) else len(text)
            note_text = text[note.end():stop]
            if (_LOCAL_SCALE.search(note_text) or _EXCEPTIONS.search(note_text)
                    or explicit_money_currencies(note_text) - {"USD"}):
                continue
            matched = [anchor for anchor in anchors if anchor in _words(note_text)]
            if len(matched) < 2:
                continue
            sections = [section for section in _NOTES.finditer(text[:note.start()])]
            if not sections:
                continue
            # A notes caption cannot cross out of the notes into a later item.
            section = sections[-1]
            between = text[section.end():note.start()]
            if re.search(r"(?im)^\s*ITEM\s+\d+[A-Z]?\b", between):
                continue
            caption = _CAPTION.match(between.lstrip())
            if caption is None:
                continue
            all_captions = list(_CAPTION.finditer(between))
            if len({m.group("scale").lower() for m in all_captions}) != 1:
                continue
            if explicit_money_currencies(text[max(0, section.start() - 200):section.end()] + caption.group()) - {"USD"}:
                continue
            recovered.append({
                "unit": "USD " + caption.group("scale").lower(),
                "quote": re.sub(r"\s+", " ", caption.group()).strip(),
                "scope": "notes_to_consolidated_financial_statements",
                "source_url": item["url"], "sha256": item["sha256"], "path": item["path"],
                "matched_note_heading": re.sub(r"\s+", " ", note.group()).strip(),
                "matched_anchors": matched,
            })
    unique = {json.dumps(item, sort_keys=True): item for item in recovered}
    return next(iter(unique.values())) if len(unique) == 1 else None
