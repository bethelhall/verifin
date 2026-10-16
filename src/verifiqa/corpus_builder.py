from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pymupdf

from verifiqa.retrieval.table_rows import table_row_chunks_from_pages
from verifiqa.types import EvidenceChunk


_LEGACY_CHUNK_ID_RE = re.compile(r"^corpus:(.+):p(\d+):c(\d+)$")
_HTML_BLOCK_TAGS = {
    "br", "div", "p", "section", "article", "header", "footer", "table",
    "thead", "tbody", "tr", "td", "th", "li", "ul", "ol", "h1", "h2",
    "h3", "h4", "h5", "h6",
}
_FILING_MARKERS = (
    "consolidated statements",
    "consolidated balance sheets",
    "consolidated statements of operations",
    "consolidated statements of cash flows",
    "statements of income",
    "statement of income",
    "balance sheet",
    "cash flows",
    "net revenue",
    "total assets",
    "cost of sales",
)
_IR_WRAPPER_MARKERS = (
    "skip to main content",
    "investor relations",
    "sec filings",
    "filing type view all",
    "select a page",
    "email alerts",
    "form 4: statement of changes in",
)


def _download_pdf(url: str, dest: Path, timeout: int = 180, retries: int = 3) -> None:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            ua = os.environ.get("VERIFIQA_SEC_UA", "VerifiQA/1.0")
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if _document_kind_from_bytes(data) not in {"pdf", "html"}:
                raise ValueError(f"Unsupported filing document (got {data[:16]!r}) from {url}")
            tmp = dest.with_suffix(".pdf.part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2 ** attempt)  # 2s, 4s, ...
    raise last_exc


def _extract_text_pages(pdf_path: Path) -> List[str]:
    kind = _document_kind(pdf_path)
    if kind == "html":
        return _extract_html_text_pages(pdf_path)
    if kind != "pdf":
        raise ValueError(f"unsupported_document_format:{kind}")
    doc = pymupdf.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    doc.close()
    return pages


def build_corpus(
    doc_names: List[str],
    doc_links: dict[str, str],
    out_dir: Path,
    progress=None,
    cached_only: bool = False,
    download_timeout: int = 180,
    download_retries: int = 3,
    supplemental_chunks: Optional[Iterable[EvidenceChunk]] = None,
) -> List[EvidenceChunk]:
    pdf_dir = out_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "corpus_chunks.jsonl"
    manifest_path = out_dir / "corpus_manifest.json"

    chunks_by_doc: Dict[str, List[EvidenceChunk]] = defaultdict(list)
    for chunk in _load_stored_corpus_chunks(chunks_path):
        chunks_by_doc[chunk.doc_name].append(chunk)

    manifest = _load_manifest(manifest_path)
    manifest.setdefault("docs", {})

    requested = list(dict.fromkeys(doc_names))
    for i, doc_name in enumerate(requested):
        pdf_path = pdf_dir / f"{doc_name}.pdf"
        existing_chunks = chunks_by_doc.get(doc_name)
        if existing_chunks and not _stored_chunks_need_rebuild(doc_name, existing_chunks, pdf_path):
            row_chunks_added = _ensure_table_row_chunks(chunks_by_doc, doc_name)
            if progress:
                suffix = f", added {row_chunks_added} table-row chunks" if row_chunks_added else ""
                progress(
                    f"  [{i+1}/{len(requested)}] SKIP {doc_name} "
                    f"(already built: {len(chunks_by_doc[doc_name])} chunks{suffix})"
                )
            manifest["docs"][doc_name] = {
                "status": "ok",
                "chunks": len(chunks_by_doc[doc_name]),
                "row_chunks": _count_source_type(chunks_by_doc[doc_name], "table_row"),
                "cached": True,
            }
            continue
        if existing_chunks:
            if progress:
                progress(
                    f"  [{i+1}/{len(requested)}] REBUILD {doc_name} "
                    f"(discarding stale low-quality chunks)"
                )
            chunks_by_doc.pop(doc_name, None)

        url = doc_links.get(doc_name)
        if not url:
            if progress:
                progress(f"  [{i+1}/{len(requested)}] SKIP {doc_name} (no url)")
            manifest["docs"][doc_name] = {"status": "missing_url", "chunks": 0}
            continue

        if not pdf_path.exists():
            if cached_only:
                if progress:
                    progress(f"  [{i+1}/{len(requested)}] SKIP {doc_name} (pdf not cached)")
                manifest["docs"][doc_name] = {"status": "pdf_not_cached", "chunks": 0}
                _write_manifest(manifest_path, manifest)
                continue
            if progress:
                progress(f"  [{i+1}/{len(requested)}] downloading {doc_name}...")
            try:
                _download_pdf(url, pdf_path, timeout=download_timeout, retries=download_retries)
                time.sleep(0.5)
            except Exception as exc:
                if progress:
                    progress(f"  [{i+1}/{len(requested)}] FAILED {doc_name}: {exc}")
                manifest["docs"][doc_name] = {
                    "status": "download_failed",
                    "chunks": 0,
                    "error": str(exc),
                }
                _write_manifest(manifest_path, manifest)
                continue
        else:
            if progress:
                progress(f"  [{i+1}/{len(requested)}] cached {doc_name}")

        source_kind = _document_kind(pdf_path)

        try:
            pages = _extract_text_pages(pdf_path)
        except Exception as exc:
            if progress:
                progress(f"  [{i+1}/{len(requested)}] extract failed {doc_name}: {exc}")
            manifest["docs"][doc_name] = {
                "status": "extract_failed",
                "chunks": 0,
                "error": str(exc),
                "source_kind": source_kind,
            }
            _write_manifest(manifest_path, manifest)
            continue
        if not pages:
            if progress:
                progress(f"  [{i+1}/{len(requested)}] no text extracted {doc_name}")
            manifest["docs"][doc_name] = {
                "status": "no_text_extracted",
                "chunks": 0,
                "cached": pdf_path.exists(),
                "source_kind": source_kind,
            }
            _write_manifest(manifest_path, manifest)
            continue

        doc_chunks = []
        for page_num, page_text in enumerate(pages, start=1):
            doc_chunks.append(EvidenceChunk(
                chunk_id=f"corpus:{doc_name}:p{page_num}",
                doc_name=doc_name,
                page=page_num,
                text=page_text,
                source_type="filing_page",
            ))

        if _primary_chunks_low_quality(doc_name, doc_chunks, source_kind=source_kind):
            if progress:
                progress(f"  [{i+1}/{len(requested)}] low-quality source {doc_name} ({source_kind})")
            manifest["docs"][doc_name] = {
                "status": "low_quality_source",
                "chunks": 0,
                "pages": len(pages),
                "cached": pdf_path.exists(),
                "source_kind": source_kind,
            }
            _write_corpus_chunks(chunks_path, chunks_by_doc)
            _write_manifest(manifest_path, manifest)
            continue

        row_chunks = table_row_chunks_from_pages(doc_chunks)
        doc_chunks.extend(row_chunks)
        chunks_by_doc[doc_name] = doc_chunks
        manifest["docs"][doc_name] = {
            "status": "ok",
            "chunks": len(doc_chunks),
            "row_chunks": len(row_chunks),
            "pages": len(pages),
            "cached": False,
            "source_kind": source_kind,
        }
        _write_corpus_chunks(chunks_path, chunks_by_doc)
        _write_manifest(manifest_path, manifest)
        if progress:
            progress(
                f"  [{i+1}/{len(requested)}] {doc_name}: {len(pages)} pages + "
                f"{len(row_chunks)} table rows → {len(doc_chunks)} chunks"
            )

    supplemental_added = _add_supplemental_chunks(chunks_by_doc, supplemental_chunks or [])
    if supplemental_added and progress:
        progress(f"  Added {supplemental_added} supplemental benchmark evidence page chunks")
    row_chunks_added = _ensure_all_table_row_chunks(chunks_by_doc)
    if row_chunks_added and progress:
        progress(f"  Added {row_chunks_added} table-row retrieval chunks")

    _write_corpus_chunks(chunks_path, chunks_by_doc)
    _write_manifest(manifest_path, manifest)
    return [chunk for doc in sorted(chunks_by_doc) for chunk in chunks_by_doc[doc]]


def evidence_page_supplements_from_records(records: Iterable[dict]) -> List[EvidenceChunk]:
    """Build corpus supplements from FinanceBench full evidence pages.

    These chunks are intentionally labeled ``benchmark_evidence_page`` so runs
    can report them separately from primary downloaded filing pages. They are
    useful when a public document link points to a filing shell but the benchmark
    evidence lives in an exhibit/attachment that the public PDF downloader did
    not ingest.
    """

    chunks: List[EvidenceChunk] = []
    seen: set[tuple[str, int | None, str]] = set()
    for record in records:
        doc_name = str(record.get("doc_name") or record.get("evidence_doc_name") or "").strip()
        company = str(record.get("company") or "").strip()
        financebench_id = str(record.get("financebench_id") or record.get("id") or "").strip()
        evidence = record.get("evidence") or []
        if isinstance(evidence, dict):
            evidence = [evidence]
        if not isinstance(evidence, list):
            continue
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                continue
            item_doc = str(item.get("doc_name") or item.get("evidence_doc_name") or doc_name).strip()
            text = str(item.get("evidence_text_full_page") or item.get("evidence_text") or "").strip()
            if not item_doc or not text:
                continue
            page = _coerce_int(item.get("evidence_page_num") or item.get("page") or item.get("page_num"))
            key = (item_doc, page, _compact_text(text[:2000]))
            if key in seen:
                continue
            seen.add(key)
            page_label = f"p{page}" if page is not None else f"e{index}"
            chunk_id = f"corpus:{item_doc}:benchmark_evidence:{page_label}:{index}"
            chunks.append(EvidenceChunk(
                chunk_id=chunk_id,
                doc_name=item_doc,
                page=page,
                text=text,
                source_type="benchmark_evidence_page",
                company=company,
                financebench_id=financebench_id,
            ))
    return chunks


def load_corpus_chunks(corpus_dir: Path) -> List[EvidenceChunk]:
    return load_corpus(corpus_dir)[0]


def _load_stored_corpus_chunks(chunks_path: Path) -> List[EvidenceChunk]:
    if not chunks_path.exists():
        return []
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(EvidenceChunk(
                chunk_id=record["chunk_id"],
                doc_name=record["doc_name"],
                page=record.get("page"),
                text=record["text"],
                source_type=record.get("source_type", "filing"),
            ))
    return chunks


def _write_corpus_chunks(chunks_path: Path, chunks_by_doc: Dict[str, List[EvidenceChunk]]) -> None:
    tmp_path = chunks_path.with_suffix(chunks_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for doc_name in sorted(chunks_by_doc):
            for chunk in sorted(chunks_by_doc[doc_name], key=lambda c: (c.page or 0, c.chunk_id)):
                handle.write(json.dumps({
                    "chunk_id": chunk.chunk_id,
                    "doc_name": chunk.doc_name,
                    "page": chunk.page,
                    "text": chunk.text,
                    "source_type": chunk.source_type,
                }, ensure_ascii=False) + "\n")
    tmp_path.replace(chunks_path)


def _ensure_all_table_row_chunks(chunks_by_doc: Dict[str, List[EvidenceChunk]]) -> int:
    added = 0
    for doc_name in list(chunks_by_doc):
        added += _ensure_table_row_chunks(chunks_by_doc, doc_name)
    return added


def _ensure_table_row_chunks(chunks_by_doc: Dict[str, List[EvidenceChunk]], doc_name: str) -> int:
    chunks = chunks_by_doc.get(doc_name) or []
    existing_ids = {chunk.chunk_id for chunk in chunks}
    row_chunks = [
        chunk
        for chunk in table_row_chunks_from_pages(chunks)
        if chunk.chunk_id not in existing_ids
    ]
    if not row_chunks:
        return 0
    chunks.extend(row_chunks)
    chunks_by_doc[doc_name] = chunks
    return len(row_chunks)


def _count_source_type(chunks: Iterable[EvidenceChunk], source_type: str) -> int:
    return sum(1 for chunk in chunks if chunk.source_type == source_type)


def _add_supplemental_chunks(
    chunks_by_doc: Dict[str, List[EvidenceChunk]],
    supplemental_chunks: Iterable[EvidenceChunk],
) -> int:
    added = 0
    existing_ids = {
        chunk.chunk_id
        for chunks in chunks_by_doc.values()
        for chunk in chunks
    }
    existing_doc_pages = {
        (chunk.doc_name, chunk.page)
        for chunks in chunks_by_doc.values()
        for chunk in chunks
        if chunk.source_type != "benchmark_evidence_page"
    }
    for chunk in supplemental_chunks:
        if not chunk.doc_name or not chunk.text.strip():
            continue
        if chunk.chunk_id in existing_ids:
            continue
        # If a primary downloaded filing page exists for the same page number,
        # keep the real page and do not add benchmark evidence for that page.
        if chunk.page is not None and (chunk.doc_name, chunk.page) in existing_doc_pages:
            continue
        chunks_by_doc[chunk.doc_name].append(chunk)
        existing_ids.add(chunk.chunk_id)
        added += 1
    return added


def _document_kind(path: Path) -> str:
    if not path.exists():
        return "missing"
    return _document_kind_from_bytes(path.read_bytes()[:4096])


def _document_kind_from_bytes(data: bytes) -> str:
    prefix = data.lstrip()[:4096].lower()
    if prefix.startswith(b"%pdf"):
        return "pdf"
    if b"<!doctype html" in prefix[:256] or b"<html" in prefix[:512]:
        return "html"
    return "unknown"


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag in _HTML_BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in _HTML_BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        joined = " ".join(self._parts)
        joined = re.sub(r"[ \t\r\f\v]+", " ", joined)
        joined = re.sub(r"\n\s+", "\n", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()


def _extract_html_text_pages(path: Path, max_chars: int = 6500, overlap: int = 400) -> List[str]:
    parser = _VisibleTextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    text = parser.text()
    if not text:
        return []
    return _split_text_windows(text, max_chars=max_chars, overlap=overlap)


def _split_text_windows(text: str, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def _stored_chunks_need_rebuild(doc_name: str, chunks: List[EvidenceChunk], source_path: Path) -> bool:
    primary_chunks = [chunk for chunk in chunks if chunk.source_type != "benchmark_evidence_page"]
    if not primary_chunks:
        return False
    source_kind = _document_kind(source_path)
    return _primary_chunks_low_quality(doc_name, primary_chunks, source_kind=source_kind)


def _primary_chunks_low_quality(
    doc_name: str,
    chunks: List[EvidenceChunk],
    *,
    source_kind: str,
) -> bool:
    if not chunks:
        return False
    # Do not classify short 8-Ks or other non-10-K filings as broken just
    # because they have few pages.  This guard targets stale 10-K corpora where
    # an IR listing page was cached in place of the filing itself.
    if "_10K" not in doc_name.upper() and "10-K" not in doc_name.upper():
        return False

    # Wrapper/nav markers live in the front matter; filing content can appear
    # anywhere (some inline-XBRL 10-Ks front-load an XBRL metadata block, pushing
    # the financial statements past the first few chunks), so scan the whole doc
    # for filing markers to avoid discarding a real filing as an IR wrapper page.
    front_text = "\n".join(chunk.text for chunk in chunks[:8]).lower()
    full_text = "\n".join(chunk.text for chunk in chunks).lower()
    filing_hits = sum(1 for marker in _FILING_MARKERS if marker in full_text)
    wrapper_hits = sum(1 for marker in _IR_WRAPPER_MARKERS if marker in front_text)

    if source_kind == "html" and wrapper_hits >= 2 and filing_hits == 0:
        return True
    if len(chunks) <= 8 and wrapper_hits >= 2 and filing_hits == 0:
        return True
    return False


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_manifest(manifest_path: Path, manifest: dict) -> None:
    docs = manifest.get("docs", {})
    manifest["summary"] = {
        "n_docs": len(docs),
        "n_ok": sum(1 for item in docs.values() if item.get("status") == "ok"),
        "n_failed": sum(1 for item in docs.values() if item.get("status") not in ("ok", None)),
    }
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(manifest_path)


def load_corpus(corpus_dir: Path) -> Tuple[List[EvidenceChunk], dict]:
    chunks_path = corpus_dir / "corpus_chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No corpus found at {chunks_path}. Run build-corpus first.")
    chunks = _load_stored_corpus_chunks(chunks_path)
    loaded = _coalesce_legacy_window_chunks(chunks)
    stats = {
        "stored_records": len(chunks),
        "loaded_chunks": len(loaded),
        "coalesced_legacy_windows": len(loaded) != len(chunks),
        "source_types": _source_type_counts(loaded),
    }
    return loaded, stats


def _coalesce_legacy_window_chunks(chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
    groups = defaultdict(list)
    passthrough = []

    for chunk in chunks:
        parsed = _parse_legacy_window_id(chunk.chunk_id)
        if parsed is None:
            passthrough.append(chunk)
            continue
        doc_name, page, chunk_idx = parsed
        groups[(doc_name, page)].append((chunk_idx, chunk))

    if not groups:
        return chunks

    page_chunks = []
    for (doc_name, page), items in sorted(groups.items()):
        text = ""
        for _, chunk in sorted(items):
            text = _append_overlapping(text, chunk.text)
        page_chunks.append(EvidenceChunk(
            chunk_id=f"corpus:{doc_name}:p{page}",
            doc_name=doc_name,
            page=page,
            text=text,
            source_type="filing_page",
        ))

    return passthrough + page_chunks


def _parse_legacy_window_id(chunk_id: str) -> Optional[Tuple[str, int, int]]:
    match = _LEGACY_CHUNK_ID_RE.match(chunk_id)
    if not match:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def _append_overlapping(existing: str, addition: str, max_overlap: int = 200) -> str:
    if not existing:
        return addition
    limit = min(len(existing), len(addition), max_overlap)
    for overlap in range(limit, 0, -1):
        if existing[-overlap:] == addition[:overlap]:
            return existing + addition[overlap:]
    return existing + addition


def _source_type_counts(chunks: List[EvidenceChunk]) -> dict:
    counts = defaultdict(int)
    for chunk in chunks:
        counts[chunk.source_type] += 1
    return dict(sorted(counts.items()))


def _coerce_int(value) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())
