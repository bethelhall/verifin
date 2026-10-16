"""
Download missing SEC filings from EDGAR and append them to the corpus.

Usage:
    python3 scripts/download_edgar_missing.py --corpus data/corpus

Targets the 11 filings that failed to download from corporate IR sites.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


def _require_sec_user_agent() -> str:
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is required for SEC EDGAR requests; set it to a "
            "descriptive value that includes contact information."
        )
    return user_agent


# (doc_name, CIK, accession_number, primary_htm_filename)
EDGAR_TARGETS = [
    ("ACTIVISIONBLIZZARD_2019_10K", "718877",  "000071887720000003", "atvi-12312019x10xk.htm"),
    ("ADOBE_2015_10K",              "796343",  None, None),   # will be looked up
    ("ADOBE_2016_10K",              "796343",  None, None),
    ("ADOBE_2017_10K",              "796343",  None, None),
    ("LOCKHEEDMARTIN_2021_10K",     "936468",  "000093646822000008", "lmt-20211231.htm"),
    ("LOCKHEEDMARTIN_2022_10K",     "936468",  "000093646823000009", "lmt-20221231.htm"),
    ("PEPSICO_2021_10K",            "77476",   "000007747622000010", "pep-20211225.htm"),
    ("PEPSICO_2022_10K",            "77476",   "000007747623000007", "pep-20221231.htm"),
]

# Adobe 10-K periods (fiscal year ends Nov/Dec)
ADOBE_PERIODS = {
    "ADOBE_2015_10K": 2015,
    "ADOBE_2016_10K": 2016,
    "ADOBE_2017_10K": 2017,
}


class _TextExtractor(HTMLParser):
    """Strip HTML tags; preserve block-level whitespace."""
    SKIP_TAGS = {"script", "style", "head", "noscript"}
    BLOCK_TAGS = {"p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "table"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        # Collapse runs of whitespace/newlines while preserving paragraph breaks
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": _require_sec_user_agent()}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _html_to_pages(html_bytes: bytes, chars_per_page: int = 3000) -> list[str]:
    """Extract text from HTML and split into fixed-size page blocks."""
    parser = _TextExtractor()
    parser.feed(html_bytes.decode("utf-8", errors="replace"))
    text = parser.get_text()
    pages = []
    for i in range(0, max(len(text), 1), chars_per_page):
        chunk = text[i : i + chars_per_page].strip()
        if chunk:
            pages.append(chunk)
    return pages


def _lookup_accession(cik: str, form: str, fiscal_year: int) -> tuple[str, str] | None:
    """Return (accession_no_dash, primary_doc) for the given filing."""
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    data = json.loads(_fetch(url))
    filings = data["filings"]["recent"]
    for i, (f, date, acc) in enumerate(zip(
        filings["form"], filings["filingDate"], filings["accessionNumber"]
    )):
        if f == form:
            yr = int(date[:4])
            # 10-K for FY N is usually filed in N+1 (Jan-Apr)
            if yr in (fiscal_year, fiscal_year + 1):
                acc_no_dash = acc.replace("-", "")
                primary = filings.get("primaryDocument", [""])[i] if "primaryDocument" in filings else ""
                return acc_no_dash, primary
    return None


def process(doc_name: str, cik: str, acc_no_dash: str | None, primary_htm: str | None,
            corpus_dir: Path, existing_docs: set[str]) -> int:
    if doc_name in existing_docs:
        print(f"  SKIP {doc_name} (already in corpus)")
        return 0

    # Resolve accession if not provided
    if acc_no_dash is None:
        fiscal_year = ADOBE_PERIODS.get(doc_name)
        result = _lookup_accession(cik.zfill(10), "10-K", fiscal_year)
        if result is None:
            print(f"  FAIL {doc_name}: could not find accession number")
            return 0
        acc_no_dash, primary_htm = result
        print(f"  Found {doc_name}: accession {acc_no_dash}, primary={primary_htm}")

    if not primary_htm:
        print(f"  FAIL {doc_name}: no primary document known")
        return 0

    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dash}/{primary_htm}"
    print(f"  Downloading {doc_name} from EDGAR...", flush=True)
    try:
        html_bytes = _fetch(url, timeout=120)
    except Exception as exc:
        print(f"  FAIL {doc_name}: {exc}")
        return 0

    pages = _html_to_pages(html_bytes)
    if not pages:
        print(f"  FAIL {doc_name}: no text extracted")
        return 0

    chunks_path = corpus_dir / "corpus_chunks.jsonl"
    with chunks_path.open("a", encoding="utf-8") as f:
        for page_num, text in enumerate(pages, start=1):
            chunk = {
                "chunk_id": f"corpus:{doc_name}:p{page_num}",
                "doc_name": doc_name,
                "page": page_num,
                "text": text,
                "source_type": "filing_page",
            }
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"  OK {doc_name}: {len(pages)} pages appended to corpus")
    return len(pages)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    args = parser.parse_args()

    _require_sec_user_agent()

    chunks_path = args.corpus / "corpus_chunks.jsonl"
    existing_docs: set[str] = set()
    if chunks_path.exists():
        with chunks_path.open() as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    existing_docs.add(d.get("doc_name", ""))
    print(f"Corpus has {len(existing_docs)} existing docs")

    total = 0
    for doc_name, cik, acc_no_dash, primary_htm in EDGAR_TARGETS:
        n = process(doc_name, cik, acc_no_dash, primary_htm, args.corpus, existing_docs)
        total += n
        time.sleep(0.3)

    print(f"\nDone. Added {total} pages total.")
    print("Re-run 'verifiqa run' to use the updated corpus (embeddings will be rebuilt).")


if __name__ == "__main__":
    main()
