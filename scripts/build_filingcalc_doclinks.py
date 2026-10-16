#!/usr/bin/env python3
"""Resolve FilingCalc filings to their primary EDGAR document URLs.

FilingCalc examples carry ``provenance.cik`` and ``provenance.accession`` but no
downloadable document link, so ``build-corpus`` has nothing to fetch. This script
reads those, asks EDGAR's per-accession ``index.json`` for the primary 10-K
document, and writes a ``financebench_document_information.jsonl`` file in the
shape ``build-corpus`` already consumes (one ``{"doc_name", "doc_link"}`` per line).

After running this, build the corpus with:

    python3 -m verifiqa.cli build-corpus \
      --data data/multicompany_provable/data \
      --examples data/multicompany_provable/data/provable_calcrequired.jsonl \
      --out data/corpus_filingcalc

EDGAR requires a descriptive User-Agent with contact info and rate-limits to
10 req/s; we send one and sleep between requests. Set SEC_USER_AGENT to your own
"Name email" before running a large job.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

_REQUEST_INTERVAL = 0.2  # ~5 req/s, comfortably under EDGAR's 10 req/s limit


def _require_sec_user_agent() -> str:
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is required for SEC EDGAR requests; set it to a "
            "descriptive value that includes contact information."
        )
    return user_agent


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _require_sec_user_agent(),
            "Accept-Encoding": "gzip, deflate",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if resp.headers.get("Content-Encoding") == "gzip":
        import gzip
        data = gzip.decompress(data)
    return data


def _load_docs(examples_path: Path) -> dict[str, dict]:
    """doc_name -> {cik, accession, ticker} from the FilingCalc examples."""
    docs: dict[str, dict] = {}
    for line in examples_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        doc = r.get("doc_name")
        prov = r.get("ground_truth", {}).get("provenance") or {}
        cik, acc = prov.get("cik") or r.get("cik"), prov.get("accession")
        if doc and cik and acc and doc not in docs:
            docs[doc] = {"cik": int(cik), "accession": acc, "ticker": (prov.get("ticker") or "").lower()}
    return docs


def _pick_primary(items: list[dict], ticker: str) -> str | None:
    """Choose the primary 10-K document from an EDGAR filing directory listing.

    Prefer a ``{ticker}-{yyyymmdd}.htm`` inline-XBRL document; otherwise the
    largest .htm that is not an exhibit (``ex...``) or an R-report fragment.
    """
    htms = [it for it in items if str(it.get("name", "")).lower().endswith((".htm", ".html"))]
    if not htms:
        return None
    if ticker:
        for it in htms:
            if re.match(rf"^{re.escape(ticker)}-\d{{8}}\.htm", it["name"].lower()):
                return it["name"]
    def _ok(name: str) -> bool:
        n = name.lower()
        return not n.startswith(("ex", "r")) and "exhibit" not in n
    def _size(it: dict) -> int:
        try:
            return int(it.get("size") or 0)
        except (ValueError, TypeError):
            return 0
    candidates = [it for it in htms if _ok(it["name"])] or htms
    candidates.sort(key=_size, reverse=True)
    return candidates[0]["name"]


def resolve(doc: str, meta: dict) -> tuple[str | None, str]:
    cik, acc = meta["cik"], meta["accession"]
    acc_nodash = acc.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"
    try:
        listing = json.loads(_get(f"{base}/index.json"))
        items = listing.get("directory", {}).get("item", [])
    except Exception as exc:  # noqa: BLE001
        return None, f"index_fetch_failed:{type(exc).__name__}"
    primary = _pick_primary(items, meta["ticker"])
    if not primary:
        return None, "no_primary_htm_found"
    return f"{base}/{primary}", "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=Path,
                    default=Path("data/multicompany_provable/data/provable_calcrequired.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/multicompany_provable/data/financebench_document_information.jsonl"))
    args = ap.parse_args()

    docs = _load_docs(args.examples)
    _require_sec_user_agent()
    print(f"Resolving {len(docs)} filings from EDGAR...", file=sys.stderr)
    rows, failed = [], []
    for i, (doc, meta) in enumerate(sorted(docs.items()), 1):
        url, status = resolve(doc, meta)
        if url:
            rows.append({"doc_name": doc, "doc_link": url})
        else:
            failed.append((doc, status))
        print(f"  [{i}/{len(docs)}] {doc}: {status}", file=sys.stderr)
        time.sleep(_REQUEST_INTERVAL)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(rows)} doc links -> {args.out}", file=sys.stderr)
    if failed:
        print(f"{len(failed)} unresolved (fix or download manually):", file=sys.stderr)
        for doc, why in failed:
            print(f"  {doc}: {why}", file=sys.stderr)


if __name__ == "__main__":
    main()
