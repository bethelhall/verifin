"""
Download SEC XBRL calculation linkbases and schemas for FinanceBench source filings.

The script resolves each source document to a SEC accession, reads the filing
directory index, downloads any calculation linkbase XML and schema files, and writes a
manifest with explicit miss/failure statuses.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACCESSION_RE = re.compile(r"\b(\d{10}-\d{2}-\d{6})\b")
CIK_RE = re.compile(r"CIK-?0*(\d{1,10})", re.IGNORECASE)
CAL_LINKBASE_RE = re.compile(r"(^|[-_])cal(?:culation)?\.xml$", re.IGNORECASE)
OTHER_LINKBASE_RE = re.compile(r"(^|[-_])(def|lab|pre|ref)\.xml$", re.IGNORECASE)
ANY_LINKBASE_RE = re.compile(r"(^|[-_])(cal|def|lab|pre|ref)\.xml$", re.IGNORECASE)
SCHEMA_RE = re.compile(r"\.xsd$", re.IGNORECASE)
INSTANCE_XML_RE = re.compile(r"\.xml$", re.IGNORECASE)
XBRL_PACKAGE_RE = re.compile(r"-xbrl\.zip$", re.IGNORECASE)


def _require_sec_user_agent(user_agent: str | None = None) -> str:
    configured = (user_agent or os.environ.get("SEC_USER_AGENT", "")).strip()
    if not configured:
        raise RuntimeError(
            "SEC_USER_AGENT is required for SEC EDGAR requests; set it to a "
            "descriptive value that includes contact information."
        )
    return configured


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _source_doc_names(questions_path: Path) -> list[str]:
    docs: set[str] = set()
    for row in _read_jsonl(questions_path):
        if row.get("doc_name"):
            docs.add(str(row["doc_name"]))
        for evidence in row.get("evidence") or []:
            if isinstance(evidence, dict) and evidence.get("doc_name"):
                docs.add(str(evidence["doc_name"]))
    return sorted(docs)


def _load_doc_info(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["doc_name"]): row for row in _read_jsonl(path) if row.get("doc_name")}


def _load_xbrl_metadata(
    path: Path, doc_names: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    summaries: dict[str, dict[str, Any]] = {}
    company_cik_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    if not path.exists():
        return summaries, {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_name = str(row.get("doc_name", ""))
            xbrl = row.get("xbrl") or {}
            cik = _normalize_cik(xbrl.get("cik"))
            company = row.get("company")
            if cik and company:
                company_cik_counts[str(company)][cik] += 1
            if doc_name not in doc_names:
                continue
            summaries[doc_name] = {
                "cik": cik,
                "form_type": _normalize_form(xbrl.get("form_type")),
                "period_end": xbrl.get("period_end"),
                "prior_period_end": xbrl.get("prior_period_end"),
                "status": xbrl.get("status"),
            }
    company_ciks = {
        company: counts.most_common(1)[0][0]
        for company, counts in company_cik_counts.items()
        if counts
    }
    return summaries, company_ciks


def _normalize_cik(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return text.zfill(10)
    match = CIK_RE.search(text)
    if match:
        return match.group(1).zfill(10)
    return None


def _normalize_form(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    mapping = {
        "10K": "10-K",
        "10-K": "10-K",
        "10K_ANNUALREPORT": "10-K",
        "10Q": "10-Q",
        "10-Q": "10-Q",
        "8K": "8-K",
        "8-K": "8-K",
    }
    return mapping.get(text, text if text else None)


def _accession_from_link(url: Any) -> str | None:
    if not url:
        return None
    match = ACCESSION_RE.search(str(url))
    return match.group(1) if match else None


def _cik_from_link(url: Any) -> str | None:
    if not url:
        return None
    match = CIK_RE.search(str(url))
    return match.group(1).zfill(10) if match else None


def _fetch_bytes(url: str, user_agent: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": _require_sec_user_agent(user_agent)}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _fetch_json(url: str, user_agent: str, timeout: int = 60) -> Any:
    return json.loads(_fetch_bytes(url, user_agent, timeout=timeout))


def _companyfacts_accession(
    cik: str,
    form_type: str | None,
    period_end: str | None,
    doc_period: int | None,
    cache_dir: Path,
) -> str | None:
    if not form_type or not period_end:
        return None
    path = cache_dir / f"{cik}.json"
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    counts: Counter[str] = Counter()
    for concept in data.values():
        units = concept.get("units") if isinstance(concept, dict) else None
        if not isinstance(units, dict):
            continue
        for facts in units.values():
            if not isinstance(facts, list):
                continue
            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                if fact.get("form") != form_type:
                    continue
                if fact.get("end") != period_end:
                    continue
                if doc_period is not None and fact.get("fy") not in (doc_period, doc_period + 1):
                    continue
                accn = fact.get("accn")
                if accn:
                    counts[str(accn)] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _sec_submissions_accession(
    cik: str,
    form_type: str | None,
    period_end: str | None,
    doc_period: int | None,
    doc_date: str | None,
    user_agent: str,
) -> str | None:
    if not form_type:
        return None
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        data = _fetch_json(submissions_url, user_agent)
    except Exception:
        return None

    filing_sets = [data.get("filings", {}).get("recent", {})]
    for file_info in data.get("filings", {}).get("files", []) or []:
        name = file_info.get("name")
        if not name:
            continue
        try:
            filing_sets.append(_fetch_json(f"https://data.sec.gov/submissions/{name}", user_agent))
        except Exception:
            continue
        time.sleep(0.1)

    candidates: list[tuple[int, str]] = []
    for filings in filing_sets:
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        period_dates = filings.get("reportDate", [])
        filing_dates = filings.get("filingDate", [])
        for i, form in enumerate(forms):
            if form != form_type:
                continue
            score = 0
            if i < len(filing_dates) and doc_date and filing_dates[i] == doc_date:
                score += 20
            if i < len(period_dates) and period_end and period_dates[i] == period_end:
                score += 10
            if i < len(filing_dates) and doc_period is not None:
                filed_year = _safe_year(filing_dates[i])
                if filed_year in (doc_period, doc_period + 1):
                    score += 3
            if score and i < len(accessions):
                candidates.append((score, str(accessions[i])))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _safe_year(value: Any) -> int | None:
    if not value:
        return None
    try:
        return int(str(value)[:4])
    except ValueError:
        return None


def _doc_date_from_name(doc_name: str) -> str | None:
    match = re.search(r"_dated-(\d{4}-\d{2}-\d{2})", doc_name)
    return match.group(1) if match else None


def _resolve_accession(
    doc_name: str,
    doc_info: dict[str, Any],
    xbrl: dict[str, Any],
    company_ciks: dict[str, str],
    cache_dir: Path,
    user_agent: str,
) -> tuple[str | None, str | None, str]:
    cik = xbrl.get("cik") or _cik_from_link(doc_info.get("doc_link"))
    if not cik and doc_info.get("company") in company_ciks:
        cik = company_ciks[str(doc_info["company"])]
    cik = _normalize_cik(cik)
    form_type = xbrl.get("form_type") or _normalize_form(doc_info.get("doc_type"))
    period_end = xbrl.get("period_end")
    doc_period = _safe_year(doc_info.get("doc_period"))
    doc_date = _doc_date_from_name(doc_name)

    if not cik:
        return None, None, "missing_cik"

    accession = _accession_from_link(doc_info.get("doc_link"))
    if accession:
        return cik, accession, "doc_link"

    accession = _companyfacts_accession(cik, form_type, period_end, doc_period, cache_dir)
    if accession:
        return cik, accession, "companyfacts_cache"

    accession = _sec_submissions_accession(
        cik, form_type, period_end, doc_period, doc_date, user_agent
    )
    if accession:
        return cik, accession, "sec_submissions"

    return cik, None, "unresolved_accession"


def _filing_index_url(cik: str, accession: str) -> str:
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/index.json"
    )


def _filing_file_url(cik: str, accession: str, filename: str) -> str:
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{filename}"
    )


def _filing_index_files(index_json: dict[str, Any], pattern: re.Pattern[str]) -> list[str]:
    directory = index_json.get("directory") if isinstance(index_json, dict) else None
    items = directory.get("item", []) if isinstance(directory, dict) else []
    filenames: list[str] = []
    for item in items:
        name = item.get("name") if isinstance(item, dict) else None
        if name and pattern.search(str(name)):
            filenames.append(str(name))
    return sorted(set(filenames))


def _calculation_files(index_json: dict[str, Any]) -> list[str]:
    return _filing_index_files(index_json, CAL_LINKBASE_RE)


def _schema_files(index_json: dict[str, Any]) -> list[str]:
    return _filing_index_files(index_json, SCHEMA_RE)


def _other_linkbase_files(index_json: dict[str, Any]) -> list[str]:
    return _filing_index_files(index_json, OTHER_LINKBASE_RE)


def _instance_files(index_json: dict[str, Any]) -> list[str]:
    candidates = _filing_index_files(index_json, INSTANCE_XML_RE)
    return [
        name
        for name in candidates
        if name.lower() != "filingsummary.xml" and not ANY_LINKBASE_RE.search(name)
    ]


def _xbrl_package_files(index_json: dict[str, Any]) -> list[str]:
    return _filing_index_files(index_json, XBRL_PACKAGE_RE)


def _download_files(
    filenames: list[str],
    doc_dir: Path,
    cik: str,
    accession: str,
    user_agent: str,
    force: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for filename in filenames:
        target = doc_dir / filename
        url = _filing_file_url(cik, accession, filename)
        file_entry = {
            "filename": filename,
            "url": url,
            "path": str(target),
            "status": "pending",
        }
        try:
            if target.exists() and not force:
                payload = target.read_bytes()
            else:
                payload = _fetch_bytes(url, user_agent)
                target.write_bytes(payload)
            file_entry["status"] = "saved"
            file_entry["bytes"] = len(payload)
        except Exception as exc:
            file_entry["status"] = f"download_error:{type(exc).__name__}:{exc}"
        results.append(file_entry)
    return results


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_for_doc(
    doc_name: str,
    doc_info_map: dict[str, dict[str, Any]],
    xbrl_map: dict[str, dict[str, Any]],
    company_ciks: dict[str, str],
    cache_dir: Path,
    out_dir: Path,
    user_agent: str,
    force: bool,
) -> dict[str, Any]:
    doc_info = doc_info_map.get(doc_name, {})
    xbrl = xbrl_map.get(doc_name, {})
    entry: dict[str, Any] = {
        "doc_name": doc_name,
        "company": doc_info.get("company"),
        "doc_type": doc_info.get("doc_type"),
        "doc_period": doc_info.get("doc_period"),
        "doc_link": doc_info.get("doc_link"),
        "xbrl": xbrl,
        "status": "pending",
        "calculation_linkbases": [],
        "schemas": [],
        "instances": [],
        "other_linkbases": [],
        "xbrl_packages": [],
    }

    doc_form = _normalize_form(doc_info.get("doc_type"))
    xbrl_form = xbrl.get("form_type")
    if (xbrl_form or doc_form) not in {"10-K", "10-Q", "8-K"}:
        entry["status"] = "not_sec_xbrl_filing"
        return entry

    cik, accession, resolution_source = _resolve_accession(
        doc_name, doc_info, xbrl, company_ciks, cache_dir, user_agent
    )
    entry["cik"] = cik
    entry["accession"] = accession
    entry["accession_resolution"] = resolution_source

    if not cik or not accession:
        entry["status"] = resolution_source
        return entry

    index_url = _filing_index_url(cik, accession)
    entry["filing_index_url"] = index_url
    try:
        index_json = _fetch_json(index_url, user_agent)
    except urllib.error.HTTPError as exc:
        entry["status"] = f"index_http_error:{exc.code}"
        return entry
    except Exception as exc:
        entry["status"] = f"index_fetch_error:{type(exc).__name__}:{exc}"
        return entry

    cal_files = _calculation_files(index_json)
    schema_files = _schema_files(index_json)
    instance_files = _instance_files(index_json)
    other_linkbase_files = _other_linkbase_files(index_json)
    xbrl_package_files = _xbrl_package_files(index_json)
    entry["index_file_count"] = len(index_json.get("directory", {}).get("item", []))
    if not any([cal_files, schema_files, instance_files, other_linkbase_files, xbrl_package_files]):
        entry["status"] = "no_xbrl_artifacts"
        return entry

    doc_dir = out_dir / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    entry["calculation_linkbases"] = _download_files(
        cal_files, doc_dir, cik, accession, user_agent, force
    )
    entry["schemas"] = _download_files(
        schema_files, doc_dir, cik, accession, user_agent, force
    )
    entry["instances"] = _download_files(
        instance_files, doc_dir, cik, accession, user_agent, force
    )
    entry["other_linkbases"] = _download_files(
        other_linkbase_files, doc_dir, cik, accession, user_agent, force
    )
    entry["xbrl_packages"] = _download_files(
        xbrl_package_files, doc_dir, cik, accession, user_agent, force
    )

    saved = sum(1 for item in entry["calculation_linkbases"] if item["status"] == "saved")
    xbrl_saved = sum(
        1
        for key in ("schemas", "instances", "other_linkbases", "xbrl_packages")
        for item in entry[key]
        if item["status"] == "saved"
    )
    entry["status"] = "saved" if saved else "calculation_download_failed"
    if saved == 0 and xbrl_saved:
        entry["status"] = "xbrl_only_no_calculation_linkbase"
    _write_json(doc_dir / "filing_metadata.json", entry)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/numerical_questions.jsonl"))
    parser.add_argument(
        "--doc-info",
        type=Path,
        default=Path("data/financebench_sample/data/financebench_document_information.jsonl"),
    )
    parser.add_argument(
        "--xbrl-augmented",
        type=Path,
        default=Path("data/financebench_sample/data/financebench_xbrl_augmented.jsonl"),
    )
    parser.add_argument("--xbrl-cache", type=Path, default=Path("artifacts/xbrl_cache"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/calculation_linkbases"))
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT"),
        help="SEC EDGAR User-Agent (default: SEC_USER_AGENT environment variable)",
    )
    parser.add_argument("--doc-name", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.user_agent:
        parser.error(
            "SEC_USER_AGENT is required for SEC EDGAR requests; set it to a "
            "descriptive value that includes contact information"
        )

    doc_names = args.doc_name or _source_doc_names(args.questions)
    if args.limit:
        doc_names = doc_names[: args.limit]

    doc_info_map = _load_doc_info(args.doc_info)
    xbrl_map, company_ciks = _load_xbrl_metadata(args.xbrl_augmented, set(doc_names))

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest_jsonl_path = args.out / "manifest.jsonl"

    results: list[dict[str, Any]] = []
    with manifest_jsonl_path.open("w", encoding="utf-8") as jsonl:
        for i, doc_name in enumerate(doc_names, start=1):
            result = download_for_doc(
                doc_name=doc_name,
                doc_info_map=doc_info_map,
                xbrl_map=xbrl_map,
                company_ciks=company_ciks,
                cache_dir=args.xbrl_cache,
                out_dir=args.out,
                user_agent=args.user_agent,
                force=args.force,
            )
            results.append(result)
            jsonl.write(json.dumps(result, sort_keys=True) + "\n")
            print(
                f"[{i}/{len(doc_names)}] {doc_name}: {result['status']}",
                file=sys.stderr,
                flush=True,
            )
            if i < len(doc_names) and args.sleep:
                time.sleep(args.sleep)

    counts = Counter(row["status"] for row in results)
    manifest = {
        "source_questions": str(args.questions),
        "source_doc_info": str(args.doc_info),
        "source_xbrl_augmented": str(args.xbrl_augmented),
        "n_docs": len(results),
        "status_counts": dict(sorted(counts.items())),
        "results": results,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"n_docs": len(results), "status_counts": manifest["status_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
