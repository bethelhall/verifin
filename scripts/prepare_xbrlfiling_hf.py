#!/usr/bin/env python3
"""Build or validate the anonymous-author XBRLFiling Hugging Face release.

This command is offline. It never logs in, uploads files, or changes source data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("data/multicompany_provable/data/provable_calcrequired.jsonl")
LINKS = Path("data/multicompany_provable/data/financebench_document_information.jsonl")
SOURCE_SHA256 = "a63655a04251bce99cc44a77af5c4abf42fb4a4f1b3e098ce03e3668f8a385ce"
LINKS_SHA256 = "6cd4594b2c1713b0e38678fff27b6c4e510708ae3724ab4cc57275ec64f8c6d9"
TEMPLATES = Path(__file__).parent / "templates/xbrlfiling_hf"
LICENSE = "cc-by-nc-4.0"
FAMILIES = {
    **dict.fromkeys(("GrossProfit", "OperatingIncomeLoss", "OperatingExpenses",
                    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                    "NetIncomeLoss"), "Income statement"),
    **dict.fromkeys(("AssetsCurrent", "AssetsNoncurrent", "Liabilities", "LiabilitiesCurrent",
                    "LiabilitiesNoncurrent", "PropertyPlantAndEquipmentNet", "StockholdersEquity"),
                   "Balance sheet"),
    **dict.fromkeys(("NetCashProvidedByUsedInOperatingActivities",
                    "NetCashProvidedByUsedInInvestingActivities",
                    "NetCashProvidedByUsedInFinancingActivities"), "Cash flow"),
}
PAYLOAD_FILES = {
    "README.md", "LICENSE.md", "NOTICE.md", "data/test.jsonl", "data/test.parquet",
    "metadata/filings.jsonl", "metadata/quality_report.json", "metadata/manifest.json",
}
PRIVATE_PATTERNS = {
    "local_path": re.compile(r"(?i)(?:/(?:home|users|tmp|workspace|mnt|root)/|file://|[a-z]:\\+)"),
    "email": re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}"),
    "workspace_identity": re.compile(r"(?i)\b(?:review_lab|review_user)\b"),
    "access_token": re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def privacy_findings(value: object) -> list[str]:
    """Return rule names only, never the possibly sensitive matched text."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return [name for name, pattern in PRIVATE_PATTERNS.items() if pattern.search(text)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_rows(source_rows: list[dict], document_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    links = {row["doc_name"]: row["doc_link"] for row in document_rows}
    require(len(links) == len(document_rows), "Duplicate filing-link identifiers")
    rows, filings = [], {}
    for source in source_rows:
        gt, evidence = source["ground_truth"], source["evidence"]
        provenance = gt["provenance"]
        require(len(evidence) == 1, "Expected exactly one evidence block per question")
        context = evidence[0]["evidence_text"]
        require(context == evidence[0]["evidence_text_full_page"], "Evidence variants differ")
        require(context.count("[redacted]") == 1, "Expected exactly one target redaction")
        require(source["answer"] == gt["answer"], "Answer disagrees with ground truth")
        require(source["company"] == provenance["ticker"] and source["cik"] == provenance["cik"],
                "Issuer identity differs from provenance")
        require(source["doc_name"] == provenance["filing"] == evidence[0]["doc_name"],
                "Filing identity differs from evidence/provenance")
        require(gt["unit"] == "USD millions", "Unexpected answer unit")
        require(gt["proof_source"] == "xbrl_calculation_linkbase", "Unexpected proof source")
        require(source["doc_name"] in links, "Missing SEC URL for a selected filing")
        url = links[source["doc_name"]]
        parsed = urlparse(url)
        expected_prefix = f"/Archives/edgar/data/{int(provenance['cik'])}/{provenance['accession'].replace('-', '')}/"
        require(parsed.scheme == "https" and parsed.hostname == "www.sec.gov"
                and parsed.path.startswith(expected_prefix), "Filing URL does not match the SEC accession")
        # Explicit field selection excludes workspace metadata and author identities.
        row = {
            "id": source["financebench_id"],
            "question": source["question"],
            "context": context,
            "answer": source["answer"],
            "answer_value": gt["value"],
            "answer_unit": gt["unit"],
            "ticker": provenance["ticker"],
            "cik": str(provenance["cik"]).zfill(10),
            "filing_id": provenance["filing"],
            "fiscal_year": provenance["fiscal_year"],
            "period_end": provenance["period_end"],
            "report_date": provenance["report_date"],
            "statement_family": FAMILIES[gt["concept"]],
            "target_concept": gt["concept"],
            "calculation": gt["identity"],
            "operands": [{key: operand[key] for key in ("concept", "weight", "value_millions")}
                         for operand in gt["operands"]],
            "source_url": url,
            "accession": provenance["accession"],
            "proof_source": gt["proof_source"],
        }
        require(all(value is not None and value != "" and value != [] for value in row.values()),
                "Missing release field")
        require(math.isfinite(row["answer_value"]), "Nonfinite answer value")
        require(not privacy_findings(row), "Private metadata detected in selected dataset fields")
        filing = {key: row[key] for key in ("filing_id", "ticker", "cik", "accession", "fiscal_year",
                                          "period_end", "report_date", "source_url")}
        prior = filings.setdefault(row["filing_id"], filing)
        require(prior == filing, "Inconsistent filing metadata")
        rows.append(row)
    require(len({row["id"] for row in rows}) == len(rows), "Duplicate question identifiers")
    require(len({row["question"] for row in rows}) == len(rows), "Duplicate question text")
    require(set(links) == set(filings), "Filing links do not exactly cover the selected questions")
    return rows, [filings[key] for key in sorted(filings)]


def quality_report(rows: list[dict], filings: list[dict]) -> dict:
    residuals = []
    for row in rows:
        total = sum((Decimal(str(op["weight"])) * Decimal(str(op["value_millions"]))
                     for op in row["operands"]), Decimal(0))
        residual = abs(total - Decimal(str(row["answer_value"])))
        tolerance = max(Decimal(1), abs(Decimal(str(row["answer_value"]))) * Decimal("0.005"))
        require(residual <= tolerance, "Stored calculation exceeds source construction tolerance")
        residuals.append(residual)
    return {
        "status": "PASS", "rows": len(rows), "unique_ids": len({row["id"] for row in rows}),
        "unique_questions": len({row["question"] for row in rows}),
        "companies": len({row["ticker"] for row in rows}), "filings": len(filings),
        "target_concepts": len({row["target_concept"] for row in rows}),
        "statement_family_counts": dict(sorted(Counter(row["statement_family"] for row in rows).items())),
        "fiscal_year_min": min(row["fiscal_year"] for row in rows),
        "fiscal_year_max": max(row["fiscal_year"] for row in rows),
        "operands": sum(len(row["operands"]) for row in rows),
        "missing_required_fields": 0, "duplicate_ids": 0, "unmatched_filing_links": 0,
        "one_target_redaction_per_question": True,
        "original_questions_answers_evidence_and_operands_preserved": True,
        "stored_calculations": {
            "nonzero_residual_rows": sum(value != 0 for value in residuals),
            "max_absolute_residual_usd_millions": float(max(residuals)),
            "source_tolerance": "max(1 USD million, 0.5% of absolute answer value)",
            "all_within_source_tolerance": True,
        },
        "anonymization": {
            "scope": "Author/lab identifiers and local paths; public issuer identities retained",
            "explicit_field_selection": True, "private_marker_scan": "PASS",
            "source_values_rewritten": False,
        },
    }


def load_source(root: Path) -> tuple[list[dict], list[dict]]:
    require(digest(root / SOURCE) == SOURCE_SHA256, "Question source differs from the frozen 600-row snapshot")
    require(digest(root / LINKS) == LINKS_SHA256, "Filing links differ from the frozen snapshot")
    return build_rows(read_jsonl(root / SOURCE), read_jsonl(root / LINKS))


def check_release(output: Path, source_root: Path) -> dict:
    import pyarrow.parquet as pq

    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    require(actual == PAYLOAD_FILES, "Release contains missing or unexpected files")
    require(not any(path.is_symlink() for path in output.rglob("*")), "Release must not contain symlinks")
    manifest = json.loads((output / "metadata/manifest.json").read_text())
    require(manifest["license"] == LICENSE, "Unexpected release license")
    require((manifest.get("dataset"), manifest.get("version"), manifest.get("split"), manifest.get("rows"))
            == ("XBRLFiling", "1.0", "test", 600), "Unexpected release identity or population")
    require(manifest.get("source_checksums") == {
        "questions_jsonl": SOURCE_SHA256, "filing_links_jsonl": LINKS_SHA256,
    }, "Source checksum provenance differs from the frozen snapshot")
    require(set(manifest["files"]) == PAYLOAD_FILES - {"metadata/manifest.json"}, "Incomplete checksum manifest")
    for relative, expected in manifest["files"].items():
        path = output / relative
        require(digest(path) == expected["sha256"] and path.stat().st_size == expected["bytes"],
                f"Release checksum mismatch: {relative}")
    rows = read_jsonl(output / "data/test.jsonl")
    require(pq.read_table(output / "data/test.parquet").to_pylist() == rows,
            "Parquet and JSONL contain different records")
    source_rows, source_filings = load_source(source_root)
    require(rows == source_rows, "Export differs from the frozen questions, answers or evidence")
    require(read_jsonl(output / "metadata/filings.jsonl") == source_filings, "Filing provenance differs")
    report = quality_report(rows, source_filings)
    require(json.loads((output / "metadata/quality_report.json").read_text()) == report, "Quality report differs")
    require((report["rows"], report["companies"], report["filings"], report["target_concepts"]) == (600, 28, 86, 15),
            "Dataset population changed")
    for relative in sorted(PAYLOAD_FILES - {"data/test.parquet"}):
        findings = privacy_findings((output / relative).read_text())
        require(not findings, f"Private metadata detected in {relative}: {findings}")
    return report


def prepare_release(source_root: Path, output: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists():
        # An existing validated export is reusable; unrelated files are never overwritten.
        return check_release(output, source_root)
    rows, filings = load_source(source_root)
    report = quality_report(rows, filings)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".xbrlfiling-", dir=output.parent))
    try:
        for name in ("README.md", "LICENSE.md", "NOTICE.md"):
            shutil.copyfile(TEMPLATES / name, staging / name)
        write_jsonl(staging / "data/test.jsonl", rows)
        table = pa.Table.from_pylist(rows).replace_schema_metadata(None)
        pq.write_table(table, staging / "data/test.parquet", compression="zstd")
        write_jsonl(staging / "metadata/filings.jsonl", filings)
        write_json(staging / "metadata/quality_report.json", report)
        manifest = {
            "dataset": "XBRLFiling", "version": "1.0", "license": LICENSE,
            "split": "test", "rows": len(rows),
            "source_checksums": {"questions_jsonl": SOURCE_SHA256, "filing_links_jsonl": LINKS_SHA256},
            "files": {relative: {"sha256": digest(staging / relative),
                                  "bytes": (staging / relative).stat().st_size}
                      for relative in sorted(PAYLOAD_FILES - {"metadata/manifest.json"})},
        }
        write_json(staging / "metadata/manifest.json", manifest)
        check_release(staging, source_root)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/xbrlfiling")
    parser.add_argument("--check-only", action="store_true", help="Validate an existing export without changing it")
    args = parser.parse_args()
    source_root, output = args.source_root.resolve(), args.output.resolve()
    if args.check_only:
        report = check_release(output, source_root)
    else:
        report = prepare_release(source_root, output)
    print(f"Validated {report['rows']} questions, {report['companies']} companies, {report['filings']} filings.")
    print(f"Release: {output}")
    print(f"License: {LICENSE}. No upload performed.")


if __name__ == "__main__":
    main()
