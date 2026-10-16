"""Build a multi-company filing-provable dataset (scales the Apple case study).

Company-agnostic. For each CIK it:
  1. fetches SEC companyfacts (authoritative XBRL values)
  2. resolves the most recent N 10-K filings (accession + primary doc)
  3. downloads each filing's calculation linkbase (_cal.xml) + XBRL instance (_htm.xml)
  4. parses the calc linkbase for additive identities (parent = sum of weighted children)
  5. keeps only fully-closed identities where parent + every operand resolve in
     companyfacts and the identity holds (residual ~ 0)
  6. emits a natural metric question per (identity, fiscal year), in two variants:
       - lookup      : evidence shows the full statement incl. the answer line
       - calcrequired: the answer line is redacted -> must be computed from components
  7. writes a combined XBRL manifest (instance + calc linkbase per doc) so the agent
     can run with calc-linkbase constraints.

Everything stays XBRL-supported: only standard us-gaap subtotal concepts, only
additive calc-linkbase identities. Ratios / YoY / averages are excluded (not
provable from the filing's own structure). Nothing is run through the agent here.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from verifiqa.verification.xbrl_linkbase import _parse_calculation_linkbase

OUT_DIR = ROOT / "data" / "multicompany_provable"
ART_DIR = OUT_DIR / "xbrl_artifacts"
EDGAR_DIR = OUT_DIR / "edgar_companyfacts"
CACHE = OUT_DIR / "_cache"


def _require_sec_user_agent() -> str:
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise RuntimeError(
            "SEC_USER_AGENT is required for SEC EDGAR requests; set it to a "
            "descriptive value that includes contact information."
        )
    return user_agent

# ticker -> CIK (diverse sectors + fiscal-year ends)
COMPANIES = {
    "AAPL": 320193, "MSFT": 789019, "GOOGL": 1652044, "AMZN": 1018724,
    "NVDA": 1045810, "META": 1326801, "WMT": 104169, "JNJ": 200406,
    "KO": 21344, "PG": 80424, "INTC": 50863, "CSCO": 858877,
    "ORCL": 1341439, "ADBE": 796343, "CRM": 1108524, "PEP": 77476,
    "NKE": 320187, "HD": 354950, "XOM": 34088, "CVX": 93410,
    "PFE": 78003, "MRK": 310158, "TXN": 97476, "QCOM": 804328,
    "COST": 909832, "IBM": 51143, "MCD": 63908, "TMO": 97745,
}
N_YEARS = 6  # most recent 10-K filings per company
TARGET_N = 600  # trim to exactly this many questions (balanced across companies)
DISPLAY_NAMES = {
    "QCOM": "Qualcomm",
    "COST": "Costco Wholesale",
}

# Standard us-gaap subtotal concepts worth asking, and which statement they live on.
PARENTS = {
    "GrossProfit": ("gross profit", "operations"),
    "OperatingExpenses": ("total operating expenses", "operations"),
    "OperatingIncomeLoss": ("operating income", "operations"),
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
        ("income before income taxes", "operations"),
    "NetIncomeLoss": ("net income", "operations"),
    "Assets": ("total assets", "balance"),
    "AssetsCurrent": ("total current assets", "balance"),
    "AssetsNoncurrent": ("total non-current assets", "balance"),
    "Liabilities": ("total liabilities", "balance"),
    "LiabilitiesCurrent": ("total current liabilities", "balance"),
    "LiabilitiesNoncurrent": ("total non-current liabilities", "balance"),
    "StockholdersEquity": ("total shareholders' equity", "balance"),
    "PropertyPlantAndEquipmentNet": ("net property, plant and equipment", "balance"),
    "NetCashProvidedByUsedInOperatingActivities": ("net cash from operating activities", "cashflow"),
    "NetCashProvidedByUsedInInvestingActivities": ("net cash from investing activities", "cashflow"),
    "NetCashProvidedByUsedInFinancingActivities": ("net cash from financing activities", "cashflow"),
}
STMT_TITLE = {
    "operations": "CONSOLIDATED STATEMENTS OF OPERATIONS",
    "balance": "CONSOLIDATED BALANCE SHEETS",
    "cashflow": "CONSOLIDATED STATEMENTS OF CASH FLOWS",
}


_NAME_DROP = {"inc", "corp", "corporation", "company", "co", "ltd", "plc", "com",
              "holdings", "group", "the", "lp", "llc", "sa", "ag", "nv", "class"}


def clean_company_name(entity: str) -> str:
    """Recognizable company name for the question (ticker possessives like 'AAPL's'
    break the classifier's metric recognition; 'Apple's' works)."""
    s = re.sub(r"/[A-Z]{2,}/?", " ", entity, flags=re.I)  # drop /DE/ etc.
    s = s.replace("/", " ")
    toks = [t for t in re.split(r"[\s,]+", s.title()) if t and t.strip(".").lower() not in _NAME_DROP]
    return " ".join(toks).strip() or entity


def fetch(url: str, dest: Path, *, is_json=False):
    """Download with on-disk cache; returns text (or parsed JSON)."""
    if dest.exists() and dest.stat().st_size > 0:
        data = dest.read_text(encoding="utf-8", errors="replace")
        return json.loads(data) if is_json else data
    req = urllib.request.Request(
        url, headers={"User-Agent": _require_sec_user_agent()}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    time.sleep(0.2)
    text = raw.decode("utf-8", errors="replace")
    return json.loads(text) if is_json else text


def companyfacts(cik: int) -> dict:
    padded = f"CIK{cik:010d}"
    data = fetch(f"https://data.sec.gov/api/xbrl/companyfacts/{padded}.json",
                 EDGAR_DIR / f"{padded}.json", is_json=True)
    return data


def recent_10ks(cik: int, n: int) -> list[dict]:
    data = fetch(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                 CACHE / f"submissions_{cik}.json", is_json=True)
    rec = data["filings"]["recent"]
    rows = []
    for form, acc, doc, rpt in zip(rec["form"], rec["accessionNumber"],
                                   rec["primaryDocument"], rec["reportDate"]):
        if form == "10-K":
            rows.append({"accession": acc, "primary": doc, "report_date": rpt,
                         "fy": int(rpt[:4])})
        if len(rows) >= n:
            break
    return rows


def stem_from_primary(primary: str) -> str:
    return primary[:-4] if primary.endswith(".htm") else primary


def fetch_artifacts(cik: int, filing: dict, doc_name: str) -> tuple[Path | None, Path | None]:
    """Download calc linkbase + XBRL instance for a filing, resolving the real
    filenames from the filing index (they don't always follow the primary-doc stem).
    Returns (cal, instance)."""
    acc_nodash = filing["accession"].replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}"
    d = ART_DIR / doc_name
    try:
        index = fetch(f"{base}/index.json", CACHE / f"index_{acc_nodash}.json", is_json=True)
        names = [it["name"] for it in index["directory"]["item"]]
    except Exception:
        return None, None
    cal_name = next((x for x in names if x.endswith("_cal.xml")), None)
    # instance = extracted inline (_htm.xml); else the main XBRL instance .xml
    inst_name = next((x for x in names if x.endswith("_htm.xml")), None)
    if inst_name is None:
        cands = [x for x in names if x.endswith(".xml") and not x.endswith(
            ("_cal.xml", "_pre.xml", "_def.xml", "_lab.xml"))]
        inst_name = cands[0] if cands else None
    cal = inst = None
    if cal_name:
        try:
            fetch(f"{base}/{cal_name}", d / cal_name)
            if (d / cal_name).stat().st_size > 2000:
                cal = d / cal_name
        except Exception:
            pass
    if inst_name:
        try:
            fetch(f"{base}/{inst_name}", d / inst_name)
            if (d / inst_name).stat().st_size > 50000:
                inst = d / inst_name
        except Exception:
            pass
    return cal, inst


def build_value_index(cf: dict):
    """concept -> list of {accn, fy, end, val} for annual 10-K facts, plus labels.

    Resolving by accession gives the value *as reported in a specific filing* and
    that filing's own fiscal-year tag — avoiding restatement duplicates and the
    fiscal-year-boundary mislabel (e.g. a fiscal year ending Jan 1 belongs to the
    prior calendar year).
    """
    idx: dict[str, list] = {}
    labels = {}
    usg = cf.get("facts", {}).get("us-gaap", {})
    for concept, cdata in usg.items():
        labels[concept] = cdata.get("label") or concept
        for unit, items in cdata.get("units", {}).items():
            if unit != "USD":
                continue
            for it in items:
                if it.get("fp") == "FY" and str(it.get("form", "")).startswith("10-K") and it.get("end"):
                    idx.setdefault(concept, []).append(
                        {"accn": it.get("accn"), "fy": it.get("fy"), "start": it.get("start"),
                         "end": it["end"], "filed": it.get("filed"), "val": it["val"]})
    return idx, labels


def fact_for_filing_period(idx: dict, concept: str, accession: str, report_date: str):
    """Target-period fact as reported in the given filing.

    A 10-K accession includes comparative-period facts. The dataset question asks
    about the filing's fiscal year, so the selected fact must match both the
    accession and the filing report date.
    """
    matches = [
        e for e in idx.get(concept, ())
        if e["accn"] == accession and e["end"] == report_date
    ]
    if not matches:
        return None
    # Companyfacts can occasionally repeat the same fact. Prefer the most
    # specific annual duration, then fall back to the first target-period value.
    return max(matches, key=lambda e: (bool(e.get("start")), e.get("filed") or ""))


def val_for_filing_period(idx: dict, concept: str, accession: str, report_date: str):
    fact = fact_for_filing_period(idx, concept, accession, report_date)
    if not fact:
        return None, None, None
    return fact["val"], fact["fy"], fact["end"]


def filing_fy(idx: dict, accession: str, report_date: str):
    """The companyfacts fiscal-year tag for a filing target period, else None."""
    for entries in idx.values():
        for e in entries:
            if e["accn"] == accession and e["end"] == report_date and e.get("fy"):
                return e["fy"]
    return None


def val_at_end(idx: dict, concept: str, end: str):
    """Modal FY/10-K value at a period end (for evidence columns / cross-year)."""
    vals = [e["val"] for e in idx.get(concept, ()) if e["end"] == end]
    if not vals:
        return None
    # modal value (most filings agree) — robust to restatements/anomalies
    return max(set(vals), key=vals.count)


def collect_identities(cal_path: Path, idx: dict, accession: str, report_date: str) -> dict:
    """Fully-closed identities as reported for this filing's target period."""
    groups, _ = _parse_calculation_linkbase(cal_path)
    out = {}
    for g in groups:
        p = g.parent_local
        if p not in PARENTS or p in out:
            continue
        pv, fy, end = val_for_filing_period(idx, p, accession, report_date)
        if pv is None:
            continue
        operands = []
        ok = True
        for c in g.children:
            v, _, child_end = val_for_filing_period(idx, c.local_name, accession, report_date)
            if v is None or child_end != end:
                ok = False
                break
            operands.append((c.local_name, float(c.weight), v))
        if not ok or len(operands) < 2:
            continue
        if abs(pv - sum(w * v for _, w, v in operands)) > max(1e6, abs(pv) * 0.005):
            continue
        out[p] = (pv, operands, g.role, fy, end)
    return out


def role_concepts(cal_path: Path, role: str) -> list[str]:
    """All concepts appearing in the given calc-linkbase role (statement), in order."""
    groups, _ = _parse_calculation_linkbase(cal_path)
    concepts = []
    for g in groups:
        if g.role != role:
            continue
        for c in g.children:
            if c.local_name not in concepts:
                concepts.append(c.local_name)
        if g.parent_local not in concepts:
            concepts.append(g.parent_local)
    return concepts


def fmt(v_millions: float) -> str:
    s = f"{abs(v_millions):,.0f}"
    return f"({s})" if v_millions < 0 else s


def answer_fmt(v_millions: float) -> str:
    s = f"{abs(v_millions):,.0f}"
    return f"-${s} million" if v_millions < 0 else f"${s} million"


def fiscal_year_of(end_date: str) -> int:
    """Fiscal-year label from a period-end date — deterministic and collision-free.

    companyfacts' own ``fy`` tag is unreliable for some January-fiscal-year filers
    (e.g. Salesforce tags its Jan-2026 period fy=2025), so we derive it from the date:
      - ends in early January (day <= 7): a 52/53-week year ending near Dec 31 —
        belongs to the prior year (JNJ 2023-01-01 -> FY2022)
      - otherwise: named for the end year (Salesforce 2026-01-31 -> FY2026,
        NVDA 2025-01-26 -> FY2025, Apple 2025-09-27 -> FY2025)
    Used for BOTH the question label and the evidence column headers so they always agree.
    """
    y, m, d = int(end_date[:4]), int(end_date[5:7]), int(end_date[8:10])
    return y - 1 if (m == 1 and d <= 7) else y


def render_statement(concepts: list[str], vidx: dict, labels: dict, ends: list[str],
                     stmt_key: str, redact_cell: tuple[str, str] | None = None) -> str:
    header = f"{STMT_TITLE[stmt_key]}\n(In millions)\n\n{'':52}" + "   ".join(str(fiscal_year_of(e)) for e in ends)
    lines = [header]
    for concept in concepts:
        cells = []
        shown = False
        for e in ends:
            v = val_at_end(vidx, concept, e)
            if v is None:
                cells.append(f"{'—':>10}")
                continue
            m = v / 1e6
            if redact_cell == (concept, e):
                cells.append(f"{'[redacted]':>10}")
            else:
                cells.append(f"{fmt(m):>10}")
                shown = True
        # keep row if it has any real value or is the (redacted) target row
        lines.append(f"{labels.get(concept, concept)[:52]:52}  " + "   ".join(cells))
    return "\n".join(lines)


def validate_selected_rows(rows: list[dict], submissions_report_dates: dict[str, str]) -> None:
    errors = []
    ids = set()
    for row in rows:
        rid = row["financebench_id"]
        if rid in ids:
            errors.append(f"{rid}: duplicate id")
        ids.add(rid)

        gt = row["ground_truth"]
        prov = gt["provenance"]
        accession = prov["accession"]
        report_date = submissions_report_dates.get(accession)
        if not report_date:
            errors.append(f"{rid}: accession {accession} missing from submissions cache")
            continue
        if prov["period_end"] != report_date:
            errors.append(
                f"{rid}: period_end {prov['period_end']} != SEC reportDate {report_date}"
            )

        calc = sum(op["weight"] * op["value_millions"] for op in gt["operands"])
        if abs(calc - gt["value"]) > max(1.0, abs(gt["value"]) * 0.005):
            errors.append(f"{rid}: operands sum {calc} != answer {gt['value']}")

        ev = row["evidence"][0]["evidence_text"]
        target_shown = fmt(gt["value"]) in ev
        target_redactions = ev.count("[redacted]")
        if rid.startswith("mc_calc_"):
            if target_redactions != 1:
                errors.append(f"{rid}: expected exactly one redaction, found {target_redactions}")
            # The calc-required variant should not leak the target display value
            # anywhere in the rendered evidence.
            if target_shown:
                errors.append(f"{rid}: calc-required evidence still contains target numeric value")
        else:
            if not target_shown:
                errors.append(f"{rid}: lookup evidence does not contain target numeric value")

    if errors:
        preview = "\n".join(errors[:25])
        more = "" if len(errors) <= 25 else f"\n... {len(errors) - 25} more"
        raise RuntimeError(f"Generated dataset failed validation:\n{preview}{more}")


def main():
    _require_sec_user_agent()
    only = os.environ.get("ONLY_TICKERS")
    if only:
        global COMPANIES
        COMPANIES = {t: COMPANIES[t] for t in only.split(",") if t in COMPANIES}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = []
    manifest_docs = {}
    per_company = {}

    for ticker, cik in COMPANIES.items():
        try:
            cf = companyfacts(cik)
        except Exception as exc:
            print(f"[{ticker}] companyfacts failed: {exc}")
            continue
        name = DISPLAY_NAMES.get(ticker) or clean_company_name(cf.get("entityName") or ticker)
        vidx, labels = build_value_index(cf)
        filings = recent_10ks(cik, N_YEARS)
        count = 0
        # true period ends (from a universally-present concept), for evidence columns
        all_ends = sorted({e["end"] for e in vidx.get("Assets", [])}, reverse=True)
        seen_fy = set()
        for filing in filings:
            fy = fiscal_year_of(filing["report_date"])
            if fy in seen_fy:   # dedupe fiscal years (year-boundary collisions)
                continue
            seen_fy.add(fy)
            doc_name = f"{ticker}_{fy}_10K"
            cal, inst = fetch_artifacts(cik, filing, doc_name)
            if not cal or not inst:
                continue
            manifest_docs[doc_name] = {
                "doc_name": doc_name, "cik": str(cik),
                "instances": [{"status": "saved", "path": str(inst.relative_to(ROOT))}],
                "calculation_linkbases": [{"status": "saved", "path": str(cal.relative_to(ROOT))}],
            }
            idents = collect_identities(cal, vidx, filing["accession"], filing["report_date"])
            for parent, (pv, operands, role, ify, end) in idents.items():
                if end != filing["report_date"]:
                    continue
                label, stmt = PARENTS[parent]
                pm = round(pv / 1e6, 3)
                terms = " ".join(f"{'+' if w > 0 else '-'}{c}" for c, w, _ in operands)
                # evidence columns: this year + up to 2 priors that exist
                ends = [e for e in all_ends if e <= end][:3]
                if end not in ends:
                    ends = [end] + ends[:2]
                concepts = role_concepts(cal, role)
                if parent not in concepts:
                    concepts = [parent] + [c for c, _, _ in operands]
                ev_look = render_statement(concepts, vidx, labels, ends, stmt)
                ev_calc = render_statement(concepts, vidx, labels, ends, stmt, redact_cell=(parent, end))
                # guard: answer must be groundable in lookup, absent in calc
                if fmt(pm) not in ev_look or ev_calc.count("[redacted]") != 1 or fmt(pm) in ev_calc:
                    continue
                count += 1
                header = (f"{ticker} (CIK {cik}) — excerpt from Form 10-K ({doc_name}), "
                          f"accession {filing['accession']}. All amounts in millions of USD.\n\n")
                base.append({
                    "company": ticker, "cik": cik, "doc_name": doc_name,
                    "dataset_subset_label": "MULTICOMPANY_FILING_PROVABLE",
                    "question_type": "metrics-generated",
                    "question": f"What was {name}'s FY{fy} {label}, in USD millions? Round to the nearest million.",
                    "answer": answer_fmt(pm),
                    "ev_look": header + ev_look, "ev_calc": header + ev_calc,
                    "ground_truth": {
                        "answer": answer_fmt(pm), "value": pm, "unit": "USD millions",
                        "proof_source": "xbrl_calculation_linkbase", "concept": parent,
                        "identity": f"{parent} = {terms}",
                        "operands": [{"concept": c, "weight": w, "value_millions": round(v / 1e6, 3)}
                                     for c, w, v in operands],
                        "provenance": {"ticker": ticker, "cik": cik, "filing": doc_name,
                                       "accession": filing["accession"],
                                       "fiscal_year": fy, "period_end": end,
                                       "report_date": filing["report_date"]},
                    },
                })
        per_company[ticker] = count
        print(f"[{ticker}] {count} questions")

    # Trim to exactly TARGET_N, balanced across companies (round-robin) so no single
    # filer dominates. Deterministic: companies in config order, questions in gen order.
    by_co: dict[str, list] = {}
    for b in base:
        by_co.setdefault(b["company"], []).append(b)
    selected = []
    if len(base) <= TARGET_N:
        selected = base
        print(f"\nWARNING: only {len(base)} questions available (< target {TARGET_N})")
    else:
        i = 0
        while len(selected) < TARGET_N:
            progressed = False
            for co in COMPANIES:
                bucket = by_co.get(co, [])
                if i < len(bucket):
                    selected.append(bucket[i]); progressed = True
                    if len(selected) >= TARGET_N:
                        break
            if not progressed:
                break
            i += 1

    lookup_rows, calc_rows = [], []
    for k, b in enumerate(selected, 1):
        common = {key: b[key] for key in ("company", "cik", "doc_name",
                  "dataset_subset_label", "question_type", "question", "answer")}
        lookup_rows.append({**common, "financebench_id": f"mc_{k:05d}",
            "question_reasoning": "Filing-provable lookup (subtotal shown)",
            "evidence": [{"doc_name": b["doc_name"], "evidence_page_num": None,
                "evidence_text": b["ev_look"], "evidence_text_full_page": b["ev_look"],
                "source": "Rendered statement from authoritative XBRL (calc-linkbase role)"}],
            "ground_truth": b["ground_truth"]})
        calc_rows.append({**common, "financebench_id": f"mc_calc_{k:05d}",
            "question_reasoning": "Filing-provable, calculation required (subtotal redacted)",
            "evidence": [{"doc_name": b["doc_name"], "evidence_page_num": None,
                "evidence_text": b["ev_calc"], "evidence_text_full_page": b["ev_calc"],
                "source": "Rendered statement with target subtotal redacted"}],
            "ground_truth": b["ground_truth"]})

    submissions_report_dates = {
        filing["accession"]: filing["report_date"]
        for cik in COMPANIES.values()
        for filing in recent_10ks(cik, N_YEARS)
    }
    validate_selected_rows(lookup_rows, submissions_report_dates)
    validate_selected_rows(calc_rows, submissions_report_dates)

    (OUT_DIR / "data").mkdir(exist_ok=True)
    look_p = OUT_DIR / "data" / "provable_lookup.jsonl"
    calc_p = OUT_DIR / "data" / "provable_calcrequired.jsonl"
    look_p.write_text("\n".join(json.dumps(r) for r in lookup_rows) + "\n")
    calc_p.write_text("\n".join(json.dumps(r) for r in calc_rows) + "\n")
    ART_DIR.mkdir(parents=True, exist_ok=True)
    (ART_DIR / "manifest.json").write_text(json.dumps({"docs": manifest_docs}, indent=2))

    from collections import Counter
    print(f"\nGenerated {len(base)} -> selected {len(lookup_rows)} (target {TARGET_N})")
    print(f"companies in final: {dict(Counter(r['company'] for r in lookup_rows))}")
    print(f"docs wired: {len(manifest_docs)}")
    print(f"lookup:  {look_p}")
    print(f"calc:    {calc_p}")


if __name__ == "__main__":
    main()
