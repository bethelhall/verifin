from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from verifiqa.agent.agent import AgentConfig, VerificationAgent
from verifiqa.agent.types import AgentResult
from verifiqa.generation.answer_generator import AnswerGenerator
from verifiqa.generation.llm_client import make_llm_client
from verifiqa.types import EvidenceChunk
from verifiqa.verification.smt_generator import SmtGenerator
from verifiqa.verification.z3_runner import Z3Runner


SEC_SLEEP_SECONDS = 0.12


def _require_sec_user_agent(user_agent: str | None = None) -> str:
    configured = (user_agent or os.environ.get("SEC_USER_AGENT", "")).strip()
    if not configured:
        raise RuntimeError(
            "SEC_USER_AGENT is required for SEC EDGAR requests; set it to a "
            "descriptive value that includes contact information."
        )
    return configured


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def _default_repo_root() -> Path:
    source_root = Path(__file__).resolve().parents[3]
    if (source_root / "pyproject.toml").is_file() and (source_root / "src" / "verifiqa").is_dir():
        return source_root
    return Path.cwd()


@dataclass
class FilingRef:
    cik: str
    accession: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str
    fiscal_year: str
    ticker: str = ""

    @property
    def cik_int(self) -> int:
        return int(self.cik)

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def doc_name(self) -> str:
        prefix = _safe_doc_part(self.ticker.upper() if self.ticker else f"CIK{self.cik}")
        form = _safe_doc_part(self.form.replace("-", ""))
        year = _safe_doc_part(self.fiscal_year or self.report_date[:4] or self.filing_date[:4] or "filing")
        return f"{prefix}_{year}_{form}"


@dataclass
class DemoBackendConfig:
    repo_root: Path
    html_path: Path
    artifact_root: Path
    user_agent: str | None = None
    llm_mode: str = "claude"
    model: str = "claude-haiku-4-5-20251001"
    vllm_url: str = "http://localhost:8000/v1"
    max_tokens: int = 1024

    @property
    def xbrl_artifacts_dir(self) -> Path:
        return self.artifact_root / "xbrl_artifacts"

    @property
    def companyfacts_dir(self) -> Path:
        return self.artifact_root / "edgar_companyfacts"


class DemoBackend:
    def __init__(self, config: DemoBackendConfig):
        self.config = config
        self.config.xbrl_artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config.companyfacts_dir.mkdir(parents=True, exist_ok=True)

    def resolve_filing(self, payload: dict[str, Any], *, download: bool = True) -> dict[str, Any]:
        ref = self._resolve_filing_ref(payload)
        artifacts = self.download_artifacts(ref) if download else {"status": "not_downloaded"}
        companyfacts = self.download_companyfacts(ref.cik)
        return {
            "status": "ok",
            "filing": _filing_ref_json(ref),
            "doc_name": ref.doc_name,
            "xbrl_artifacts_dir": str(self.config.xbrl_artifacts_dir),
            "artifacts": artifacts,
            "companyfacts": companyfacts,
        }

    def ask_and_verify(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("missing question")

        filing_text = str(payload.get("filing_text") or "").strip()
        ref = self._resolve_filing_ref(payload)
        artifacts = self.download_artifacts(ref)
        companyfacts = self.download_companyfacts(ref.cik)

        if not filing_text:
            filing_text = self._download_primary_document_text(ref)
        if not filing_text:
            raise ValueError("missing filing_text and primary filing text could not be downloaded")

        # Keep the live demo responsive. The verifier still sees enough context for
        # the visible filing excerpt, while avoiding multi-megabyte prompt payloads.
        evidence_text = _compact_text(filing_text, limit=int(payload.get("evidence_char_limit") or 42000))
        llm_mode = str(payload.get("llm_mode") or self.config.llm_mode)
        model = str(payload.get("model") or self.config.model)
        llm = self._make_llm(llm_mode=llm_mode, model=model)

        answer = str(payload.get("answer") or "").strip()
        if not answer:
            answer = AnswerGenerator(llm).generate(
                question,
                [EvidenceChunk(f"{ref.doc_name}:uploaded", ref.doc_name, None, evidence_text)],
            )

        agent = VerificationAgent(
            llm_client=llm,
            smt_generator=SmtGenerator(llm),
            z3_runner=Z3Runner(),
            config=AgentConfig(
                max_workers=int(payload.get("workers") or 4),
                xbrl_artifacts_dir=self.config.xbrl_artifacts_dir,
                require_claimspec_consensus=bool(payload.get("claimspec_consensus") or False),
                require_smt_consensus=bool(payload.get("smt_consensus") or False),
            ),
        )
        result = agent.run(question, answer, evidence_text, doc_name=ref.doc_name)
        return {
            "status": "ok",
            "filing": _filing_ref_json(ref),
            "doc_name": ref.doc_name,
            "answer": answer,
            "verdict": _ui_verdict(result.status),
            "agent_status": result.status,
            "result": _agent_result_json(result),
            "xbrl_artifacts_dir": str(self.config.xbrl_artifacts_dir),
            "artifacts": artifacts,
            "companyfacts": companyfacts,
            "trace": _trace(result),
            "evidence": _evidence_lines(result, evidence_text),
            "concepts": _concepts(result),
        }

    def download_companyfacts(self, cik: str) -> dict[str, Any]:
        normalized = _normalize_cik(cik)
        path = self.config.companyfacts_dir / f"CIK{normalized}.json"
        if path.exists():
            return {"status": "cached", "path": str(path)}
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalized}.json"
        data = self._fetch_json(url)
        path.write_text(json.dumps(data), encoding="utf-8")
        return {"status": "saved", "path": str(path), "url": url}

    def download_artifacts(self, ref: FilingRef) -> dict[str, Any]:
        doc_dir = self.config.xbrl_artifacts_dir / ref.doc_name
        doc_dir.mkdir(parents=True, exist_ok=True)
        index = self._filing_index(ref)

        names = [
            str(item.get("name") or "")
            for item in index.get("directory", {}).get("item", [])
            if isinstance(item, dict)
        ]
        cal_files = _files_matching(names, ("_cal.xml", "-cal.xml"))
        label_files = _files_matching(names, ("_lab.xml", "-lab.xml"))
        instance_files = [
            name for name in _files_matching(names, (".xml",))
            if not _is_linkbase_or_summary(name)
        ]

        entry = {
            "doc_name": ref.doc_name,
            "cik": str(ref.cik_int),
            "accession": ref.accession,
            "form": ref.form,
            "filing_date": ref.filing_date,
            "report_date": ref.report_date,
            "primary_document": ref.primary_document,
            "instances": [],
            "calculation_linkbases": [],
            "label_linkbases": [],
        }
        if instance_files:
            entry["instances"].append(self._download_archive_file(ref, instance_files[0], doc_dir))
        if cal_files:
            entry["calculation_linkbases"].append(self._download_archive_file(ref, cal_files[0], doc_dir))
        if label_files:
            entry["label_linkbases"].append(self._download_archive_file(ref, label_files[0], doc_dir))
        entry["status"] = (
            "saved"
            if entry["instances"] and entry["calculation_linkbases"]
            and all(item.get("status") == "saved" for item in entry["instances"] + entry["calculation_linkbases"])
            else "partial"
        )
        self._update_manifest(ref.doc_name, entry)
        return {
            "status": entry["status"],
            "doc_name": ref.doc_name,
            "instance_count": len(entry["instances"]),
            "calculation_linkbase_count": len(entry["calculation_linkbases"]),
            "label_linkbase_count": len(entry["label_linkbases"]),
        }

    def _resolve_filing_ref(self, payload: dict[str, Any]) -> FilingRef:
        cik = _normalize_cik(str(payload.get("cik") or ""))
        if not cik:
            raise ValueError("missing cik")
        form_type = str(payload.get("form_type") or payload.get("form") or "10-K").upper().strip()
        fiscal_year = str(payload.get("filing_year") or payload.get("year") or "").strip()
        accession = str(payload.get("accession") or "").strip()
        ticker = str(payload.get("ticker") or "").strip().upper()

        submissions = self._submissions(cik)
        rows = self._submission_rows(submissions)
        if accession:
            for row in rows:
                if row.accession == accession:
                    row.ticker = ticker
                    row.fiscal_year = fiscal_year or row.report_date[:4] or row.filing_date[:4]
                    return row
            raise ValueError(f"accession not found for CIK: {accession}")

        candidates = [row for row in rows if _form_matches(row.form, form_type)]
        if fiscal_year:
            report_matches = [row for row in candidates if (row.report_date or "")[:4] == fiscal_year]
            candidates = report_matches or [
                row for row in candidates if _fallback_year_matches(row, fiscal_year, form_type)
            ]
        if not candidates:
            raise ValueError(f"no {form_type} filing found for CIK {cik} year {fiscal_year or 'latest'}")

        candidates.sort(key=lambda row: (row.report_date, row.filing_date, row.accession), reverse=True)
        ref = candidates[0]
        ref.ticker = ticker
        ref.fiscal_year = fiscal_year or ref.report_date[:4] or ref.filing_date[:4]
        return ref

    def _submissions(self, cik: str) -> dict[str, Any]:
        return self._fetch_json(f"https://data.sec.gov/submissions/CIK{cik}.json")

    def _submission_rows(self, submissions: dict[str, Any]) -> list[FilingRef]:
        out = _rows_from_submission_block(
            submissions.get("filings", {}).get("recent", {}),
            cik=_normalize_cik(str(submissions.get("cik") or "")),
        )
        for page in submissions.get("filings", {}).get("files", []) or []:
            name = page.get("name") if isinstance(page, dict) else ""
            if not name:
                continue
            try:
                time.sleep(SEC_SLEEP_SECONDS)
                data = self._fetch_json(f"https://data.sec.gov/submissions/{name}")
                out.extend(_rows_from_submission_block(data, cik=_normalize_cik(str(submissions.get("cik") or ""))))
            except Exception:
                continue
        return out

    def _filing_index(self, ref: FilingRef) -> dict[str, Any]:
        url = f"https://www.sec.gov/Archives/edgar/data/{ref.cik_int}/{ref.accession_nodash}/index.json"
        return self._fetch_json(url)

    def _download_archive_file(self, ref: FilingRef, filename: str, dest_dir: Path) -> dict[str, Any]:
        url = f"https://www.sec.gov/Archives/edgar/data/{ref.cik_int}/{ref.accession_nodash}/{filename}"
        path = dest_dir / filename
        if path.exists():
            return {"status": "saved", "path": str(path), "url": url, "bytes": path.stat().st_size}
        payload = self._fetch_bytes(url)
        path.write_bytes(payload)
        return {"status": "saved", "path": str(path), "url": url, "bytes": len(payload)}

    def _download_primary_document_text(self, ref: FilingRef) -> str:
        if not ref.primary_document:
            return ""
        url = f"https://www.sec.gov/Archives/edgar/data/{ref.cik_int}/{ref.accession_nodash}/{ref.primary_document}"
        try:
            raw = self._fetch_bytes(url).decode("utf-8", errors="replace")
        except Exception:
            return ""
        return _html_to_text(raw) if "<html" in raw[:2000].lower() or "<ix:" in raw[:2000].lower() else raw

    def _update_manifest(self, doc_name: str, entry: dict[str, Any]) -> None:
        path = self.config.xbrl_artifacts_dir / "manifest.json"
        if path.exists():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        else:
            manifest = {}
        docs = manifest.get("docs")
        if not isinstance(docs, dict):
            docs = {}
        docs[doc_name] = entry
        manifest["docs"] = docs
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _make_llm(self, *, llm_mode: str, model: str):
        config: dict[str, str] = {
            "model": model,
            "max_tokens": str(self.config.max_tokens),
        }
        if llm_mode == "vllm":
            config["base_url"] = self.config.vllm_url
        return make_llm_client(llm_mode, config)

    def _fetch_json(self, url: str) -> Any:
        return json.loads(self._fetch_bytes(url).decode("utf-8"))

    def _fetch_bytes(self, url: str, timeout: int = 45) -> bytes:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _require_sec_user_agent(self.config.user_agent)},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} fetching {url}: {body[:240]}") from exc


def serve(config: DemoBackendConfig, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    backend = DemoBackend(config)

    class Handler(BaseHTTPRequestHandler):
        server_version = "VerifiQADemo/0.1"

        def do_OPTIONS(self) -> None:
            self._send_json({"ok": True})

        def do_GET(self) -> None:
            if self.path in {"/", "/filingguard_demo.html"}:
                self._send_file(config.html_path, "text/html; charset=utf-8")
                return
            if self.path == "/api/health":
                self._send_json({"status": "ok"})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                if self.path == "/api/resolve-filing":
                    self._send_json(backend.resolve_filing(payload, download=True))
                    return
                if self.path == "/api/ask-and-verify":
                    self._send_json(backend.ask_and_verify(payload))
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:
                self._send_json(
                    {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                    status=HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            data = json.loads(raw or "{}")
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, f"Missing file: {path}")
                return
            payload = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._headers(content_type=content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self._headers(content_type="application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _headers(self, *, content_type: str) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Cache-Control", "no-store")

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"VerifiQA demo backend serving {config.html_path}")
    print(f"Open: http://{host}:{port}/")
    print(f"EDGAR cache: {config.artifact_root}")
    return httpd


def main(argv: list[str] | None = None) -> None:
    repo_root = _default_repo_root()
    _load_dotenv(repo_root / ".env")

    parser = argparse.ArgumentParser(description="Serve the VerifiQA interactive demo backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--html",
        type=Path,
        default=Path(__file__).resolve().parent / "static" / "index.html",
        help="HTML page to serve (default: bundled demo page)",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/demo_edgar"))
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT"),
        help="SEC EDGAR User-Agent (default: SEC_USER_AGENT environment variable)",
    )
    parser.add_argument("--llm-mode", choices=["claude", "anthropic", "vllm"], default=os.environ.get("VERIFIQA_LLM_MODE", "claude"))
    parser.add_argument("--model", default=os.environ.get("VERIFIQA_MODEL", "claude-haiku-4-5-20251001"))
    parser.add_argument("--vllm-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    args = parser.parse_args(argv)

    if not args.user_agent:
        parser.error(
            "SEC_USER_AGENT is required for SEC EDGAR requests; set it to a "
            "descriptive value that includes contact information"
        )
    config = DemoBackendConfig(
        repo_root=repo_root,
        html_path=(repo_root / args.html).resolve(),
        artifact_root=(repo_root / args.artifact_root).resolve(),
        user_agent=args.user_agent,
        llm_mode=args.llm_mode,
        model=args.model,
        vllm_url=args.vllm_url,
    )
    server = serve(config, host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo backend.")
    finally:
        server.server_close()


def _normalize_cik(cik: str) -> str:
    digits = re.sub(r"\D+", "", cik or "")
    if not digits:
        return ""
    return str(int(digits)).zfill(10)


def _safe_doc_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value or "").strip("_")
    return value or "DOC"


def _rows_from_submission_block(block: dict[str, Any], *, cik: str) -> list[FilingRef]:
    forms = block.get("form") or []
    accessions = block.get("accessionNumber") or []
    filing_dates = block.get("filingDate") or []
    report_dates = block.get("reportDate") or []
    primary_docs = block.get("primaryDocument") or []
    rows = []
    for i, accession in enumerate(accessions):
        if not accession:
            continue
        rows.append(
            FilingRef(
                cik=cik,
                accession=str(accession),
                form=str(forms[i] if i < len(forms) else ""),
                filing_date=str(filing_dates[i] if i < len(filing_dates) else ""),
                report_date=str(report_dates[i] if i < len(report_dates) else ""),
                primary_document=str(primary_docs[i] if i < len(primary_docs) else ""),
                fiscal_year=str(report_dates[i] if i < len(report_dates) else "")[:4],
            )
        )
    return rows


def _form_matches(actual: str, requested: str) -> bool:
    actual = (actual or "").upper()
    requested = (requested or "").upper()
    if actual == requested:
        return True
    if requested == "10-K":
        return actual in {"10-K", "10-K/A", "10-K405", "10-KT"}
    if requested == "10-Q":
        return actual in {"10-Q", "10-Q/A", "10-QT"}
    return False


def _fallback_year_matches(row: FilingRef, year: str, form_type: str) -> bool:
    report_year = row.report_date[:4] if row.report_date else ""
    filing_year = row.filing_date[:4] if row.filing_date else ""
    if report_year == year:
        return True
    if form_type.upper() == "10-K":
        try:
            return filing_year in {year, str(int(year) + 1)}
        except ValueError:
            return filing_year == year
    return filing_year == year


def _files_matching(names: list[str], suffixes: tuple[str, ...]) -> list[str]:
    return [name for name in names if any(name.lower().endswith(suffix) for suffix in suffixes)]


def _is_linkbase_or_summary(name: str) -> bool:
    lower = name.lower()
    return (
        lower == "filingsummary.xml"
        or lower.endswith(("_cal.xml", "_def.xml", "_lab.xml", "_pre.xml"))
        or lower.endswith(("-cal.xml", "-def.xml", "-lab.xml", "-pre.xml"))
    )


def _filing_ref_json(ref: FilingRef) -> dict[str, str]:
    return {
        "cik": ref.cik,
        "accession": ref.accession,
        "form": ref.form,
        "filing_date": ref.filing_date,
        "report_date": ref.report_date,
        "primary_document": ref.primary_document,
        "fiscal_year": ref.fiscal_year,
        "ticker": ref.ticker,
        "doc_name": ref.doc_name,
    }


def _agent_result_json(result: AgentResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "question": result.question,
        "answer": result.answer,
        "metric": result.metric,
        "formula": result.formula,
        "formula_source": result.formula_source,
        "claimed_value": result.claimed_value,
        "solver_status": result.solver_status,
        "failure_reason": result.failure_reason,
        "diagnostics": result.diagnostics or {},
        "facts": {
            name: {
                "value": fact.value,
                "unit": fact.unit,
                "period": fact.period,
                "row_label": fact.row_label,
                "source_quote": fact.source_quote,
                "fact_type": fact.fact_type,
            }
            for name, fact in (result.facts or {}).items()
        },
    }


def _ui_verdict(status: str) -> str:
    if status == "VERIFIED":
        return "VERIFIED"
    if status == "VIOLATED":
        return "VIOLATED"
    return "ABSTAINED"


def _trace(result: AgentResult) -> dict[str, str]:
    return {
        "formula_authority": result.formula_source or "unresolved",
        "formula": result.formula or "unresolved",
        "evidence_binding": ", ".join((result.facts or {}).keys()) or "none",
        "solver": result.solver_status or result.failure_reason or result.status,
    }


def _evidence_lines(result: AgentResult, evidence_text: str) -> list[str]:
    quotes = [
        fact.source_quote
        for fact in (result.facts or {}).values()
        if fact.source_quote
    ]
    if quotes:
        return quotes[:5]
    lines = [line.strip() for line in evidence_text.splitlines() if line.strip()]
    return lines[:3]


def _concepts(result: AgentResult) -> list[str]:
    concepts = []
    if result.metric:
        concepts.append(_humanize(result.metric))
    if result.formula_source:
        concepts.append(result.formula_source)
    if result.solver_status:
        concepts.append(f"solver_{result.solver_status.lower()}")
    if result.facts:
        concepts.append("evidence_binding")
    return concepts or [result.status.lower()]


def _humanize(value: str) -> str:
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    return value.strip().title()


def _compact_text(text: str, *, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.72)]
    tail = text[-int(limit * 0.18):]
    return f"{head}\n\n[... filing text truncated for demo prompt ...]\n\n{tail}"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return "\n".join(parser.parts)


if __name__ == "__main__":
    main()
