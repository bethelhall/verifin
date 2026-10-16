from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional


FINANCEBENCH_SAMPLE_FILES: Dict[str, str] = {
    "data/financebench_open_source.jsonl": (
        "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
    ),
    "data/financebench_document_information.jsonl": (
        "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_document_information.jsonl"
    ),
}


class DownloadError(RuntimeError):
    pass


def download_financebench_sample(
    out_dir: Path,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    for relative_path, url in FINANCEBENCH_SAMPLE_FILES.items():
        destination = out_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            if progress:
                progress(f"exists {destination}")
            downloaded[relative_path] = "exists"
            continue
        if progress:
            progress(f"download {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise DownloadError(f"Failed to download {url}: {exc}") from exc
        destination.write_bytes(content)
        downloaded[relative_path] = "downloaded"
    summary = summarize_sample(out_dir)
    (out_dir / "sample_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return downloaded


def summarize_sample(sample_dir: Path) -> Dict[str, object]:
    questions_path = sample_dir / "data" / "financebench_open_source.jsonl"
    docs_path = sample_dir / "data" / "financebench_document_information.jsonl"
    summary = {
        "questions_path": str(questions_path),
        "document_information_path": str(docs_path),
        "questions": _count_jsonl(questions_path),
        "documents": _count_jsonl(docs_path),
        "source": "https://github.com/patronus-ai/financebench",
    }
    return summary


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count
