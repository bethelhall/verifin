from __future__ import annotations

import json
from pathlib import Path

from verifiqa.eval.metrics import summarize_results
from verifiqa.types import RavResult


def write_report(results, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path

