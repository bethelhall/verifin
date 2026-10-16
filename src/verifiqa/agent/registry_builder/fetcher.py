"""Fetch source text for metric definitions.

Serves from local offline snapshots first (guaranteed, reproducible).
Falls back to live HTTP fetch if no local file is provided.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_SOURCE_TEXTS_DIR = Path(__file__).parent / "source_texts"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; verifiqa-registry-builder/1.0)",
    "Accept": "text/html,application/xhtml+xml",
}

_TIMEOUT = 15.0


def fetch_text(url: str, local_file: Optional[str] = None) -> Optional[str]:
    """Return source text for a metric definition page.

    Checks local_file in source_texts/ first. Falls back to live HTTP fetch.
    Returns None if both fail.
    """
    if local_file:
        path = _SOURCE_TEXTS_DIR / local_file
        if path.exists():
            return path.read_text(encoding="utf-8")

    return _fetch_live(url)


def _fetch_live(url: str) -> Optional[str]:
    """Fetch URL and return cleaned plain text. Returns None on failure."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        return _strip_html(html)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None


def _strip_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    html = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text[:6000]
