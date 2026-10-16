from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


@dataclass
class TaxonomySource:
    source_id: str
    standard: str
    publisher: str
    version: str
    prefixes: tuple[str, ...]
    package_url: str


def load_taxonomy_concepts(
    sources: list[TaxonomySource],
    *,
    cache_dir: Path,
    fetch: bool = True,
    timeout: int = 30,
) -> set[str]:
    """Return the set of prefixed XBRL concept qnames from taxonomy ZIP packages."""
    concepts: set[str] = set()
    for source in sources:
        zip_path = cache_dir / f"{source.source_id}.zip"
        if not zip_path.exists():
            if fetch:
                _fetch_zip(source.package_url, zip_path, timeout=timeout)
            else:
                continue
        concepts.update(_concepts_from_zip(zip_path, source.prefixes))
    return concepts


def build_policy_source_report(
    *,
    data_dir: Path,
    cache_dir: Path,
    fetch: bool = True,
    timeout: int = 30,
) -> dict[str, Any]:
    """Verify that every xbrl_concept in semantic_concepts.json exists in a known taxonomy."""
    sources_path = data_dir / "taxonomy_sources.json"
    concepts_path = data_dir / "semantic_concepts.json"

    sources: list[TaxonomySource] = []
    if sources_path.exists():
        for entry in json.loads(sources_path.read_text(encoding="utf-8")):
            sources.append(TaxonomySource(
                source_id=entry["source_id"],
                standard=entry["standard"],
                publisher=entry["publisher"],
                version=entry["version"],
                prefixes=tuple(entry.get("prefixes", [])),
                package_url=entry["package_url"],
            ))

    taxonomy_concepts = load_taxonomy_concepts(sources, cache_dir=cache_dir, fetch=fetch, timeout=timeout)

    seeded_qnames: list[str] = []
    if concepts_path.exists():
        for entry in json.loads(concepts_path.read_text(encoding="utf-8")):
            seeded_qnames.extend(entry.get("xbrl_concepts", []))

    missing = [q for q in seeded_qnames if q not in taxonomy_concepts]

    return {
        "status": "ok",
        "xbrl_concepts": len(seeded_qnames),
        "verified_xbrl_concepts": len(seeded_qnames) - len(missing),
        "missing_xbrl_concepts": missing,
    }


def write_policy_source_report(
    *,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    out: Path | None = None,
    fetch: bool = True,
    timeout: int = 30,
) -> dict[str, Any]:
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "data" / "taxonomy_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)
    report = build_policy_source_report(
        data_dir=data_dir, cache_dir=cache_dir, fetch=fetch, timeout=timeout
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _concepts_from_zip(zip_path: Path, prefixes: tuple[str, ...]) -> set[str]:
    XS = "http://www.w3.org/2001/XMLSchema"
    concepts: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".xsd"):
                continue
            try:
                data = zf.read(name)
                root = ElementTree.fromstring(data)
                for el in root.iter(f"{{{XS}}}element"):
                    elem_name = el.get("name")
                    if elem_name:
                        for prefix in prefixes:
                            concepts.add(f"{prefix}:{elem_name}")
            except ElementTree.ParseError:
                continue
    return concepts


def _fetch_zip(url: str, dest: Path, *, timeout: int) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
