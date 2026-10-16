"""The public dataset preserves the frozen benchmark and excludes private metadata."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from scripts import prepare_xbrlfiling_hf as release


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source():
    return release.read_jsonl(ROOT / release.SOURCE), release.read_jsonl(ROOT / release.LINKS)


@pytest.fixture
def one_question(source):
    rows, links = source
    row = deepcopy(rows[0])
    link = next(deepcopy(item) for item in links if item["doc_name"] == row["doc_name"])
    return [row], [link]


def test_normalization_preserves_every_benchmark_value_and_public_issuer(source):
    originals, links = source
    rows, filings = release.build_rows(originals, links)
    assert len(rows) == 600 and len(filings) == 86
    for original, row in zip(originals, rows):
        truth = original["ground_truth"]
        assert row["id"] == original["financebench_id"]
        assert row["question"] == original["question"]
        assert row["answer"] == original["answer"] == truth["answer"]
        assert row["answer_value"] == truth["value"]
        assert row["answer_unit"] == truth["unit"]
        assert row["context"] == original["evidence"][0]["evidence_text"]
        assert row["operands"] == truth["operands"]
        assert row["calculation"] == truth["identity"]
        assert row["ticker"] == original["company"]
        assert row["cik"] == str(original["cik"]).zfill(10)
        assert row["accession"] == truth["provenance"]["accession"]
    assert "Apple" in rows[0]["question"] and rows[0]["ticker"] == "AAPL"
    report = release.quality_report(rows, filings)
    assert report["statement_family_counts"] == {"Income statement": 226, "Balance sheet": 279, "Cash flow": 95}
    assert report["stored_calculations"]["nonzero_residual_rows"] == 29
    assert report["stored_calculations"]["max_absolute_residual_usd_millions"] == 2.0


def test_unselected_author_and_workspace_metadata_are_omitted(one_question):
    rows, links = one_question
    expected = release.build_rows(rows, links)
    rows[0]["author"] = "researcher@example.org"
    rows[0]["ground_truth"]["local_path"] = "/home/researcher/work"
    rows[0]["ground_truth"]["provenance"]["lab"] = "review_lab"
    rows[0]["ground_truth"]["operands"][0]["author"] = "researcher@example.org"
    rows[0]["evidence"][0]["source_path"] = "/home/researcher/filing.xml"
    links[0]["author"] = "researcher@example.org"
    assert release.build_rows(rows, links) == expected


@pytest.mark.parametrize("private", [
    "/home/researcher/project", "file:///tmp/private.txt", r"C:\Users\researcher\data",
    "researcher@example.org", "review_lab", "hf_" + "a" * 30,
])
def test_private_text_in_exported_content_is_rejected(one_question, private):
    rows, links = one_question
    rows[0]["question"] += " " + private
    with pytest.raises(ValueError, match="Private metadata"):
        release.build_rows(rows, links)


def test_raw_windows_path_is_detected_without_json_escaping():
    assert "local_path" in release.privacy_findings(r"C:\Users\researcher\data")


@pytest.mark.parametrize("mutation, message", [
    ("duplicate_id", "Duplicate question identifiers"),
    ("duplicate_link", "Duplicate filing-link identifiers"),
    ("missing_link", "filing|Filing"),
    ("wrong_host", "SEC accession"),
    ("wrong_accession", "SEC accession"),
    ("no_redaction", "target redaction"),
    ("two_redactions", "target redaction"),
])
def test_invalid_release_inputs_are_rejected(one_question, mutation, message):
    rows, links = one_question
    if mutation == "duplicate_id":
        rows.append(deepcopy(rows[0]))
    elif mutation == "duplicate_link":
        links.append(deepcopy(links[0]))
    elif mutation == "missing_link":
        links.clear()
    elif mutation == "wrong_host":
        links[0]["doc_link"] = links[0]["doc_link"].replace("www.sec.gov", "example.org")
    elif mutation == "wrong_accession":
        accession = rows[0]["ground_truth"]["provenance"]["accession"].replace("-", "")
        links[0]["doc_link"] = links[0]["doc_link"].replace(accession, "000000000000000000")
    else:
        evidence = rows[0]["evidence"][0]
        context = evidence["evidence_text"].replace("[redacted]", "0") if mutation == "no_redaction" else evidence["evidence_text"] + " [redacted]"
        evidence["evidence_text"] = evidence["evidence_text_full_page"] = context
    with pytest.raises(ValueError, match=message):
        release.build_rows(rows, links)


@pytest.fixture(scope="module")
def production_export(tmp_path_factory):
    pytest.importorskip("pyarrow")
    output = tmp_path_factory.mktemp("xbrlfiling") / "release"
    release.prepare_release(ROOT, output)
    return output


def test_full_production_export_roundtrip_and_idempotence(production_export):
    output = production_export
    before = {str(path.relative_to(output)): release.digest(path) for path in output.rglob("*") if path.is_file()}
    report = release.check_release(output, ROOT)
    assert (report["rows"], report["companies"], report["filings"]) == (600, 28, 86)
    assert release.prepare_release(ROOT, output) == report
    assert before == {str(path.relative_to(output)): release.digest(path) for path in output.rglob("*") if path.is_file()}


@pytest.mark.parametrize("tamper", ["answer", "unexpected_file", "manifest_source_hash"])
def test_export_integrity_rejects_changes(production_export, tmp_path, tamper):
    output = tmp_path / "release"
    shutil.copytree(production_export, output)
    if tamper == "answer":
        rows = release.read_jsonl(output / "data/test.jsonl")
        rows[0]["answer"] = "$0 million"
        release.write_jsonl(output / "data/test.jsonl", rows)
    elif tamper == "unexpected_file":
        (output / "author_notes.txt").write_text("Private notes")
    else:
        path = output / "metadata/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["source_checksums"]["questions_jsonl"] = "0" * 64
        release.write_json(path, manifest)
    with pytest.raises(ValueError):
        release.check_release(output, ROOT)
