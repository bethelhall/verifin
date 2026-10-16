# Publish XBRLFiling to Hugging Face

The prepared dataset is `verifin/datasets/xbrlfiling/`. Upload only this folder.
It contains 600 questions, embedded redacted evidence, reference answers,
calculation annotations, 86 SEC filing links, a dataset card, checksums, and
the selected **CC BY-NC 4.0** license.

The release excludes author/lab identifiers and local paths. Public company
names, tickers, CIKs, accession numbers, and SEC URLs remain. For anonymous
review, use an anonymous Hugging Face account and namespace: the repository
owner and uploader are account metadata, separate from the dataset files.

## Upload

Run from the project root. Replace `YOUR_NAMESPACE` with the intended account
or organization. The upload command creates a public dataset repository if it
does not already exist; it does not change an existing repository's visibility.

```bash
cd /path/to/verifin-review

# Verify every release file and its relationship to the frozen source.
.venv/bin/python verifin/scripts/prepare_xbrlfiling_hf.py --check-only

# Authenticate interactively; do not paste tokens into shell commands.
.venv/bin/python .venv/bin/hf auth login

# Upload the complete prepared dataset, with a neutral commit message.
.venv/bin/python .venv/bin/hf upload \
  YOUR_NAMESPACE/XBRLFiling \
  verifin/datasets/xbrlfiling . \
  --repo-type dataset --no-private \
  --commit-message "Release XBRLFiling v1.0"
```

The existing `hf` executable has a stale interpreter path. Invoking it with
the working `.venv/bin/python` avoids that issue without reinstalling packages.
On another machine with a normal installation, use `hf auth login` and
`hf upload ...` directly. Install the validated dependencies with
`python -m pip install -r requirements-dataset.txt`. See the official
[Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

The resulting repository URL is
`https://huggingface.co/datasets/YOUR_NAMESPACE/XBRLFiling`.

## Rebuild or validate offline

```bash
# Build a fresh release at the default location, or validate it if it exists.
.venv/bin/python verifin/scripts/prepare_xbrlfiling_hf.py

# Build a separate copy without overwriting the prepared release.
.venv/bin/python verifin/scripts/prepare_xbrlfiling_hf.py \
  --output /tmp/xbrlfiling-release
```

The exporter uses explicit fields, preserves the original question/answer/
evidence/operand values, and checks the two frozen source hashes. It verifies
unique IDs and questions, complete SEC accession links, one target redaction
per question, calculation tolerances, JSONL/Parquet equivalence, all release
checksums, and private-marker scans. Existing unrelated output files cause a
validation failure instead of being overwritten or silently uploaded.

The single `test` split is the existing 600-question benchmark. Raw XML,
companyfacts snapshots, retrieval corpora, model outputs, and weights are not
part of this compact provided-evidence release. Its dataset card describes
the additional artifacts needed for full verifier/retrieval reproduction.
