# Dataset release documentation

This directory stages the documentation for the two evaluation datasets used in
the VeriFin paper. It intentionally contains no dataset records, filing copies,
model outputs, credentials, or personal information. The release assembler
should place these documents beside the corresponding versioned data assets.

## Releases covered

| Release slug | Paper population | Origin | License status |
|---|---:|---|---|
| `xbrlbench-600/v1.0` | 600 questions | Derived from SEC EDGAR XBRL facts and filing calculation linkbases | Pending an explicit choice by the dataset authors |
| `financebench-67/v1.0` | 67 questions | Unmodified subset of the 150-example FinanceBench open-source sample | CC BY-NC 4.0 |

`XBRLBench`, `XBRLFiling`, and `XBRLBench-600` have all appeared in working
materials. Select one canonical name before release and use it consistently in
the paper, repository, archive metadata, and citation record.

## Recommended public layout

```text
datasets/
  xbrlbench-600/v1.0/
    README.md
    DATASET_CARD.md
    NOTICE.md
    LICENSE                 # add only after the authors choose one
    questions.jsonl
    documents.jsonl
    provenance.json
    CHECKSUMS.sha256
  financebench-67/v1.0/
    README.md
    DATASET_CARD.md
    NOTICE.md
    LICENSE-CC-BY-NC-4.0
    questions.jsonl
    documents.jsonl
    selection.json
    CHECKSUMS.sha256
```

Keep large, immutable inputs in versioned GitHub Release or DOI-backed archive
assets rather than normal Git history. Suggested assets are:

- `xbrlbench-xbrl.tar.gz`, `xbrlbench-companyfacts.tar.gz`, and
  `xbrlbench-corpus.tar.gz`;
- `financebench67-xbrl.tar.gz`, `financebench67-companyfacts.tar.gz`, and
  `financebench67-corpus.tar.gz`.

The release must publish SHA-256 checksums for every file and archive. Preserve
the repository-relative paths encoded in the current XBRL manifests, or rewrite
and test the manifests together with the archive layout.

## Release blockers

1. Choose a license for XBRLBench-600. This documentation does not grant one.
2. If the paper characterizes how FinanceBench-67 was sampled, recover and
   document that historical rule. `selection.json` already freezes the exact
   IDs and states that the rule is currently unknown.
3. Recover the XBRLBench source cutoff date. `documents.jsonl` pins all 86 SEC
   filing URLs/accessions, but the current builder's "latest six 10-Ks" rule
   will drift if run without the frozen inputs.
4. Generate any large reproduction archives and their final checksums; the payload digests
   in the dataset cards describe the audited source files, not as-yet-unbuilt
   archives.
5. Keep dataset licenses separate from the repository's software license.

The licensing notes are a release checklist, not legal advice.
