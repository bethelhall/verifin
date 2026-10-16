# Dataset card: XBRLBench-600 v1.0

## Summary

XBRLBench-600, released as **XBRLFiling**, evaluates numerical answer generation and verification over
issuer-filed financial statements. It contains 600 calculation-required
questions derived from 86 Form 10-K filings from 28 public companies. For each
question, a filing-declared additive XBRL calculation relationship is selected,
the target subtotal is removed from a rendered statement, and the remaining
line items are supplied as evidence.

The intended grain is one question per filing, fiscal period, and requested
subtotal concept. Question IDs are the primary key.

## Composition

| Statement family | Metrics | Questions |
|---|---:|---:|
| Income statement | 5 | 226 |
| Balance sheet | 7 | 279 |
| Cash flow statement | 3 | 95 |
| **Total** | **15** | **600** |

The 15 concepts and row counts are:

- income statement: `GrossProfit` (35),
  `IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest`
  (60), `NetIncomeLoss` (67), `OperatingExpenses` (10), and
  `OperatingIncomeLoss` (54);
- balance sheet: `AssetsCurrent` (64), `AssetsNoncurrent` (4), `Liabilities`
  (43), `LiabilitiesCurrent` (60), `LiabilitiesNoncurrent` (4),
  `PropertyPlantAndEquipmentNet` (44), and `StockholdersEquity` (60);
- cash flow: `NetCashProvidedByUsedInFinancingActivities` (43),
  `NetCashProvidedByUsedInInvestingActivities` (24), and
  `NetCashProvidedByUsedInOperatingActivities` (28).

The represented fiscal-year labels span 2021 through 2026. They describe the
frozen v1.0 snapshot and should not be recomputed from a later EDGAR state.

## Record schema

Top-level fields:

| Field | Description |
|---|---|
| `financebench_id` | Unique identifier of the form `mc_calc_00001` through `mc_calc_00600` |
| `company`, `cik`, `doc_name` | Issuer and filing identifiers |
| `dataset_subset_label`, `question_type`, `question_reasoning` | Dataset metadata |
| `question`, `answer` | Natural-language question and gold answer |
| `evidence` | One rendered, target-redacted statement |
| `ground_truth` | Computation and source provenance |

Each evidence object contains `doc_name`, `evidence_page_num`, `evidence_text`,
`evidence_text_full_page`, and `source`. `evidence_page_num` is null because the
statement is rendered from XBRL rather than extracted from a PDF page.

Each `ground_truth` object contains `answer`, `value`, `unit`, `proof_source`,
`concept`, `identity`, `operands`, and `provenance`. Every operand records an
XBRL concept, calculation weight, and value in USD millions. Provenance records
the ticker, CIK, filing name, accession, fiscal year, period end, and report
date.

## Construction and provenance

The builder:

1. obtains issuer facts from the SEC companyfacts API;
2. identifies up to six recent 10-K filings per configured issuer;
3. obtains each filing's XBRL instance and calculation linkbase;
4. retains supported additive relationships whose parent and all children
   resolve for the filing period and satisfy the declared calculation within
   the construction tolerance;
5. renders a multi-period financial statement and redacts the target cell; and
6. deterministically round-robins across companies to select 600 rows.

All 600 records declare `xbrl_calculation_linkbase` as their proof source. The
builder is not temporally stable because "recent" filings change. The v1.0
release therefore must include the exact 86 accessions, report dates, source
cutoff, builder revision, and checksums rather than relying on a future rebuild
to recreate the same rows.

Primary sources:

- SEC EDGAR APIs and filing archives: <https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data>
- SEC developer and fair-access guidance: <https://www.sec.gov/about/developer-resources>

## Audited integrity

The publication audit found:

- 600 rows, 600 unique IDs, and 600 unique question strings;
- no missing answers, evidence blocks, ground-truth objects, or required
  provenance fields;
- exactly one target redaction and one evidence object per row;
- 2,490 operands total, with 2 to 16 operands per calculation;
- all 86 dataset filings present in the local document-link file and XBRL
  manifest; and
- all stored equations within the builder tolerance.

Operand and answer values are stored to three decimal places in USD millions.
Recomputing the weighted sums from the stored operands leaves a nonzero residual
in 29 rows, with a maximum absolute residual of USD 2 million. All rows satisfy
the construction tolerance: the larger of USD 1 million and 0.5% of the absolute
parent value. These residuals are not attributed solely to decimal rounding.
Use the frozen raw XBRL facts when reproducing fact binding.

## Audited files and payloads

| Payload | Count or size | SHA-256 |
|---|---:|---|
| `provable_calcrequired.jsonl` | 600 rows; 3,917,327 bytes | `a63655a04251bce99cc44a77af5c4abf42fb4a4f1b3e098ce03e3668f8a385ce` |
| `financebench_document_information.jsonl` | 86 rows; 10,977 bytes | `6cd4594b2c1713b0e38678fff27b6c4e510708ae3724ab4cc57275ec64f8c6d9` |
| Used XBRL instances and calculation linkbases | 172 files; 339,055,942 bytes | checksum-list root: `18a72bee24d5df74c140c0210181fc56564f8e1510d355422c8ea3f5b8bf523c` |
| SEC companyfacts snapshots | 28 files; 116,625,636 bytes | checksum-list root: `74b8d927c2a2a22ac3da38f7ba724a207b58a31ba94bd8d084398a1138ea7726` |
| Frozen retrieval chunks | 139,252 rows; 111,651,373 bytes | `2d47ea2d7481d056b9c539d64441b9029cf1166bf7a7f53bb5e542861538b147` |
| Retrieval corpus manifest | 10,569 bytes | `549547fcbec1d60a0875fb323b52c9be9ef23ce119af8959b416835e2b7029b1` |
| Retrieval source PDFs | 86 files; 305,632,038 bytes | checksum-list root: `e77468e85c41dffd420a61d4e7b1d58784e3d92586567c7fb5cb6580adfd25fd` |

A "checksum-list root" is the SHA-256 of the newline-terminated, sorted list of
`<file-sha256><two spaces><repository-relative-path>` entries audited in the
source tree. The final release must also provide the individual checksum list
and hashes of the built archives.

The source XBRL manifest contains 149 documents, of which 86 are used by this
dataset. Publish a pruned 86-document manifest with the v1.0 release. Do not
ship unrelated filings merely because they are present in the working cache.

## Reproducibility tiers

- Dataset inspection and oracle-evidence answer generation require the JSONL
  dataset only.
- Exact deterministic fact binding and calculation-linkbase verification
  require the 86-document XBRL payload and 28 companyfacts snapshots.
- Exact retrieval experiments require the frozen corpus chunks. Embeddings can
  be regenerated with the declared retrieval model and should not be stored in
  ordinary Git history.
- Rebuilding the benchmark from EDGAR requires the pinned accessions, builder
  revision, source cutoff, and SEC-compliant network access.

## Limitations

- The benchmark contains only additive relationships declared in filing XBRL
  calculation linkbases; it does not represent all financial reasoning.
- It covers 28 configured issuers and is not a statistically representative
  sample of all filers.
- Filing taxonomies, restatements, and EDGAR contents can change after the
  frozen source date.
- The evidence is rendered from XBRL and differs from a user's experience of a
  complete filing or PDF.

## License

The dataset compilation and original contributions are released under
[CC BY-NC 4.0](LICENSE.md), with attribution to **XBRLFiling contributors**.
See [NOTICE.md](NOTICE.md) for source attribution and third-party rights.
