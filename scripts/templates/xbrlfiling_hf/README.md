---
pretty_name: XBRLFiling
language:
  - en
license: cc-by-nc-4.0
task_categories:
  - question-answering
size_categories:
  - n<1K
tags:
  - finance
  - xbrl
  - financial-reasoning
  - numerical-reasoning
  - tabular
  - datasets
configs:
  - config_name: default
    data_files:
      - split: test
        path: data/test.parquet
---

# XBRLFiling

**600 financial calculation questions · 28 public companies · 86 annual filings**

XBRLFiling evaluates numerical question answering over public-company financial
statements. Each question asks for a financial subtotal whose target value has
been removed from the supplied statement. The remaining line items provide the
evidence needed to calculate the answer. A filing-declared XBRL calculation
relationship supplies the reference calculation and provenance.

This release preserves the existing benchmark questions, evidence, answers,
and numeric values. Workspace metadata containing author or laboratory
identifiers and local filesystem paths is excluded; public company names,
tickers, SEC identifiers, and source links remain available for traceability.

## Quick start

Load the test split through the [anonymous review link](https://anonymous-hf.com/a/dc78cftylin7/).
No Hugging Face account or token is required.

```python
from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "test": "https://anonymous-hf.com/api/a/dc78cftylin7/resolve/data/test.parquet"
    },
    split="test",
    token=False,
)
example = dataset[0]

# Give the answer generator only the question and the redacted evidence.
model_input = {
    "question": example["question"],
    "context": example["context"],
}

# Keep the reference answer and proof separate for evaluation.
reference_answer = example["answer"]
```

The single **test** split contains all 600 benchmark examples. No training or
validation split is introduced. The dataset card explicitly selects the Parquet
file, so the equivalent JSONL copy does not create duplicate examples when
loaded with `datasets`.

## Coverage

| Statement family | Target concepts | Questions |
|---|---:|---:|
| Income statement | 5 | 226 |
| Balance sheet | 7 | 279 |
| Cash flow statement | 3 | 95 |
| **Total** | **15** | **600** |

The frozen source records have fiscal-year labels from **2021 through 2026**.
The unit is **USD millions**. Each question specifies its requested rounding.
There are 2,490 reference operands, with 2–16 operands per calculation.

## Fields

One row represents one question and its reference answer.

| Field | Description |
|---|---|
| `id` | Stable benchmark question identifier. |
| `question` | Original natural-language question. |
| `context` | Rendered financial statement with the target cell redacted. |
| `answer` | Original formatted reference answer. |
| `answer_value` | Numeric reference value, in `answer_unit`. |
| `answer_unit` | Unit of the reference value: USD millions. |
| `ticker` | Public-company ticker symbol. |
| `cik` | SEC Central Index Key. |
| `filing_id` | Stable filing label from the source benchmark. |
| `fiscal_year` | Fiscal-year label associated with the question. |
| `period_end` | Source financial period end. |
| `report_date` | Report-date field preserved from the source metadata. |
| `statement_family` | Income statement, balance sheet, or cash flow family. |
| `target_concept` | XBRL concept requested by the question. |
| `calculation` | Reference additive relationship declared by the filing. |
| `operands` | Reference operands, including concept, weight, and value in USD millions. |
| `source_url` | Public SEC source-filing URL. |
| `accession` | SEC filing accession number. |
| `proof_source` | `xbrl_calculation_linkbase` for every example. |

**Evaluation boundary:** `answer`, `answer_value`, `calculation`, and `operands`
are reference labels or proof material. Do not include them in the answer
generator's prompt. Use `question` and `context` for the provided-evidence task;
report retrieval-based experiments separately.

## Files

```text
.
├── README.md
├── LICENSE.md
├── NOTICE.md
├── data/
│   ├── test.parquet
│   └── test.jsonl
└── metadata/
    ├── filings.jsonl
    ├── quality_report.json
    └── manifest.json
```

`test.parquet` and `test.jsonl` contain the same 600 rows. The filing index
contains the 86 source filings and public URLs. The quality report records
release validation checks, and the manifest records SHA-256 file checksums.

This compact release contains the evidence needed for the provided-evidence
benchmark. Raw XBRL XML, SEC companyfacts snapshots, retrieval corpora, model
weights, and experiment outputs are not included. Reproducing the original
fact-binding or retrieval pipeline requires those additional source artifacts.

## Construction and sources

The source benchmark selected additive relationships from Form 10-K XBRL
calculation linkbases. It resolved the parent and child facts for the filing
period, checked the calculation against a construction tolerance, rendered a
multi-period statement, and redacted the target cell. Deterministic round-robin
selection across companies produced the 600-question benchmark.

The financial facts originate in issuer filings available through
[SEC EDGAR](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).
Each record retains its filing accession and source URL. Consult the
[SEC developer resources](https://www.sec.gov/about/developer-resources) before
automatically downloading source filings.

## Quality and limitations

- The source audit found 600 unique question identifiers and question strings,
  one evidence block and target redaction per row, and no missing answers or
  required provenance fields. Release checks are recorded in
  `metadata/quality_report.json`.
- Stored reference operands and values use three decimal places in USD
  millions. In 29 rows, recomputing the weighted sum from the stored operands
  leaves a nonzero residual; the maximum absolute residual is USD 2 million.
  All rows satisfy the original construction tolerance: the larger of USD
  1 million and 0.5% of the absolute parent value. These residuals are not
  attributed solely to decimal rounding, and exact equality of every stored
  operand sum is not guaranteed.
- The benchmark covers additive XBRL relationships and 28 selected issuers;
  it is not representative of all financial reasoning or all public companies.
- Evidence is rendered from XBRL rather than taken directly from a complete
  filing or PDF. A provided-evidence evaluation does not measure retrieval.
- Filings, taxonomies, and restatements may change. This release describes the
  frozen benchmark; rebuilding from newer EDGAR data can yield different rows.

## License and attribution

The dataset compilation and original contributions are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Credit **XBRLFiling contributors**, link to this dataset and the license, and
indicate changes when sharing an adaptation. See [LICENSE.md](LICENSE.md) for
the complete terms and [NOTICE.md](NOTICE.md) for source attribution and scope.

Public-company filings and their identifiers retain their source attribution.
The release license grants only rights held by the dataset contributors; it
does not relicense third-party material or imply SEC or issuer endorsement.
