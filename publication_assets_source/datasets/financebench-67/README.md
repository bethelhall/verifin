# FinanceBench-67 v1.0

FinanceBench-67 is the exact 67-question FinanceBench evaluation subset used in
the VeriFin paper. All 67 records are retained in the benchmark denominator,
including questions for which an answer generator emits no parseable numerical
candidate.

The audited source file is `data/numerical_questions.jsonl`:

- rows: 67;
- bytes: 466,529;
- SHA-256: `37639aefbd212518e601685a448463304b6e9f8818487f2d229b7460f55472c2`;
- unique question IDs: 67;
- companies: 28;
- source documents: 55;
- embedded evidence objects: 90.

The records are an exact, order-preserving, unmodified subset of Patronus AI's
150-example FinanceBench open-source sample. The subset is distributed under
CC BY-NC 4.0 and requires FinanceBench attribution. The machine-readable
[`selection.json`](selection.json) freezes all 67 IDs, their order, and the
source revision and hashes. [`documents.jsonl`](documents.jsonl) contains the
55 unique document metadata records referenced by the subset. The historical
selection rule is not recoverable from the current source tree, so the file
records that limitation instead of inventing a criterion.

See [DATASET_CARD.md](DATASET_CARD.md) for schema and integrity details and
[NOTICE.md](NOTICE.md) for the required attribution and license notice.
