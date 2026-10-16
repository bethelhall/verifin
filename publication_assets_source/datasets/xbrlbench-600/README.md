# XBRLBench-600 v1.0

XBRLBench-600, released as **XBRLFiling**, is the 600-question,
calculation-required filing benchmark used in the VeriFin paper. The historical
directory name is retained for reproducibility. Each question asks for a financial-statement subtotal.
Its evidence is a rendered statement in which the target subtotal is redacted,
so the answer must be reconstructed from reported components. The ground-truth
record identifies the issuer filing, XBRL concept, calculation relationship,
operands, weights, period, unit, and answer.

The audited source file is
`data/multicompany_provable/data/provable_calcrequired.jsonl`:

- rows: 600;
- bytes: 3,917,327;
- SHA-256: `a63655a04251bce99cc44a77af5c4abf42fb4a4f1b3e098ce03e3668f8a385ce`;
- unique question IDs: 600;
- companies: 28;
- filings and SEC accessions: 86;
- requested subtotal concepts: 15.

[`financebench_document_information.jsonl`](../../../data/multicompany_provable/data/financebench_document_information.jsonl) pins the 86 SEC filing URLs used by the
snapshot (86 unique rows; SHA-256
`6cd4594b2c1713b0e38678fff27b6c4e510708ae3724ab4cc57275ec64f8c6d9`).

The dataset alone contains the question, answer, rendered evidence, and
ground-truth plan. Exact verifier reproduction additionally requires the frozen
SEC companyfacts snapshots and the XBRL instance and calculation-linkbase files
listed in the versioned provenance records. Exact retrieval reproduction also
requires the frozen corpus chunks or their source filing documents.

See [DATASET_CARD.md](DATASET_CARD.md) for schema and validation details and
[NOTICE.md](NOTICE.md) for provenance and licensing boundaries.

The dataset compilation and original contributions are licensed under
[CC BY-NC 4.0](LICENSE.md), with attribution to **XBRLFiling contributors**.
Third-party source material retains its separate rights and attribution.
