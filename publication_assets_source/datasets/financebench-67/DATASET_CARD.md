# Dataset card: FinanceBench-67 v1.0

## Summary

FinanceBench-67 is the fixed subset of the public FinanceBench sample evaluated
in the VeriFin paper. It contains 67 question-answer records concerning 55
financial documents from 28 companies. The records include human-annotated
answers and one or more evidence excerpts or full pages.

The intended grain is one FinanceBench question. `financebench_id` is the
primary key.

## Relationship to FinanceBench

All 67 records match the local 150-record Patronus FinanceBench source snapshot
byte-for-byte at the parsed JSON-object level and retain their original relative
order. No field was added, removed, or edited in the subset file.

The upstream source is pinned to Hugging Face revision
`e04404e3a97f69f79c14d42f24981a1c9c3bcd18`. Its
`financebench_merged.jsonl` has SHA-256
`7a1c81789e0fd2f1c37057a7ec0097756d726b05e7228e68e57db8e18c54fd0b`.
The 150-record source snapshot audited locally is an order-preserving field
projection of that file and has SHA-256
`a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877`;
after projecting the pinned upstream objects to the local schema, all 150
objects match exactly.

The source tree does not contain a script, selection manifest, or written rule
that explains why these particular 67 IDs were selected. Do not describe the
selection as systematic, random, or exhaustive until the authors document the
actual criterion. The safe description is "the fixed 67-question evaluation
subset used in the paper."

## Composition and schema

Question types:

| Question type | Rows |
|---|---:|
| `metrics-generated` | 47 |
| `novel-generated` | 16 |
| `domain-relevant` | 4 |
| **Total** | **67** |

Top-level fields are `financebench_id`, `company`, `doc_name`, `question_type`,
`question_reasoning`, `domain_question_num`, `question`, `answer`,
`justification`, `dataset_subset_label`, and `evidence`.

Each evidence object contains `doc_name`, `evidence_page_num`, `evidence_text`,
and `evidence_text_full_page`. The 67 records contain 90 evidence objects, or
one to two per question.

## Audited integrity and expected missingness

The publication audit found:

- 67 rows, 67 unique IDs, and 67 unique question strings;
- all 67 IDs present in the 150-example source snapshot;
- all 55 referenced documents present in the local source metadata;
- no missing answers or evidence blocks;
- no mismatches between row document names and evidence document names;
- integer evidence page indices ranging from 0 through 303;
- 15 null `justification` fields;
- 16 null `question_reasoning` fields; and
- 3 gold-answer strings containing no digit.

The null annotations and nonnumeric gold strings are properties of the source
records, not corruption introduced by subsetting. For this reason,
FinanceBench-67 should not be described as a dataset in which every gold answer
is numeric. Four questions that produced no numerical candidate in the paper's
answer-generation runs are still part of the 67-question population; that is a
run outcome, not a dataset exclusion.

The full local 361-row FinanceBench document metadata snapshot contains one
duplicate document identifier outside this subset. The release therefore
includes only the 55 unique metadata records required by the 67 questions.

## Audited files and payloads

| Payload | Count or size | SHA-256 |
|---|---:|---|
| `numerical_questions.jsonl` | 67 rows; 466,529 bytes | `37639aefbd212518e601685a448463304b6e9f8818487f2d229b7460f55472c2` |
| `documents.jsonl` | 55 rows; 13,894 bytes | `143280c89ec7c30d219eaffbfe52994fa84d5171cdff45351cc2b5a93957bd70` |
| Pinned upstream `financebench_merged.jsonl` | 150 rows | `7a1c81789e0fd2f1c37057a7ec0097756d726b05e7228e68e57db8e18c54fd0b` |
| 150-row FinanceBench source snapshot | 150 rows | `a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877` |
| Frozen retrieval chunks | 153,332 rows; 106,715,237 bytes | `2ba73e202633567a5be5a23495efe37b5448352e93a988d562b03398b3e5952e` |
| Retrieval corpus manifest | 5,791 bytes | `7bdbc1c03387bab99428827f3d07b0b2bc7e1a23d697e1f4ae1808ccd645fbe5` |
| Retrieval source PDFs | 45 files; 82,567,978 bytes | checksum-list root: `9e6168773e0f66655dc8a5cade888aaa7f701770ff056dfd21808ddb20bf1d57` |
| XBRL artifact manifest | 165,236 bytes | `c4e7746ca9e98864dddeca22a1039b0437c3302dfb6d4342b499e7deeb55ad62` |
| Saved XBRL instances and calculation linkbases | 100 files; 269,704,985 bytes | checksum-list root: `c17f0392e8adf4635c69238e490db2b77e2fc5676a6ae129888001cd9c436f8f` |
| SEC companyfacts snapshots | 49 files; 240,406,393 bytes | checksum-list root: `e064d5d8479f2017c95f17e80ec80345f5ff791f3fb2fec4e4b1c10631d353dd` |

A "checksum-list root" is the SHA-256 of the newline-terminated, sorted list of
`<file-sha256><two spaces><repository-relative-path>` entries audited in the
source tree. The final release must include the individual checksum list and
hashes for each built archive.

The XBRL manifest covers the 55 documents as follows: 48 have a saved XBRL
instance and calculation linkbase, 4 have an instance but no calculation
linkbase, and 3 are non-SEC documents without SEC XBRL. These are expected
coverage limits and should remain visible rather than being silently excluded.

## Reproducibility tiers

- Reproducing oracle-evidence experiments requires the 67-row JSONL file.
- Reproducing retrieval experiments requires the frozen corpus chunks or the
  exact source documents and build configuration. Regenerate embeddings from
  the frozen chunks with the declared retrieval model rather than committing a
  large platform-specific array.
- Reproducing XBRL grounding requires the saved XBRL artifacts and companyfacts
  snapshots. Questions whose documents lack an applicable calculation linkbase
  must retain that status.
- The 67 IDs and their source snapshot must remain fixed across all method
  comparisons.

## Limitations

- The subset-selection rationale is currently undocumented.
- The subset is small and is not a statistically representative sample of all
  financial questions or filings.
- Some source annotations are intentionally null and some answers are prose.
- FinanceBench's CC BY-NC 4.0 license prohibits uses outside its terms.
- Financial documents and filing contents remain attributable to their original
  sources; no issuer or Patronus AI endorsement is implied.
