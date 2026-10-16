# TAT-DQA arithmetic SEC-XBRL pilot

This handoff contains the frozen dataset, the supplied TAT-DQA PDF excerpts,
independently retrieved SEC XBRL packages, evaluator-only answers, and source
matching provenance needed to run the existing FinanceBench-style VeriFin path.

## Frozen population

- 407 TAT-DQA questions met the document/XBRL match and `answer_type=arithmetic`
  rules.
- 4 were rejected by VeriFin's existing pre-run numerical-question filter.
- The executable pilot therefore contains 403 questions from 157 PDF excerpts and
  76 exact SEC filing matches.
- Eligibility was fixed before running VeriFin and never depends on whether
  VeriFin can retrieve facts, verify a formula, or produce a correct answer.

The exact exclusion and every included ID are recorded in
`provenance/arithmetic_eligibility_manifest.json`.

## Directory map

- `dataset/financebench_xbrl_augmented.jsonl`: 403-row input for the unchanged
  FinanceBench loader. It contains the answer only because the existing loader
  carries that field into result scoring; generation and verification prompts do
  not consume it.
- `dataset/tatdqa_arithmetic_inference.jsonl`: the same 403 questions with no
  answer, derivation, annotated facts, or block mappings.
- `dataset/tatdqa_arithmetic_candidates_answer_free.jsonl`: all 407 matched
  arithmetic candidates before the unchanged numerical-question gate.
- `evaluator_only/tatdqa_arithmetic_pilot_gold.jsonl`: minimal scoring sidecar.
- `documents/`: the 157 supplied TAT-DQA PDF excerpts and their extracted page
  JSON files.
- `xbrl/`: 76 complete filing packages and a VeriFin-compatible `manifest.json`.
- `provenance/`: matching records, frozen eligibility, and file checksums.

No TAT-DQA gold derivation, annotated fact list, or block mapping is included.

## Validate before running

Run these commands from this handoff directory:

```bash
python3 -m json.tool xbrl/manifest.json >/dev/null
python3 - <<'PY'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path('dataset/financebench_xbrl_augmented.jsonl').read_text().splitlines()]
inference = [json.loads(line) for line in Path('dataset/tatdqa_arithmetic_inference.jsonl').read_text().splitlines()]
manifest = json.loads(Path('xbrl/manifest.json').read_text())
assert len(rows) == len(inference) == 403
assert len(manifest['docs']) == 76
assert all('answer' not in row and 'gold_answer' not in row for row in inference)
assert all(Path(row['source_pdf']).exists() for row in inference)
print('Validated: 403 questions, 157 PDF excerpts, 76 VeriFin XBRL packages')
PY
```

## Run the unchanged FinanceBench-style agent pipeline

Use the same LLM and retrieval flags as the FinanceBench experiment. From this
handoff directory, point `PYTHONPATH` at the verified-finance repository:

```bash
PYTHONPATH=/absolute/path/to/verified-finance/src python3 -m verifiqa.agent.cli run   --data dataset/financebench_xbrl_augmented.jsonl   --xbrl-artifacts xbrl   --out run_output   --llm-mode vllm   --model YOUR_FINANCEBENCH_QWEN_MODEL   --vllm-url http://127.0.0.1:8000/v1   --claim-source generated_answer   --deterministic-smt   --workers 4   --verbose
```

For a one-question plumbing check, add `--limit 1`. Then remove it and add
`--resume` to retain the completed smoke-test row. Omit `--deterministic-smt`
only when intentionally reproducing the LLM-SMT ablation. Do not add
`--finqa-dir`; that selects the separate FinQA program-oracle path.

The command must be launched from this handoff directory because paths inside
the portable XBRL manifest are relative to it.

## Why this avoids the benchmark circularity

The reference answers originate in TAT-DQA's document-linked annotations. The
SEC XBRL is retrieved separately only as VeriFin evidence. The benchmark gold
was not constructed from those calculation linkbases, source matching did not
use gold answers, and no gold derivation is supplied to VeriFin. XBRL and the
PDF may encode the same underlying financial facts; that shared source fact is
not circular label construction.
