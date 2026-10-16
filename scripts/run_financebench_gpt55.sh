#!/usr/bin/env bash
# Run the 67-question FinanceBench subset with GPT-5.5 through the Responses API.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
output_dir="${RUN_OUTPUT_DIR:-$repo_root/results/reruns/financebench_gpt5.5}"

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Usage: bash scripts/run_financebench_gpt55.sh [agent run options]

Runs all 67 FinanceBench questions and resumes completed rows automatically.
Add --limit 1 for a one-question smoke test.
Results: results/reruns/financebench_gpt5.5/
Saved paper notebooks keep using the frozen results/new_submissions inputs.

Requires OPENAI_API_KEY in your environment or the repository .env.
Uses the OpenAI Responses API with API billing and requests gpt-5.5 explicitly.
Defaults: low reasoning effort, 8192 output tokens, one extraction worker.
HELP
    exit 0 ;;
esac

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$repo_root"

# Check generation before the benchmark can save provider failures as abstentions.
"$python_bin" - <<'PY'
from verifiqa.agent import cli  # Load the repository .env without printing secrets.
from verifiqa.generation.llm_client import ChatMessage, OpenAIResponsesClient

client = OpenAIResponsesClient(model="gpt-5.5", reasoning_effort="low",
                               max_output_tokens=512, max_retries=0)
answer = client.chat([ChatMessage(role="user", content="Reply with only OK.")])
print(f"GPT-5.5 API readiness check: {answer.strip()}")
PY

mkdir -p "$output_dir"
# FinanceBench uses its supplied text evidence. The optional XBRL cache is unvalidated.
"$python_bin" -m verifiqa.agent.cli run \
  --data data/numerical_questions.jsonl \
  --out "$output_dir" \
  --source-plan-cache "$output_dir/source_plans" \
  --llm-mode openai \
  --model gpt-5.5 \
  --openai-reasoning-effort low \
  --openai-max-output-tokens 8192 \
  --claim-source generated_answer \
  --deterministic-smt \
  --workers 1 \
  --scoring-view source_reviewed \
  --scoring-root "$repo_root" \
  --resume --verbose "$@" 2>&1 | tee -a "$output_dir/run.log"
