#!/usr/bin/env bash
# Run the 403-question TAT-DQA handoff with local Llama 3.1 8B Instruct.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
output_dir="${RUN_OUTPUT_DIR:-$repo_root/results/reruns/tatdqa_llama3.1_8b}"
model=llama3.1-8b-tatdqa:latest
base_model=llama3.1:8b

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Usage: bash scripts/run_tatdqa_llama31_8b.sh [agent run options]

Runs all 403 questions and resumes completed rows automatically.
Add --limit 1 for a one-question smoke test.
Results: results/reruns/tatdqa_llama3.1_8b/
Models: Ollama's normal model storage (not the results folder).
Requires the repository .venv and a running local Ollama server.
HELP
    exit 0 ;;
esac

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Local Ollama needs no provider credential.
export VLLM_API_KEY=EMPTY
export OLLAMA_HOST=http://127.0.0.1:11434

"$python_bin" -c 'import verifiqa.agent.cli'
ollama list >/dev/null
if ! ollama show "$model" >/dev/null 2>&1; then
  if ! ollama show "$base_model" >/dev/null 2>&1; then
    ollama pull "$base_model"
  fi
  ollama create "$model" -f "$repo_root/configs/llama3_1_8b_tatdqa.Modelfile"
fi

# Fail before saving benchmark rows if the server cannot generate an answer.
"$python_bin" - "$model" <<'PY'
import sys
from verifiqa.generation.llm_client import ChatMessage, VllmOpenAIClient

client = VllmOpenAIClient(base_url="http://127.0.0.1:11434/v1",
                         model=sys.argv[1], max_retries=0, max_tokens=16)
answer = client.chat([ChatMessage(role="user", content="Reply with only OK.")])
if not answer or not answer.strip():
    raise SystemExit("Ollama returned no answer; stopped before writing results.")
print(f"Llama readiness check: {answer.strip()}")
PY

mkdir -p "$output_dir"
if [[ ! -f "$output_dir/ollama_model.txt" ]]; then
  ollama show "$model" --modelfile > "$output_dir/ollama_model.txt"
fi

# The handoff's PDF and XBRL manifest paths are relative to this directory.
cd "$repo_root/data/tatdqa_403_question_handoff_v1"
"$python_bin" -m verifiqa.agent.cli run \
  --data dataset/financebench_xbrl_augmented.jsonl \
  --xbrl-artifacts xbrl \
  --out "$output_dir" \
  --source-plan-cache "$output_dir/source_plans" \
  --llm-mode vllm \
  --model "$model" \
  --vllm-url http://127.0.0.1:11434/v1 \
  --claim-source generated_answer \
  --deterministic-smt \
  --workers 1 \
  --scoring-view source_reviewed \
  --scoring-root "$repo_root" \
  --resume --verbose "$@" 2>&1 | tee -a "$output_dir/run.log"
