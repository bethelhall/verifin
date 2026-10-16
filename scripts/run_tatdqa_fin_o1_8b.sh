#!/usr/bin/env bash
# Run the 403-question TAT-DQA handoff with the Qwen3-based Fin-o1-8B.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.venv/bin/python"
output_dir="${RUN_OUTPUT_DIR:-$repo_root/results/reruns/tatdqa_fin_o1_8b}"
model=fin-o1-8b-tatdqa:latest

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Usage: bash scripts/run_tatdqa_fin_o1_8b.sh [agent run options]

Runs all 403 questions and resumes completed rows automatically.
Add --limit 1 for a one-question smoke test.
Results: results/reruns/tatdqa_fin_o1_8b/
Saved paper notebooks keep using the frozen results/new_submissions inputs.
Model: TheFinAI/Fin-o1-8B via mradermacher's pinned Q4_K_M GGUF.
Weights stay in Ollama's normal model storage, outside the results directory.
Requires the repository .venv and a running local Ollama server.
HELP
    exit 0 ;;
esac

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export VLLM_API_KEY=EMPTY
export OLLAMA_HOST=http://127.0.0.1:11434

# Validate model identity and a real structured response before saving any rows.
"$python_bin" "$repo_root/scripts/setup_fin_o1_8b.py" \
  --provenance "$output_dir/model_setup.json"
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
