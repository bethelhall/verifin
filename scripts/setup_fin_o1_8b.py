"""Install and check the pinned public Fin-o1-8B quantization in local Ollama."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HOST = "http://127.0.0.1:11434"
CONFIG = ROOT / "configs/fin_o1_8b_model.json"
MODELFILE = ROOT / "configs/fin_o1_8b_tatdqa.Modelfile"


def show(model: str) -> dict | None:
    request = urllib.request.Request(
        HOST + "/api/show", data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def validate_model(details: dict, config: dict) -> None:
    source = re.search(r"^FROM\s+.*sha256-([0-9a-f]{64})\s*$",
                       details.get("modelfile", ""), re.MULTILINE)
    if source is None or source.group(1) != config["quantization_sha256"]:
        raise ValueError("Fin-o1 GGUF checksum differs from the pinned model; refusing to run.")
    info = details.get("model_info", {})
    if info.get("general.architecture") != config["architecture"]:
        raise ValueError("Unexpected Fin-o1 model architecture")
    if info.get("general.parameter_count") != config["parameter_count"]:
        raise ValueError("Unexpected Fin-o1 parameter count")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path,
                        default=ROOT / "artifacts/model_setup/fin_o1_8b.json")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    details = show(config["base_model"])
    if details is None:
        subprocess.run(["ollama", "pull", config["base_model"]], check=True)
        details = show(config["base_model"])
    if details is None:
        raise SystemExit("Ollama could not load the downloaded Fin-o1 model.")
    validate_model(details, config)
    subprocess.run(["ollama", "create", config["model"], "-f", str(MODELFILE)], check=True)
    configured = show(config["model"])
    if configured is None:
        raise SystemExit("Ollama could not load the configured Fin-o1 model.")
    validate_model(configured, config)
    if not re.search(r"^num_ctx\s+32768\s*$", configured.get("parameters", ""), re.MULTILINE):
        raise SystemExit("Fin-o1 context does not match the 32K run configuration.")

    from verifiqa.generation.llm_client import ChatMessage, VllmOpenAIClient
    client = VllmOpenAIClient(base_url=HOST + "/v1", model=config["model"],
                             max_retries=0, max_tokens=2048)
    answer = client.chat([ChatMessage(role="user", content=
        'What is 12.5% of 480? Return only a JSON object with one string field "answer".')])
    if not answer or not answer.strip():
        raise SystemExit("Fin-o1 returned no final answer; stopped before writing benchmark rows.")
    parsed = json.loads(answer)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str) or not parsed["answer"].strip():
        raise SystemExit("Fin-o1 did not return the required answer object; benchmark rows were not written.")
    record = {
        **config, "modelfile_sha256": hashlib.sha256(MODELFILE.read_bytes()).hexdigest(),
        "ollama_model_info": configured["model_info"],
        "ollama_parameters": configured["parameters"],
        "ollama_template_sha256": hashlib.sha256(configured["template"].encode()).hexdigest(),
        "readiness_answer": parsed["answer"],
    }
    if args.provenance.exists():
        previous = json.loads(args.provenance.read_text())
        for field in ("checkpoint", "model", "quantization_sha256", "modelfile_sha256",
                      "ollama_template_sha256", "context_tokens"):
            if previous.get(field) != record[field]:
                raise SystemExit(f"Existing run has different {field}; use a new output directory.")
    else:
        args.provenance.parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Fin-o1 ready: {config['model']} (Q4_K_M, 32768 context); answer={parsed['answer']}")
    print(f"Model provenance: {args.provenance}")


if __name__ == "__main__":
    main()
