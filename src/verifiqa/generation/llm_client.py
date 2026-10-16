from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# Reasoning-tuned models (e.g. TheFinAI/Fino1-8B) prepend a chain-of-thought and
# wrap structured output in markdown fences. The downstream JSON parsers do
# json.loads() on the raw string, so a "## Thinking" preamble or a ```json fence
# raises JSONDecodeError (a ValueError). Normalise the reply before returning.
_THINK_TAG = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)
_THINK_HEADER = re.compile(
    r"^\s*#{1,6}\s*(?:thinking|thought|reasoning)\b.*?(?=\n\s*#{1,6}\s+\S)",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_HEADER = re.compile(
    r"^\s*#{1,6}\s*(?:response|answer|final(?:\s+answer)?|output|result)\b[:\s]*",
    re.IGNORECASE,
)
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove <think> blocks / '## Thinking' preambles and unwrap code fences.

    Conservative: only strips a leading reasoning section when a later header
    follows it, and only unwraps a fence when exactly one is present, so plain
    replies (e.g. "$89,311 million") pass through untouched.
    """
    if not text:
        return text
    out = _THINK_TAG.sub("", text)
    m = _THINK_HEADER.match(out)
    if m:
        out = _ANSWER_HEADER.sub("", out[m.end():], count=1)
    fences = _FENCE.findall(out)
    if len(fences) == 1 and out.strip().startswith("```"):
        out = fences[0]
    return out.strip()


@dataclass
class ChatMessage:
    role: str
    content: str


class LlmError(RuntimeError):
    pass


def _resolve_codex_executable(executable: str) -> str:
    """Find Codex even when an external terminal lacks the IDE's PATH entries."""
    if os.sep in executable:
        return str(Path(executable).expanduser().resolve())
    on_path = shutil.which(executable)
    if on_path:
        return str(Path(on_path).resolve())
    # Respect explicit custom names; discovery only applies to the default.
    if executable != "codex" or platform.system() != "Darwin":
        return executable
    home_dir = Path.home()
    app_paths = [Path("/Applications/Codex.app/Contents/Resources/codex"),
                 home_dir / "Applications/Codex.app/Contents/Resources/codex"]
    for candidate in app_paths:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    architecture = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64"}.get(platform.machine())
    if architecture is None:
        return executable
    candidates = []
    for editor in (".vscode", ".vscode-insiders"):
        extension_root = home_dir / editor / "extensions"
        for candidate in extension_root.glob(f"openai.chatgpt-*/bin/macos-{architecture}/codex"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                candidates.append(candidate)
    if candidates:
        return str(max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p))).resolve())
    return executable


class CodexCliClient:
    """One-shot text requests through Codex's saved ChatGPT authentication.

    Codex is an agent harness, not the Chat Completions API: temperature and
    max_tokens are accepted for caller compatibility but are not enforced.
    Each request starts a separate ephemeral session outside the repository.
    """

    def __init__(self, model: str = "gpt-6.1-sol", executable: str = "codex",
                 timeout: float = 300.0, reasoning_effort: str = "low"):
        if not model or not executable or timeout <= 0:
            raise ValueError("Codex requires a model, executable, and positive timeout")
        if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported Codex reasoning effort")
        self.model = model
        self.executable = _resolve_codex_executable(executable)
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        try:
            version = subprocess.run([self.executable, "--version"], capture_output=True,
                                     text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LlmError(
                f"Cannot run Codex CLI at {self.executable!r}: {exc}. "
                "Install it or set --codex-bin to its executable path, then run codex login."
            ) from exc
        if version.returncode or not version.stdout.strip().startswith("codex-cli "):
            raise LlmError("The configured executable did not report a Codex CLI version")
        self.run_metadata = {
            "transport": "codex_exec", "model": model, "authentication": "chatgpt",
            "codex_version": version.stdout.strip(),
            "reasoning_effort": reasoning_effort, "temperature": None,
            "max_tokens": None, "tools_enabled": False,
        }

    def chat(self, messages: List[ChatMessage], temperature: float = 0.0,
             stage: str = "unknown", max_tokens: Optional[int] = None) -> str:
        if not messages or any(m.role not in {"system", "developer", "user", "assistant"}
                               or not isinstance(m.content, str) for m in messages):
            raise LlmError("Codex requires nonempty text messages with supported roles")
        # Feed all evidence through stdin, never via shell interpolation or argv.
        prompt = (
            "Complete the following conversation. Respect its system/developer instructions "
            "and return only the requested assistant answer, without commentary or tool use.\n"
            + json.dumps([{"role": m.role, "content": m.content} for m in messages],
                         ensure_ascii=False)
        )
        command = [self.executable, "exec", "--ignore-user-config", "--ephemeral",
                   "--skip-git-repo-check", "--sandbox", "read-only", "--json",
                   "--color", "never", "--model", self.model]
        overrides = {
            "forced_login_method": "chatgpt", "model_provider": "openai",
            "model_reasoning_effort": self.reasoning_effort,
            "web_search": "disabled", "project_doc_max_bytes": 0,
            "history.persistence": "none",
            "developer_instructions": (
                "You are a text-only financial question-answering component. Use only the "
                "conversation supplied on stdin. Do not access files, tools, memories, "
                "external sources, or other conversations. Return only the requested answer."
            ),
        }
        for feature in ("shell_tool", "apps", "multi_agent", "multi_agent_v2", "plugins",
                        "hooks", "memories", "browser_use", "computer_use", "image_generation",
                        "view_image", "skill_search", "code_mode"):
            overrides[f"features.{feature}"] = False
        for key, value in overrides.items():
            command.extend(["-c", f"{key}={json.dumps(value)}"])
        command.append("-")
        environment = os.environ.copy()
        # An inherited API key must not silently change the user's billing route.
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_THREAD_ID"):
            environment.pop(key, None)
        try:
            with tempfile.TemporaryDirectory(prefix="verifiqa-codex-") as workdir:
                completed = subprocess.run(command, input=prompt, text=True,
                                           capture_output=True, cwd=workdir, env=environment,
                                           timeout=self.timeout, check=False)
        except FileNotFoundError as exc:
            raise LlmError("Codex CLI not found. Install it or set --codex-bin, then run codex login.") from exc
        except subprocess.TimeoutExpired as exc:
            raise LlmError(f"Codex request timed out after {self.timeout:g}s ({stage}).") from exc
        except OSError as exc:
            raise LlmError(f"Could not start Codex CLI: {exc}") from exc
        if completed.returncode:
            detail = completed.stderr.strip()[-1500:]
            raise LlmError(f"Codex CLI exited with {completed.returncode}: {detail}")
        answer = None
        finished = False
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise LlmError("Codex returned invalid JSON events; update the CLI.") from exc
            if not isinstance(event, dict):
                raise LlmError("Codex returned an invalid event object")
            kind = event.get("type")
            if kind in {"error", "turn.failed"}:
                raise LlmError(f"Codex request failed: {event.get('message') or event.get('error')}")
            if kind in {"item.started", "item.updated", "item.completed"}:
                item = event.get("item", {})
                if not isinstance(item, dict):
                    raise LlmError("Codex returned an invalid item object")
                if item.get("type") == "error":
                    raise LlmError(f"Codex reported an error: {item.get('message')}")
                if item.get("type") not in {"agent_message", "reasoning"}:
                    raise LlmError("Codex attempted tool use; refusing a result with extra evidence")
                if kind == "item.completed" and item.get("type") == "agent_message":
                    answer = item.get("text")
            finished |= kind == "turn.completed"
        if not finished or not isinstance(answer, str) or not answer.strip():
            raise LlmError("Codex did not complete a turn with a final text answer")
        answer = strip_reasoning(answer)
        if not answer:
            raise LlmError("Codex returned an empty final answer")
        return answer


class OpenAIResponsesClient:
    """Text-only OpenAI Responses API requests, billed to OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-6.1-sol", timeout: float = 180.0,
                 reasoning_effort: str = "low", max_output_tokens: int = 8192,
                 max_retries: int = 4, retry_backoff: float = 2.0):
        self._api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not self._api_key:
            raise LlmError("Missing OpenAI API key. Set OPENAI_API_KEY.")
        if not model or timeout <= 0 or max_output_tokens <= 0 or max_retries < 0 or retry_backoff < 0:
            raise ValueError("OpenAI requires a model, positive timeout/token limit and nonnegative retry settings")
        if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported OpenAI reasoning effort")
        self.model, self.timeout = model, timeout
        self.reasoning_effort, self.max_output_tokens = reasoning_effort, max_output_tokens
        self.max_retries, self.retry_backoff = max_retries, retry_backoff
        self.run_metadata = {
            "transport": "openai_responses", "authentication": "api_key", "model": model,
            "reasoning_effort": reasoning_effort, "max_output_tokens": max_output_tokens,
            "temperature": None, "tools_enabled": False,
        }

    def chat(self, messages: List[ChatMessage], temperature: float = 0.0,
             stage: str = "unknown", max_tokens: Optional[int] = None) -> str:
        if not messages or any(m.role not in {"system", "developer", "user", "assistant"}
                               or not isinstance(m.content, str) for m in messages):
            raise LlmError("OpenAI requires nonempty text messages with supported roles")
        output_tokens = int(max_tokens) if max_tokens is not None else self.max_output_tokens
        if output_tokens <= 0:
            raise ValueError("OpenAI max_output_tokens must be positive")
        payload = {
            "model": self.model, "input": [_message_dict(message) for message in messages],
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": output_tokens, "store": False,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}",
                     "User-Agent": "verifiqa/1.0"}, method="POST",
        )
        transient = {408, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                # Error bodies may echo request details. Do not log credentials,
                # provider response bodies or request headers through exceptions.
                error = LlmError(f"OpenAI request failed: HTTP {exc.code}")
                if exc.code not in transient or attempt == self.max_retries:
                    raise error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.max_retries:
                    raise LlmError("OpenAI request failed: network error or timeout") from exc
            except (ValueError, UnicodeError) as exc:
                raise LlmError("OpenAI returned malformed response JSON") from exc
            time.sleep(self.retry_backoff * (2 ** attempt))
        if not isinstance(data, dict) or data.get("status") != "completed" or data.get("error"):
            raise LlmError("OpenAI did not return a completed response")
        output = data.get("output")
        if not isinstance(output, list):
            raise LlmError("OpenAI returned malformed output")
        texts = []
        for item in output:
            if not isinstance(item, dict):
                raise LlmError("OpenAI returned malformed output item")
            if item.get("type") == "reasoning":
                continue
            if (item.get("type") != "message" or item.get("role") != "assistant"
                    or item.get("status") != "completed" or not isinstance(item.get("content"), list)):
                raise LlmError("OpenAI returned unexpected nonfinal or nonassistant output")
            for part in item["content"]:
                if not isinstance(part, dict):
                    raise LlmError("OpenAI returned malformed message content")
                if part.get("type") == "output_text":
                    if not isinstance(part.get("text"), str):
                        raise LlmError("OpenAI returned malformed answer text")
                    texts.append(part["text"])
        answer = strip_reasoning("".join(texts))
        if not answer:
            raise LlmError("OpenAI returned no final assistant text")
        return answer


class VllmOpenAIClient:
    """Small OpenAI-compatible chat client for a local vLLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: str = "EMPTY",
        timeout: float = 180.0,
        max_tokens: int = 1024,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
        strip_reasoning: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.strip_reasoning = strip_reasoning

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.0,
        stage: str = "unknown",
        max_tokens: Optional[int] = None,
    ) -> str:
        output_tokens = int(max_tokens) if max_tokens is not None else self.max_tokens
        payload = {
            "model": self.model,
            "messages": [message.__dict__ for message in messages],
            "temperature": temperature,
            "max_tokens": output_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # Some hosted OpenAI-compatible endpoints (e.g. the HuggingFace
                # router) sit behind Cloudflare, which 403s the default
                # "Python-urllib" User-Agent. Send an explicit one.
                "User-Agent": "verifiqa/1.0",
            },
            method="POST",
        )
        # Hosted routers (e.g. HuggingFace + featherless-ai) intermittently return
        # transient errors when a model is cold or overloaded. Retry those with
        # exponential backoff; fail fast on non-transient errors (auth, bad request).
        _TRANSIENT = {408, 429, 500, 502, 503, 504}
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                last_exc = LlmError(f"vLLM request failed: HTTP Error {exc.code}: {exc.reason}")
                if exc.code not in _TRANSIENT or attempt == self.max_retries:
                    raise last_exc from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = LlmError(f"vLLM request failed: {exc}")
                if attempt == self.max_retries:
                    raise last_exc from exc
            time.sleep(self.retry_backoff * (2 ** attempt))
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"Unexpected vLLM response shape: {data}") from exc
        return strip_reasoning(content) if self.strip_reasoning else content


class AnthropicClaudeClient:
    """Small Anthropic Messages API client for Claude experiments."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com/v1",
        anthropic_version: str = "2023-06-01",
        timeout: float = 60.0,
        max_tokens: int = 1024,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.0,
        stage: str = "unknown",
        max_tokens: Optional[int] = None,
        output_config: Optional[dict] = None,
    ) -> str:
        if not self.api_key:
            raise LlmError(f"Missing Anthropic API key. Set {self.api_key_env}.")
        output_tokens = int(max_tokens) if max_tokens is not None else self.max_tokens
        system_parts = [message.content for message in messages if message.role == "system"]
        user_messages = [
            {
                "role": "assistant" if message.role == "assistant" else "user",
                "content": message.content,
            }
            for message in messages
            if message.role != "system"
        ]
        payload = {
            "model": self.model,
            "messages": user_messages,
            "max_tokens": output_tokens,
            "temperature": temperature,
        }
        if output_config is not None:
            payload["output_config"] = output_config
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        request = urllib.request.Request(
            f"{self.base_url}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            },
            method="POST",
        )
        # The Anthropic API and the network path to it intermittently return transient
        # errors (rate limits, 5xx, DNS blips). Retry those with exponential backoff;
        # fail fast on non-transient errors (bad request, auth, not-found).
        _TRANSIENT = {408, 429, 500, 502, 503, 504}
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_exc = LlmError(f"Claude request failed: HTTP {exc.code}: {body}")
                if exc.code not in _TRANSIENT or attempt == self.max_retries:
                    raise last_exc from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = LlmError(f"Claude request failed: {exc}")
                if attempt == self.max_retries:
                    raise last_exc from exc
            time.sleep(self.retry_backoff * (2 ** attempt))
        try:
            return "".join(part.get("text", "") for part in data["content"] if part.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LlmError(f"Unexpected Claude response shape: {data}") from exc


class LoggedLlmClient:
    """Logs every LLM input/output to stderr and optionally JSONL."""

    def __init__(
        self,
        inner,
        terminal: bool = True,
        jsonl_path: Optional[Path] = None,
        reset_jsonl: bool = True,
    ):
        self.inner = inner
        self.terminal = terminal
        self.jsonl_path = jsonl_path
        self.call_index = 0
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            if reset_jsonl and self.jsonl_path.exists():
                self.jsonl_path.unlink()

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.0,
        stage: str = "unknown",
        max_tokens: Optional[int] = None,
        output_config: Optional[dict] = None,
    ) -> str:
        self.call_index += 1
        call_id = self.call_index
        started = time.time()
        self._print_start(call_id, stage, messages, temperature, max_tokens)
        kwargs = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if output_config is not None:
            kwargs["output_config"] = output_config
        try:
            response = self.inner.chat(messages, **kwargs)
        except Exception as exc:
            duration = time.time() - started
            record = {
                "call_id": call_id,
                "stage": stage,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [_message_dict(message) for message in messages],
                "response": "",
                "error": f"{type(exc).__name__}: {exc}",
                "duration_seconds": duration,
            }
            self._write_record(record)
            self._print_error(call_id, stage, exc, duration)
            raise
        duration = time.time() - started
        record = {
            "call_id": call_id,
            "stage": stage,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [_message_dict(message) for message in messages],
            "response": response,
            "error": "",
            "duration_seconds": duration,
        }
        self._write_record(record)
        self._print_response(call_id, stage, response, duration)
        return response

    def _write_record(self, record: Dict[str, object]) -> None:
        if self.jsonl_path is None:
            return
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _print_start(
        self,
        call_id: int,
        stage: str,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: Optional[int],
    ) -> None:
        if not self.terminal:
            return
        sys.stderr.write(f"\n\n========== LLM CALL {call_id}: {stage} INPUT ==========\n")
        sys.stderr.write(f"temperature: {temperature}\n")
        if max_tokens is not None:
            sys.stderr.write(f"max_tokens: {max_tokens}\n")
        for index, message in enumerate(messages, start=1):
            sys.stderr.write(f"\n--- message {index} role={message.role} ---\n")
            sys.stderr.write(message.content)
            if not message.content.endswith("\n"):
                sys.stderr.write("\n")
        sys.stderr.write(f"========== END LLM CALL {call_id}: {stage} INPUT ==========\n")
        sys.stderr.flush()

    def _print_response(self, call_id: int, stage: str, response: str, duration: float) -> None:
        if not self.terminal:
            return
        sys.stderr.write(f"\n========== LLM CALL {call_id}: {stage} OUTPUT ({duration:.2f}s) ==========\n")
        sys.stderr.write(response)
        if not response.endswith("\n"):
            sys.stderr.write("\n")
        sys.stderr.write(f"========== END LLM CALL {call_id}: {stage} OUTPUT ==========\n")
        sys.stderr.flush()

    def _print_error(self, call_id: int, stage: str, exc: Exception, duration: float) -> None:
        if not self.terminal:
            return
        sys.stderr.write(f"\n========== LLM CALL {call_id}: {stage} ERROR ({duration:.2f}s) ==========\n")
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.stderr.write(f"========== END LLM CALL {call_id}: {stage} ERROR ==========\n")
        sys.stderr.flush()


def _message_dict(message: ChatMessage) -> Dict[str, str]:
    return {"role": message.role, "content": message.content}


def make_llm_client(mode: str = "claude", config: Optional[Dict[str, str]] = None):
    config = config or {}
    if mode == "openai":
        return OpenAIResponsesClient(
            model=config.get("model", "gpt-6.1-sol"),
            timeout=float(config.get("timeout", 180)),
            reasoning_effort=config.get("reasoning_effort", "low"),
            max_output_tokens=int(config.get("max_output_tokens", 8192)),
            max_retries=int(config.get("max_retries", 4)),
            retry_backoff=float(config.get("retry_backoff", 2.0)),
        )
    if mode == "codex":
        return CodexCliClient(
            model=config.get("model", "gpt-6.1-sol"),
            executable=config.get("executable", "codex"),
            timeout=float(config.get("timeout", 300)),
            reasoning_effort=config.get("reasoning_effort", "low"),
        )
    if mode == "vllm":
        return VllmOpenAIClient(
            base_url=config.get("base_url", "http://localhost:8000/v1"),
            model=config.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            api_key=config.get("api_key", "EMPTY"),
            timeout=float(config.get("timeout", 180)),
            max_tokens=int(config.get("max_tokens", 2048)),
            max_retries=int(config.get("max_retries", 4)),
            retry_backoff=float(config.get("retry_backoff", 2.0)),
            strip_reasoning=bool(config.get("strip_reasoning", True)),
        )
    if mode in {"claude", "anthropic"}:
        return AnthropicClaudeClient(
            base_url=config.get("base_url", "https://api.anthropic.com/v1"),
            model=config.get("model", "claude-sonnet-4-20250514"),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env", "ANTHROPIC_API_KEY"),
            timeout=float(config.get("timeout", 60)),
            max_tokens=int(config.get("max_tokens", 2048)),
            max_retries=int(config.get("max_retries", 4)),
            retry_backoff=float(config.get("retry_backoff", 2.0)),
        )
    raise ValueError(f"Unsupported LLM mode: {mode}")
