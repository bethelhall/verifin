from __future__ import annotations

import ast
import json
from typing import Any

from verifiqa.answerers.cot import format_evidence
from verifiqa.experiments.schemas import AnswerOutcome, SelectedEvidence
from verifiqa.generation.llm_client import ChatMessage
from verifiqa.types import ABSTENTION_MESSAGE, FinanceBenchExample


_PROMPT = """\
Answer the financial question using only the evidence provided.

Return JSON only, no markdown:
{{
  "reasoning": "<brief explanation of which evidence values are used>",
  "python": "<safe Python arithmetic only; assign the final value to variable answer>",
  "answer_unit": "<unit for answer, e.g. USD millions, percent, ratio>"
}}

Python rules:
- Use only numeric literals, arithmetic, assignments, and calls to round, abs,
  min, max, or sum.
- Do not import modules.
- Do not read files, use network access, call attributes, or use names starting
  with underscores.
- The final line must define variable answer.

Question:
{question}

Evidence:
{evidence}
"""


class PotAnswerer:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def answer(self, example: FinanceBenchExample, evidence: SelectedEvidence) -> AnswerOutcome:
        if not evidence.chunks:
            return AnswerOutcome(
                answer=ABSTENTION_MESSAGE,
                answerer="pot",
                error=evidence.status,
            )
        content = self.llm_client.chat(
            [ChatMessage(role="user", content=_PROMPT.format(
                question=example.question,
                evidence=format_evidence(evidence),
            ))],
            temperature=0.0,
            stage="experiment_pot_answer",
        ).strip()
        if not content:
            return AnswerOutcome(answer=ABSTENTION_MESSAGE, answerer="pot", error="empty_answer")
        try:
            data = json.loads(_extract_json(content))
            code = str(data.get("python", "")).strip()
            unit = str(data.get("answer_unit", "")).strip()
            value = execute_safe_python(code)
            suffix = f" {unit}" if unit else ""
            return AnswerOutcome(
                answer=f"{_format_value(value)}{suffix}",
                answerer="pot",
                raw={"reasoning": data.get("reasoning", ""), "python": code, "answer_unit": unit},
            )
        except Exception as exc:
            return AnswerOutcome(
                answer=ABSTENTION_MESSAGE,
                answerer="pot",
                raw={"raw_response": content},
                error=f"pot_failed:{type(exc).__name__}:{exc}",
            )


def execute_safe_python(code: str) -> Any:
    if not code:
        raise ValueError("empty_python")
    tree = ast.parse(code, mode="exec")
    _SafePythonValidator().visit(tree)
    env: dict[str, Any] = {}
    builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "sum": sum,
    }
    exec(compile(tree, "<pot>", "exec"), {"__builtins__": builtins}, env)
    if "answer" not in env:
        raise ValueError("missing_answer_variable")
    answer = env["answer"]
    if not isinstance(answer, (int, float)):
        raise ValueError("answer_not_numeric")
    return float(answer)


class _SafePythonValidator(ast.NodeVisitor):
    allowed_nodes = (
        ast.Module,
        ast.Assign,
        ast.Expr,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.List,
        ast.Tuple,
    )
    allowed_calls = {"abs", "min", "max", "round", "sum"}

    def generic_visit(self, node):
        if not isinstance(node, self.allowed_nodes):
            raise ValueError(f"unsafe_python_node:{type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id.startswith("_"):
            raise ValueError(f"unsafe_name:{node.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("unsafe_call")
        if node.func.id not in self.allowed_calls:
            raise ValueError(f"unsafe_call:{node.func.id}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if not isinstance(target, ast.Name):
                raise ValueError("unsafe_assignment_target")
        self.generic_visit(node)


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no_json_object")
    return text[start: end + 1]


def _format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.12g}"
