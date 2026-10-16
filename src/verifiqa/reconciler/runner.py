from __future__ import annotations

from typing import Any

from verifiqa.generation.llm_client import ChatMessage
from verifiqa.policy.registry import DEFAULT_POLICY_REGISTRY, PolicyRegistry
from verifiqa.types import FinanceBenchExample, VerificationIR
from verifiqa.verification.z3_runner import Z3Runner

from .compiler import compile_reconciler_spec
from .context import build_reconciler_context
from .prompt import build_reconciler_prompt
from .spec import ReconcilerSpecError, parse_reconciler_spec


class ReconcilerRunner:
    def __init__(
        self,
        llm_client,
        *,
        registry: PolicyRegistry | None = None,
        z3_runner: Z3Runner | None = None,
        relative_tolerance: float = 0.01,
    ):
        self.llm_client = llm_client
        self.registry = registry or DEFAULT_POLICY_REGISTRY
        self.z3_runner = z3_runner or Z3Runner()
        self.relative_tolerance = relative_tolerance

    def run(self, example: FinanceBenchExample, ir: VerificationIR) -> dict[str, Any]:
        context = build_reconciler_context(example, ir, self.registry)
        prompt = build_reconciler_prompt(context)
        response = self.llm_client.chat(
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            stage="reconciler_spec",
            max_tokens=1024,
        )

        base: dict[str, Any] = {
            "enabled": True,
            "mode": "reconciler_shadow",
            "input_hash": context.input_hash,
            "source_fact_count": len(context.source_facts),
            "raw_response": response,
        }
        try:
            spec = parse_reconciler_spec(response)
        except ReconcilerSpecError as exc:
            return {
                **base,
                "status": "invalid_spec",
                "valid": None,
                "reason": str(exc),
            }

        compiled = compile_reconciler_spec(
            spec,
            context,
            ir,
            self.registry,
            relative_tolerance=self.relative_tolerance,
        )
        result = {
            **base,
            "spec": spec.to_dict(),
            "compiler": compiled.to_dict(),
        }
        if compiled.status != "compiled":
            return {
                **result,
                "status": compiled.status,
                "valid": None,
                "reason": compiled.reason,
            }

        solver = self.z3_runner.run(
            compiled.smtlib,
            label=f"{example.financebench_id or 'example'}_reconciler",
            debug_unsat_core=True,
        )
        if solver.solver_status == "SAT":
            status = "verified"
            valid: bool | None = True
            reason = "reconciler_smt_sat"
        elif solver.solver_status == "UNSAT":
            status = "violated"
            valid = False
            reason = "reconciler_smt_unsat"
        else:
            status = "solver_failed"
            valid = None
            reason = solver.error or solver.solver_status
        return {
            **result,
            "status": status,
            "valid": valid,
            "reason": reason,
            "solver": {
                "solver_status": solver.solver_status,
                "smt_path": solver.smt_path,
                "error": solver.error,
                "unsat_core": solver.unsat_core,
                "model": solver.model if solver.solver_status == "SAT" else "",
            },
        }
