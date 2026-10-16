from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from verifiqa.types import SolverResult


class Z3Runner:
    def __init__(
        self,
        z3_binary: str = "z3",
        timeout_seconds: float = 10.0,
        save_dir: Optional[Path] = None,
        debug_unsat_cores: bool = False,
    ):
        self.z3_binary = z3_binary
        self.timeout_seconds = timeout_seconds
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.debug_unsat_cores = debug_unsat_cores
        self._save_index = 0

    def run(
        self,
        smtlib: str,
        label: str = "",
        *,
        consistency_only: bool = False,
        debug_unsat_core: Optional[bool] = None,
    ) -> SolverResult:
        include_core = self.debug_unsat_cores if debug_unsat_core is None else debug_unsat_core
        if include_core and not consistency_only:
            smtlib = _inject_unsat_core_options(smtlib)
        path = self._write_smt(smtlib, label)
        smt_path = str(path) if self.save_dir is not None else ""
        try:
            proc = subprocess.run(
                [self.z3_binary, str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return SolverResult(solver_status="UNKNOWN", error=f"z3_timeout:{exc}", smt_path=smt_path)
        except OSError as exc:
            return _run_with_python_z3(
                smtlib,
                smt_path=smt_path,
                timeout_seconds=self.timeout_seconds,
                include_core=include_core,
                unavailable_error=exc,
            )
        finally:
            if self.save_dir is None:
                try:
                    path.unlink()
                except OSError:
                    pass
        output = (proc.stdout or "").strip()
        error = (proc.stderr or "").strip()
        lines = output.splitlines()
        first = lines[0].strip().lower() if lines else ""
        if first == "unsat":
            core = _parse_unsat_core(lines[1] if len(lines) > 1 else "")
            return SolverResult(solver_status="UNSAT", raw_output=output, error=error, smt_path=smt_path, unsat_core=core)
        if first == "sat":
            model = "\n".join(lines[1:]).strip()
            return SolverResult(solver_status="SAT", raw_output=output, model=model, error=error, smt_path=smt_path)
        if first == "unknown":
            return SolverResult(solver_status="UNKNOWN", raw_output=output, error=error, smt_path=smt_path)
        return SolverResult(solver_status="INVALID", raw_output=output, error=error or f"z3_exit_code:{proc.returncode}", smt_path=smt_path)

    def _write_smt(self, smtlib: str, label: str) -> Path:
        if self.save_dir is None:
            with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False, encoding="utf-8") as handle:
                handle.write(smtlib)
                return Path(handle.name)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._save_index += 1
        stem = _safe_stem(label) or "query"
        path = self.save_dir / f"{self._save_index:04d}_{stem}.smt2"
        path.write_text(smtlib, encoding="utf-8")
        return path


def _run_with_python_z3(
    smtlib: str,
    *,
    smt_path: str,
    timeout_seconds: float,
    include_core: bool,
    unavailable_error: OSError,
) -> SolverResult:
    try:
        import z3
    except ImportError as exc:
        return SolverResult(
            solver_status="UNKNOWN",
            error=f"z3_unavailable:{unavailable_error}; z3_python_unavailable:{exc}",
            smt_path=smt_path,
        )

    solver = z3.Solver()
    solver.set(timeout=max(1, int(timeout_seconds * 1000)))
    if include_core:
        solver.set(unsat_core=True)

    try:
        solver.from_string(_smtlib_for_z3py(smtlib))
        status = solver.check()
    except z3.Z3Exception as exc:
        return SolverResult(
            solver_status="INVALID",
            error=f"z3_python_error:{exc}",
            smt_path=smt_path,
        )

    if status == z3.unsat:
        core = [str(item) for item in solver.unsat_core()] if include_core else []
        raw_output = "unsat"
        if include_core:
            raw_output = f"{raw_output}\n({' '.join(core)})"
        return SolverResult(
            solver_status="UNSAT",
            raw_output=raw_output,
            smt_path=smt_path,
            unsat_core=core,
        )
    if status == z3.sat:
        model = str(solver.model())
        return SolverResult(
            solver_status="SAT",
            raw_output=f"sat\n{model}".strip(),
            model=model,
            smt_path=smt_path,
        )
    return SolverResult(
        solver_status="UNKNOWN",
        raw_output="unknown",
        error=str(solver.reason_unknown()),
        smt_path=smt_path,
    )


def _smtlib_for_z3py(smtlib: str) -> str:
    smtlib = re.sub(r"\(\s*(check-sat|get-model|get-unsat-core)\s*\)", "", smtlib)
    smtlib = re.sub(r"\(\s*set-option\s+:[^)]*\)", "", smtlib)
    return smtlib


def _inject_unsat_core_options(smtlib: str) -> str:
    if "(set-option :produce-unsat-cores true)" not in smtlib:
        smtlib = re.sub(
            r"(\(set-logic\s+\S+\s*\))",
            "\\1\n(set-option :produce-unsat-cores true)",
            smtlib,
            count=1,
        )
    if "(get-unsat-core)" not in smtlib:
        smtlib = re.sub(
            r"(\(\s*check-sat\s*\))",
            "\\1\n(get-unsat-core)",
            smtlib,
            count=1,
        )
    return smtlib


def _parse_unsat_core(line: str) -> list[str]:
    line = line.strip()
    if not line.startswith("(") or not line.endswith(")"):
        return []
    return line[1:-1].split()


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return stem.strip("._")[:120]


# ---------------------------------------------------------------------------
# Unsat-core minimization and interpretation. Used only to build --repair
# feedback (see agent/cli.py); not called from the default (no --repair)
# pipeline, so it never changes behavior for a run made without --repair.
# ---------------------------------------------------------------------------

_NAMED_ASSERT_RE = re.compile(
    r"^\(assert\s+\(!\s+(?P<expr>.*?)\s+:named\s+(?P<name>[A-Za-z0-9_]+)\)\)\s*$"
)
_DECLARE_RE = re.compile(r"^\(declare-const\b.*\)\s*$")
_HEADER_RE = re.compile(r"^\((set-logic|set-option)\b.*\)\s*$")


def _extract_named_asserts(smtlib: str) -> dict[str, str]:
    """Map assertion name -> its full source line, for named top-level asserts."""
    out: dict[str, str] = {}
    for line in smtlib.splitlines():
        m = _NAMED_ASSERT_RE.match(line.strip())
        if m:
            out[m.group("name")] = line.strip()
    return out


def _smtlib_with_only(smtlib: str, keep_names: set[str]) -> str:
    """Rebuild an SMT script keeping only declare-consts, header options, and the
    named assertions in keep_names, ending in a plain (check-sat) (no core request:
    this is used only to test satisfiability of a candidate subset)."""
    named = _extract_named_asserts(smtlib)
    lines: list[str] = []
    for line in smtlib.splitlines():
        s = line.strip()
        if _HEADER_RE.match(s) or _DECLARE_RE.match(s):
            lines.append(s)
    for name in keep_names:
        if name in named:
            lines.append(named[name])
    lines.append("(check-sat)")
    return "\n".join(lines)


def minimize_unsat_core(
    runner: "Z3Runner",
    smtlib: str,
    core: list[str],
) -> list[str]:
    """Deletion-based minimization of an unsat core.

    A core returned by Z3 is *a* sufficient unsatisfiable subset, not necessarily
    the smallest one. For each member, try dropping it and re-check with just the
    remaining candidates; keep it removed only if the reduced set is still UNSAT.
    Costs at most len(core) extra solver calls (cheap relative to an LLM call).
    Not called anywhere in the default pipeline.
    """
    candidates = list(dict.fromkeys(core))  # de-dup, preserve order
    working = set(candidates)
    for name in candidates:
        trial = working - {name}
        if not trial:
            continue
        reduced = _smtlib_with_only(smtlib, trial)
        result = runner.run(reduced, label="core_min", consistency_only=True)
        if result.solver_status == "UNSAT":
            working = trial  # name wasn't needed; drop it for good
    # preserve original relative order for readability
    return [n for n in candidates if n in working]


def interpret_core_direction(core: list[str]) -> str:
    """Cheap, deterministic (no LLM) read of which tolerance bound is implicated.

    claim_upper in the core means (computed - claim) hit the +tolerance bound, i.e.
    computed exceeds the claim by more than tolerance -> the claim is too LOW.
    claim_lower means the reverse -> the claim is too HIGH. Both/neither is reported
    as unclear rather than guessed.
    """
    has_upper = "claim_upper" in core
    has_lower = "claim_lower" in core
    if has_upper and not has_lower:
        return "too_low"
    if has_lower and not has_upper:
        return "too_high"
    return "unclear"
