"""Bounded arithmetic interpreter for model-produced PoT programs.

``evaluate_program(source)`` returns a finite float, or ``None`` for invalid
programs. ``evaluate_program_detailed(source)`` additionally returns an error
code suitable for saving with the complete model response/program in results.

This is a small Python-syntax arithmetic language, not a Python sandbox. It
supports assignments, augmented assignments, numeric literals, arithmetic,
and abs/round/min/max/sum/pow. Lists and tuples may hold numeric operands for
aggregation; pow accepts two arguments. All other syntax is rejected. No
generated code is passed to exec/eval. Arithmetic uses Python integers and
floats, with finite outputs and explicit resource bounds; it is not exact
real arithmetic.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import operator


MAX_SOURCE_CHARS = 16_384
MAX_AST_NODES = 512
MAX_AST_DEPTH = 48
MAX_STATEMENTS = 64
MAX_SEQUENCE_ITEMS = 128
MAX_INTEGER_BITS = 1024
MAX_EXPONENT = 1000
MAX_ROUND_DIGITS = 308
MAX_IDENTIFIER_CHARS = 64

_FUNCTIONS = frozenset({"abs", "round", "min", "max", "sum", "pow"})
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


@dataclass(frozen=True)
class ProgramEvaluation:
    value: float | None
    error: str | None = None


class _Rejected(ValueError):
    pass


def _number(value):
    # bool is a subclass of int, but is not a numerical answer.
    if type(value) is int:
        if value.bit_length() > MAX_INTEGER_BITS:
            raise _Rejected("integer_limit")
    elif type(value) is float:
        if not math.isfinite(value):
            raise _Rejected("non_finite")
    else:
        raise _Rejected("non_numeric")
    return value


def _integer(value):
    value = _number(value)
    if type(value) is not int:
        raise _Rejected("integer_argument_required")
    return value


def _check_tree(tree):
    # Iterative traversal bounds recursion before recursive interpretation.
    pending = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_AST_DEPTH:
            raise _Rejected("ast_limit")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


class _Interpreter:
    def __init__(self):
        self.variables = {}

    def _name(self, node):
        if not isinstance(node, ast.Name):
            raise _Rejected("name_required")
        if len(node.id) > MAX_IDENTIFIER_CHARS:
            raise _Rejected("identifier_limit")
        return node.id

    def _assign(self, target, value):
        name = self._name(target)
        if name in _FUNCTIONS:
            raise _Rejected("reserved_function_name")
        self.variables[name] = value

    def statement(self, node):
        if isinstance(node, ast.Assign):
            value = self.expression(node.value)
            for target in node.targets:
                self._assign(target, value)
        elif isinstance(node, ast.AugAssign):
            name = self._name(node.target)
            if name not in self.variables:
                raise _Rejected("undefined_name")
            self._assign(node.target, self.binary(
                node.op, self.variables[name], self.expression(node.value)
            ))
        else:
            raise _Rejected("unsupported_statement")

    def expression(self, node):
        if isinstance(node, ast.Constant):
            return _number(node.value)
        if isinstance(node, ast.Name):
            name = self._name(node)
            if name not in self.variables:
                raise _Rejected("undefined_name")
            return self.variables[name]
        if isinstance(node, ast.UnaryOp):
            value = _number(self.expression(node.operand))
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return _number(-value)
            raise _Rejected("unsupported_operator")
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.expression(node.left), self.expression(node.right))
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > MAX_SEQUENCE_ITEMS:
                raise _Rejected("sequence_limit")
            return tuple(_number(self.expression(child)) for child in node.elts)
        if isinstance(node, ast.Call):
            return self.call(node)
        raise _Rejected("unsupported_expression")

    def binary(self, operation, left, right):
        left, right = _number(left), _number(right)
        if isinstance(operation, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise _Rejected("exponent_limit")
            # Bound work *before* evaluating integer exponentiation. All other
            # integer operations have at most twice the operand bit limit.
            if type(left) is int and type(right) is int and right >= 0:
                if abs(left) > 1 and left.bit_length() * right > MAX_INTEGER_BITS:
                    raise _Rejected("integer_limit")
            return _number(pow(left, right))
        function = _BINARY.get(type(operation))
        if function is None:
            raise _Rejected("unsupported_operator")
        return _number(function(left, right))

    def call(self, node):
        name = self._name(node.func)
        if name not in _FUNCTIONS:
            raise _Rejected("unsupported_function")
        args = [self.expression(arg) for arg in node.args]
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg in kwargs:
                raise _Rejected("unsupported_arguments")
            kwargs[keyword.arg] = self.expression(keyword.value)
        if name == "abs" and len(args) == 1 and not kwargs:
            return _number(abs(_number(args[0])))
        if name == "round" and 1 <= len(args) <= 2:
            if kwargs and (set(kwargs) != {"ndigits"} or len(args) == 2):
                raise _Rejected("unsupported_arguments")
            digits = args[1] if len(args) == 2 else kwargs.get("ndigits", 0)
            if abs(_integer(digits)) > MAX_ROUND_DIGITS:
                raise _Rejected("round_limit")
            return _number(round(_number(args[0]), digits))
        if name in {"min", "max"} and args and not kwargs:
            values = args[0] if len(args) == 1 else args
            if not isinstance(values, (tuple, list)) or not values:
                raise _Rejected("sequence_required")
            return _number((min if name == "min" else max)([_number(v) for v in values]))
        if name == "sum" and 1 <= len(args) <= 2:
            if kwargs and (set(kwargs) != {"start"} or len(args) == 2):
                raise _Rejected("unsupported_arguments")
            if not isinstance(args[0], tuple):
                raise _Rejected("sequence_required")
            total = _number(args[1] if len(args) == 2 else kwargs.get("start", 0))
            for value in args[0]:
                total = _number(total + _number(value))
            return total
        if name == "pow" and len(args) == 2 and not kwargs:
            return self.binary(ast.Pow(), args[0], args[1])
        raise _Rejected("unsupported_arguments")


def evaluate_program_detailed(source: str) -> ProgramEvaluation:
    """Interpret a bounded numeric program; failures never produce a value.

    Resource bounds cover source size, AST size/depth, statement and sequence
    counts, integer size, exponents and rounding. There are no loops, recursion,
    attributes, indexing, imports, comprehensions, or user-defined functions.
    As a result, a program cannot invoke an unbounded operation or external I/O.
    """
    if not isinstance(source, str):
        return ProgramEvaluation(None, "empty_program")
    if len(source) > MAX_SOURCE_CHARS:
        return ProgramEvaluation(None, "source_limit")
    if not source.strip():
        return ProgramEvaluation(None, "empty_program")
    try:
        tree = ast.parse(source, mode="exec")
        _check_tree(tree)
        if len(tree.body) > MAX_STATEMENTS:
            raise _Rejected("statement_limit")
        interpreter = _Interpreter()
        for statement in tree.body:
            interpreter.statement(statement)
        if "result" not in interpreter.variables:
            raise _Rejected("missing_result")
        result = float(_number(interpreter.variables["result"]))
        if not math.isfinite(result):
            raise _Rejected("non_finite")
        return ProgramEvaluation(result)
    except _Rejected as error:
        return ProgramEvaluation(None, str(error))
    except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError):
        return ProgramEvaluation(None, "invalid_arithmetic_or_syntax")


def evaluate_program(source: str) -> float | None:
    """Return the final finite numeric result, or None on any rejection."""
    return evaluate_program_detailed(source).value
