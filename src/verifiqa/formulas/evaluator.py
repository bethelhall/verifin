from __future__ import annotations

import ast
import re
from typing import Dict, Iterable, List, Set


class FormulaError(ValueError):
    pass


def result_is_money(formula: str, money_variables: Set[str]) -> bool:
    """Dimensional inference: does this formula's result have a monetary dimension?

    Money variables carry a $ dimension; constants are dimensionless. Money/money
    cancels to a ratio; money*scalar or money±money stays money. Used to tell a
    genuine dollar answer (keep/label as money) from a ratio, without keyword
    guessing. Returns False when the dimension can't be determined.
    """
    try:
        tree = ast.parse(normalize_formula(formula), mode="eval")
    except SyntaxError:
        return False
    return _money_dim(tree.body, set(money_variables)) == "money"


def _money_dim(node: ast.AST, money_vars: Set[str]) -> str:
    if isinstance(node, ast.Name):
        return "money" if node.id in money_vars else "unknown"
    if isinstance(node, ast.Constant):
        return "scalar"
    if isinstance(node, ast.UnaryOp):
        return _money_dim(node.operand, money_vars)
    if isinstance(node, ast.BinOp):
        left = _money_dim(node.left, money_vars)
        right = _money_dim(node.right, money_vars)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if "money" in (left, right):
                return "money"
            return "scalar" if left == right == "scalar" else "unknown"
        if isinstance(node.op, ast.Mult):
            if "money" in (left, right) and {left, right} <= {"money", "scalar"}:
                return "money"
            return "scalar" if left == right == "scalar" else "unknown"
        if isinstance(node.op, ast.Div):
            if left == "money" and right == "money":
                return "scalar"
            if left == "money" and right == "scalar":
                return "money"
            return "scalar" if left == right == "scalar" else "unknown"
    return "unknown"


def formula_variables(formula: str) -> Set[str]:
    tree = ast.parse(normalize_formula(formula), mode="eval")
    variables: Set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            variables.add(node.id)
            return
        if isinstance(node, ast.Call):
            for arg in node.args:
                visit(arg)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree.body)
    return variables


def formula_constants(formula: str) -> List[float]:
    normalized = normalize_formula(formula)
    tree = ast.parse(normalized, mode="eval")
    constants = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            constants.append(float(node.value))
    cagr = _cagr_call(tree.body)
    if cagr is not None:
        constants.append(1.0)
        if cagr == "cagr_percent":
            constants.append(100.0)
            constants.append(-100.0)
        else:
            constants.append(-1.0)
    if _coverage_ratio_call(tree.body) is not None:
        constants.append(0.0)
    return constants


def evaluate_formula(formula: str, values: Dict[str, float]) -> float:
    tree = ast.parse(normalize_formula(formula), mode="eval")
    return float(_eval_node(tree.body, values))


def normalize_formula(formula: str, output_unit: str = "") -> str:
    """Canonicalize verifier formulas before parsing.

    The formalizer is allowed to describe CAGR using analyst notation such as
    ``(end / start) ^ (1 / years) - 1``. Python parses ``^`` as bitwise XOR,
    and SMT-LIB has no direct fractional exponent in linear arithmetic. We
    canonicalize that family into explicit functions that the verifier encodes
    algebraically.
    """
    text = str(formula or "").strip()
    if not text:
        return text

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        tree = None
    if tree is not None and _cagr_call(tree.body) is not None:
        return text

    compact = re.sub(r"\s+", "", text)
    compact = _strip_outer_parens(compact)
    percent_multiplier = False
    if compact.endswith("*100"):
        percent_multiplier = True
        compact = _strip_outer_parens(compact[:-4])
    elif compact.startswith("100*"):
        percent_multiplier = True
        compact = _strip_outer_parens(compact[4:])

    compact = _strip_outer_parens(compact)
    match = re.fullmatch(
        r"\(?([A-Za-z_][A-Za-z0-9_]*)/([A-Za-z_][A-Za-z0-9_]*)\)?"
        r"(?:\^|\*\*)"
        r"\(?(?:1/([0-9]+)|0\.5)\)?"
        r"-1",
        compact,
    )
    if match:
        end_var, start_var, years = match.groups()
        years_value = int(years or 2)
        function = (
            "cagr_percent"
            if percent_multiplier or output_unit in {"percent", "percentage_points"}
            else "cagr"
        )
        return f"{function}({end_var}, {start_var}, {years_value})"

    return text


def formula_assertion_smt(computed_variable: str, formula: str, computed_unit: str = "") -> str:
    normalized = normalize_formula(formula, computed_unit)
    tree = ast.parse(normalized, mode="eval")
    coverage = _coverage_ratio_call(tree.body)
    if coverage is not None:
        numerator, denominator = _coverage_ratio_args(tree.body)
        return (
            f"(and (=> (<= {numerator} 0) (= {computed_variable} 0)) "
            f"(=> (> {numerator} 0) (= {computed_variable} (/ {numerator} {denominator}))))"
        )

    cagr = _cagr_call(tree.body)
    if cagr is None:
        return f"(= {computed_variable} {_node_to_smt(tree.body)})"

    end_expr, start_expr, years = _cagr_args(tree.body)
    base = (
        f"(+ 1 (/ {computed_variable} 100))"
        if cagr == "cagr_percent"
        else f"(+ 1 {computed_variable})"
    )
    product_terms = [start_expr] + [base for _ in range(years)]
    return f"(= (* {' '.join(product_terms)}) {end_expr})"


def formula_domain_constraints_smt(
    computed_variable: str,
    formula: str,
    computed_unit: str = "",
) -> List[str]:
    normalized = normalize_formula(formula, computed_unit)
    tree = ast.parse(normalized, mode="eval")
    constraints = _division_domain_constraints(tree.body)
    coverage = _coverage_ratio_call(tree.body)
    if coverage is not None:
        _numerator, denominator = _coverage_ratio_args(tree.body)
        constraints.append(f"(or (> {denominator} 0) (< {denominator} 0))")
    cagr = _cagr_call(tree.body)
    if cagr == "cagr_percent":
        constraints.append(f"(> {computed_variable} -100)")
    if cagr == "cagr":
        constraints.append(f"(> {computed_variable} -1)")
    return constraints


def _division_domain_constraints(node: ast.AST) -> List[str]:
    constraints: List[str] = []

    def visit(current: ast.AST) -> None:
        if isinstance(current, ast.BinOp):
            visit(current.left)
            visit(current.right)
            if isinstance(current.op, ast.Div) and not _is_nonzero_numeric_constant(current.right):
                denominator = _node_to_smt(current.right)
                constraints.append(f"(or (> {denominator} 0) (< {denominator} 0))")
        elif isinstance(current, ast.UnaryOp):
            visit(current.operand)
        elif isinstance(current, ast.Call):
            for arg in current.args:
                visit(arg)

    visit(node)
    return constraints


def _is_nonzero_numeric_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and float(node.value) != 0.0


def _eval_node(node: ast.AST, values: Dict[str, float]) -> float:
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise FormulaError(f"Missing value for variable {node.id}")
        return float(values[node.id])
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand, values)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, values)
        right = _eval_node(node.right, values)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise FormulaError("Division by zero")
            return left / right
    if isinstance(node, ast.Call):
        coverage = _coverage_ratio_call(node)
        if coverage is not None:
            numerator = _eval_node(node.args[0], values)
            denominator = _eval_node(node.args[1], values)
            if numerator <= 0:
                return 0.0
            if denominator == 0:
                raise FormulaError("Division by zero")
            return numerator / denominator
        function = _cagr_call(node)
        if function is not None:
            end = _eval_node(node.args[0], values)
            start = _eval_node(node.args[1], values)
            years = _constant_positive_int(node.args[2])
            if start == 0:
                raise FormulaError("Division by zero")
            result = (end / start) ** (1.0 / years) - 1.0
            return result * 100.0 if function == "cagr_percent" else result
    raise FormulaError(f"Unsupported formula syntax: {ast.dump(node)}")


def formula_to_smt(formula: str) -> str:
    normalized = normalize_formula(formula)
    tree = ast.parse(normalized, mode="eval")
    if _cagr_call(tree.body) is not None:
        raise FormulaError("CAGR formula requires formula_assertion_smt")
    if _coverage_ratio_call(tree.body) is not None:
        raise FormulaError("Coverage-ratio formula requires formula_assertion_smt")
    return _node_to_smt(tree.body)


def _node_to_smt(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _format_number(float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(- {_node_to_smt(node.operand)})"
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            return _flatten_nary("+", node)
        if isinstance(node.op, ast.Mult):
            return _flatten_nary("*", node)
        if isinstance(node.op, ast.Sub):
            return f"(- {_node_to_smt(node.left)} {_node_to_smt(node.right)})"
        if isinstance(node.op, ast.Div):
            return f"(/ {_node_to_smt(node.left)} {_node_to_smt(node.right)})"
    raise FormulaError(f"Unsupported formula syntax: {ast.dump(node)}")


def _coverage_ratio_call(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id != "coverage_ratio":
        return None
    if len(node.args) != 2:
        raise FormulaError("coverage_ratio_requires_two_args")
    return node.func.id


def _coverage_ratio_args(node: ast.Call) -> tuple[str, str]:
    return (_node_to_smt(node.args[0]), _node_to_smt(node.args[1]))


def _cagr_call(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id not in {"cagr", "cagr_percent"}:
        return None
    if len(node.args) != 3:
        raise FormulaError(f"{node.func.id}_requires_three_args")
    _constant_positive_int(node.args[2])
    return node.func.id


def _cagr_args(node: ast.Call) -> tuple[str, str, int]:
    return (
        _node_to_smt(node.args[0]),
        _node_to_smt(node.args[1]),
        _constant_positive_int(node.args[2]),
    )


def _constant_positive_int(node: ast.AST) -> int:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, (int, float)):
        raise FormulaError("cagr_years_must_be_constant")
    value = float(node.value)
    if not value.is_integer() or value <= 0:
        raise FormulaError("cagr_years_must_be_positive_integer")
    return int(value)


def _strip_outer_parens(value: str) -> str:
    text = value
    while text.startswith("(") and text.endswith(")") and _outer_parens_wrap(text):
        text = text[1:-1]
    return text


def _outer_parens_wrap(value: str) -> bool:
    depth = 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(value) - 1:
                return False
            if depth < 0:
                return False
    return depth == 0


def _flatten_nary(operator: str, node: ast.BinOp) -> str:
    parts: List[str] = []
    _collect_nary(operator, node, parts)
    return f"({operator} {' '.join(parts)})"


def _collect_nary(operator: str, node: ast.AST, parts: List[str]) -> None:
    op_class = ast.Add if operator == "+" else ast.Mult
    if isinstance(node, ast.BinOp) and isinstance(node.op, op_class):
        _collect_nary(operator, node.left, parts)
        _collect_nary(operator, node.right, parts)
    else:
        parts.append(_node_to_smt(node))


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    if 0 < abs(value) < 1e-4:
        return f"{value:.18f}".rstrip("0").rstrip(".")
    return f"{value:.17g}"


def format_number(value: float) -> str:
    return _format_number(float(value))


def normalize_smt(value: str) -> str:
    return "".join(value.split())


def required_values(values: Dict[str, float], names: Iterable[str]) -> Dict[str, float]:
    missing = [name for name in names if name not in values]
    if missing:
        raise FormulaError(f"Missing variables: {', '.join(missing)}")
    return {name: float(values[name]) for name in names}
