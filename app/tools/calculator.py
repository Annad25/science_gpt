"""
Calculator tool for numeric computations.

Evaluates safe mathematical expressions using Python's ``ast`` module
to prevent code-injection attacks.  Only arithmetic operators, common
math functions, and numeric literals are allowed.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool

from app.logging_cfg import get_logger

logger = get_logger(__name__)

# Whitelisted operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelisted math functions
_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node using only whitelisted operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _OPERATORS[op_type](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            args = [_safe_eval(a) for a in node.args]
            return _FUNCTIONS[node.func.id](*args)
        raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")
    if isinstance(node, ast.Name):
        if node.id in _FUNCTIONS:
            return _FUNCTIONS[node.id]
        raise ValueError(f"Unsupported name: {node.id}")
    raise ValueError(f"Unsupported node type: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports arithmetic (+, -, *, /, //, %, **) and common math functions
    (sqrt, log, exp, ceil, floor, abs, round, min, max, sum, pow).

    Args:
        expression: A mathematical expression string, e.g. "sqrt(144) + 3 * 2".

    Returns:
        The numeric result as a string, or an error message.
    """
    logger.info("[Calculator] Evaluating: %s", expression)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        logger.info("[Calculator] Result: %s", result)
        return str(result)
    except Exception as exc:
        error = f"Calculation error: {exc}"
        logger.warning("[Calculator] %s", error)
        return error
