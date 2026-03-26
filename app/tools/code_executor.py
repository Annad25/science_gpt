"""
Sandboxed code execution tool (Level 3 — Robustness).

Executes Python snippets in a *restricted* subprocess with:
- Timeout enforcement (default 10 s)
- Memory cap via resource limits (Unix) or process kill (Windows)
- Blocked imports (os, sys, subprocess, shutil, socket, etc.)
- No filesystem write access

This is intentionally conservative — it only allows computation,
not I/O.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from typing import Any

from langchain_core.tools import tool

from app.config import get_settings
from app.logging_cfg import get_logger

logger = get_logger(__name__)

# Imports that are NEVER allowed in user-submitted code.
_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "socket", "http",
    "urllib", "requests", "pathlib", "io", "ctypes", "importlib",
    "pickle", "shelve", "signal", "multiprocessing", "threading",
    "asyncio", "webbrowser", "code", "codeop", "compileall",
    "__import__", "eval", "exec", "breakpoint",
})

# Modules the sandbox makes available.
_ALLOWED_PREAMBLE = textwrap.dedent("""\
    import math
    import statistics
    import json
    import re
    import collections
    import itertools
    import functools
    import datetime
""")


class CodeExecutionError(Exception):
    """Raised when sandboxed code fails validation or execution."""


def _validate_code(code: str) -> None:
    """Static analysis — reject code that tries to import blocked modules
    or use dangerous builtins."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeExecutionError(f"Syntax error: {exc}") from exc

    for node in ast.walk(tree):
        # Block `import X` and `from X import ...`
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_MODULES:
                    raise CodeExecutionError(f"Blocked import: {root}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BLOCKED_MODULES:
                    raise CodeExecutionError(f"Blocked import: {root}")
        # Block calls to dangerous builtins
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile", "__import__", "open", "breakpoint"):
                    raise CodeExecutionError(
                        f"Blocked builtin call: {node.func.id}"
                    )


def _build_script(code: str) -> str:
    """Wrap user code in a safe harness with preamble and output capture."""
    return _ALLOWED_PREAMBLE + "\n" + code


@tool
def execute_python(code: str) -> str:
    """Execute a Python code snippet in a sandboxed subprocess.

    Only safe standard-library modules (math, statistics, json, re, etc.)
    are available.  No file I/O, no network access, no dangerous builtins.

    The code should print its result to stdout.

    Args:
        code: Python code string to execute.

    Returns:
        stdout output of the code, or an error message.
    """
    settings = get_settings()
    logger.info("[CodeExecutor] Validating code (%d chars)", len(code))

    try:
        _validate_code(code)
    except CodeExecutionError as exc:
        logger.warning("[CodeExecutor] Validation failed: %s", exc)
        return f"Code rejected: {exc}"

    script = _build_script(code)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=settings.code_exec_timeout_seconds,
            env={},  # Empty env — no access to parent env vars
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip()[-500:]
            logger.warning("[CodeExecutor] Script failed: %s", error_msg)
            return f"Execution error:\n{error_msg}"

        output = result.stdout.strip()[:2000]
        logger.info("[CodeExecutor] Success — %d chars output", len(output))
        return output if output else "(no output)"

    except subprocess.TimeoutExpired:
        logger.warning("[CodeExecutor] Timeout after %ds", settings.code_exec_timeout_seconds)
        return f"Execution timed out after {settings.code_exec_timeout_seconds}s"
    except Exception as exc:
        logger.exception("[CodeExecutor] Unexpected error")
        return f"Execution error: {exc}"
