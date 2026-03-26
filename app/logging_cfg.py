"""
Structured logging configuration.

Provides a consistent JSON-ish log format with trace IDs so that every
request can be followed across pipeline stages.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

# Per-request trace ID propagated via contextvars.
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="no-trace")


class TraceFormatter(logging.Formatter):
    """Formatter that injects the current ``trace_id`` into every record."""

    def format(self, record: logging.LogRecord) -> str:
        record.trace_id = trace_id_var.get()  # type: ignore[attr-defined]
        return super().format(record)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with structured output to *stderr*."""
    handler = logging.StreamHandler(sys.stderr)
    fmt = TraceFormatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(trace_id)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (child of root)."""
    return logging.getLogger(name)
