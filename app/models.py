"""
Pydantic models shared across the application.

Defines request / response schemas for the API and internal data
transfer objects used between pipeline stages.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    """Classification label emitted by the Router agent."""
    TEXT = "text"
    STRUCTURED = "structured"
    HYBRID = "hybrid"


class EvalVerdict(str, Enum):
    """Verdict produced by the Evaluator agent."""
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


# ── Request / Response ───────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Payload accepted by ``POST /query``."""
    query: str = Field(..., min_length=1, max_length=2000, description="User question")


class SourceReference(BaseModel):
    """A single source backing a claim in the answer."""
    source: str = Field(..., description="Document title or dataset name")
    chunk_id: str | None = Field(None, description="Chunk identifier when applicable")
    relevance_score: float | None = Field(None, ge=0.0, le=1.0)
    text_snippet: str = Field("", description="Relevant excerpt")


class LatencyBreakdown(BaseModel):
    """Timing (ms) for every pipeline stage."""
    routing_ms: float = 0.0
    retrieval_ms: float = 0.0
    dynamic_ingestion_ms: float = 0.0
    reasoning_ms: float = 0.0
    evaluation_ms: float = 0.0
    total_ms: float = 0.0


class EvaluationResult(BaseModel):
    """Output of the Evaluator agent."""
    verdict: EvalVerdict = EvalVerdict.SUPPORTED
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Payload returned by ``POST /query``."""
    answer: str
    query_type: QueryType
    sources: list[SourceReference] = Field(default_factory=list)
    evaluation: EvaluationResult = Field(default_factory=EvaluationResult)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    cached: bool = False


# ── Internal DTOs ────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A chunk returned by a retrieval agent."""
    text: str
    source: str
    chunk_id: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingDecision(BaseModel):
    """Output of the Router agent."""
    query_type: QueryType
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = ""
