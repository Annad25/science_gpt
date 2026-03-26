"""
Systematic evaluation harness (Level 3).

Runs a predefined set of test questions through the pipeline and
scores each answer using the Evaluator agent.  Produces a JSON
report with per-question verdicts, aggregate pass rate, and
latency statistics.

Usage (standalone):
    python -m app.evaluation.systematic
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from app.agents.evaluator import EvaluatorAgent
from app.agents.reasoner import ReasoningAgent
from app.agents.retriever import (
    StructuredRetriever,
    VectorRetriever,
    run_retrieval,
)
from app.agents.router import RouterAgent
from app.config import get_settings
from app.embeddings.encoder import EmbeddingEncoder
from app.ingestion.pipeline import run_ingestion
from app.ingestion.who_loader import WHODataStore
from app.logging_cfg import get_logger, setup_logging
from app.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)

# Evaluation question bank — covers text, structured, and hybrid queries.
EVAL_QUESTIONS: list[dict[str, str]] = [
    {
        "query": "What is machine learning and how is it applied in healthcare?",
        "expected_type": "text",
    },
    {
        "query": "What are the latest findings on climate change impacts?",
        "expected_type": "text",
    },
    {
        "query": "What is the life expectancy in Japan?",
        "expected_type": "structured",
    },
    {
        "query": "Which country has the highest mortality rate?",
        "expected_type": "structured",
    },
    {
        "query": "How does AI research relate to health outcomes in developed countries?",
        "expected_type": "hybrid",
    },
    {
        "query": "Explain the transformer architecture in neural networks.",
        "expected_type": "text",
    },
    {
        "query": "Compare health expenditure across G7 countries.",
        "expected_type": "structured",
    },
    {
        "query": "What does recent research say about the relationship between air pollution and respiratory diseases, and what are the WHO statistics?",
        "expected_type": "hybrid",
    },
]


async def run_evaluation() -> dict[str, Any]:
    """Execute the full evaluation suite.

    Returns:
        Dict with ``results`` (per-question), ``summary`` (aggregates),
        and ``latencies``.
    """
    setup_logging("INFO")
    logger.info("── Starting systematic evaluation ──")
    settings = get_settings()

    # Initialise pipeline components
    encoder = EmbeddingEncoder()
    qdrant = QdrantStore()
    await qdrant.ensure_collection()
    who_store = WHODataStore()

    await run_ingestion(qdrant=qdrant, encoder=encoder, who_store=who_store)

    router = RouterAgent(who_columns=who_store.columns)
    vector_ret = VectorRetriever(qdrant, encoder)
    structured_ret = StructuredRetriever(who_store)
    reasoner = ReasoningAgent()
    evaluator = EvaluatorAgent()

    results: list[dict[str, Any]] = []
    latencies: list[float] = []

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        query = q["query"]
        logger.info("[Eval %d/%d] %s", i, len(EVAL_QUESTIONS), query)
        t0 = time.perf_counter()

        try:
            routing = await router.route(query)
            chunks = await run_retrieval(
                query, routing.query_type, vector_ret, structured_ret
            )
            answer = await reasoner.reason(query, chunks)
            evaluation = await evaluator.evaluate(query, answer, chunks)

            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

            results.append({
                "query": query,
                "expected_type": q["expected_type"],
                "routed_type": routing.query_type.value,
                "routing_correct": routing.query_type.value == q["expected_type"],
                "answer_preview": answer[:300],
                "verdict": evaluation.verdict.value,
                "confidence": evaluation.confidence,
                "issues": evaluation.issues,
                "latency_ms": round(elapsed, 1),
                "num_sources": len(chunks),
            })
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            results.append({
                "query": query,
                "error": str(exc),
                "latency_ms": round(elapsed, 1),
            })
            logger.exception("[Eval %d] Failed", i)

    # Aggregate
    supported = sum(1 for r in results if r.get("verdict") == "supported")
    partial = sum(1 for r in results if r.get("verdict") == "partially_supported")
    routing_correct = sum(1 for r in results if r.get("routing_correct"))
    total = len(results)

    summary = {
        "total_questions": total,
        "fully_supported": supported,
        "partially_supported": partial,
        "not_supported": total - supported - partial,
        "pass_rate": round(supported / total, 2) if total else 0,
        "routing_accuracy": round(routing_correct / total, 2) if total else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
    }

    report = {"results": results, "summary": summary}
    logger.info("── Evaluation complete: %s ──", summary)
    return report


if __name__ == "__main__":
    report = asyncio.run(run_evaluation())
    print(json.dumps(report, indent=2))
