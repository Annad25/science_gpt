"""
FastAPI application — the single entry-point for the Science GPT service.

Exposes:
- ``POST /query``     — main question-answering endpoint
- ``POST /ingest``    — trigger data re-ingestion
- ``POST /cache/invalidate`` — manual cache bust
- ``GET  /health``    — liveness check
- ``GET  /stats``     — pipeline statistics
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agents.entity_linker import EntityLinker
from app.agents.evaluator import EvaluatorAgent
from app.agents.reasoner import ReasoningAgent
from app.agents.retriever import (
    StructuredRetriever,
    VectorRetriever,
    run_retrieval,
)
from app.agents.router import RouterAgent
from app.cache import QueryCache
from app.config import get_settings
from app.embeddings.encoder import EmbeddingEncoder
from app.evaluation.benchmark import (
    BenchmarkRequest,
    BenchmarkResult,
    QuestionResult,
    compute_benchmark_metrics,
    DEFAULT_BENCHMARK_QUESTIONS,
)
from app.ingestion.dynamic import dynamic_ingest_and_retry
from app.ingestion.pipeline import run_ingestion
from app.ingestion.who_loader import WHODataStore
from app.logging_cfg import get_logger, setup_logging, trace_id_var
from app.models import (
    EvaluationResult,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
    QueryType,
    SourceReference,
)
from app.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


# ── Shared state (initialised in lifespan) ───────────────────────────────────

class AppState:
    """Mutable container for services initialised at startup."""

    encoder: EmbeddingEncoder
    qdrant: QdrantStore
    who_store: WHODataStore
    router: RouterAgent
    vector_retriever: VectorRetriever
    structured_retriever: StructuredRetriever
    entity_linker: EntityLinker
    reasoner: ReasoningAgent
    evaluator: EvaluatorAgent
    cache: QueryCache
    ingestion_stats: dict[str, int]


state = AppState()


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise all services and run ingestion pipeline."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("── Science GPT starting up ──")

    # Core services
    state.encoder = EmbeddingEncoder()
    state.qdrant = QdrantStore()
    await state.qdrant.ensure_collection()
    state.who_store = WHODataStore()

    # Run ingestion
    try:
        state.ingestion_stats = await run_ingestion(
            qdrant=state.qdrant,
            encoder=state.encoder,
            who_store=state.who_store,
        )
    except Exception:
        logger.exception("Ingestion failed — service will start with empty data")
        state.ingestion_stats = {}
        # Still load WHO data if possible
        try:
            await state.who_store.load()
        except Exception:
            logger.exception("WHO data also failed to load")

    # Agents
    state.router = RouterAgent(
        who_columns=state.who_store.columns if state.who_store._df is not None else []
    )
    state.vector_retriever = VectorRetriever(state.qdrant, state.encoder)
    state.structured_retriever = StructuredRetriever(state.who_store)
    state.entity_linker = EntityLinker(state.who_store)
    state.reasoner = ReasoningAgent()
    state.evaluator = EvaluatorAgent()
    state.cache = QueryCache()

    logger.info("── Science GPT ready ──")
    yield
    logger.info("── Science GPT shutting down ──")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Science GPT",
    description="Multi-Agent RAG Pipeline over Scientific Data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Answer a scientific question using the multi-agent RAG pipeline.

    Pipeline stages: Router → Retriever(s) → Reasoner → Evaluator.
    Each stage is timed and the latency breakdown is included in the
    response.
    """
    trace_id = str(uuid.uuid4())[:8]
    trace_id_var.set(trace_id)
    logger.info("[API] Received query: '%s'", req.query[:120])
    pipeline_start = time.perf_counter()

    # ── Cache check ──────────────────────────────────────────────────
    cached = await state.cache.get(req.query)
    if cached is not None:
        logger.info("[API] Returning cached response")
        return QueryResponse(**cached, cached=True)

    try:
        settings = get_settings()

        # ── Stage 1: Routing ─────────────────────────────────────────
        t0 = time.perf_counter()
        routing = await state.router.route(req.query)
        routing_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 2: Retrieval ───────────────────────────────────────
        t0 = time.perf_counter()
        chunks = await run_retrieval(
            query=req.query,
            query_type=routing.query_type,
            vector_retriever=state.vector_retriever,
            structured_retriever=state.structured_retriever,
        )

        # ── Stage 2b: Dynamic ingestion if relevance is low ─────────
        dynamic_ms = 0.0
        if chunks and routing.query_type in (QueryType.TEXT, QueryType.HYBRID):
            # Check if the vector results are relevant enough
            vector_chunks = [c for c in chunks if c.source.startswith("arXiv:")]
            avg = state.vector_retriever.avg_score(vector_chunks)
            threshold = settings.dynamic_ingestion_score_threshold

            if avg < threshold and avg > 0:
                logger.info(
                    "[API] Low relevance detected (avg=%.3f < threshold=%.3f) — triggering dynamic ingestion",
                    avg,
                    threshold,
                )
                t_dyn = time.perf_counter()
                dynamic_chunks = await dynamic_ingest_and_retry(
                    query=req.query,
                    qdrant=state.qdrant,
                    encoder=state.encoder,
                    top_k=settings.retrieval_top_k,
                )
                dynamic_ms = (time.perf_counter() - t_dyn) * 1000

                if dynamic_chunks:
                    # Replace the low-relevance vector chunks with the fresh ones
                    non_vector = [c for c in chunks if not c.source.startswith("arXiv:")]
                    chunks = dynamic_chunks + non_vector
                    logger.info(
                        "[API] Replaced with %d dynamic chunks (took %.0fms)",
                        len(dynamic_chunks),
                        dynamic_ms,
                    )

        retrieval_ms = (time.perf_counter() - t0) * 1000

        if not chunks:
            logger.warning("[API] No context retrieved — returning fallback")
            return QueryResponse(
                answer="This question cannot be answered from the available content.",
                query_type=routing.query_type,
                sources=[],
                evaluation=EvaluationResult(verdict="supported", confidence=1.0, issues=[]),
                latency=LatencyBreakdown(
                    routing_ms=routing_ms,
                    retrieval_ms=retrieval_ms,
                    total_ms=(time.perf_counter() - pipeline_start) * 1000,
                ),
            )

        # ── Stage 2c: Entity linking (cross-source traversal) ────────
        try:
            chunks = await state.entity_linker.enrich_with_structured_data(
                chunks, req.query
            )
        except Exception:
            logger.warning("[API] Entity linking failed — continuing without enrichment")

        # ── Stage 3: Reasoning ───────────────────────────────────────
        t0 = time.perf_counter()
        answer = await state.reasoner.reason(req.query, chunks)
        reasoning_ms = (time.perf_counter() - t0) * 1000

        # ── Stage 4: Evaluation ──────────────────────────────────────
        t0 = time.perf_counter()
        evaluation = await state.evaluator.evaluate(req.query, answer, chunks)
        evaluation_ms = (time.perf_counter() - t0) * 1000

        # If answer is not supported, return a safe fallback
        if evaluation.verdict == "not_supported":
            logger.warning("[API] Evaluator rejected answer — returning safe fallback")
            answer = (
                "I'm sorry, I cannot provide a reliable answer from the given sources. "
                f"Issues found: {'; '.join(evaluation.issues)}"
            )

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        # Build response
        sources = [
            SourceReference(
                source=c.source,
                chunk_id=c.chunk_id,
                relevance_score=c.score,
                text_snippet=c.text[:200],
            )
            for c in chunks
        ]

        response = QueryResponse(
            answer=answer,
            query_type=routing.query_type,
            sources=sources,
            evaluation=evaluation,
            latency=LatencyBreakdown(
                routing_ms=round(routing_ms, 1),
                retrieval_ms=round(retrieval_ms, 1),
                dynamic_ingestion_ms=round(dynamic_ms, 1),
                reasoning_ms=round(reasoning_ms, 1),
                evaluation_ms=round(evaluation_ms, 1),
                total_ms=round(total_ms, 1),
            ),
        )

        # Cache the result
        await state.cache.put(req.query, response.model_dump(exclude={"cached"}))

        logger.info(
            "[API] Response ready — total=%.0fms (route=%.0f, retrieve=%.0f, reason=%.0f, eval=%.0f)",
            total_ms, routing_ms, retrieval_ms, reasoning_ms, evaluation_ms,
        )
        return response

    except Exception:
        logger.exception("[API] Pipeline error for query: '%s'", req.query[:80])
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your query.",
        )


@app.post("/ingest")
async def ingest_endpoint() -> dict[str, Any]:
    """Re-run the ingestion pipeline and invalidate the cache."""
    trace_id_var.set(f"ingest-{uuid.uuid4().hex[:6]}")
    logger.info("[API] Manual ingestion triggered")

    try:
        await state.cache.invalidate()
        stats = await run_ingestion(
            qdrant=state.qdrant,
            encoder=state.encoder,
            who_store=state.who_store,
        )
        state.ingestion_stats = stats
        return {"status": "ok", "stats": stats}
    except Exception:
        logger.exception("[API] Re-ingestion failed")
        raise HTTPException(status_code=500, detail="Ingestion failed")


@app.post("/cache/invalidate")
async def cache_invalidate(query: str | None = None) -> dict[str, Any]:
    """Invalidate a specific cached query or the entire cache."""
    removed = await state.cache.invalidate(query)
    return {"removed": removed}


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/stats")
async def stats() -> dict[str, Any]:
    """Return pipeline statistics."""
    vector_count = 0
    try:
        vector_count = await state.qdrant.count()
    except Exception:
        pass

    return {
        "ingestion": state.ingestion_stats,
        "vectors_in_store": vector_count,
        "cache_size": state.cache.size,
    }


# ── Systematic evaluation endpoint (Level 3) ────────────────────────

_REFUSAL_PHRASES = {
    "cannot be answered from the available content",
    "i don't have enough information",
    "could not find relevant information",
}


@app.post("/evaluate", response_model=BenchmarkResult)
async def evaluate_endpoint(req: BenchmarkRequest | None = None) -> BenchmarkResult:
    """Run the pipeline on a benchmark question set and return aggregate metrics.

    If no questions are provided, the default benchmark set is used.
    This endpoint is for systematic quality evaluation (Level 3).
    """
    trace_id_var.set(f"eval-{uuid.uuid4().hex[:6]}")
    logger.info("[Eval] Starting systematic evaluation")

    questions = (
        [{"query": q.query, "expected_type": q.expected_type} for q in req.questions]
        if req and req.questions
        else DEFAULT_BENCHMARK_QUESTIONS
    )

    settings = get_settings()
    results: list[QuestionResult] = []

    for i, q in enumerate(questions, 1):
        query = q["query"]
        expected = q.get("expected_type", "")
        logger.info("[Eval %d/%d] %s", i, len(questions), query[:80])
        t0 = time.perf_counter()

        try:
            # Run the full pipeline
            routing = await state.router.route(query)

            chunks = await run_retrieval(
                query=query,
                query_type=routing.query_type,
                vector_retriever=state.vector_retriever,
                structured_retriever=state.structured_retriever,
            )

            # Dynamic ingestion check
            dynamic_triggered = False
            if chunks and routing.query_type in (QueryType.TEXT, QueryType.HYBRID):
                vector_chunks = [c for c in chunks if c.source.startswith("arXiv:")]
                avg = state.vector_retriever.avg_score(vector_chunks)
                if avg < settings.dynamic_ingestion_score_threshold and avg > 0:
                    dynamic_triggered = True
                    dynamic_chunks = await dynamic_ingest_and_retry(
                        query=query,
                        qdrant=state.qdrant,
                        encoder=state.encoder,
                        top_k=settings.retrieval_top_k,
                    )
                    if dynamic_chunks:
                        non_vector = [c for c in chunks if not c.source.startswith("arXiv:")]
                        chunks = dynamic_chunks + non_vector

            # Entity linking
            try:
                chunks = await state.entity_linker.enrich_with_structured_data(chunks, query)
            except Exception:
                pass

            if not chunks:
                answer = "This question cannot be answered from the available content."
                verdict = "supported"
                confidence = 1.0
            else:
                answer = await state.reasoner.reason(query, chunks)
                evaluation = await state.evaluator.evaluate(query, answer, chunks)
                verdict = evaluation.verdict.value
                confidence = evaluation.confidence

            elapsed = (time.perf_counter() - t0) * 1000
            answer_lower = answer.lower()
            is_refusal = any(p in answer_lower for p in _REFUSAL_PHRASES)

            avg_score = (
                sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
            )

            results.append(
                QuestionResult(
                    query=query,
                    expected_type=expected,
                    actual_type=routing.query_type.value,
                    answer_snippet=answer[:300],
                    verdict=verdict,
                    verdict_confidence=confidence,
                    avg_retrieval_score=round(avg_score, 3),
                    latency_ms=round(elapsed, 1),
                    dynamic_ingestion_triggered=dynamic_triggered,
                    is_refusal=is_refusal,
                )
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.exception("[Eval %d] Failed: %s", i, exc)
            results.append(
                QuestionResult(
                    query=query,
                    expected_type=expected,
                    actual_type="error",
                    answer_snippet=f"ERROR: {exc}",
                    verdict="not_supported",
                    verdict_confidence=0.0,
                    avg_retrieval_score=0.0,
                    latency_ms=round(elapsed, 1),
                    dynamic_ingestion_triggered=False,
                    is_refusal=False,
                )
            )

    benchmark = compute_benchmark_metrics(results)
    logger.info(
        "[Eval] Complete — support=%.1f%% partial=%.1f%% refusal=%.1f%% routing=%.1f%%",
        benchmark.support_rate,
        benchmark.partial_rate,
        benchmark.refusal_rate,
        benchmark.routing_accuracy,
    )
    return benchmark
