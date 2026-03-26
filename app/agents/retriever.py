"""
Retrieval Agents — vector and structured search.

Two independent agents that can run in parallel via ``asyncio.gather``:

* **VectorRetriever** — semantic similarity search over Qdrant.
* **StructuredRetriever** — filter/aggregate queries on the WHO DataFrame.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.embeddings.encoder import EmbeddingEncoder
from app.ingestion.who_loader import WHODataStore
from app.logging_cfg import get_logger
from app.models import QueryType, RetrievedChunk
from app.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


# ── Vector Retriever ─────────────────────────────────────────────────────────

class VectorRetriever:
    """Retrieves top-k semantically similar chunks from Qdrant."""

    def __init__(self, qdrant: QdrantStore, encoder: EmbeddingEncoder) -> None:
        self._qdrant = qdrant
        self._encoder = encoder
        self._settings = get_settings()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Embed the query and search Qdrant.

        Args:
            query: Natural-language question.
            top_k: Number of results (defaults to config value).
            min_score: Minimum similarity score.  Chunks below this
                threshold are filtered out to avoid feeding irrelevant
                context to the Reasoner.

        Returns:
            Ranked list of ``RetrievedChunk`` objects (may be empty if
            all results fall below ``min_score``).
        """
        logger.info("[VectorRetriever] Searching for: '%s'", query[:100])
        query_vec = await self._encoder.encode(query)
        hits = await self._qdrant.search(query_vector=query_vec, top_k=top_k)

        chunks: list[RetrievedChunk] = []
        for h in hits:
            score = h.get("score", 0.0)
            chunks.append(
                RetrievedChunk(
                    text=h["text"],
                    source=h["source"],
                    chunk_id=h.get("chunk_id", ""),
                    score=score,
                    metadata={
                        "title": h.get("title", ""),
                        "authors": h.get("payload", {}).get("authors", []),
                        "published": h.get("payload", {}).get("published", ""),
                    },
                )
            )

        # Log raw scores for observability
        raw_scores = [c.score for c in chunks]
        if raw_scores:
            logger.info(
                "[VectorRetriever] Score range: min=%.3f max=%.3f avg=%.3f",
                min(raw_scores),
                max(raw_scores),
                sum(raw_scores) / len(raw_scores),
            )

        # Filter by minimum relevance score
        if min_score is not None:
            before = len(chunks)
            chunks = [c for c in chunks if c.score >= min_score]
            if len(chunks) < before:
                logger.info(
                    "[VectorRetriever] Filtered %d→%d chunks (min_score=%.3f)",
                    before,
                    len(chunks),
                    min_score,
                )

        logger.info("[VectorRetriever] Returned %d chunks", len(chunks))
        return chunks

    def avg_score(self, chunks: list[RetrievedChunk]) -> float:
        """Compute the average similarity score for a set of chunks."""
        if not chunks:
            return 0.0
        return sum(c.score for c in chunks) / len(chunks)


# ── Structured Retriever ─────────────────────────────────────────────────────

_PARSE_PROMPT = """\
You are a query parser for a WHO health statistics dataset.
Given a user query, extract structured filter parameters.

Available columns: {columns}

Return ONLY a JSON object with these optional keys:
- "filters": dict of column_name -> value for equality matching
- "columns": list of column names to select
- "agg_column": column to aggregate (if query asks for totals/averages)
- "group_by": column to group by (if aggregation requested)
- "agg_func": one of "mean", "sum", "min", "max", "count"
- "text_search": a keyword to search across all text columns (fallback)

Example: {{"filters": {{"country": "Germany"}}, "columns": ["country", "year", "value"]}}
If you cannot parse the query, return: {{"text_search": "<key terms>"}}
"""


class StructuredRetriever:
    """Retrieves data from the WHO structured store."""

    def __init__(self, who_store: WHODataStore) -> None:
        self._store = who_store
        self._settings = get_settings()
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-init the OpenAI async client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._client

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Parse the query into structured params and run against the store.

        Falls back to text search when:
        - LLM parsing fails entirely
        - LLM-parsed filters return zero rows (e.g. misspelled country)

        This two-stage fallback ensures we almost always return data
        if the dataset contains anything relevant.
        """
        logger.info("[StructuredRetriever] Processing: '%s'", query[:100])

        result_csv: str = ""
        params: dict[str, Any] = {}

        # Stage 1: Try LLM-parsed structured query
        try:
            params = await self._parse_query(query)
            logger.info("[StructuredRetriever] Parsed params: %s", params)

            if "agg_column" in params and "group_by" in params:
                result_csv = await self._store.aggregate(
                    group_by=params["group_by"],
                    agg_column=params["agg_column"],
                    agg_func=params.get("agg_func", "mean"),
                )
            elif "filters" in params or "columns" in params:
                result_csv = await self._store.query(
                    filters=params.get("filters"),
                    columns=params.get("columns"),
                )
            elif "text_search" in params:
                result_csv = await self._store.text_search(params["text_search"])
        except Exception:
            logger.warning("[StructuredRetriever] Parse failed — will try text search")

        # Stage 2: If structured query returned nothing, fall back to
        # word-level text search using the original query.  This catches
        # cases where the LLM misspelled a country name, etc.
        is_empty = (
            not result_csv.strip()
            or result_csv.strip() == "No matching rows found."
        )
        if is_empty:
            logger.info(
                "[StructuredRetriever] Structured query returned nothing — "
                "falling back to text search"
            )
            result_csv = await self._store.text_search(query)

        if not result_csv.strip() or result_csv.strip() == "No matching rows found.":
            logger.info("[StructuredRetriever] No results found (even after text fallback)")
            return []

        chunk = RetrievedChunk(
            text=result_csv,
            source="WHO Global Health Statistics",
            chunk_id="who_query_result",
            score=1.0,
            metadata={"query_params": params, "type": "structured"},
        )
        logger.info("[StructuredRetriever] Returned structured data (%d chars)", len(result_csv))
        return [chunk]

    async def _parse_query(self, query: str) -> dict[str, Any]:
        """Use GPT to extract structured parameters from the query."""
        client = self._get_client()
        prompt = _PARSE_PROMPT.format(columns=self._store.columns)

        response = await client.chat.completions.create(
            model=self._settings.llm_model,
            max_completion_tokens=300,
            temperature=1,  # gpt-5-mini only supports default (1)
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )

        raw = response.choices[0].message.content or "{}"
        raw = re.sub(r"```json?\s*", "", raw).replace("```", "").strip()
        return json.loads(raw)


# ── Parallel dispatch ────────────────────────────────────────────────────────

async def run_retrieval(
    query: str,
    query_type: QueryType,
    vector_retriever: VectorRetriever,
    structured_retriever: StructuredRetriever,
) -> list[RetrievedChunk]:
    """Dispatch to one or both retrievers based on routing decision.

    Runs both in parallel when ``query_type`` is ``HYBRID``.

    Returns:
        Combined list of retrieved chunks.
    """
    logger.info("[Retrieval] Dispatching for query_type=%s", query_type.value)

    if query_type == QueryType.TEXT:
        return await vector_retriever.retrieve(query)

    if query_type == QueryType.STRUCTURED:
        return await structured_retriever.retrieve(query)

    # HYBRID — run both in parallel
    text_chunks, struct_chunks = await asyncio.gather(
        vector_retriever.retrieve(query),
        structured_retriever.retrieve(query),
    )
    combined = text_chunks + struct_chunks
    logger.info(
        "[Retrieval] Hybrid: %d text + %d structured = %d total",
        len(text_chunks),
        len(struct_chunks),
        len(combined),
    )
    return combined
