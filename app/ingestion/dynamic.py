"""
Dynamic on-demand ingestion.

When the vector retriever returns low-relevance results (all below a
configured threshold), this module fetches arXiv papers matching the
user's query, extracts text, chunks, embeds, and indexes them — then
retries the search with freshly relevant data.

This ensures the system adapts to topics not covered by the seed queries
without requiring a manual restart or re-ingestion.
"""

from __future__ import annotations

from app.config import get_settings
from app.embeddings.encoder import EmbeddingEncoder
from app.ingestion.arxiv_loader import fetch_arxiv_papers
from app.ingestion.chunker import chunk_documents
from app.logging_cfg import get_logger
from app.models import RetrievedChunk
from app.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


async def dynamic_ingest_and_retry(
    query: str,
    qdrant: QdrantStore,
    encoder: EmbeddingEncoder,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fetch papers for ``query`` from arXiv, index them, and re-search.

    This is the "Approach B" fallback: when seed data doesn't cover the
    user's topic, we dynamically expand the index on demand.

    Args:
        query: The user's original question (used as arXiv search term).
        qdrant: Initialised Qdrant store to upsert new vectors into.
        encoder: Embedding encoder for vectorising new chunks.
        top_k: Number of results for the retry search.

    Returns:
        List of ``RetrievedChunk`` from the re-search (may still be
        empty if arXiv has no relevant papers either).
    """
    settings = get_settings()

    if not settings.dynamic_ingestion_enabled:
        logger.info("[DynamicIngest] Disabled — skipping")
        return []

    logger.info("[DynamicIngest] Low relevance detected — fetching papers for: '%s'", query[:100])

    try:
        # 1. Fetch papers using the user's query as the arXiv search term
        papers = await fetch_arxiv_papers(
            queries=[query],
            max_papers_per_query=settings.dynamic_ingestion_max_papers,
        )

        if not papers:
            logger.warning("[DynamicIngest] No papers found on arXiv for query")
            return []

        logger.info("[DynamicIngest] Fetched %d papers", len(papers))

        # 2. Chunk the new papers
        chunks = chunk_documents(papers)
        logger.info("[DynamicIngest] Created %d chunks", len(chunks))

        if not chunks:
            return []

        # 3. Embed and upsert
        texts = [c["text"] for c in chunks]
        embeddings = await encoder.encode_batch(texts)

        payloads = [
            {
                "text": c["text"],
                "source": c.get("source", ""),
                "title": c.get("title", ""),
                "chunk_id": c.get("chunk_id", ""),
                "chunk_index": c.get("chunk_index", 0),
                "authors": c.get("authors", []),
                "published": c.get("published", ""),
                "categories": c.get("categories", []),
                "dynamic": True,  # mark as dynamically ingested
            }
            for c in chunks
        ]

        await qdrant.upsert(embeddings=embeddings, payloads=payloads)
        logger.info("[DynamicIngest] Upserted %d vectors into Qdrant", len(chunks))

        # 4. Re-search with the freshly indexed content
        query_vec = await encoder.encode(query)
        hits = await qdrant.search(query_vector=query_vec, top_k=top_k)

        result_chunks = [
            RetrievedChunk(
                text=h["text"],
                source=h["source"],
                chunk_id=h.get("chunk_id", ""),
                score=h.get("score", 0.0),
                metadata={
                    "title": h.get("title", ""),
                    "authors": h.get("payload", {}).get("authors", []),
                    "published": h.get("payload", {}).get("published", ""),
                },
            )
            for h in hits
        ]

        scores = [c.score for c in result_chunks]
        if scores:
            logger.info(
                "[DynamicIngest] Re-search scores: min=%.3f max=%.3f avg=%.3f",
                min(scores),
                max(scores),
                sum(scores) / len(scores),
            )

        logger.info(
            "[DynamicIngest] Re-search returned %d chunks after dynamic ingestion",
            len(result_chunks),
        )
        return result_chunks

    except Exception:
        logger.exception("[DynamicIngest] Dynamic ingestion failed")
        return []
