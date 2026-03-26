"""
Ingestion orchestrator.

Coordinates arXiv fetching → chunking → embedding → Qdrant upsert, and
WHO CSV loading.  Designed to be called once at startup and optionally
re-triggered via an admin endpoint for incremental refresh.
"""

from __future__ import annotations

from app.embeddings.encoder import EmbeddingEncoder
from app.ingestion.arxiv_loader import fetch_arxiv_papers
from app.ingestion.chunker import chunk_documents
from app.ingestion.who_loader import WHODataStore
from app.logging_cfg import get_logger
from app.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


async def run_ingestion(
    qdrant: QdrantStore,
    encoder: EmbeddingEncoder,
    who_store: WHODataStore,
) -> dict[str, int]:
    """Execute the full ingestion pipeline.

    Args:
        qdrant: Initialised Qdrant store.
        encoder: Sentence-transformer encoder.
        who_store: WHO structured data store.

    Returns:
        Dict with counts: ``papers_fetched``, ``chunks_created``,
        ``vectors_upserted``, ``who_rows``.
    """
    logger.info("[Pipeline] ── Starting ingestion pipeline ──")

    # 1. Fetch arXiv papers
    papers = await fetch_arxiv_papers()
    logger.info("[Pipeline] Fetched %d papers from arXiv", len(papers))

    # 2. Chunk documents
    chunks = chunk_documents(papers)
    logger.info("[Pipeline] Created %d chunks", len(chunks))

    # 3. Encode chunks and upsert into Qdrant
    if chunks:
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
            }
            for c in chunks
        ]

        await qdrant.upsert(embeddings=embeddings, payloads=payloads)
        logger.info("[Pipeline] Upserted %d vectors into Qdrant", len(chunks))

    # 4. Load WHO structured data
    await who_store.load()
    who_rows = len(who_store.df)

    stats = {
        "papers_fetched": len(papers),
        "chunks_created": len(chunks),
        "vectors_upserted": len(chunks),
        "who_rows": who_rows,
    }
    logger.info("[Pipeline] ── Ingestion complete: %s ──", stats)
    return stats
