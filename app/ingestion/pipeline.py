"""
Ingestion orchestrator.

Coordinates arXiv fetching → chunking → embedding → Qdrant upsert, and
WHO CSV loading.  Designed to be called once at startup and optionally
re-triggered via an admin endpoint for incremental refresh.

**Smart caching**: On startup, checks if Qdrant already has vectors
from a previous run.  If so, skips the full ingestion (PDF download
+ embedding) and only loads WHO data.  ``POST /ingest?force=true``
bypasses this check and forces a full rebuild.
"""

from __future__ import annotations

from app.embeddings.encoder import EmbeddingEncoder
from app.ingestion.arxiv_loader import fetch_arxiv_papers
from app.ingestion.chunker import chunk_documents
from app.ingestion.paper_cache import PaperCache
from app.ingestion.who_loader import WHODataStore
from app.logging_cfg import get_logger
from app.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


async def run_ingestion(
    qdrant: QdrantStore,
    encoder: EmbeddingEncoder,
    who_store: WHODataStore,
    force: bool = False,
) -> dict[str, int]:
    """Execute the full ingestion pipeline with smart caching.

    On a normal startup:
    1. Checks if Qdrant already has indexed vectors.
    2. If yes → skips arXiv fetch + embedding (fast start).
    3. If no  → fetches papers, chunks, embeds, and upserts.

    The paper cache (JSON on disk) avoids re-downloading PDFs even
    during a forced re-ingestion; only re-embedding is needed.

    Args:
        qdrant: Initialised Qdrant store.
        encoder: Sentence-transformer encoder.
        who_store: WHO structured data store.
        force: If ``True``, wipe and rebuild everything.

    Returns:
        Dict with counts: ``papers_fetched``, ``chunks_created``,
        ``vectors_upserted``, ``who_rows``, ``skipped``.
    """
    logger.info("[Pipeline] ── Starting ingestion pipeline (force=%s) ──", force)

    # ── Paper cache ──────────────────────────────────────────────────
    paper_cache = PaperCache()
    paper_cache.load()

    # ── Skip check: if Qdrant already has data and not forced ────────
    if not force:
        existing_count = await qdrant.count()
        if existing_count > 0:
            logger.info(
                "[Pipeline] Qdrant already has %d vectors — skipping arXiv ingestion "
                "(use POST /ingest?force=true to rebuild)",
                existing_count,
            )
            # Still load WHO data (it's fast and always needed)
            await who_store.load()
            who_rows = len(who_store.df)

            stats = {
                "papers_fetched": 0,
                "chunks_created": 0,
                "vectors_upserted": 0,
                "who_rows": who_rows,
                "skipped": True,
                "existing_vectors": existing_count,
                "cached_papers": paper_cache.size,
            }
            logger.info("[Pipeline] ── Ingestion skipped (cached): %s ──", stats)
            return stats

    # ── Force rebuild: wipe existing collection ──────────────────────
    if force:
        logger.info("[Pipeline] Force mode — deleting and recreating collection")
        await qdrant.delete_collection()
        await qdrant.ensure_collection()

    # ── 1. Fetch arXiv papers (use cache where possible) ─────────────
    papers = await _fetch_with_cache(paper_cache)
    logger.info("[Pipeline] Total papers ready: %d", len(papers))

    # ── 2. Chunk documents ───────────────────────────────────────────
    chunks = chunk_documents(papers)
    logger.info("[Pipeline] Created %d chunks", len(chunks))

    # ── 3. Encode chunks and upsert into Qdrant ─────────────────────
    vectors_upserted = 0
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

        vectors_upserted = await qdrant.upsert(embeddings=embeddings, payloads=payloads)
        logger.info("[Pipeline] Upserted %d vectors into Qdrant", vectors_upserted)

    # ── 4. Load WHO structured data ──────────────────────────────────
    await who_store.load()
    who_rows = len(who_store.df)

    stats = {
        "papers_fetched": len(papers),
        "chunks_created": len(chunks),
        "vectors_upserted": vectors_upserted,
        "who_rows": who_rows,
        "skipped": False,
        "cached_papers": paper_cache.size,
    }
    logger.info("[Pipeline] ── Ingestion complete: %s ──", stats)
    return stats


async def _fetch_with_cache(paper_cache: PaperCache) -> list[dict]:
    """Fetch papers, preferring the disk cache over network.

    For each configured query:
    - Calls arXiv search to get candidate IDs.
    - For papers already in the cache, reuses the cached text.
    - For new papers, downloads PDF, extracts text, and caches the result.
    - If arXiv is unavailable but cache exists, falls back to cached papers.

    Returns:
        Combined list of all paper dicts (cached + freshly fetched).
    """
    cached_papers = {
        paper["arxiv_id"]: paper
        for paper in paper_cache.get_all()
        if paper.get("arxiv_id")
    }

    try:
        fresh_papers = await fetch_arxiv_papers(cached_papers=cached_papers)
    except Exception:
        if paper_cache.size > 0:
            logger.warning(
                "[Pipeline] arXiv fetch failed - falling back to %d cached papers",
                paper_cache.size,
            )
            return paper_cache.get_all()
        raise

    # Merge with cache: add any newly fetched papers to cache
    new_count = paper_cache.put_many(fresh_papers)
    paper_cache.save()
    if new_count > 0:
        logger.info("[Pipeline] Added %d new papers to cache", new_count)
    else:
        logger.info("[Pipeline] All %d papers were already cached", len(fresh_papers))

    # Return all cached papers (includes both seed + dynamically ingested)
    all_papers = paper_cache.get_all()
    logger.info(
        "[Pipeline] Using %d total papers (%d from cache, %d freshly fetched)",
        len(all_papers),
        paper_cache.size - new_count,
        new_count,
    )
    return all_papers
