"""
Qdrant vector store abstraction.

Supports in-memory, file-based (persistent), Docker, and cloud backends
via a single interface.  All public methods are async-safe (Qdrant's
gRPC client is sync, so we wrap calls in ``asyncio.to_thread``).

**File mode** (default) stores vectors on disk so they survive restarts.
This avoids re-downloading PDFs and re-embedding on every server start.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.logging_cfg import get_logger

logger = get_logger(__name__)


class QdrantStore:
    """Async-safe Qdrant abstraction for the RAG pipeline."""

    def __init__(self) -> None:
        settings = get_settings()
        self._collection = settings.qdrant_collection
        self._dimension = settings.embedding_dimension

        if settings.qdrant_mode == "memory":
            logger.info("[Qdrant] Initialising in-memory store")
            self._client = QdrantClient(location=":memory:")

        elif settings.qdrant_mode == "file":
            db_path = Path(settings.qdrant_path)
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info("[Qdrant] Initialising file-based store at %s", db_path)
            self._client = QdrantClient(path=str(db_path))

        elif settings.qdrant_mode == "docker":
            logger.info(
                "[Qdrant] Connecting to Docker at %s:%d",
                settings.qdrant_host,
                settings.qdrant_port,
            )
            self._client = QdrantClient(
                host=settings.qdrant_host, port=settings.qdrant_port
            )
        else:
            logger.info("[Qdrant] Connecting to Qdrant Cloud")
            self._client = QdrantClient(
                url=settings.qdrant_cloud_url, api_key=settings.qdrant_api_key
            )

    async def ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        exists = await asyncio.to_thread(
            self._client.collection_exists, self._collection
        )
        if not exists:
            await asyncio.to_thread(
                self._client.create_collection,
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                "[Qdrant] Created collection '%s' (dim=%d, cosine)",
                self._collection,
                self._dimension,
            )
        else:
            logger.info("[Qdrant] Collection '%s' already exists", self._collection)

    async def delete_collection(self) -> None:
        """Delete the collection (used for forced re-ingestion)."""
        try:
            await asyncio.to_thread(
                self._client.delete_collection, self._collection
            )
            logger.info("[Qdrant] Deleted collection '%s'", self._collection)
        except Exception:
            logger.warning("[Qdrant] Collection '%s' did not exist — nothing to delete", self._collection)

    async def upsert(
        self,
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> int:
        """Upsert vectors with payloads into the collection.

        Args:
            embeddings: List of embedding vectors.
            payloads: Matching list of metadata dicts.

        Returns:
            Number of points upserted.
        """
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=pay,
            )
            for vec, pay in zip(embeddings, payloads)
        ]

        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self._collection,
            points=points,
        )
        logger.info("[Qdrant] Upserted %d points", len(points))
        return len(points)

    async def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Similarity search returning top-k results.

        Returns:
            List of dicts with ``text``, ``source``, ``score``, and full
            ``payload``.
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k

        results = await asyncio.to_thread(
            self._client.query_points,
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )

        hits: list[dict[str, Any]] = []
        for point in results.points:
            payload = point.payload or {}
            hits.append(
                {
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "title": payload.get("title", ""),
                    "chunk_id": payload.get("chunk_id", ""),
                    "score": point.score,
                    "payload": payload,
                }
            )
        logger.info("[Qdrant] Search returned %d hits (top_k=%d)", len(hits), top_k)
        return hits

    async def count(self) -> int:
        """Return the total number of points in the collection."""
        info = await asyncio.to_thread(
            self._client.get_collection, self._collection
        )
        return info.points_count or 0
