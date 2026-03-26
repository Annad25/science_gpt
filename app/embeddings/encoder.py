"""
Sentence-transformer embedding encoder.

Wraps ``sentence-transformers`` to provide async-safe batch encoding
that runs the model in a thread-pool so it never blocks the event loop.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.logging_cfg import get_logger

logger = get_logger(__name__)


class EmbeddingEncoder:
    """Lazy-loaded sentence-transformer encoder."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or get_settings().embedding_model
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """Load the model on first use (heavy import)."""
        if self._model is None:
            logger.info("[Encoder] Loading model '%s' …", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("[Encoder] Model loaded (%d dim)", self.dimension)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._load_model().get_sentence_embedding_dimension()  # type: ignore[return-value]

    async def encode(self, text: str) -> list[float]:
        """Encode a single text string.

        Returns:
            List of floats (embedding vector).
        """
        vectors = await self.encode_batch([text])
        return vectors[0]

    async def encode_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Encode a batch of texts in a worker thread.

        Args:
            texts: Texts to embed.
            batch_size: Internal mini-batch size for the transformer.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        model = self._load_model()
        total = len(texts)
        num_batches = (total + batch_size - 1) // batch_size

        logger.info(
            "[Encoder] Encoding %d texts in ~%d batches (batch_size=%d) …",
            total,
            num_batches,
            batch_size,
        )

        def _encode() -> np.ndarray:
            return model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=total > 100,  # show bar for large batches
                normalize_embeddings=True,
            )

        embeddings: np.ndarray = await asyncio.to_thread(_encode)
        logger.info("[Encoder] Encoding complete — %d vectors produced", len(embeddings))
        return embeddings.tolist()
