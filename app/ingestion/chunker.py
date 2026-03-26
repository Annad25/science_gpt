"""
Text chunking for vector indexing.

Splits documents into overlapping chunks of configurable size using
LangChain's ``RecursiveCharacterTextSplitter`` — the recommended
splitter for general-purpose text because it respects paragraph and
sentence boundaries before falling back to character-level splits.
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.logging_cfg import get_logger

logger = get_logger(__name__)


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """Split a list of document dicts into smaller chunks.

    Each input document must have at least a ``text`` key.  Metadata keys
    (``title``, ``source``, etc.) are propagated to every resulting chunk.

    Args:
        documents: Raw documents from a loader.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunk dicts with added ``chunk_index`` metadata.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    for doc in documents:
        text = doc.get("text", "")
        if not text.strip():
            continue

        splits = splitter.split_text(text)
        metadata = {k: v for k, v in doc.items() if k != "text"}

        for idx, split_text in enumerate(splits):
            chunks.append(
                {
                    "text": split_text,
                    "chunk_index": idx,
                    "chunk_id": f"{metadata.get('source', 'unknown')}::chunk_{idx}",
                    **metadata,
                }
            )

    logger.info(
        "[Chunker] Split %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
