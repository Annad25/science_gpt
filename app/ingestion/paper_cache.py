"""
Local disk cache for arXiv papers.

Saves fetched papers (metadata + extracted text) to a JSON file so
they don't need to be re-downloaded and re-extracted on every restart.
Each paper is keyed by its unique ``arxiv_id``.

Cache invalidation:
- Startup and re-ingestion reuse cached papers when possible.
- Papers fetched via dynamic ingestion are also persisted here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.logging_cfg import get_logger

logger = get_logger(__name__)


class PaperCache:
    """JSON-file-backed cache for arXiv paper documents."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        settings = get_settings()
        self._path = Path(cache_path or settings.arxiv_cache_path)
        self._papers: dict[str, dict[str, Any]] = {}
        self._dirty = False

    def load(self) -> None:
        """Load the cache from disk.  Silently starts empty if missing."""
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self._papers = {p["arxiv_id"]: p for p in data if "arxiv_id" in p}
                logger.info(
                    "[PaperCache] Loaded %d cached papers from %s",
                    len(self._papers),
                    self._path,
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning("[PaperCache] Cache file corrupt — starting fresh")
                self._papers = {}
        else:
            logger.info("[PaperCache] No cache file at %s — starting fresh", self._path)

    def save(self) -> None:
        """Persist the current cache to disk (only if modified)."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(list(self._papers.values()), f, ensure_ascii=False, indent=2)
        self._dirty = False
        logger.info("[PaperCache] Saved %d papers to %s", len(self._papers), self._path)

    def get(self, arxiv_id: str) -> dict[str, Any] | None:
        """Retrieve a cached paper by its arXiv ID."""
        return self._papers.get(arxiv_id)

    def has(self, arxiv_id: str) -> bool:
        """Check if a paper is cached."""
        return arxiv_id in self._papers

    def put(self, paper: dict[str, Any]) -> None:
        """Add or update a paper in the cache."""
        arxiv_id = paper.get("arxiv_id", "")
        if not arxiv_id:
            return
        if self._papers.get(arxiv_id) != paper:
            self._papers[arxiv_id] = paper
            self._dirty = True

    def put_many(self, papers: list[dict[str, Any]]) -> int:
        """Add multiple papers.  Returns count of newly added ones."""
        added = 0
        for paper in papers:
            aid = paper.get("arxiv_id", "")
            if aid and aid not in self._papers:
                added += 1
            if aid and self._papers.get(aid) != paper:
                self._papers[aid] = paper
                self._dirty = True
        return added

    def get_all(self) -> list[dict[str, Any]]:
        """Return all cached papers as a list."""
        return list(self._papers.values())

    def get_uncached_ids(self, arxiv_ids: list[str]) -> list[str]:
        """Return IDs that are NOT in the cache."""
        return [aid for aid in arxiv_ids if aid not in self._papers]

    def clear(self) -> None:
        """Wipe the cache (in memory; call save() to persist)."""
        self._papers.clear()
        self._dirty = True
        logger.info("[PaperCache] Cleared all cached papers")

    @property
    def size(self) -> int:
        """Number of cached papers."""
        return len(self._papers)
