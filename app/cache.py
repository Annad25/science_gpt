"""
In-memory TTL cache with size limits and manual invalidation.

Uses ``cachetools.TTLCache`` under the hood, wrapped with an async-safe
interface and helper methods for cache-busting on data refresh.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from cachetools import TTLCache

from app.config import get_settings
from app.logging_cfg import get_logger

logger = get_logger(__name__)


class QueryCache:
    """Thread-safe, TTL-bound query result cache."""

    def __init__(self, max_size: int | None = None, ttl: int | None = None) -> None:
        settings = get_settings()
        self._ttl = ttl or settings.cache_ttl_seconds
        self._max_size = max_size or settings.cache_max_size
        self._store: TTLCache[str, dict[str, Any]] = TTLCache(
            maxsize=self._max_size, ttl=self._ttl
        )
        self._lock = asyncio.Lock()

    @staticmethod
    def _make_key(query: str) -> str:
        """Normalise and hash the query to produce a stable cache key."""
        normalised = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalised.encode()).hexdigest()[:32]

    async def get(self, query: str) -> dict[str, Any] | None:
        """Return cached result or ``None`` on miss."""
        key = self._make_key(query)
        async with self._lock:
            result = self._store.get(key)
        if result is not None:
            logger.info("Cache HIT for key=%s", key[:8])
        else:
            logger.debug("Cache MISS for key=%s", key[:8])
        return result

    async def put(self, query: str, value: dict[str, Any]) -> None:
        """Store a result in the cache."""
        key = self._make_key(query)
        async with self._lock:
            self._store[key] = value
        logger.debug("Cached result for key=%s", key[:8])

    async def invalidate(self, query: str | None = None) -> int:
        """Invalidate a specific query or the entire cache.

        Returns the number of entries removed.
        """
        async with self._lock:
            if query is None:
                count = len(self._store)
                self._store.clear()
                logger.info("Full cache invalidation — removed %d entries", count)
                return count
            key = self._make_key(query)
            if key in self._store:
                del self._store[key]
                logger.info("Invalidated cache key=%s", key[:8])
                return 1
            return 0

    @property
    def size(self) -> int:
        """Current number of live entries."""
        return len(self._store)
