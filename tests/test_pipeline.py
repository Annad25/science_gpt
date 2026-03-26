"""
Integration and unit tests for the Science GPT pipeline.

Uses pytest-asyncio for async tests and httpx for API testing.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure a dummy API key for tests
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")


def _workspace_temp_file(name: str) -> Path:
    """Create a writable test path inside the workspace."""
    root = Path(".test_artifacts")
    root.mkdir(exist_ok=True)
    return root / f"{name}-{uuid4().hex}.json"


# ── Unit tests ───────────────────────────────────────────────────────────────

class TestCalculator:
    """Tests for the sandboxed calculator tool."""

    def test_basic_arithmetic(self):
        from app.tools.calculator import calculator

        assert calculator.invoke("2 + 3") == "5"
        assert calculator.invoke("10 / 3") == str(10 / 3)
        assert calculator.invoke("2 ** 10") == "1024"

    def test_math_functions(self):
        from app.tools.calculator import calculator

        assert calculator.invoke("sqrt(144)") == "12.0"
        assert calculator.invoke("abs(-42)") == "42"

    def test_blocked_expression(self):
        from app.tools.calculator import calculator

        result = calculator.invoke("__import__('os')")
        assert "error" in result.lower()


class TestCodeExecutor:
    """Tests for the sandboxed code execution tool."""

    def test_safe_code(self):
        from app.tools.code_executor import execute_python

        result = execute_python.invoke("print(2 + 2)")
        assert "4" in result

    def test_blocked_import(self):
        from app.tools.code_executor import execute_python

        result = execute_python.invoke("import os; print(os.listdir('.'))")
        assert "rejected" in result.lower() or "blocked" in result.lower()

    def test_blocked_open(self):
        from app.tools.code_executor import execute_python

        result = execute_python.invoke("open('/etc/passwd').read()")
        assert "rejected" in result.lower() or "blocked" in result.lower()


class TestChunker:
    """Tests for the text chunking module."""

    def test_chunking(self):
        from app.ingestion.chunker import chunk_documents

        docs = [{"text": "A " * 1000, "source": "test"}]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        for c in chunks:
            assert "text" in c
            assert "source" in c
            assert "chunk_id" in c

    def test_empty_document(self):
        from app.ingestion.chunker import chunk_documents

        chunks = chunk_documents([{"text": "", "source": "empty"}])
        assert len(chunks) == 0


class TestCache:
    """Tests for the in-memory TTL cache."""

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        from app.cache import QueryCache

        cache = QueryCache(max_size=10, ttl=60)
        await cache.put("hello world", {"answer": "test"})
        result = await cache.get("hello world")
        assert result is not None
        assert result["answer"] == "test"

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        from app.cache import QueryCache

        cache = QueryCache(max_size=10, ttl=60)
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate(self):
        from app.cache import QueryCache

        cache = QueryCache(max_size=10, ttl=60)
        await cache.put("test", {"data": 1})
        removed = await cache.invalidate("test")
        assert removed == 1
        assert await cache.get("test") is None

    @pytest.mark.asyncio
    async def test_invalidate_all(self):
        from app.cache import QueryCache

        cache = QueryCache(max_size=10, ttl=60)
        await cache.put("a", {"x": 1})
        await cache.put("b", {"x": 2})
        removed = await cache.invalidate()
        assert removed == 2
        assert cache.size == 0


class TestPaperCache:
    """Tests for the disk-backed arXiv paper cache."""

    def test_save_and_load_roundtrip(self):
        from app.ingestion.paper_cache import PaperCache

        cache_path = _workspace_temp_file("arxiv_cache")
        paper = {
            "arxiv_id": "1234.5678",
            "title": "Cached Paper",
            "text": "cached text",
            "source": "arXiv:test",
            "authors": ["A. Researcher"],
            "published": "2026-01-01T00:00:00",
            "categories": ["cs.AI"],
        }

        cache = PaperCache(cache_path=cache_path)
        cache.put(paper)
        cache.save()

        restored = PaperCache(cache_path=cache_path)
        restored.load()

        assert restored.size == 1
        assert restored.get("1234.5678") == paper


class TestPipelineCaching:
    """Tests for startup ingestion reuse of cached papers."""

    @pytest.mark.asyncio
    async def test_fetch_with_cache_falls_back_to_cached_papers(self, monkeypatch):
        from app.ingestion.paper_cache import PaperCache
        from app.ingestion import pipeline

        cache_path = _workspace_temp_file("arxiv_cache")
        cached_paper = {
            "arxiv_id": "1234.5678",
            "title": "Cached Paper",
            "text": "cached text",
            "source": "arXiv:test",
            "authors": ["A. Researcher"],
            "published": "2026-01-01T00:00:00",
            "categories": ["cs.AI"],
        }

        cache = PaperCache(cache_path=cache_path)
        cache.put(cached_paper)
        cache.save()
        cache.load()

        async def failing_fetch(**_: dict):
            raise RuntimeError("network unavailable")

        monkeypatch.setattr(pipeline, "fetch_arxiv_papers", failing_fetch)

        papers = await pipeline._fetch_with_cache(cache)
        assert papers == [cached_paper]

    @pytest.mark.asyncio
    async def test_fetch_with_cache_passes_cached_papers_and_saves_new_ones(
        self,
        monkeypatch,
    ):
        from app.ingestion.paper_cache import PaperCache
        from app.ingestion import pipeline

        cache_path = _workspace_temp_file("arxiv_cache")
        cached_paper = {
            "arxiv_id": "1234.5678",
            "title": "Cached Paper",
            "text": "cached text",
            "source": "arXiv:test",
            "authors": ["A. Researcher"],
            "published": "2026-01-01T00:00:00",
            "categories": ["cs.AI"],
        }
        new_paper = {
            "arxiv_id": "9999.0001",
            "title": "New Paper",
            "text": "new text",
            "source": "arXiv:new",
            "authors": ["B. Researcher"],
            "published": "2026-01-02T00:00:00",
            "categories": ["cs.LG"],
        }

        cache = PaperCache(cache_path=cache_path)
        cache.put(cached_paper)
        cache.save()
        cache.load()

        observed: dict[str, dict] = {}

        async def fake_fetch(**kwargs):
            observed["cached_papers"] = kwargs["cached_papers"]
            return [cached_paper, new_paper]

        monkeypatch.setattr(pipeline, "fetch_arxiv_papers", fake_fetch)

        papers = await pipeline._fetch_with_cache(cache)
        assert len(papers) == 2
        assert "1234.5678" in observed["cached_papers"]

        restored = PaperCache(cache_path=cache_path)
        restored.load()
        assert restored.size == 2


class TestModels:
    """Tests for Pydantic models."""

    def test_query_request_validation(self):
        from app.models import QueryRequest

        req = QueryRequest(query="What is AI?")
        assert req.query == "What is AI?"

    def test_query_request_empty(self):
        from pydantic import ValidationError

        from app.models import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_routing_decision(self):
        from app.models import QueryType, RoutingDecision

        rd = RoutingDecision(query_type=QueryType.HYBRID, confidence=0.9)
        assert rd.query_type == QueryType.HYBRID


class TestRouterHeuristics:
    """Tests for the Router's fallback heuristic classification."""

    def test_text_query(self):
        from app.agents.router import RouterAgent

        router = RouterAgent()
        decision = router._heuristic_classify("Explain the mechanism of CRISPR gene editing")
        assert decision.query_type.value == "text"

    def test_structured_query(self):
        from app.agents.router import RouterAgent

        router = RouterAgent()
        decision = router._heuristic_classify("What is the mortality rate in Nigeria?")
        assert decision.query_type.value in ("structured", "hybrid")

    def test_hybrid_query(self):
        from app.agents.router import RouterAgent

        router = RouterAgent()
        decision = router._heuristic_classify(
            "Explain the research on life expectancy statistics"
        )
        assert decision.query_type.value == "hybrid"


class TestWHODataStore:
    """Tests for the WHO structured data loader."""

    @pytest.mark.asyncio
    async def test_load_and_query(self):
        from app.ingestion.who_loader import WHODataStore

        store = WHODataStore()
        await store.load()
        assert len(store.df) > 0
        assert "country" in store.columns

        result = await store.query(filters={"country": "Japan"})
        assert "Japan" in result

    @pytest.mark.asyncio
    async def test_text_search(self):
        from app.ingestion.who_loader import WHODataStore

        store = WHODataStore()
        await store.load()
        result = await store.text_search("Germany")
        assert "Germany" in result

    @pytest.mark.asyncio
    async def test_text_search_normalizes_possessives(self):
        from app.ingestion.who_loader import WHODataStore

        store = WHODataStore()
        await store.load()
        assert "Germany" in await store.text_search("Germany's life expectancy")
        assert "Germany" in await store.text_search("germanys life expectancy")

    @pytest.mark.asyncio
    async def test_invalid_filter_column_raises(self):
        from app.ingestion.who_loader import WHODataStore

        store = WHODataStore()
        await store.load()
        with pytest.raises(ValueError):
            await store.query(filters={"nation": "Germany"})

    @pytest.mark.asyncio
    async def test_aggregate(self):
        from app.ingestion.who_loader import WHODataStore

        store = WHODataStore()
        await store.load()
        result = await store.aggregate(
            group_by="region",
            agg_column="life_expectancy",
            agg_func="mean",
        )
        assert "region" in result


class TestStructuredRetriever:
    """Tests for hybrid decomposition and structured fallback behavior."""

    @pytest.mark.asyncio
    async def test_build_query_parts_for_hybrid(self):
        from app.agents.retriever import StructuredRetriever
        from app.ingestion.who_loader import WHODataStore
        from app.models import QueryType

        store = WHODataStore()
        await store.load()
        retriever = StructuredRetriever(store)

        parts = retriever.build_query_parts(
            "what is germanys life expectancy and recent development in its healthcare",
            QueryType.HYBRID,
        )

        assert parts.structured_query == "life expectancy of Germany"
        assert "Germany" in parts.text_query
        assert "life" not in parts.text_query.lower()
        assert "expectancy" not in parts.text_query.lower()

    @pytest.mark.asyncio
    async def test_empty_parse_uses_deterministic_structured_fallback(self, monkeypatch):
        from app.agents.retriever import StructuredRetriever
        from app.ingestion.who_loader import WHODataStore

        store = WHODataStore()
        await store.load()
        retriever = StructuredRetriever(store)

        async def fake_parse(_: str) -> dict:
            return {}

        monkeypatch.setattr(retriever, "_parse_query", fake_parse)

        results = await retriever.retrieve(
            "what is germanys life expectancy and recent development in its healthcare"
        )

        assert len(results) == 1
        assert "Germany" in results[0].text
        assert "year" in results[0].text.splitlines()[0]
        assert "life_expectancy" in results[0].text.splitlines()[0]

    @pytest.mark.asyncio
    async def test_year_is_preserved_for_country_metric_queries(self, monkeypatch):
        from app.agents.retriever import StructuredRetriever
        from app.ingestion.who_loader import WHODataStore

        store = WHODataStore()
        await store.load()
        retriever = StructuredRetriever(store)

        async def fake_parse(_: str) -> dict:
            return {
                "filters": {"country": "Germany"},
                "columns": ["country", "life_expectancy"],
            }

        monkeypatch.setattr(retriever, "_parse_query", fake_parse)

        results = await retriever.retrieve("life expectancy of Germany")

        assert len(results) == 1
        header = results[0].text.splitlines()[0]
        assert header == "country,year,life_expectancy"
