"""
Integration and unit tests for the Science GPT pipeline.

Uses pytest-asyncio for async tests and httpx for API testing.
"""

from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure a dummy API key for tests
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")


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
