"""
Centralised configuration loaded from environment variables.

Every tunable knob lives here so the rest of the codebase never calls
``os.getenv`` directly.  Pydantic-settings validates at startup and
surfaces missing / malformed values early.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from ``.env`` or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_model: str = "gpt-5-mini"
    llm_temperature: float = 1
    llm_max_tokens: int = 4096
    llm_timeout_seconds: int = 60

    # ── Embeddings ───────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Qdrant ───────────────────────────────────────────────────────────
    qdrant_mode: Literal["memory", "docker", "cloud"] = "memory"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "science_chunks"
    qdrant_cloud_url: str = ""
    qdrant_api_key: str = ""

    # ── Retrieval ────────────────────────────────────────────────────────
    retrieval_top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── arXiv ingestion ──────────────────────────────────────────────────
    arxiv_max_papers: int = 8
    arxiv_queries: list[str] = [
        "large language model inference optimization",
        "retrieval augmented generation",
        "transformer architecture survey",
        "machine learning healthcare",
        "climate change science",
    ]

    # ── Dynamic ingestion (on-demand fetch for unknown topics) ────────
    dynamic_ingestion_enabled: bool = True
    dynamic_ingestion_score_threshold: float = 0.45
    dynamic_ingestion_max_papers: int = 5

    # ── Cache ────────────────────────────────────────────────────────────
    cache_ttl_seconds: int = 600  # 10 minutes
    cache_max_size: int = 256

    # ── API ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ── Sandboxed execution ──────────────────────────────────────────────
    code_exec_timeout_seconds: int = 10
    code_exec_max_memory_mb: int = 128


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
