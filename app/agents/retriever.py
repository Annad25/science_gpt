"""
Retrieval Agents - vector and structured search.

Two independent agents that can run in parallel via ``asyncio.gather``:

* **VectorRetriever** - semantic similarity search over Qdrant.
* **StructuredRetriever** - filter/aggregate queries on the WHO DataFrame.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.embeddings.encoder import EmbeddingEncoder
from app.ingestion.who_loader import WHODataStore
from app.logging_cfg import get_logger
from app.models import (
    HybridQueryParts,
    QueryType,
    RetrievedChunk,
    StructuredQuerySpec,
)
from app.vectorstore.qdrant_store import QdrantStore

logger = get_logger(__name__)


class VectorRetriever:
    """Retrieves top-k semantically similar chunks from Qdrant."""

    def __init__(self, qdrant: QdrantStore, encoder: EmbeddingEncoder) -> None:
        self._qdrant = qdrant
        self._encoder = encoder
        self._settings = get_settings()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Embed the query and search Qdrant."""
        logger.info("[VectorRetriever] Searching for: '%s'", query[:100])
        query_vec = await self._encoder.encode(query)
        hits = await self._qdrant.search(query_vector=query_vec, top_k=top_k)

        chunks: list[RetrievedChunk] = []
        for hit in hits:
            score = hit.get("score", 0.0)
            chunks.append(
                RetrievedChunk(
                    text=hit["text"],
                    source=hit["source"],
                    chunk_id=hit.get("chunk_id", ""),
                    score=score,
                    metadata={
                        "title": hit.get("title", ""),
                        "authors": hit.get("payload", {}).get("authors", []),
                        "published": hit.get("payload", {}).get("published", ""),
                    },
                )
            )

        raw_scores = [chunk.score for chunk in chunks]
        if raw_scores:
            logger.info(
                "[VectorRetriever] Score range: min=%.3f max=%.3f avg=%.3f",
                min(raw_scores),
                max(raw_scores),
                sum(raw_scores) / len(raw_scores),
            )

        if min_score is not None:
            before = len(chunks)
            chunks = [chunk for chunk in chunks if chunk.score >= min_score]
            if len(chunks) < before:
                logger.info(
                    "[VectorRetriever] Filtered %d->%d chunks (min_score=%.3f)",
                    before,
                    len(chunks),
                    min_score,
                )

        logger.info("[VectorRetriever] Returned %d chunks", len(chunks))
        return chunks

    def avg_score(self, chunks: list[RetrievedChunk]) -> float:
        """Compute the average similarity score for a set of chunks."""
        if not chunks:
            return 0.0
        return sum(chunk.score for chunk in chunks) / len(chunks)


_PARSE_PROMPT = """\
You are a query parser for a WHO health statistics dataset.
Given a user query, extract the part that can be answered from the WHO dataset.

If the user query mixes structured data with open-ended research or explanation,
IGNORE the non-structured clauses and extract ONLY the WHO-relevant part.

Available columns: {columns}

Return ONLY a JSON object with these optional keys:
- "filters": dict of column_name -> value for equality matching
- "columns": list of column names to select
- "agg_column": column to aggregate (if query asks for totals/averages)
- "group_by": column to group by (if aggregation requested)
- "agg_func": one of "mean", "sum", "min", "max", "count"
- "text_search": a keyword to search across all text columns (fallback)

Example: {{"filters": {{"country": "Germany"}}, "columns": ["country", "year", "life_expectancy"]}}
If you can only partially parse the query, return the structured part you can identify.
If you cannot parse any structured part, return: {{"text_search": "<country/metric keywords>"}}
"""

_METRIC_ALIASES: dict[str, str] = {
    "life expectancy": "life_expectancy",
    "mortality rate": "mortality_rate_per_1000",
    "mortality": "mortality_rate_per_1000",
    "health expenditure": "health_expenditure_pct_gdp",
    "healthcare expenditure": "health_expenditure_pct_gdp",
    "health spending": "health_expenditure_pct_gdp",
    "population": "population_millions",
    "infant mortality": "infant_mortality_per_1000",
    "physician density": "physicians_per_10000",
    "physicians": "physicians_per_10000",
    "physician": "physicians_per_10000",
    "income group": "income_group",
}

_COLUMN_ALIASES: dict[str, str] = {
    "country": "country",
    "year": "year",
    "region": "region",
    "income_group": "income_group",
    "income group": "income_group",
    "life_expectancy": "life_expectancy",
    "life expectancy": "life_expectancy",
    "mortality_rate": "mortality_rate_per_1000",
    "mortality rate": "mortality_rate_per_1000",
    "mortality_rate_per_1000": "mortality_rate_per_1000",
    "health_expenditure": "health_expenditure_pct_gdp",
    "health expenditure": "health_expenditure_pct_gdp",
    "health_expenditure_pct_gdp": "health_expenditure_pct_gdp",
    "population": "population_millions",
    "population_millions": "population_millions",
    "infant_mortality": "infant_mortality_per_1000",
    "infant mortality": "infant_mortality_per_1000",
    "infant_mortality_per_1000": "infant_mortality_per_1000",
    "physicians": "physicians_per_10000",
    "physician": "physicians_per_10000",
    "physicians_per_10000": "physicians_per_10000",
}

_DISPLAY_BY_COLUMN = {
    "life_expectancy": "life expectancy",
    "mortality_rate_per_1000": "mortality rate",
    "health_expenditure_pct_gdp": "health expenditure",
    "population_millions": "population",
    "infant_mortality_per_1000": "infant mortality",
    "physicians_per_10000": "physicians",
    "income_group": "income group",
}

_TEXT_FOCUS_STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "of", "for", "and", "in", "to",
    "compare", "comparison", "statistic", "statistics", "data", "number",
    "numbers", "rate", "rates", "average", "mean", "total", "year", "years",
}


class StructuredRetriever:
    """Retrieves data from the WHO structured store."""

    def __init__(self, who_store: WHODataStore) -> None:
        self._store = who_store
        self._settings = get_settings()
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-init the OpenAI async client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def build_query_parts(self, query: str, query_type: QueryType) -> HybridQueryParts:
        """Split a mixed query into structured and text-focused subqueries."""
        if query_type != QueryType.HYBRID:
            return HybridQueryParts(
                original_query=query,
                structured_query=query,
                text_query=query,
            )

        countries = self._store.find_countries(query)
        metric_phrase, metric_column = self._extract_metric(query)

        structured_query = query
        if metric_column or countries:
            parts: list[str] = []
            if metric_column:
                parts.append(_DISPLAY_BY_COLUMN.get(metric_column, metric_phrase))
            if countries:
                country_text = ", ".join(countries)
                connector = "of" if metric_column else "for"
                parts.append(f"{connector} {country_text}")
            structured_query = " ".join(parts).strip() or query

        text_query = self._derive_text_focus(query, countries, metric_phrase)
        return HybridQueryParts(
            original_query=query,
            structured_query=structured_query or query,
            text_query=text_query or query,
        )

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Parse the query into structured params and run against the store."""
        logger.info("[StructuredRetriever] Processing: '%s'", query[:100])

        result_csv = ""
        params = StructuredQuerySpec()

        try:
            raw_params = await self._parse_query(query)
            logger.info("[StructuredRetriever] Parsed params: %s", raw_params)
            params = self._normalise_spec(raw_params)
            result_csv = await self._execute_spec(params)
        except Exception as exc:
            logger.warning(
                "[StructuredRetriever] Parse failed (%s) - will try deterministic fallback",
                exc,
            )

        if not result_csv.strip() or result_csv.strip() == "No matching rows found.":
            direct_spec = self._build_direct_spec(query)
            if direct_spec is not None:
                logger.info(
                    "[StructuredRetriever] Deterministic fallback spec: %s",
                    direct_spec.model_dump(exclude_none=True),
                )
                try:
                    direct_csv = await self._execute_spec(direct_spec)
                    if direct_csv.strip() and direct_csv.strip() != "No matching rows found.":
                        result_csv = direct_csv
                        params = direct_spec
                except Exception as exc:
                    logger.warning(
                        "[StructuredRetriever] Deterministic fallback failed (%s)",
                        exc,
                    )

        is_empty = (
            not result_csv.strip()
            or result_csv.strip() == "No matching rows found."
        )
        if is_empty:
            logger.info(
                "[StructuredRetriever] Structured query returned nothing - "
                "falling back to text search"
            )
            search_terms = self._build_text_search_terms(query)
            result_csv = await self._store.text_search(search_terms)

        if not result_csv.strip() or result_csv.strip() == "No matching rows found.":
            logger.info("[StructuredRetriever] No results found (even after text fallback)")
            return []

        chunk = RetrievedChunk(
            text=result_csv,
            source="WHO Global Health Statistics",
            chunk_id="who_query_result",
            score=1.0,
            metadata={
                "query_params": params.model_dump(exclude_none=True),
                "type": "structured",
            },
        )
        logger.info("[StructuredRetriever] Returned structured data (%d chars)", len(result_csv))
        return [chunk]

    def _extract_metric(self, query: str) -> tuple[str, str | None]:
        """Return the best metric phrase and canonical column for ``query``."""
        normalized = self._store.normalize_text(query)
        for phrase in sorted(_METRIC_ALIASES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(phrase)}\b", normalized):
                return phrase, _METRIC_ALIASES[phrase]
        return "", None

    def _derive_text_focus(
        self,
        query: str,
        countries: list[str],
        metric_phrase: str,
    ) -> str:
        """Strip structured-only tokens while keeping topical context."""
        country_map = {
            self._store.normalize_text(country): country
            for country in countries
        }
        metric_terms = set(metric_phrase.split()) if metric_phrase else set()

        kept: list[str] = []
        seen: set[str] = set()
        for term in self._store.build_search_terms(query):
            if term in metric_terms or term in _TEXT_FOCUS_STOP_WORDS:
                continue

            canonical_country = country_map.get(term)
            value = canonical_country if canonical_country else term
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(value)

        return " ".join(kept)

    def _resolve_column(self, name: str | None) -> str | None:
        """Map a free-form column name or alias to the WHO schema."""
        if not name:
            return None

        normalized = self._store.normalize_text(name)
        alias = _COLUMN_ALIASES.get(normalized)
        if alias:
            return alias

        underscored = normalized.replace(" ", "_")
        if underscored in self._store.columns:
            return underscored
        return None

    def _build_text_search_terms(self, query: str) -> str:
        """Reduce a noisy user query to the highest-signal structured terms."""
        countries = self._store.find_countries(query)
        metric_phrase, _ = self._extract_metric(query)

        pieces: list[str] = []
        if countries:
            pieces.extend(countries)
        if metric_phrase:
            pieces.append(metric_phrase)

        if not pieces:
            pieces.extend(self._store.build_search_terms(query)[:6])
        return " ".join(pieces) or query

    def _build_direct_spec(self, query: str) -> StructuredQuerySpec | None:
        """Build a deterministic structured plan from country/metric cues."""
        countries = self._store.find_countries(query)
        _, metric_column = self._extract_metric(query)

        if not countries and not metric_column:
            terms = self._build_text_search_terms(query)
            return StructuredQuerySpec(text_search=terms) if terms else None

        filters: dict[str, Any] = {}
        if countries:
            filters["country"] = countries[0] if len(countries) == 1 else countries

        columns: list[str] = []
        if metric_column:
            if "country" in self._store.columns:
                columns.append("country")
            if "year" in self._store.columns:
                columns.append("year")
            columns.append(metric_column)

        return StructuredQuerySpec(filters=filters, columns=columns)

    def _ensure_context_columns(self, spec: StructuredQuerySpec) -> StructuredQuerySpec:
        """Keep disambiguating columns like year when returning repeated metrics."""
        columns = list(spec.columns)
        metric_columns = {
            "life_expectancy",
            "mortality_rate_per_1000",
            "health_expenditure_pct_gdp",
            "population_millions",
            "infant_mortality_per_1000",
            "physicians_per_10000",
        }

        if (
            spec.filters.get("country")
            and any(column in metric_columns for column in columns)
            and "year" in self._store.columns
            and "year" not in columns
        ):
            insert_at = columns.index("country") + 1 if "country" in columns else 0
            columns.insert(insert_at, "year")

        return StructuredQuerySpec(
            filters=spec.filters,
            columns=columns,
            agg_column=spec.agg_column,
            group_by=spec.group_by,
            agg_func=spec.agg_func,
            text_search=spec.text_search,
        )

    def _normalise_spec(self, raw: dict[str, Any]) -> StructuredQuerySpec:
        """Validate and canonicalise the parsed LLM response."""
        spec = StructuredQuerySpec.model_validate(raw or {})
        if spec.is_empty:
            raise ValueError("empty structured spec")

        filters: dict[str, Any] = {}
        for key, value in spec.filters.items():
            column = self._resolve_column(key)
            if column is None:
                raise ValueError(f"unknown filter column: {key}")
            if isinstance(value, (list, tuple, set)):
                filters[column] = [
                    self._store.canonicalize_value(column, item)
                    for item in value
                ]
            else:
                filters[column] = self._store.canonicalize_value(column, value)

        columns: list[str] = []
        for column in spec.columns:
            resolved = self._resolve_column(column)
            if resolved and resolved not in columns:
                columns.append(resolved)

        agg_column = self._resolve_column(spec.agg_column)
        group_by = self._resolve_column(spec.group_by)
        agg_func = spec.agg_func.lower() if spec.agg_func else None
        if agg_func and agg_func not in {"mean", "sum", "min", "max", "count"}:
            raise ValueError(f"invalid agg_func: {agg_func}")

        text_search = spec.text_search.strip() if spec.text_search else None

        if not filters and not columns and not agg_column and not group_by and not text_search:
            raise ValueError("normalised spec is empty")

        return self._ensure_context_columns(
            StructuredQuerySpec(
                filters=filters,
                columns=columns,
                agg_column=agg_column,
                group_by=group_by,
                agg_func=agg_func,
                text_search=text_search,
            )
        )

    async def _execute_spec(self, spec: StructuredQuerySpec) -> str:
        """Execute a validated structured query spec."""
        if spec.agg_column and spec.group_by:
            return await self._store.aggregate(
                group_by=spec.group_by,
                agg_column=spec.agg_column,
                agg_func=spec.agg_func or "mean",
            )
        if spec.filters or spec.columns:
            return await self._store.query(
                filters=spec.filters or None,
                columns=spec.columns or None,
            )
        if spec.text_search:
            return await self._store.text_search(spec.text_search)
        return ""

    async def _parse_query(self, query: str) -> dict[str, Any]:
        """Use GPT to extract structured parameters from the query."""
        client = self._get_client()
        prompt = _PARSE_PROMPT.format(columns=self._store.columns)

        response = await client.chat.completions.create(
            model=self._settings.llm_model,
            max_completion_tokens=300,
            temperature=1,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )

        raw = response.choices[0].message.content or "{}"
        raw = re.sub(r"```json?\s*", "", raw).replace("```", "").strip()
        return json.loads(raw)


async def run_retrieval(
    query: str,
    query_type: QueryType,
    vector_retriever: VectorRetriever,
    structured_retriever: StructuredRetriever,
    query_parts: HybridQueryParts | None = None,
) -> list[RetrievedChunk]:
    """Dispatch to one or both retrievers based on routing decision."""
    logger.info("[Retrieval] Dispatching for query_type=%s", query_type.value)
    query_parts = query_parts or structured_retriever.build_query_parts(query, query_type)

    if query_type == QueryType.TEXT:
        return await vector_retriever.retrieve(query_parts.text_query)

    if query_type == QueryType.STRUCTURED:
        return await structured_retriever.retrieve(query_parts.structured_query)

    text_chunks, struct_chunks = await asyncio.gather(
        vector_retriever.retrieve(query_parts.text_query),
        structured_retriever.retrieve(query_parts.structured_query),
    )
    combined = text_chunks + struct_chunks
    logger.info(
        "[Retrieval] Hybrid: %d text + %d structured = %d total",
        len(text_chunks),
        len(struct_chunks),
        len(combined),
    )
    return combined
