"""
Entity Linker Agent — cross-source relationship traversal.

Extracts named entities from vector search results (e.g. country names,
diseases, technologies) and looks them up in the structured data store.
This bridges the gap between unstructured papers and structured datasets,
enabling the Reasoner to synthesise answers that combine both sources.

Example: a paper mentions "Germany's healthcare system" → entity linker
finds "Germany" → fetches all WHO rows for Germany → appends as extra
context alongside the paper chunks.
"""

from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from app.config import get_settings
from app.ingestion.who_loader import WHODataStore
from app.logging_cfg import get_logger
from app.models import RetrievedChunk

logger = get_logger(__name__)

_ENTITY_EXTRACTION_PROMPT = """\
Extract named entities from the text below that could be looked up in a
structured health/science dataset.  Focus on:
- Country names
- Disease or condition names
- Specific metrics (e.g. "life expectancy", "mortality rate")
- Year references

Return ONLY a JSON object:
{{"entities": ["entity1", "entity2", ...]}}

If no entities are found, return: {{"entities": []}}
"""


class EntityLinker:
    """Traverses relationships between vector results and structured data."""

    def __init__(self, who_store: WHODataStore) -> None:
        self._store = who_store
        self._settings = get_settings()
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-init the async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._client

    async def enrich_with_structured_data(
        self,
        chunks: list[RetrievedChunk],
        query: str,
    ) -> list[RetrievedChunk]:
        """Extract entities from vector chunks and fetch matching structured data.

        Args:
            chunks: Retrieved vector chunks (from arXiv papers).
            query: Original user query (also mined for entities).

        Returns:
            Original chunks plus any structured-data chunks found via
            entity linking.  Returns original chunks unchanged if no
            entities are found or linking fails.
        """
        if self._store.df.empty:
            return chunks

        try:
            entities = await self._extract_entities(chunks, query)
            if not entities:
                logger.info("[EntityLinker] No linkable entities found")
                return chunks

            logger.info("[EntityLinker] Extracted entities: %s", entities)

            linked_chunks: list[RetrievedChunk] = []
            seen: set[str] = set()

            for entity in entities:
                csv_data = await self._store.get_entity_data(entity)
                if not csv_data or entity in seen:
                    continue
                seen.add(entity)
                linked_chunks.append(
                    RetrievedChunk(
                        text=f"Structured data for '{entity}':\n{csv_data}",
                        source=f"Structured Data (entity: {entity})",
                        chunk_id=f"entity_link:{entity}",
                        score=0.9,  # high score — exact entity match
                        metadata={"type": "entity_link", "entity": entity},
                    )
                )
                logger.info(
                    "[EntityLinker] Linked entity '%s' → %d chars of structured data",
                    entity,
                    len(csv_data),
                )

            if linked_chunks:
                logger.info(
                    "[EntityLinker] Enriched context with %d entity-linked chunks",
                    len(linked_chunks),
                )
                return chunks + linked_chunks

            return chunks

        except Exception:
            logger.exception("[EntityLinker] Entity linking failed — returning original chunks")
            return chunks

    async def _extract_entities(
        self,
        chunks: list[RetrievedChunk],
        query: str,
    ) -> list[str]:
        """Use LLM to extract named entities from the query and top chunks."""
        # Combine query + first 2 chunk texts for entity extraction
        combined = f"User query: {query}\n\n"
        for i, c in enumerate(chunks[:2], 1):
            combined += f"Source {i}: {c.text[:500]}\n\n"

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._settings.llm_model,
                max_completion_tokens=200,
                temperature=1,
                messages=[
                    {"role": "system", "content": _ENTITY_EXTRACTION_PROMPT},
                    {"role": "user", "content": combined},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            raw = re.sub(r"```json?\s*", "", raw).replace("```", "").strip()
            data = json.loads(raw)
            return data.get("entities", [])
        except Exception:
            logger.warning("[EntityLinker] LLM entity extraction failed — using regex fallback")
            return self._regex_extract(query)

    def _regex_extract(self, text: str) -> list[str]:
        """Simple regex fallback: find capitalised multi-word names."""
        # Match country-like names and capitalised terms
        known_countries = {
            "germany", "india", "brazil", "japan", "china", "usa",
            "united states", "united kingdom", "france", "canada",
            "australia", "south africa", "nigeria", "mexico", "russia",
            "indonesia", "pakistan", "bangladesh", "ethiopia", "egypt",
            "italy", "spain",
        }
        text_lower = text.lower()
        found = [c.title() for c in known_countries if c in text_lower]
        return found[:5]  # cap at 5 entities
