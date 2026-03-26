"""
Router Agent — query classification and dispatch.

Uses a lightweight LLM call (or heuristics as fallback) to classify
incoming queries into ``text``, ``structured``, or ``hybrid`` and
decide which retrieval agents to invoke.
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.logging_cfg import get_logger
from app.models import QueryType, RoutingDecision

logger = get_logger(__name__)

# ── Keyword patterns for heuristic classification ────────────────────────────

# Keywords that strongly suggest structured-data queries.
_STRUCTURED_KEYWORDS = re.compile(
    r"\b(statistic|data|number|count|rate|percent|average|mean|total|"
    r"population|mortality|life expectancy|gdp|emission|expenditure|"
    r"who\b|country|countries|rank|top \d+|compare|comparison|trend|"
    r"year|dataset|infant|physician|income)\b",
    re.IGNORECASE,
)

# Keywords that suggest free-text / scientific literature queries.
_TEXT_KEYWORDS = re.compile(
    r"\b(explain|describe|what is|how does|research|paper|study|"
    r"theory|mechanism|approach|method|algorithm|review|survey|"
    r"finding|conclusion|hypothesis|abstract|technique|framework|"
    r"architecture|neural|transformer|llm|machine learning|deep learning|"
    r"ai\b|artificial intelligence|inference|training|optimization|"
    r"quantization|retrieval|generation|healthcare|climate)\b",
    re.IGNORECASE,
)

_ROUTER_SYSTEM_PROMPT = """\
You are a query router for a scientific knowledge system with two data sources:

1. **Scientific papers** (arXiv) — research on AI, ML, healthcare, climate, etc.
2. **Structured dataset** (CSV/WHO) — country-level health statistics with
   columns like: country, year, life_expectancy, mortality_rate, population,
   health_expenditure, infant_mortality, physicians, region, income_group.

Classify the user query into exactly ONE category:

- "text": The query is ONLY about concepts, explanations, or research findings
  from scientific papers.  No statistics or country data needed.
- "structured": The query is ONLY about specific statistics, numbers, rankings,
  or comparisons from the structured dataset.  No paper knowledge needed.
- "hybrid": The query needs BOTH paper knowledge AND structured data.  This
  includes any question that mentions both a scientific topic AND a measurable
  statistic, country comparison, or data-driven aspect.

IMPORTANT: When in doubt between "text" and "hybrid", prefer "hybrid".
A query like "countries with high mortality exploring AI in healthcare" is
clearly hybrid — it needs mortality data AND AI research papers.

Respond with ONLY a JSON object:
{"type": "<text|structured|hybrid>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}
"""


class RouterAgent:
    """Classifies queries and decides retrieval strategy."""

    def __init__(self, who_columns: list[str] | None = None) -> None:
        self._settings = get_settings()
        self._who_columns = who_columns or []
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-init the OpenAI async client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._client

    async def route(self, query: str) -> RoutingDecision:
        """Classify ``query`` and return a routing decision.

        Attempts an LLM-based classification first; falls back to
        keyword heuristics if the LLM call fails.  Additionally,
        applies a safety net: if the LLM says "text" but heuristics
        detect structured keywords, we upgrade to "hybrid".
        """
        logger.info("[Router] Classifying query: '%s'", query[:120])

        heuristic = self._heuristic_classify(query)

        try:
            llm_decision = await self._llm_classify(query)
            logger.info(
                "[Router] LLM decision: type=%s confidence=%.2f reason='%s'",
                llm_decision.query_type.value,
                llm_decision.confidence,
                llm_decision.reasoning,
            )

            # Safety net: if LLM says "text" but heuristic detects
            # structured keywords too, upgrade to hybrid.
            if (
                llm_decision.query_type == QueryType.TEXT
                and heuristic.query_type in (QueryType.STRUCTURED, QueryType.HYBRID)
            ):
                logger.info(
                    "[Router] Safety net: LLM said 'text' but heuristic detected "
                    "structured keywords — upgrading to 'hybrid'"
                )
                return RoutingDecision(
                    query_type=QueryType.HYBRID,
                    confidence=max(llm_decision.confidence, heuristic.confidence),
                    reasoning=f"Upgraded: LLM={llm_decision.query_type.value}, "
                              f"heuristic={heuristic.query_type.value}",
                )

            # Similarly, if LLM says "structured" but heuristic sees text keywords
            if (
                llm_decision.query_type == QueryType.STRUCTURED
                and heuristic.query_type in (QueryType.TEXT, QueryType.HYBRID)
            ):
                logger.info(
                    "[Router] Safety net: LLM said 'structured' but heuristic detected "
                    "text keywords — upgrading to 'hybrid'"
                )
                return RoutingDecision(
                    query_type=QueryType.HYBRID,
                    confidence=max(llm_decision.confidence, heuristic.confidence),
                    reasoning=f"Upgraded: LLM={llm_decision.query_type.value}, "
                              f"heuristic={heuristic.query_type.value}",
                )

            return llm_decision

        except Exception:
            logger.warning("[Router] LLM classification failed — using heuristics")
            return heuristic

    async def _llm_classify(self, query: str) -> RoutingDecision:
        """Use GPT to classify the query."""
        client = self._get_client()

        column_hint = ""
        if self._who_columns:
            column_hint = (
                f"\n\nThe structured dataset has these columns: {self._who_columns}"
            )

        response = await client.chat.completions.create(
            model=self._settings.llm_model,
            max_completion_tokens=200,
            temperature=1,  # gpt-5-mini only supports default (1)
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT + column_hint},
                {"role": "user", "content": query},
            ],
        )

        raw = response.choices[0].message.content or "{}"
        # Strip markdown fences if present
        raw = re.sub(r"```json?\s*", "", raw).replace("```", "").strip()
        data = json.loads(raw)

        return RoutingDecision(
            query_type=QueryType(data.get("type", "text")),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
        )

    def _heuristic_classify(self, query: str) -> RoutingDecision:
        """Rule-based fallback classification.

        Uses keyword matching with a bias toward hybrid: if BOTH
        structured and text keywords are found (even one each),
        the query is classified as hybrid.
        """
        structured_hits = _STRUCTURED_KEYWORDS.findall(query)
        text_hits = _TEXT_KEYWORDS.findall(query)
        structured_score = len(structured_hits)
        text_score = len(text_hits)

        # Any overlap → hybrid (this is the key fix)
        if structured_score > 0 and text_score > 0:
            qtype = QueryType.HYBRID
        elif structured_score > 0:
            qtype = QueryType.STRUCTURED
        elif text_score > 0:
            qtype = QueryType.TEXT
        else:
            # No keywords matched — default to text (most queries
            # are about papers)
            qtype = QueryType.TEXT

        total = structured_score + text_score or 1
        confidence = min(1.0, max(structured_score, text_score) / total)

        logger.info(
            "[Router] Heuristic decision: type=%s (text=%d%s, struct=%d%s)",
            qtype.value,
            text_score,
            f" {text_hits}" if text_hits else "",
            structured_score,
            f" {structured_hits}" if structured_hits else "",
        )
        return RoutingDecision(
            query_type=qtype,
            confidence=confidence,
            reasoning=(
                f"Heuristic: text_keywords={text_score}, "
                f"structured_keywords={structured_score}"
            ),
        )
