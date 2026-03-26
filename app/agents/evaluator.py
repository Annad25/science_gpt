"""
Evaluator Agent — hallucination detection and answer verification.

Uses an LLM-as-judge pattern: a separate LLM call checks whether every
claim in the generated answer is supported by the retrieved sources.
This is the "jury" step that catches confidently-wrong outputs.
"""

from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from app.config import get_settings
from app.logging_cfg import get_logger
from app.models import EvalVerdict, EvaluationResult, RetrievedChunk

logger = get_logger(__name__)

_EVALUATOR_SYSTEM_PROMPT = """\
You are a strict fact-checker for a scientific question-answering system.
Your job is to verify whether an AI-generated answer is fully supported
by the provided source material.

For each claim in the answer:
1. Check if it is directly stated or logically implied by the sources.
2. Flag any claim that is not supported, exaggerated, or fabricated.
3. Pay special attention to numbers, dates, statistics, and named entities.

Respond with ONLY a JSON object:
{{
  "verdict": "supported" | "partially_supported" | "not_supported",
  "confidence": <0.0 to 1.0>,
  "issues": ["<description of each unsupported claim>"]
}}

Rules:
- "supported": every claim in the answer traces back to the sources.
- "partially_supported": most claims are supported but some are unsupported.
- "not_supported": the answer contains significant fabrications or the
  sources do not cover the question at all.
- If the answer says "I don't know", "insufficient information", or
  "cannot be answered from the available content", that is
  considered "supported" (honest refusal).
- If the answer addresses some parts and honestly states which parts
  lack sufficient source data, that is "supported" (partial with
  honest disclosure).
"""


class EvaluatorAgent:
    """Verifies generated answers against source material."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-init the async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._client

    async def evaluate(
        self,
        query: str,
        answer: str,
        sources: list[RetrievedChunk],
    ) -> EvaluationResult:
        """Judge whether ``answer`` is supported by ``sources``.

        Args:
            query: Original user question.
            answer: Generated answer to verify.
            sources: Retrieved chunks that were provided as context.

        Returns:
            An ``EvaluationResult`` with verdict, confidence, and issues.
        """
        logger.info("[Evaluator] Verifying answer (%d chars) against %d sources", len(answer), len(sources))

        # Format sources for the judge
        source_text = self._format_sources(sources)

        user_prompt = (
            f"## User Question\n{query}\n\n"
            f"## AI Answer\n{answer}\n\n"
            f"## Source Material\n{source_text}\n\n"
            f"Evaluate the AI answer against the source material."
        )

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._settings.llm_model,
                max_completion_tokens=500,
                temperature=1,  # gpt-5-mini only supports default (1)
                messages=[
                    {"role": "system", "content": _EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw = response.choices[0].message.content or "{}"
            raw = re.sub(r"```json?\s*", "", raw).replace("```", "").strip()
            data = json.loads(raw)

            result = EvaluationResult(
                verdict=EvalVerdict(data.get("verdict", "supported")),
                confidence=float(data.get("confidence", 0.5)),
                issues=data.get("issues", []),
            )
            logger.info(
                "[Evaluator] Verdict=%s confidence=%.2f issues=%d",
                result.verdict.value,
                result.confidence,
                len(result.issues),
            )
            return result

        except Exception:
            logger.exception("[Evaluator] Evaluation failed — defaulting to partial support")
            return EvaluationResult(
                verdict=EvalVerdict.PARTIALLY_SUPPORTED,
                confidence=0.0,
                issues=["Evaluation failed — could not verify answer"],
            )

    @staticmethod
    def _format_sources(sources: list[RetrievedChunk], max_chars: int = 8000) -> str:
        """Format source chunks for the judge, respecting a char budget."""
        parts: list[str] = []
        total = 0
        for i, s in enumerate(sources, 1):
            entry = f"[Source {i}] ({s.source})\n{s.text}\n"
            if total + len(entry) > max_chars:
                parts.append(f"... ({len(sources) - i + 1} sources omitted)")
                break
            parts.append(entry)
            total += len(entry)
        return "\n".join(parts)
