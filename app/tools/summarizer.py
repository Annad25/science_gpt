"""
Summarizer tool.

Condenses long text passages via the configured LLM.  Useful when
retrieved context exceeds the reasoning agent's effective window or
when the user asks for a concise overview.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.config import get_settings
from app.logging_cfg import get_logger

logger = get_logger(__name__)

_SUMMARIZE_PROMPT = """\
Summarise the following text concisely, preserving all key facts, numbers,
and conclusions.  Do not add information that is not in the text.

TEXT:
{text}

SUMMARY:"""


async def summarize_text(text: str, max_tokens: int = 512) -> str:
    """Summarise ``text`` using the configured LLM.

    Args:
        text: Long passage to condense.
        max_tokens: Cap on summary length.

    Returns:
        A concise summary string.
    """
    settings = get_settings()
    logger.info("[Summarizer] Summarising %d chars", len(text))

    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=settings.llm_model,
            max_completion_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise summariser.  Never hallucinate.",
                },
                {"role": "user", "content": _SUMMARIZE_PROMPT.format(text=text)},
            ],
        )
        summary = response.choices[0].message.content or ""
        logger.info("[Summarizer] Produced %d char summary", len(summary))
        return summary.strip()
    except Exception:
        logger.exception("[Summarizer] Failed — returning truncated original")
        return text[:1500] + "\n\n[Summary truncated due to error]"
