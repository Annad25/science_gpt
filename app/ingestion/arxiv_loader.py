"""
arXiv paper loader.

Fetches scientific papers from the arXiv API, downloads full PDFs,
extracts text via PyMuPDF (fitz), and returns them as document dicts
ready for chunking.  Falls back to abstract-only when PDF extraction
fails (network error, corrupt PDF, etc.).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import arxiv

from app.config import get_settings
from app.logging_cfg import get_logger

logger = get_logger(__name__)


async def fetch_arxiv_papers(
    queries: list[str] | None = None,
    max_papers_per_query: int | None = None,
    cached_papers: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Fetch papers from arXiv for each configured search query.

    Uses arXiv search metadata to determine candidate papers first.
    For papers already present in ``cached_papers``, reuses the cached
    document instead of re-downloading the PDF.  For uncached papers,
    downloads the full PDF and extracts the body text.  If PDF extraction
    fails for any paper, it falls back to using the abstract only
    (prefixed by the title).

    Args:
        queries: Search terms.  Falls back to ``Settings.arxiv_queries``.
        max_papers_per_query: Cap per query.  Falls back to
            ``Settings.arxiv_max_papers``.
        cached_papers: Optional arXiv-ID to paper-document map used to
            skip PDF downloads for already cached papers.

    Returns:
        List of document dicts with keys ``title``, ``text``, ``source``,
        ``authors``, ``published``, ``arxiv_id``, ``categories``.
    """
    settings = get_settings()
    queries = queries or settings.arxiv_queries
    max_papers = max_papers_per_query or settings.arxiv_max_papers
    cached_papers = cached_papers or {}

    documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for query in queries:
        logger.info("[Ingestion] Fetching arXiv papers for query='%s'", query)
        try:
            papers = await asyncio.to_thread(
                _search_arxiv,
                query,
                max_papers,
                cached_papers,
            )
        except Exception as e:
            logger.exception("arXiv search failed for query='%s'", query)
            raise e

        for paper in papers:
            if paper["arxiv_id"] in seen_ids:
                continue
            seen_ids.add(paper["arxiv_id"])
            documents.append(paper)

    logger.info(
        "[Ingestion] Fetched %d unique arXiv papers across %d queries",
        len(documents),
        len(queries),
    )
    return documents


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF (fitz).

    Args:
        pdf_path: Path to the downloaded PDF file.

    Returns:
        Concatenated text from all pages, or empty string on failure.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning(
            "[Ingestion] PyMuPDF not installed — cannot extract PDF text. "
            "Install with: pip install PyMuPDF"
        )
        return ""

    try:
        doc = fitz.open(pdf_path)
        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text.strip())
        doc.close()
        full_text = "\n\n".join(pages)
        logger.debug(
            "[Ingestion] Extracted %d chars from %d pages in %s",
            len(full_text),
            len(pages),
            pdf_path,
        )
        return full_text
    except Exception:
        logger.exception("[Ingestion] PDF text extraction failed for %s", pdf_path)
        return ""


def _search_arxiv(
    query: str,
    max_results: int,
    cached_papers: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Synchronous wrapper around the ``arxiv`` library (runs in a thread).

    For each paper, checks whether the arXiv ID already exists in
    ``cached_papers``.  If so, reuses the cached document.  Otherwise,
    downloads the PDF to a temp directory and extracts full text.
    Falls back to abstract if extraction fails.
    """
    cached_papers = cached_papers or {}
    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="arxiv_pdfs_") as tmpdir:
        for paper in client.results(search):
            cached = cached_papers.get(paper.entry_id)
            if cached is not None:
                results.append(cached)
                logger.info(
                    "[Ingestion] Reused cached paper '%s' (%s)",
                    paper.title[:60],
                    paper.entry_id,
                )
                continue

            # Always have the abstract as baseline
            abstract_text = f"Title: {paper.title}\n\nAbstract: {paper.summary}"

            # Attempt full PDF download and extraction
            full_text = ""
            try:
                pdf_filename = paper.get_short_id().replace("/", "_") + ".pdf"
                pdf_path = paper.download_pdf(dirpath=tmpdir, filename=pdf_filename)
                logger.info(
                    "[Ingestion] Downloaded PDF for '%s' (%s)",
                    paper.title[:60],
                    pdf_filename,
                )
                full_text = _extract_pdf_text(str(pdf_path))
            except Exception:
                logger.warning(
                    "[Ingestion] PDF download failed for '%s' — using abstract only",
                    paper.title[:60],
                )

            # Use full text if extraction succeeded and is substantial,
            # otherwise fall back to abstract.  Prepend title either way.
            if full_text and len(full_text) > len(abstract_text):
                text = f"Title: {paper.title}\n\n{full_text}"
                text_source = "full_pdf"
            else:
                text = abstract_text
                text_source = "abstract_only"

            results.append(
                {
                    "title": paper.title,
                    "text": text,
                    "source": f"arXiv:{paper.entry_id}",
                    "authors": [a.name for a in paper.authors],
                    "published": paper.published.isoformat() if paper.published else "",
                    "arxiv_id": paper.entry_id,
                    "categories": list(paper.categories),
                    "text_source": text_source,
                }
            )
            logger.info(
                "[Ingestion] Paper '%s' — %s (%d chars)",
                paper.title[:60],
                text_source,
                len(text),
            )

    return results
