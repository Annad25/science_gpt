"""
Entry-point script for running the Science GPT service.

Usage:
    python run.py
"""

import uvicorn

from app.config import get_settings


def main() -> None:
    """Launch the FastAPI server."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
