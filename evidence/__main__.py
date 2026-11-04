"""`python -m evidence` serves the app with Waitress."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from waitress import serve

from .app import create_app
from .config import load_settings


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = load_settings()
    app = create_app(settings)
    logging.getLogger("evidence").info(
        "metacare-evidence listening on :%d model=%s claude_configured=%s ncbi_key=%s origins=%d",
        settings.port, settings.model, settings.has_anthropic_key, bool(settings.ncbi_api_key), len(settings.allowed_origins),
    )
    serve(app, host="0.0.0.0", port=settings.port, threads=8, channel_timeout=180)


if __name__ == "__main__":
    main()
