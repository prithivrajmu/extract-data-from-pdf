"""Project-wide logging helpers."""

from __future__ import annotations

import logging
import os
from typing import Optional


DEFAULT_LOG_LEVEL = os.getenv("EC_APP_LOG_LEVEL", "INFO")
_configured = False


def configure_logging(level: Optional[str] = None) -> None:
    """Configure the root logger once with a structured format."""

    global _configured
    if _configured:
        return

    resolved_level = level or DEFAULT_LOG_LEVEL
    if isinstance(resolved_level, str):
        resolved_level = resolved_level.upper()

    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, ensuring configuration is applied."""

    configure_logging()
    return logging.getLogger(name)

