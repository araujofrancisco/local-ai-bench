"""Structured logging helpers."""

from __future__ import annotations

import logging
import sys

_LOGGER_NAME = "ollama_bench"


def setup_logging(level: str = "info") -> None:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")