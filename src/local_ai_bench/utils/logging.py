"""Structured logging helpers."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "local_ai_bench"
_DEFAULT_LOG_DIR = "logs"


def setup_logging(level: str = "info", log_file: str | None = None) -> None:
    """Configure stderr + optional rotating file logging.

    ``log_file`` may be an absolute path or a directory. Environment variable
    ``LOG_FILE`` is honored (defaults to ``logs/local-ai-bench.log``) so the API
    process captures orchestrator warnings to disk, not just stderr.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return
    level_name = level.upper()
    logger.setLevel(level_name)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is None:
        log_file = os.getenv("LOG_FILE", os.path.join(_DEFAULT_LOG_DIR, "local-ai-bench.log"))
    try:
        log_path = Path(log_file)
        if log_path.is_dir():
            log_path = log_path / "local-ai-bench.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Logging must never break startup; stderr still works.
        pass


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")