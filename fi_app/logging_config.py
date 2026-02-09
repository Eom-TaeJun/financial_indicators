#!/usr/bin/env python3
"""Logging configuration for financial_indicators."""

from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_logging() -> None:
    """Configure root logger once for CLI execution."""
    root = logging.getLogger()
    if getattr(root, "_fi_logging_configured", False):
        return

    level_name = os.getenv("FINANCIAL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [stream_handler]

    log_dir = Path("outputs") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "financial_indicators.log")
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers)
    root._fi_logging_configured = True
