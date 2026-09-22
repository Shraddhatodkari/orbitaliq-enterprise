"""Structured logging setup built on loguru.

Enterprise deployments typically ship logs to a centralized sink (ELK,
Datadog, CloudWatch); routing there is a one-line ``logger.add(...)`` change
because every call site already uses the shared ``logger`` from this module.
"""
from __future__ import annotations

import sys

from loguru import logger

from orbitaliq.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )
    _CONFIGURED = True


__all__ = ["logger", "configure_logging"]
