"""
Module: dashboard.backend.app.core.logging

Purpose:
Configures structured, application-wide logging.
Ensures consistency in log formats across the API, allowing easier integration
with log aggregators (e.g., ELK stack or DataDog).

TODO:
- [ ] Switch to a JSON-based log formatter for production.
"""

import logging
import sys


def setup_logging() -> None:
    """Configures root logger with standard formatting."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Silence third-party noisy loggers if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
