"""
Module: utils.logger

Purpose:
Production Centralized Structured Logging Engine for FedMed v2.0.
Supports JSON logs, colored ANSI console logs, rotating file logging (logs/fedmed.log),
and contextual fields (request_id, experiment_id, hospital_id, round_id).
"""

import os
import sys
import json
import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON objects for SIEM / ELK / Grafana Loki ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "subsystem": getattr(record, "subsystem", "FEDMED"),
            "experiment_id": getattr(record, "experiment_id", "N/A"),
            "hospital_id": getattr(record, "hospital_id", "N/A"),
            "round_id": getattr(record, "round_id", "N/A"),
            "request_id": getattr(record, "request_id", "N/A"),
            "filename": record.filename,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


class ColoredConsoleFormatter(logging.Formatter):
    """Formats log records with ANSI colors for readable terminal logs."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        sub = getattr(record, "subsystem", "FEDMED")
        h_id = getattr(record, "hospital_id", None)
        r_id = getattr(record, "round_id", None)

        prefix = f"{ts} [{color}{record.levelname.center(7)}{self.RESET}] [{sub.center(12)}]"
        if h_id and h_id != "N/A":
            prefix += f" [{h_id}]"
        if r_id and r_id != "N/A":
            prefix += f" [Round {r_id}]"

        return f"{prefix} {record.getMessage()}"


def setup_structured_logger(
    name: str = "fedmed",
    log_dir: str = "logs",
    log_filename: str = "fedmed.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configures and returns a thread-safe structured logger with JSON and console handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    # Ensure log directory exists
    log_path = os.path.abspath(log_dir)
    os.makedirs(log_path, exist_ok=True)
    full_file_path = os.path.join(log_path, log_filename)

    # 1. Console Handler (Colored text)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredConsoleFormatter())
    logger.addHandler(console_handler)

    # 2. Rotating File Handler (Structured JSON)
    file_handler = RotatingFileHandler(
        full_file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger


# Default shared logger instance
get_logger = setup_structured_logger
