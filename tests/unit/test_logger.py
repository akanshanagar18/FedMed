"""
Module: tests.unit.test_logger

Purpose:
Unit test suite for centralized structured logger (utils/logger.py).
"""

import os
import pytest
from utils.logger import setup_structured_logger


def test_structured_logger_creation_and_file_logging(tmp_path):
    log_dir = str(tmp_path / "logs")
    logger = setup_structured_logger(name="test_logger", log_dir=log_dir, log_filename="test.log")

    logger.info("Test structured info log message", extra={"subsystem": "TEST", "experiment_id": "exp_123"})
    log_file = os.path.join(log_dir, "test.log")

    assert os.path.exists(log_file)
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Test structured info log message" in content
        assert "exp_123" in content
