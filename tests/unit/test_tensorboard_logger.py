"""
Module: tests.unit.test_tensorboard_logger

Purpose:
Unit test suite for TensorBoardLogger module.
"""

import os
import pytest
from utils.tensorboard_logger import TensorBoardLogger


def test_tensorboard_logger_creation(tmp_path):
    log_dir = str(tmp_path / "runs")
    logger = TensorBoardLogger(experiment_id="test_exp_001", log_dir=log_dir)
    assert os.path.exists(logger.log_path)

    logger.log_scalar("Loss/train", 0.45, 1)
    logger.log_round_metrics(
        global_step=1,
        training_loss=0.45,
        val_loss=0.42,
        dice=0.88,
        iou=0.79,
        learning_rate=1e-4,
        privacy_budget=1.2,
        gpu_memory_mb=2048.0,
    )
    logger.flush()
    logger.close()
