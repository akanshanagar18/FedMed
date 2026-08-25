"""
Module: tests.unit.test_calibration_metrics

Purpose:
Unit test suite for CalibrationMetrics.
"""

import numpy as np
import pytest
from calibration.metrics import CalibrationMetrics


def test_ece_and_temperature_scaling():
    cm = CalibrationMetrics(num_bins=5)
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    acc = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    res = cm.compute_ece_and_brier(conf, acc)
    assert "expected_calibration_error" in res
    assert "brier_score" in res

    logits = np.array([[2.0, 1.0]], dtype=np.float32)
    scaled = cm.apply_temperature_scaling(logits, temperature=1.5)
    assert scaled.shape == (1, 2)
