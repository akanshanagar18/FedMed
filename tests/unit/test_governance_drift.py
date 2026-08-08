"""
Module: tests.unit.test_governance_drift

Purpose:
Unit test suite for FederatedDriftDetector (MMD, KS-test, Wasserstein, PSI, and diagnostic report generation).
"""

import pytest
import numpy as np
from governance.drift import FederatedDriftDetector


def test_drift_detector_initialization():
    detector = FederatedDriftDetector(significance_level=0.05, psi_threshold=0.2)
    assert detector.significance_level == 0.05
    assert detector.psi_threshold == 0.2


def test_compute_mmd_identical_distributions():
    detector = FederatedDriftDetector()
    np.random.seed(42)
    data = np.random.normal(0, 1, size=(50, 10))
    mmd = detector.compute_mmd(data, data)
    assert mmd == pytest.approx(0.0, abs=1e-3)


def test_compute_mmd_different_distributions():
    detector = FederatedDriftDetector()
    np.random.seed(42)
    data1 = np.random.normal(0, 1, size=(50, 10))
    data2 = np.random.normal(2, 1, size=(50, 10))
    mmd = detector.compute_mmd(data1, data2)
    assert mmd > 0.10


def test_compute_ks_test():
    detector = FederatedDriftDetector()
    np.random.seed(42)
    data1 = np.random.normal(0, 1, size=100)
    data2 = np.random.normal(0, 1, size=100)
    ks_res = detector.compute_ks_test(data1, data2)
    assert "ks_statistic" in ks_res
    assert "p_value" in ks_res
    assert ks_res["is_drift"] is False


def test_compute_wasserstein():
    detector = FederatedDriftDetector()
    data1 = np.array([1.0, 2.0, 3.0, 4.0])
    data2 = np.array([2.0, 3.0, 4.0, 5.0])
    w1 = detector.compute_wasserstein(data1, data2)
    assert w1 == pytest.approx(1.0, abs=0.1)


def test_compute_psi():
    detector = FederatedDriftDetector()
    np.random.seed(42)
    data1 = np.random.normal(0, 1, size=500)
    data2 = np.random.normal(0, 1, size=500)
    psi = detector.compute_psi(data1, data2)
    assert psi < 0.10


def test_detect_drift_full_diagnostic():
    detector = FederatedDriftDetector()
    np.random.seed(42)
    ref = np.random.normal(0, 1, size=(100, 16))
    cur = np.random.normal(0, 1, size=(100, 16))
    report = detector.detect_drift(ref, cur, node_id="hospital_alpha")
    assert report["node_id"] == "hospital_alpha"
    assert "drift_detected" in report
    assert "risk_level" in report
    assert "metrics" in report
    assert "mmd" in report["metrics"]
    assert "psi" in report["metrics"]
