"""
Module: tests.unit.test_telemetry

Purpose:
Unit test suite for system resource metrics and Prometheus text generation (utils/telemetry.py).
"""

import pytest
from utils.telemetry import get_system_resource_metrics, prometheus_registry


def test_get_system_resource_metrics():
    metrics = get_system_resource_metrics()
    assert "cpu_percent" in metrics
    assert "ram_percent" in metrics
    assert "disk_percent" in metrics
    assert "gpu" in metrics
    assert metrics["cpu_count"] > 0


def test_prometheus_registry_exposition():
    prometheus_registry.set_metric("fedmed_test_metric", 42.5)
    prom_text = prometheus_registry.generate_prometheus_text()

    assert "fedmed_system_cpu_percent" in prom_text
    assert "fedmed_test_metric 42.5" in prom_text
