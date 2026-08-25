"""
Module: tests.unit.test_helm_chart

Purpose:
Unit test suite validating Helm chart declaration, values, and templates under helm/fedmed/.
"""

import os
import glob
import yaml
import pytest


def test_helm_chart_declaration_and_values():
    chart_path = "helm/fedmed/Chart.yaml"
    values_path = "helm/fedmed/values.yaml"

    assert os.path.exists(chart_path)
    assert os.path.exists(values_path)

    with open(chart_path, "r", encoding="utf-8") as f:
        chart_data = yaml.safe_load(f)
        assert chart_data["name"] == "fedmed"
        assert chart_data["version"] == "2.0.0"
        assert chart_data["apiVersion"] == "v2"

    with open(values_path, "r", encoding="utf-8") as f:
        values_data = yaml.safe_load(f)
        assert "backend" in values_data
        assert "flowerServer" in values_data
        assert "dashboard" in values_data
        assert "hospitals" in values_data


def test_helm_templates_exist():
    templates = glob.glob("helm/fedmed/templates/*.yaml")
    assert len(templates) >= 5
