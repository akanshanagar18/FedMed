"""
Module: tests.unit.test_model_exporter

Purpose:
Unit test suite for ProductionModelExporter (TorchScript export and governance certificate).
"""

import os
import pytest
from deployment.exporter import ProductionModelExporter, global_model_exporter


def test_production_model_exporter():
    exporter = global_model_exporter
    res = exporter.export_model(model_name="test_brats_unet", version="v1.0-test")
    assert res["success"] is True
    assert os.path.exists(res["torchscript_path"])
    assert "governance_certificate" in res
    assert res["governance_certificate"]["hipaa_compliant"] is True
