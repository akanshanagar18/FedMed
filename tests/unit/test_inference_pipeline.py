"""
Module: tests.unit.test_inference_pipeline

Purpose:
Unit test suite for BraTSInferenceEngine (predict, history, sliding window MONAI inference).
"""

import pytest
from inference.pipeline import BraTSInferenceEngine, global_inference_engine


def test_brats_inference_engine_prediction():
    engine = global_inference_engine
    res = engine.predict(patient_id="TEST_PATIENT_101", model_version="v2.0-test")
    assert res["status"] == "SUCCESS"
    assert res["patient_id"] == "TEST_PATIENT_101"
    assert "enhancing_tumor_et" in res["tumor_volumes_mm3"]

    history = engine.get_history()
    assert len(history) > 0
