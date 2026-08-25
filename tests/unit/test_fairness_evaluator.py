"""
Module: tests.unit.test_fairness_evaluator

Purpose:
Unit test suite for FairnessEvaluator.
"""

import pytest
from fairness.evaluator import FairnessEvaluator


def test_fairness_evaluation():
    evaluator = FairnessEvaluator()
    hospital_metrics = {
        "hospital_alpha": {"dice_score": 0.92},
        "hospital_beta": {"dice_score": 0.90},
        "hospital_gamma": {"dice_score": 0.88},
    }

    res = evaluator.evaluate_fairness(hospital_metrics)
    assert res["disparate_impact_ratio"] >= 0.80
    assert res["equal_opportunity_met"] is True
    assert "scanner_breakdown" in res
