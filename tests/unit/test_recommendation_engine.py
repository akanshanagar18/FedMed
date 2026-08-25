"""
Module: tests.unit.test_recommendation_engine

Purpose:
Unit test suite for OperationalRecommendationEngine (trigger HPO, reduce LR, promote model).
"""

import pytest
from analytics.recommendation_engine import OperationalRecommendationEngine


def test_recommendation_engine_generation():
    engine = OperationalRecommendationEngine()
    recs = engine.generate_recommendations(
        current_dice=0.80,
        drift_mmd=0.14,
        epsilon_consumed=7.0,
        avg_latency_ms=150.0,
        candidate_dice=0.87,
    )
    assert len(recs) >= 3
    actions = [r["action"] for r in recs]
    assert "TRIGGER_HPO" in actions
    assert "REDUCE_LEARNING_RATE" in actions
    assert "PROMOTE_CANDIDATE_MODEL" in actions
