"""
Module: tests.unit.test_statistical_analysis

Purpose:
Unit tests for StatisticalAnalysisEngine, hypothesis testing (paired t-test, Wilcoxon), and effect sizes.
"""

import pytest
from utils.statistical_analysis import (
    compute_descriptive_stats,
    compute_cohens_d,
    compute_cliffs_delta,
    perform_hypothesis_tests,
    StatisticalAnalysisEngine,
)


def test_compute_descriptive_stats():
    values = [0.85, 0.88, 0.89, 0.87, 0.90]
    stats = compute_descriptive_stats(values)
    assert stats["mean"] == 0.878
    assert stats["std"] > 0
    assert stats["ci_lower"] < stats["mean"] < stats["ci_upper"]
    assert stats["count"] == 5


def test_effect_sizes():
    sample_a = [0.85, 0.88, 0.89, 0.87]
    sample_b = [0.75, 0.78, 0.79, 0.77]
    d = compute_cohens_d(sample_a, sample_b)
    delta = compute_cliffs_delta(sample_a, sample_b)

    assert d > 0
    assert delta == 1.0  # complete stochastic dominance


def test_perform_hypothesis_tests():
    group_a = [0.88, 0.89, 0.90, 0.91]
    group_b = [0.78, 0.79, 0.80, 0.81]
    res = perform_hypothesis_tests(group_a, group_b, "FedProx", "FedAvg")

    assert res["t_statistic"] > 0
    assert res["p_value_ttest"] < 0.05
    assert res["statistically_significant"] is True


def test_statistical_analysis_engine_rankings():
    results = [
        {"strategy_name": "FedAvg", "best_dice": 0.82, "avg_loss": 0.20, "runtime_sec": 10.0, "convergence_round": 3},
        {"strategy_name": "FedProx", "best_dice": 0.89, "avg_loss": 0.12, "runtime_sec": 12.0, "convergence_round": 2},
        {"strategy_name": "FedAvg", "best_dice": 0.84, "avg_loss": 0.18, "runtime_sec": 10.5, "convergence_round": 3},
        {"strategy_name": "FedProx", "best_dice": 0.91, "avg_loss": 0.10, "runtime_sec": 11.5, "convergence_round": 2},
    ]

    engine = StatisticalAnalysisEngine(results)
    ranks = engine.rank_strategies()

    assert len(ranks) == 2
    assert ranks[0]["strategy"] == "FedProx"  # higher mean dice
    assert ranks[0]["rank"] == 1
