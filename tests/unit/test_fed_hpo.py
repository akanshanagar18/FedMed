"""
Module: tests.unit.test_fed_hpo

Purpose:
Unit test suite for HyperparameterSpace and FederatedHPOEngine (Successive Halving & Bayesian Suggestion).
"""

import pytest
from hpo.hpo_engine import HyperparameterSpace, FederatedHPOEngine


def test_hyperparameter_space_sampling():
    space = HyperparameterSpace()
    cfg = space.sample_config()
    assert "learning_rate" in cfg
    assert "proximal_mu" in cfg
    assert "ewc_lambda" in cfg
    assert "dp_noise_multiplier" in cfg
    assert "strategy" in cfg
    assert 1e-4 <= cfg["learning_rate"] <= 1e-2
    assert cfg["strategy"] in space.strategies


def test_hpo_engine_successive_halving():
    engine = FederatedHPOEngine(seed=42)
    study = engine.run_successive_halving(num_initial_configs=6, max_rounds=9)
    assert "best_config" in study
    assert "best_dice" in study
    assert study["best_dice"] > 0.0
    assert study["total_trials_evaluated"] > 0


def test_hpo_engine_bayesian_suggest():
    engine = FederatedHPOEngine(seed=42)
    history = [
        {"val_dice": 0.81, "config": engine.search_space.sample_config()},
        {"val_dice": 0.85, "config": engine.search_space.sample_config()},
    ]
    next_cand = engine.bayesian_suggest_next(history)
    assert "learning_rate" in next_cand
    assert "strategy" in next_cand
