"""
Module: tests.unit.test_policy_engine

Purpose:
Unit test suite for EnterprisePolicyEngine (loading configs, policy lookup, hot-reloading).
"""

import pytest
from configs.policy_engine import EnterprisePolicyEngine, global_policy_engine


def test_policy_engine_loading_and_retrieval():
    engine = EnterprisePolicyEngine()
    gov_policy = engine.get_governance_policy()
    assert "mmd_drift_warning_threshold" in gov_policy
    assert engine.get_policy("governance", "significance_level") == 0.05

    sla_policy = engine.get_sla_policy()
    assert sla_policy["max_epsilon"] == 10.0


def test_policy_engine_reload():
    initial_count = global_policy_engine.reload_count
    policies = global_policy_engine.reload_policies()
    assert global_policy_engine.reload_count == initial_count + 1
    assert "governance" in policies
