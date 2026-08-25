"""
Unit tests for server.strategies.registry StrategyRegistry.
Verifies registration, decorator, auto-discovery, creation, and metadata listing.
"""

import pytest
from server.strategies.base import BaseStrategy, StrategyMetadata, FitResult, EvaluateResult
from server.strategies.registry import StrategyRegistry, StrategyRegistryError, register_strategy


def test_strategy_registry_list_and_discovery():
    """Verify built-in strategies FedAvg and FedProx are auto-discovered and listed."""
    strategies = StrategyRegistry.list_strategies()
    assert "FedAvg" in strategies
    assert "FedProx" in strategies
    assert strategies["FedAvg"].name == "FedAvg"
    assert strategies["FedProx"].name == "FedProx"


def test_strategy_registry_get_case_insensitive():
    """Verify strategy retrieval is case-insensitive."""
    cls1 = StrategyRegistry.get("FedAvg")
    cls2 = StrategyRegistry.get("fedavg")
    assert cls1 == cls2


def test_strategy_registry_create_instance():
    """Verify instantiation via StrategyRegistry.create."""
    fedavg_instance = StrategyRegistry.create("FedAvg", min_fit_clients=3)
    assert fedavg_instance.min_fit_clients == 3
    meta = fedavg_instance.get_metadata()
    assert meta.name == "FedAvg"

    fedprox_instance = StrategyRegistry.create("FedProx", proximal_mu=0.05)
    assert fedprox_instance.proximal_mu == 0.05
    meta_prox = fedprox_instance.get_metadata()
    assert meta_prox.name == "FedProx"


def test_strategy_registry_unknown_strategy_raises_error():
    """Verify requesting an unknown strategy raises StrategyRegistryError."""
    with pytest.raises(StrategyRegistryError):
        StrategyRegistry.get("NonExistentStrategyAlgorithm")


def test_register_custom_strategy_decorator():
    """Verify custom experimental strategy can be registered via @register_strategy."""
    @register_strategy("ExperimentalStrategy")
    class ExperimentalStrategy(BaseStrategy):
        def get_metadata(self) -> StrategyMetadata:
            return StrategyMetadata(
                name="ExperimentalStrategy",
                description="Custom experimental algorithm",
            )
        def aggregate_fit(self, server_round, results, failures):
            return None, {}
        def aggregate_evaluate(self, server_round, results, failures):
            return None, {}

    retrieved = StrategyRegistry.get("ExperimentalStrategy")
    assert retrieved == ExperimentalStrategy
    inst = StrategyRegistry.create("ExperimentalStrategy")
    assert inst.get_metadata().name == "ExperimentalStrategy"
