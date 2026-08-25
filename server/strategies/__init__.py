"""
Package: server.strategies

Exposes Strategy Engine core contracts, strategies, registry, and framework adapters.
"""

from server.strategies.base import BaseStrategy, StrategyMetadata, FitResult, EvaluateResult
from server.strategies.registry import StrategyRegistry, StrategyRegistryError, register_strategy
from server.strategies.fedavg import FedAvg
from server.strategies.fedprox import FedProx
from server.strategies.adapters.flower_adapter import FlowerStrategyAdapter

__all__ = [
    "BaseStrategy",
    "StrategyMetadata",
    "FitResult",
    "EvaluateResult",
    "StrategyRegistry",
    "StrategyRegistryError",
    "register_strategy",
    "FedAvg",
    "FedProx",
    "FlowerStrategyAdapter",
]

