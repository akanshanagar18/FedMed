"""
Module: server.strategies.adapters

Exports Flower framework adapters bridging BaseStrategy implementations to flwr server strategies.
"""

from server.strategies.adapters.flower_adapter import FlowerStrategyAdapter

__all__ = ["FlowerStrategyAdapter"]
