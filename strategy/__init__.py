"""
Module: strategy

Purpose:
Adaptive Strategy Selector Engine for automated dynamic switching between FedAvg, FedProx,
FedNova, FedAdam, FedYogi, Scaffold, and Mime algorithms.
"""

from strategy.adaptive_engine import AdaptiveStrategySelector

__all__ = ["AdaptiveStrategySelector"]
