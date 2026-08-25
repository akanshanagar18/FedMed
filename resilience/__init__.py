"""
Module: resilience

Purpose:
Self-Healing Recovery Engine for automated runtime resilience, node reconnection,
aggregation retries, and checkpoint restoration in FedMed v2.0.
"""

from resilience.self_healing import SelfHealingRecoveryEngine

__all__ = ["SelfHealingRecoveryEngine"]
