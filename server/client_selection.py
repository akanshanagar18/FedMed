"""
Module: server.client_selection

Purpose:
Client selection engine for federated learning systems.
Implements policies (Random, Resource-Aware, Data-Aware, Fair Scheduling, Availability-Based, Reputation-Based)
to select optimal hospital client subsets under heterogeneous edge environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class ClientProfile:
    """Metadata container for a hospital client node."""

    def __init__(
        self,
        cid: str,
        num_examples: int = 100,
        cpu_capacity: float = 1.0,
        ram_gb: float = 8.0,
        gpu_available: bool = False,
        uptime_ratio: float = 0.99,
        avg_latency_ms: float = 50.0,
        historical_failures: int = 0,
        participation_count: int = 0,
    ):
        self.cid = cid
        self.num_examples = num_examples
        self.cpu_capacity = cpu_capacity
        self.ram_gb = ram_gb
        self.gpu_available = gpu_available
        self.uptime_ratio = uptime_ratio
        self.avg_latency_ms = avg_latency_ms
        self.historical_failures = historical_failures
        self.participation_count = participation_count


class BaseClientSelector(ABC):
    """Abstract Base Class for client selection policies."""

    @abstractmethod
    def select_clients(self, available_clients: List[ClientProfile], num_to_select: int) -> List[ClientProfile]:
        """Selects a subset of num_to_select clients from available_clients."""
        pass


class RandomSelector(BaseClientSelector):
    """Uniform random client selection policy."""

    def select_clients(self, available_clients: List[ClientProfile], num_to_select: int) -> List[ClientProfile]:
        if not available_clients:
            return []
        num = min(len(available_clients), num_to_select)
        indices = np.random.choice(len(available_clients), size=num, replace=False)
        return [available_clients[i] for i in indices]


class ResourceAwareSelector(BaseClientSelector):
    """Resource-aware client selection based on CPU, RAM, and GPU capacity."""

    def select_clients(self, available_clients: List[ClientProfile], num_to_select: int) -> List[ClientProfile]:
        if not available_clients:
            return []
        scores = []
        for c in available_clients:
            gpu_score = 2.0 if c.gpu_available else 1.0
            score = (c.cpu_capacity * 0.4) + (c.ram_gb * 0.4) + (gpu_score * 0.2)
            scores.append(score)

        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = sorted_indices[:num_to_select]
        return [available_clients[i] for i in selected_indices]


class DataAwareSelector(BaseClientSelector):
    """Data-volume aware client selection favoring clients with larger datasets."""

    def select_clients(self, available_clients: List[ClientProfile], num_to_select: int) -> List[ClientProfile]:
        if not available_clients:
            return []
        scores = [c.num_examples for c in available_clients]
        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = sorted_indices[:num_to_select]
        return [available_clients[i] for i in selected_indices]


class FairScheduler(BaseClientSelector):
    """Fair scheduling policy prioritizing clients with lowest historical participation count."""

    def select_clients(self, available_clients: List[ClientProfile], num_to_select: int) -> List[ClientProfile]:
        if not available_clients:
            return []
        scores = [c.participation_count for c in available_clients]
        sorted_indices = np.argsort(scores)  # ascending order: least participated first
        selected_indices = sorted_indices[:num_to_select]
        selected = [available_clients[i] for i in selected_indices]
        for s in selected:
            s.participation_count += 1
        return selected


class ReputationBasedSelector(BaseClientSelector):
    """Reputation-based policy penalizing high latency and past node failures."""

    def select_clients(self, available_clients: List[ClientProfile], num_to_select: int) -> List[ClientProfile]:
        if not available_clients:
            return []
        scores = []
        for c in available_clients:
            rep = (c.uptime_ratio * 100.0) - (c.avg_latency_ms * 0.1) - (c.historical_failures * 10.0)
            scores.append(rep)

        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = sorted_indices[:num_to_select]
        return [available_clients[i] for i in selected_indices]


class ClientSelectionEngine:
    """Master Client Selection Engine factory."""

    _POLICIES = {
        "random": RandomSelector,
        "resource_aware": ResourceAwareSelector,
        "data_aware": DataAwareSelector,
        "fair_scheduling": FairScheduler,
        "reputation_based": ReputationBasedSelector,
    }

    @classmethod
    def get_selector(cls, policy_name: str = "random") -> BaseClientSelector:
        policy_cls = cls._POLICIES.get(policy_name.lower(), RandomSelector)
        return policy_cls()
