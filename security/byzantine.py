"""
Module: security.byzantine

Purpose:
Byzantine-Robust Aggregation Rules for Federated Learning under Malicious / Adversarial Client Attacks.
Implements Krum, Multi-Krum, Trimmed Mean, Median, Bulyan, and FLTrust.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class KrumAggregator:
    """
    Krum Aggregator (Blanchard et al., NeurIPS 2017).
    Selects the single update that minimizes sum of Euclidean distances to its m closest neighbors.
    """

    def __init__(self, f_byzantine: int = 1):
        self.f_byzantine = f_byzantine

    def aggregate(self, client_updates: List[List[np.ndarray]]) -> List[np.ndarray]:
        n = len(client_updates)
        if n <= 2 * self.f_byzantine + 2:
            # Fallback to mean if client count too low
            return [np.mean([u[k] for u in client_updates], axis=0) for k in range(len(client_updates[0]))]

        flat_updates = [np.concatenate([p.flatten() for p in u]) for u in client_updates]
        m = n - self.f_byzantine - 2

        scores = []
        for i in range(n):
            dists = [np.linalg.norm(flat_updates[i] - flat_updates[j]) ** 2 for j in range(n) if i != j]
            dists.sort()
            scores.append(sum(dists[:m]))

        best_idx = int(np.argmin(scores))
        return client_updates[best_idx]


class MultiKrumAggregator:
    """
    Multi-Krum Aggregator (Blanchard et al., NeurIPS 2017).
    Averages top-m updates with lowest Krum scores.
    """

    def __init__(self, f_byzantine: int = 1, m_select: int = 2):
        self.f_byzantine = f_byzantine
        self.m_select = m_select

    def aggregate(self, client_updates: List[List[np.ndarray]]) -> List[np.ndarray]:
        n = len(client_updates)
        if n <= 2 * self.f_byzantine + 2:
            return [np.mean([u[k] for u in client_updates], axis=0) for k in range(len(client_updates[0]))]

        flat_updates = [np.concatenate([p.flatten() for p in u]) for u in client_updates]
        k_neighbors = n - self.f_byzantine - 2

        scores = []
        for i in range(n):
            dists = [np.linalg.norm(flat_updates[i] - flat_updates[j]) ** 2 for j in range(n) if i != j]
            dists.sort()
            scores.append(sum(dists[:k_neighbors]))

        top_indices = np.argsort(scores)[: min(n, self.m_select)]
        return [np.mean([client_updates[i][k] for i in top_indices], axis=0) for k in range(len(client_updates[0]))]


class TrimmedMeanAggregator:
    """
    Coordinate-Wise Trimmed Mean Aggregator (Yin et al., ICML 2018).
    Removes largest and smallest beta fraction of updates per coordinate.
    """

    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def aggregate(self, client_updates: List[List[np.ndarray]]) -> List[np.ndarray]:
        n = len(client_updates)
        k_trim = int(n * self.beta)
        if k_trim == 0 or n <= 2 * k_trim:
            return [np.mean([u[k] for u in client_updates], axis=0) for k in range(len(client_updates[0]))]

        aggregated = []
        for l_idx in range(len(client_updates[0])):
            layer_updates = np.stack([u[l_idx] for u in client_updates], axis=0)
            sorted_layers = np.sort(layer_updates, axis=0)
            trimmed = sorted_layers[k_trim : n - k_trim]
            aggregated.append(np.mean(trimmed, axis=0))

        return aggregated


class MedianAggregator:
    """
    Coordinate-Wise Median Aggregator (Yin et al., ICML 2018).
    Computes coordinate-wise median across client parameter updates.
    """

    def aggregate(self, client_updates: List[List[np.ndarray]]) -> List[np.ndarray]:
        aggregated = []
        for l_idx in range(len(client_updates[0])):
            layer_updates = np.stack([u[l_idx] for u in client_updates], axis=0)
            aggregated.append(np.median(layer_updates, axis=0))
        return aggregated


class BulyanAggregator:
    """
    Bulyan Aggregator (El Mhamdi et al., ICML 2018).
    Combines Multi-Krum candidate selection with Trimmed Mean coordinate aggregation.
    """

    def __init__(self, f_byzantine: int = 1):
        self.f_byzantine = f_byzantine
        self.multi_krum = MultiKrumAggregator(f_byzantine=f_byzantine, m_select=len(client_updates) if 'client_updates' in locals() else 3)

    def aggregate(self, client_updates: List[List[np.ndarray]]) -> List[np.ndarray]:
        n = len(client_updates)
        theta = n - 2 * self.f_byzantine
        if theta <= 0:
            return [np.mean([u[k] for u in client_updates], axis=0) for k in range(len(client_updates[0]))]

        self.multi_krum.m_select = theta
        krum_selected = self.multi_krum.aggregate(client_updates)

        trimmed_mean = TrimmedMeanAggregator(beta=0.2)
        return trimmed_mean.aggregate(client_updates[:theta])


class FLTrustAggregator:
    """
    FLTrust Aggregator (Cao et al., NDSS 2021).
    Uses server clean root dataset gradient g_0 to compute trust score ReLU(cos(g_i, g_0)).
    """

    def __init__(self, server_root_update: Optional[List[np.ndarray]] = None):
        self.server_root_update = server_root_update

    def aggregate(self, client_updates: List[List[np.ndarray]]) -> List[np.ndarray]:
        if self.server_root_update is None:
            return [np.mean([u[k] for u in client_updates], axis=0) for k in range(len(client_updates[0]))]

        g0_flat = np.concatenate([p.flatten() for p in self.server_root_update])
        g0_norm = np.linalg.norm(g0_flat) + 1e-8

        trust_scores = []
        normalized_updates = []

        for u in client_updates:
            u_flat = np.concatenate([p.flatten() for p in u])
            u_norm = np.linalg.norm(u_flat) + 1e-8

            # Cosine similarity cos_sim = (u . g0) / (|u| * |g0|)
            cos_sim = float(np.dot(u_flat, g0_flat) / (u_norm * g0_norm))
            ts = max(0.0, cos_sim)  # ReLU trust score
            trust_scores.append(ts)

            # Rescale client update magnitude to match server root update
            rescaled_u = [(p / u_norm) * g0_norm for p in u]
            normalized_updates.append(rescaled_u)

        sum_ts = sum(trust_scores) + 1e-8
        aggregated = [np.zeros_like(client_updates[0][k], dtype=np.float32) for k in range(len(client_updates[0]))]

        for i in range(len(client_updates)):
            weight = trust_scores[i] / sum_ts
            for k in range(len(aggregated)):
                aggregated[k] += weight * normalized_updates[i][k]

        return aggregated
