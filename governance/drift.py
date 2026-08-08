"""
Module: governance.drift

Purpose:
Real-Time Data & Model Concept/Covariate Drift Engine for Cross-Silo Medical FL.
Computes Maximum Mean Discrepancy (MMD), Kolmogorov-Smirnov (KS) test, Wasserstein Distance,
and Population Stability Index (PSI) across hospital datasets and feature embeddings.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class FederatedDriftDetector:
    """
    Detects distribution drift between baseline reference data (e.g. initial hospital cohort)
    and current client features/predictions during federated training rounds.
    """

    def __init__(self, significance_level: float = 0.05, psi_threshold: float = 0.2):
        self.significance_level = significance_level
        self.psi_threshold = psi_threshold

    @staticmethod
    def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Computes Radial Basis Function (RBF) Gaussian kernel matrix between X and Y."""
        X_norm = np.sum(X**2, axis=1)[:, np.newaxis]
        Y_norm = np.sum(Y**2, axis=1)[np.newaxis, :]
        dist_sq = X_norm + Y_norm - 2.0 * np.dot(X, Y.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-gamma * dist_sq)

    def compute_mmd(
        self,
        reference_features: np.ndarray,
        current_features: np.ndarray,
        gamma: float = 1.0,
    ) -> float:
        """
        Computes Maximum Mean Discrepancy (MMD) with RBF kernel between reference and current embeddings.
        """
        ref = np.atleast_2d(reference_features)
        cur = np.atleast_2d(current_features)

        m = ref.shape[0]
        n = cur.shape[0]

        K_XX = self._rbf_kernel(ref, ref, gamma=gamma)
        K_YY = self._rbf_kernel(cur, cur, gamma=gamma)
        K_XY = self._rbf_kernel(ref, cur, gamma=gamma)

        mmd_sq = (np.sum(K_XX) / (m * m)) + (np.sum(K_YY) / (n * n)) - (2.0 * np.sum(K_XY) / (m * n))
        return float(np.sqrt(max(mmd_sq, 0.0)))

    def compute_ks_test(
        self,
        reference_scores: np.ndarray,
        current_scores: np.ndarray,
    ) -> Dict[str, float]:
        """
        Computes Kolmogorov-Smirnov (KS) two-sample test statistic and approximate p-value.
        """
        ref = np.sort(reference_scores.flatten())
        cur = np.sort(current_scores.flatten())

        n1 = len(ref)
        n2 = len(cur)

        if n1 == 0 or n2 == 0:
            return {"ks_statistic": 0.0, "p_value": 1.0, "is_drift": False}

        data_all = np.concatenate([ref, cur])
        cdf1 = np.searchsorted(ref, data_all, side="right") / n1
        cdf2 = np.searchsorted(cur, data_all, side="right") / n2

        ks_stat = float(np.max(np.abs(cdf1 - cdf2)))

        # Approximate p-value based on asymptotic Kolmogorov distribution
        en = np.sqrt(n1 * n2 / (n1 + n2))
        lambda_val = (en + 0.12 + 0.11 / en) * ks_stat
        
        # Kolmogorov cumulative distribution function approximation
        p_val = 0.0
        for k in range(1, 101):
            term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k * lambda_val) ** 2)
            p_val += term
            if abs(term) < 1e-8:
                break
        p_val = float(np.clip(p_val, 0.0, 1.0))

        return {
            "ks_statistic": round(ks_stat, 5),
            "p_value": round(p_val, 5),
            "is_drift": bool(p_val < self.significance_level),
        }

    def compute_wasserstein(
        self,
        reference_dist: np.ndarray,
        current_dist: np.ndarray,
    ) -> float:
        """
        Computes 1D Wasserstein distance (Earth Mover's Distance).
        """
        ref = np.sort(reference_dist.flatten())
        cur = np.sort(current_dist.flatten())

        n1 = len(ref)
        n2 = len(cur)

        all_vals = np.unique(np.concatenate([ref, cur]))

        cdf1 = np.searchsorted(ref, all_vals, side="right") / n1
        cdf2 = np.searchsorted(cur, all_vals, side="right") / n2

        deltas = np.diff(all_vals)
        w1 = np.sum(np.abs(cdf1[:-1] - cdf2[:-1]) * deltas)
        return float(round(w1, 5))

    def compute_psi(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        num_bins: int = 10,
    ) -> float:
        """
        Computes Population Stability Index (PSI) between reference and current feature distributions.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant drift detected
        """
        ref = reference_data.flatten()
        cur = current_data.flatten()

        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref, percentiles)
        bins[0] -= 1e-5
        bins[-1] += 1e-5

        ref_counts, _ = np.histogram(ref, bins=bins)
        cur_counts, _ = np.histogram(cur, bins=bins)

        ref_pct = np.maximum(ref_counts / len(ref), 1e-4)
        cur_pct = np.maximum(cur_counts / len(cur), 1e-4)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(round(psi, 5))

    def detect_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        node_id: str = "hospital_alpha",
    ) -> Dict[str, Any]:
        """
        Runs comprehensive multi-metric drift evaluation and generates structured governance diagnostic.
        """
        ref = np.asarray(reference_data)
        cur = np.asarray(current_data)

        # MMD computation (flattening or embedding space)
        ref_2d = ref.reshape(ref.shape[0], -1) if ref.ndim > 2 else ref
        cur_2d = cur.reshape(cur.shape[0], -1) if cur.ndim > 2 else cur
        
        # Subsample if dimension is large
        if ref_2d.shape[1] > 64:
            ref_2d = ref_2d[:, :64]
            cur_2d = cur_2d[:, :64]

        mmd_score = self.compute_mmd(ref_2d, cur_2d)
        ks_res = self.compute_ks_test(ref, cur)
        wasserstein_dist = self.compute_wasserstein(ref, cur)
        psi_score = self.compute_psi(ref, cur)

        # Risk Assessment
        drift_detected = bool(ks_res["is_drift"] or psi_score >= self.psi_threshold or mmd_score > 0.3)

        if psi_score >= 0.25 or mmd_score > 0.4:
            risk_level = "CRITICAL"
            action = "Trigger urgent model re-training and isolate node data"
        elif drift_detected:
            risk_level = "HIGH"
            action = "Alert hospital administrator & adjust local learning rate"
        elif psi_score >= 0.1:
            risk_level = "MODERATE"
            action = "Monitor next federated round for drift progression"
        else:
            risk_level = "LOW"
            action = "No intervention required"

        return {
            "node_id": node_id,
            "drift_detected": drift_detected,
            "risk_level": risk_level,
            "recommended_action": action,
            "metrics": {
                "mmd": round(mmd_score, 4),
                "ks_statistic": ks_res["ks_statistic"],
                "ks_p_value": ks_res["p_value"],
                "wasserstein_distance": wasserstein_dist,
                "psi": psi_score,
            },
        }
