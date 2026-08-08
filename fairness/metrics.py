"""
Module: fairness.metrics

Purpose:
Fairness metric calculations: Equal Opportunity, Equalized Odds, Demographic Parity,
Calibration Gap, Subgroup Dice, Domain Fairness.
"""

from typing import Any, Dict, List
import numpy as np


def equal_opportunity_gap(tpr_a: float, tpr_b: float) -> float:
    """Absolute difference in True Positive Rates between two groups."""
    return abs(tpr_a - tpr_b)


def equalized_odds_gap(tpr_a: float, tpr_b: float, fpr_a: float, fpr_b: float) -> float:
    """Max of TPR gap and FPR gap between two groups."""
    return max(abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))


def demographic_parity_ratio(positive_rate_a: float, positive_rate_b: float) -> float:
    """Min(rate_a/rate_b, rate_b/rate_a) — 1.0 is perfect parity."""
    if positive_rate_a == 0 or positive_rate_b == 0:
        return 0.0
    return min(positive_rate_a / positive_rate_b, positive_rate_b / positive_rate_a)


def calibration_gap(confidence: np.ndarray, accuracy: np.ndarray, num_bins: int = 10) -> float:
    """Mean absolute calibration gap across bins."""
    bins = np.linspace(0, 1, num_bins + 1)
    gaps = []
    for i in range(num_bins):
        mask = (confidence > bins[i]) & (confidence <= bins[i + 1])
        if np.sum(mask) > 0:
            gaps.append(abs(np.mean(accuracy[mask]) - np.mean(confidence[mask])))
    return float(np.mean(gaps)) if gaps else 0.0


def subgroup_dice_scores(hospital_dices: Dict[str, float]) -> Dict[str, Any]:
    """Computes subgroup Dice analysis."""
    vals = list(hospital_dices.values())
    return {
        "min_dice": float(np.min(vals)),
        "max_dice": float(np.max(vals)),
        "mean_dice": float(np.mean(vals)),
        "std_dice": float(np.std(vals)),
        "range": float(np.max(vals) - np.min(vals)),
    }
