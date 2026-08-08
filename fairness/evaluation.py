"""
Module: fairness.evaluation

Purpose:
End-to-end fairness evaluation pipeline across federated hospital nodes.
"""

from typing import Any, Dict
import numpy as np
from fairness.metrics import subgroup_dice_scores, demographic_parity_ratio


class FairnessReport:
    """Generates a comprehensive fairness evaluation report."""

    def __init__(self, scanner_groups: Dict[str, str] = None):
        self.scanner_groups = scanner_groups or {
            "hospital_alpha": "Siemens PRISMA 3T",
            "hospital_beta": "GE Discovery MR750 3T",
            "hospital_gamma": "Philips Ingenia 1.5T",
        }

    def evaluate(self, hospital_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        dices = {h: m.get("dice_score", 0.0) for h, m in hospital_metrics.items()}
        subgroup = subgroup_dice_scores(dices)
        disparate_impact = float(subgroup["min_dice"] / subgroup["max_dice"]) if subgroup["max_dice"] > 0 else 1.0

        breakdown = {}
        mean_dice = subgroup["mean_dice"]
        for h, d in dices.items():
            breakdown[h] = {
                "scanner": self.scanner_groups.get(h, "Unknown"),
                "dice": d,
                "gap_from_mean": float(abs(d - mean_dice)),
            }

        return {
            "subgroup_analysis": subgroup,
            "disparate_impact": disparate_impact,
            "equal_opportunity_met": disparate_impact >= 0.80,
            "fairness_index": float(1.0 - subgroup["range"]),
            "hospital_breakdown": breakdown,
        }
