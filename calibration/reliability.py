"""
Module: calibration.reliability

Purpose:
Reliability diagram data generation and confidence histogram computation.
"""

import numpy as np
from typing import Dict, List


def reliability_diagram_data(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> Dict[str, List[float]]:
    """Returns bin midpoints, average accuracy per bin, and average confidence per bin."""
    bins = np.linspace(0, 1, num_bins + 1)
    midpoints = []
    avg_acc = []
    avg_conf = []
    counts = []
    for i in range(num_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        midpoints.append(float((bins[i] + bins[i + 1]) / 2))
        if np.sum(mask) > 0:
            avg_acc.append(float(np.mean(accuracies[mask])))
            avg_conf.append(float(np.mean(confidences[mask])))
            counts.append(int(np.sum(mask)))
        else:
            avg_acc.append(0.0)
            avg_conf.append(0.0)
            counts.append(0)
    return {"midpoints": midpoints, "avg_accuracy": avg_acc, "avg_confidence": avg_conf, "counts": counts}
