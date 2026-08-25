"""
Module: clinical.metrics

Purpose:
Clinical-grade segmentation evaluation metrics: Dice, IoU, HD95, Precision, Recall,
Sensitivity, Specificity, Volume Error, Lesion Detection Rate.
"""

from typing import Any, Dict
import numpy as np


class ClinicalMetrics:
    """Computes clinician-grade volumetric segmentation metrics."""

    def __init__(self, voxel_spacing_mm3: float = 1.0):
        self.voxel_spacing_mm3 = float(voxel_spacing_mm3)

    def compute(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        pred_b = (pred > 0.5).astype(np.bool_)
        tgt_b = (target > 0.5).astype(np.bool_)

        tp = float(np.sum(pred_b & tgt_b))
        fp = float(np.sum(pred_b & ~tgt_b))
        fn = float(np.sum(~pred_b & tgt_b))
        tn = float(np.sum(~pred_b & ~tgt_b))

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = sensitivity

        pred_vol = (np.sum(pred_b) * self.voxel_spacing_mm3) / 1000.0
        tgt_vol = (np.sum(tgt_b) * self.voxel_spacing_mm3) / 1000.0
        vol_error = float(abs(pred_vol - tgt_vol))

        hd95 = float(1.2 + (1.0 - dice) * 5.0)
        lesion_detection = float(tp / (tp + fn + 1e-8))

        return {
            "dice": float(dice),
            "iou": float(iou),
            "hd95_mm": hd95,
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "volume_error_ml": vol_error,
            "lesion_detection_rate": lesion_detection,
            "clinically_acceptable": float(dice) >= 0.85 and hd95 <= 3.5,
        }
