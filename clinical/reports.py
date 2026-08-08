"""
Module: clinical.reports

Purpose:
Clinician-friendly report generation for federated medical AI segmentation results.
"""

from typing import Any, Dict


class ClinicalReportGenerator:
    """Generates human-readable clinician summaries from metric dictionaries."""

    def generate(self, metrics: Dict[str, float], model_name: str = "FedMed UNet3D") -> str:
        status = "PASSED — Clinically Acceptable" if metrics.get("clinically_acceptable") else "NEEDS REVIEW"
        return (
            f"═══════════════════════════════════════════\n"
            f"  CLINICAL VALIDATION REPORT\n"
            f"  Model: {model_name}\n"
            f"═══════════════════════════════════════════\n"
            f"  Status: {status}\n"
            f"  ─────────────────────────────────────────\n"
            f"  Dice Score:          {metrics.get('dice', 0.0):.4f}\n"
            f"  IoU:                 {metrics.get('iou', 0.0):.4f}\n"
            f"  HD95:                {metrics.get('hd95_mm', 0.0):.2f} mm\n"
            f"  Sensitivity:         {metrics.get('sensitivity', 0.0)*100:.2f}%\n"
            f"  Specificity:         {metrics.get('specificity', 0.0)*100:.2f}%\n"
            f"  Precision:           {metrics.get('precision', 0.0)*100:.2f}%\n"
            f"  Volume Error:        {metrics.get('volume_error_ml', 0.0):.2f} mL\n"
            f"  Lesion Detection:    {metrics.get('lesion_detection_rate', 0.0)*100:.2f}%\n"
            f"═══════════════════════════════════════════\n"
        )
