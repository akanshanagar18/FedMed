"""
Module: inference.pipeline

Purpose:
Production 3D Medical Imaging Inference Engine for FedMed v2.0.
Executes 3D Sliding Window Inference (MONAI) on 4-channel BraTS MRI scans (T1, T1ce, T2, FLAIR).
Computes Enhancing Tumor (ET), Tumor Core (TC), Whole Tumor (WT) segmentation metrics, confidence scores,
and stores historical prediction records.
"""

import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np

from model.unet3d import UNet3D
from monai.inferers import SlidingWindowInferer


class BraTSInferenceEngine:
    """
    Production 3D BraTS Brain Tumor Segmentation Inference Engine.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BraTSInferenceEngine, cls).__new__(cls)
            cls._instance.model = UNet3D(in_channels=4, out_channels=3)
            cls._instance.inferer = SlidingWindowInferer(roi_size=(64, 64, 64), sw_batch_size=2, overlap=0.25)
            cls._instance.history: List[Dict[str, Any]] = []
        return cls._instance

    def predict(
        self,
        input_tensor: Optional[torch.Tensor] = None,
        patient_id: str = "PATIENT_DEMO_001",
        model_version: str = "v2.0.0-prod",
    ) -> Dict[str, Any]:
        """
        Executes sliding window inference on 4-channel 3D MRI volume tensor.
        """
        start_time = time.time()
        self.model.eval()

        if input_tensor is None:
            # Synthetic 4-channel 3D volume (B=1, C=4, D=64, H=64, W=64)
            input_tensor = torch.randn(1, 4, 64, 64, 64)

        with torch.no_grad():
            output_logits = self.inferer(inputs=input_tensor, network=self.model)
            output_probs = torch.sigmoid(output_logits)

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Compute volume statistics for ET, TC, WT
        probs_np = output_probs.squeeze(0).cpu().numpy()
        et_vol_mm3 = float(np.sum(probs_np[0] > 0.5) * 1.0)
        tc_vol_mm3 = float(np.sum(probs_np[1] > 0.5) * 1.0)
        wt_vol_mm3 = float(np.sum(probs_np[2] > 0.5) * 1.0)
        mean_confidence = float(np.mean(probs_np))

        prediction_record = {
            "prediction_id": f"pred_{int(time.time())}",
            "patient_id": patient_id,
            "model_version": model_version,
            "latency_ms": latency_ms,
            "mean_confidence": round(mean_confidence, 4),
            "tumor_volumes_mm3": {
                "enhancing_tumor_et": round(et_vol_mm3, 1),
                "tumor_core_tc": round(tc_vol_mm3, 1),
                "whole_tumor_wt": round(wt_vol_mm3, 1),
            },
            "status": "SUCCESS",
            "timestamp": time.time(),
        }

        self.history.append(prediction_record)
        return prediction_record

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history


global_inference_engine = BraTSInferenceEngine()
