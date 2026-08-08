"""
Module: explainability.gradcam

Purpose:
3D Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al., ICCV 2017).
Generates visual saliency maps for medical volumetric segmentation predictions.
"""

from typing import Dict, Tuple
import numpy as np


class GradCAM3D:
    """
    3D Grad-CAM activation map generator.
    """

    def __init__(self, target_layer_name: str = "conv_final"):
        self.target_layer_name = target_layer_name

    def generate_heatmap(self, feature_activations: np.ndarray, feature_gradients: np.ndarray) -> np.ndarray:
        """
        Computes 3D Grad-CAM heatmap:
        alpha_k = Mean( grad_k )
        L_GradCAM = ReLU( sum alpha_k * activation_k )
        """
        # Channel-wise mean gradient weights alpha_k
        weights = np.mean(feature_gradients, axis=(2, 3, 4), keepdims=True)  # (B, C, 1, 1, 1)

        # Weighted combination of feature maps
        cam = np.sum(weights * feature_activations, axis=1)  # (B, D, H, W)

        # ReLU non-linearity
        cam = np.maximum(0, cam)

        # Min-max normalization into [0.0, 1.0]
        cam_min = np.min(cam, axis=(1, 2, 3), keepdims=True)
        cam_max = np.max(cam, axis=(1, 2, 3), keepdims=True)
        diff = np.maximum(1e-8, cam_max - cam_min)

        return (cam - cam_min) / diff
