"""
Module: foundation.dinov2_adapter

Purpose:
DINOv2 Self-Supervised Vision Transformer Feature Extractor Adapter (Oquab et al., 2023).
Extracts dense visual representations for medical domain adaptation and cross-scanner generalization.
"""

from typing import Any, Dict
import numpy as np


class DINOv2Adapter:
    """
    DINOv2 ViT Feature Extractor Adapter.
    """

    def __init__(self, variant: str = "vit_b14", frozen_backbone: bool = True):
        self.variant = variant
        self.frozen_backbone = frozen_backbone
        self.backbone_params = 86_000_000 if variant == "vit_b14" else 300_000_000
        self.head_params = 1_200_000

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Simulates DINOv2 visual patch embedding extraction."""
        batch_size = image.shape[0] if image.ndim == 4 else 1
        embed_dim = 768 if self.variant == "vit_b14" else 1024
        return np.random.randn(batch_size, embed_dim).astype(np.float32)

    def get_parameter_stats(self) -> Dict[str, Any]:
        trainable = self.head_params if self.frozen_backbone else (self.backbone_params + self.head_params)
        total = self.backbone_params + self.head_params
        return {
            "model_name": f"DINOv2-{self.variant.upper()}",
            "frozen_backbone": self.frozen_backbone,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_percent": float(trainable / total) * 100.0,
        }
