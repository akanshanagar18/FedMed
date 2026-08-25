"""
Module: foundation.sam_adapter

Purpose:
Segment Anything Model (SAM) foundation model encoder & prompt adapter.
Supports frozen vision transformer encoders with lightweight promptable mask decoder tuning.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class SAMAdapter:
    """
    Segment Anything (SAM) Vision Foundation Model Adapter.
    """

    def __init__(self, model_variant: str = "vit_b", frozen_encoder: bool = True):
        self.model_variant = model_variant
        self.frozen_encoder = frozen_encoder
        self.encoder_params_count = 86_000_000 if model_variant == "vit_b" else 307_000_000
        self.decoder_params_count = 4_000_000

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Simulates frozen SAM image encoder embedding extraction."""
        batch_size = image.shape[0] if image.ndim == 4 else 1
        return np.random.randn(batch_size, 256, 64, 64).astype(np.float32)

    def decode_mask(self, features: np.ndarray, prompt: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Simulates promptable mask decoder output logits."""
        batch_size = features.shape[0]
        return np.random.randn(batch_size, 3, 128, 128).astype(np.float32)

    def get_parameter_stats(self) -> Dict[str, Any]:
        trainable = self.decoder_params_count if self.frozen_encoder else (self.encoder_params_count + self.decoder_params_count)
        total = self.encoder_params_count + self.decoder_params_count
        return {
            "model_name": f"SAM-{self.model_variant.upper()}",
            "frozen_encoder": self.frozen_encoder,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_percent": float(trainable / total) * 100.0,
        }
