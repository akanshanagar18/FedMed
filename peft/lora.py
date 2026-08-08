"""
Module: peft.lora

Purpose:
Low-Rank Adaptation (LoRA, Hu et al., ICLR 2022) for Parameter-Efficient Fine-Tuning.
Injects trainable rank decomposition matrices A in R^(r x d) and B in R^(k x r) into linear/conv layers:
W = W_0 + (alpha / r) * B * A
"""

from typing import Dict, Tuple
import numpy as np


class LoRALayer:
    """
    Low-Rank Adaptation (LoRA) matrix layer wrapper.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: float = 16.0):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha / r)

        # Base frozen weights W_0
        self.W_0 = np.random.randn(out_features, in_features).astype(np.float32) * 0.01

        # Trainable LoRA low-rank decomposition matrices A and B
        self.A = np.random.randn(r, in_features).astype(np.float32) * (1.0 / np.sqrt(in_features))
        self.B = np.zeros((out_features, r), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = x @ (W_0 + (alpha/r) * B @ A).T
        """
        w_eff = self.W_0 + self.scaling * (self.B @ self.A)
        return x @ w_eff.T

    def get_parameter_counts(self) -> Dict[str, int]:
        frozen_params = self.W_0.size
        trainable_params = self.A.size + self.B.size
        return {
            "frozen_parameters": frozen_params,
            "trainable_parameters": trainable_params,
            "total_parameters": frozen_params + trainable_params,
        }
