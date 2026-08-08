"""
Module: explainability.attention_maps

Purpose:
Self-Attention Heatmap visualizer for ViT and Transformer-based foundation models (SAM/DINOv2).
"""

from typing import Dict
import numpy as np


class AttentionVisualizer:
    """
    Self-Attention Map Visualizer.
    """

    def __init__(self, num_heads: int = 12):
        self.num_heads = num_heads

    def process_attention_weights(self, attn_weights: np.ndarray) -> np.ndarray:
        """
        Averages multi-head self-attention maps and normalizes.
        attn_weights shape: (Batch, Heads, Tokens, Tokens)
        """
        # Average across attention heads
        mean_attn = np.mean(attn_weights, axis=1)  # (Batch, Tokens, Tokens)

        # Min-Max normalize
        a_min = np.min(mean_attn, axis=(-2, -1), keepdims=True)
        a_max = np.max(mean_attn, axis=(-2, -1), keepdims=True)
        return (mean_attn - a_min) / np.maximum(1e-8, a_max - a_min)
