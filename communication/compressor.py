"""
Module: communication.compressor

Purpose:
Framework-agnostic Abstract Base Class contract and factory for gradient & model weight compression.
Reduces network bandwidth payloads during distributed federated learning communication.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np


class CompressedPayload:
    """Container for compressed communication payload and metadata."""

    def __init__(self, compressed_data: Any, original_shapes: List[Tuple[int, ...]], original_size_bytes: int, compressed_size_bytes: int, compression_type: str):
        self.compressed_data = compressed_data
        self.original_shapes = original_shapes
        self.original_size_bytes = original_size_bytes
        self.compressed_size_bytes = compressed_size_bytes
        self.compression_type = compression_type

    @property
    def compression_ratio(self) -> float:
        if self.compressed_size_bytes == 0:
            return 1.0
        return float(self.original_size_bytes / self.compressed_size_bytes)

    @property
    def bandwidth_savings_pct(self) -> float:
        if self.original_size_bytes == 0:
            return 0.0
        return float(1.0 - (self.compressed_size_bytes / self.original_size_bytes)) * 100.0


class BaseCompressor(ABC):
    """
    Abstract Base Class for communication compressors.
    """

    @abstractmethod
    def compress(self, weights: List[np.ndarray]) -> CompressedPayload:
        """Compresses numpy array weights into a CompressedPayload."""
        pass

    @abstractmethod
    def decompress(self, payload: CompressedPayload) -> List[np.ndarray]:
        """Decompresses CompressedPayload back into numpy weight arrays."""
        pass
