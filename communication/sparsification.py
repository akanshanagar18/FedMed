"""
Module: communication.sparsification

Purpose:
Top-K and Random-K gradient sparsification compressor.
Transmits only the top ratio k (e.g. k=0.10 for top 10% magnitude) of non-zero gradient values.
"""

from typing import List, Tuple
import numpy as np

from communication.compressor import BaseCompressor, CompressedPayload


class Sparsifier(BaseCompressor):
    """
    Sparsification compressor supporting Top-K and Random-K strategies.
    """

    def __init__(self, ratio: float = 0.1, method: str = "topk"):
        assert 0.0 < ratio <= 1.0, "Ratio must be in (0.0, 1.0]."
        assert method in ("topk", "randomk"), "Method must be 'topk' or 'randomk'."
        self.ratio = ratio
        self.method = method

    def compress(self, weights: List[np.ndarray]) -> CompressedPayload:
        compressed_layers = []
        original_shapes = []
        orig_bytes = 0
        comp_bytes = 0

        for w in weights:
            orig_bytes += w.nbytes
            shape = w.shape
            original_shapes.append(shape)
            flat_w = w.flatten()
            total_elements = flat_w.size
            k_count = max(1, int(total_elements * self.ratio))

            if self.method == "topk":
                top_indices = np.argpartition(np.abs(flat_w), -k_count)[-k_count:]
            else:
                top_indices = np.random.choice(total_elements, size=k_count, replace=False)

            top_values = flat_w[top_indices]

            # Index coordinates (INT32) + float values (FP32)
            layer_comp_bytes = (top_indices.nbytes + top_values.nbytes)
            comp_bytes += layer_comp_bytes

            compressed_layers.append({
                "indices": top_indices,
                "values": top_values,
                "shape": shape,
                "size": total_elements,
            })

        return CompressedPayload(
            compressed_data=compressed_layers,
            original_shapes=original_shapes,
            original_size_bytes=orig_bytes,
            compressed_size_bytes=comp_bytes,
            compression_type=f"Sparsification({self.method.upper()}, k={self.ratio*100:.0f}%)",
        )

    def decompress(self, payload: CompressedPayload) -> List[np.ndarray]:
        decompressed_weights = []

        for layer in payload.compressed_data:
            indices = layer["indices"]
            values = layer["values"]
            shape = layer["shape"]
            size = layer["size"]

            decomp_flat = np.zeros(size, dtype=np.float32)
            decomp_flat[indices] = values
            decompressed_weights.append(decomp_flat.reshape(shape))

        return decompressed_weights
