"""
Module: communication.quantization

Purpose:
Uniform 8-bit (INT8) and 4-bit (INT4) weight & gradient quantization compressor.
Scales floating point values (FP32) into quantized integers with zero-point and scale metadata.
"""

from typing import List, Tuple
import numpy as np

from communication.compressor import BaseCompressor, CompressedPayload


class UniformQuantizer(BaseCompressor):
    """
    Uniform Min-Max Quantizer supporting 8-bit (INT8) and 4-bit (INT4) quantization.
    """

    def __init__(self, bits: int = 8):
        assert bits in (4, 8), "UniformQuantizer supports 4-bit or 8-bit quantization."
        self.bits = bits
        self.qmin = 0
        self.qmax = (2 ** bits) - 1

    def compress(self, weights: List[np.ndarray]) -> CompressedPayload:
        quantized_layers = []
        original_shapes = []
        orig_bytes = 0
        comp_bytes = 0

        for w in weights:
            orig_bytes += w.nbytes
            original_shapes.append(w.shape)

            w_min = float(np.min(w))
            w_max = float(np.max(w))

            if abs(w_max - w_min) < 1e-8:
                scale = 1.0
                zero_point = 0
                q_arr = np.zeros_like(w, dtype=np.uint8)
            else:
                scale = (w_max - w_min) / float(self.qmax - self.qmin)
                zero_point = int(np.round((0 - w_min) / scale))
                zero_point = max(self.qmin, min(self.qmax, zero_point))

                q_arr = np.round((w - w_min) / scale).astype(np.uint8)
                q_arr = np.clip(q_arr, self.qmin, self.qmax)

            # Metadata bytes + INT8 array bytes
            layer_comp_bytes = q_arr.nbytes if self.bits == 8 else (q_arr.nbytes // 2 + 1)
            layer_comp_bytes += 16  # scale + zero_point + min float metadata
            comp_bytes += layer_comp_bytes

            quantized_layers.append({
                "q_arr": q_arr,
                "scale": scale,
                "w_min": w_min,
                "zero_point": zero_point,
            })

        return CompressedPayload(
            compressed_data=quantized_layers,
            original_shapes=original_shapes,
            original_size_bytes=orig_bytes,
            compressed_size_bytes=comp_bytes,
            compression_type=f"Quantization({self.bits}-bit)",
        )

    def decompress(self, payload: CompressedPayload) -> List[np.ndarray]:
        decompressed_weights = []

        for layer in payload.compressed_data:
            q_arr = layer["q_arr"]
            scale = layer["scale"]
            w_min = layer["w_min"]

            # Dequantize: w = q_arr * scale + w_min
            w_dequant = (q_arr.astype(np.float32) * scale) + w_min
            decompressed_weights.append(w_dequant)

        return decompressed_weights
