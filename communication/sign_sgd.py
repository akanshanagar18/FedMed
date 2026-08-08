"""
Module: communication.sign_sgd

Purpose:
SignSGD 1-bit sign compression with local error feedback accumulation buffers.
Compresses gradient values into single-bit signs {+1, -1} and accumulates quantization residual errors locally.
"""

from typing import List, Optional, Tuple
import numpy as np

from communication.compressor import BaseCompressor, CompressedPayload


class SignSGDCompressor(BaseCompressor):
    """
    SignSGD compressor with bit-packing and local error feedback buffer.
    """

    def __init__(self, use_error_feedback: bool = True):
        self.use_error_feedback = use_error_feedback
        self.error_buffers: Optional[List[np.ndarray]] = None

    def compress(self, weights: List[np.ndarray]) -> CompressedPayload:
        if self.error_buffers is None and self.use_error_feedback:
            self.error_buffers = [np.zeros_like(w, dtype=np.float32) for w in weights]

        compressed_layers = []
        original_shapes = []
        orig_bytes = 0
        comp_bytes = 0

        for idx, w in enumerate(weights):
            orig_bytes += w.nbytes
            original_shapes.append(w.shape)

            # Apply error feedback buffer: w_accum = w + e_t
            if self.use_error_feedback and self.error_buffers is not None:
                w_target = w + self.error_buffers[idx]
            else:
                w_target = w

            # Compute sign {-1.0, +1.0} and mean absolute magnitude scale
            sign_w = np.sign(w_target)
            sign_w[sign_w == 0] = 1.0  # default 0 to +1
            scale = float(np.mean(np.abs(w_target)))

            # Compressed representation: boolean sign array packed as bits
            bit_packed = np.packbits((sign_w > 0).astype(np.uint8))
            layer_comp_bytes = bit_packed.nbytes + 8  # scale (float64)

            comp_bytes += layer_comp_bytes

            # Update local error feedback buffer: e_{t+1} = w_target - scale * sign_w
            if self.use_error_feedback and self.error_buffers is not None:
                self.error_buffers[idx] = w_target - (scale * sign_w)

            compressed_layers.append({
                "bit_packed": bit_packed,
                "scale": scale,
                "shape": w.shape,
                "size": w.size,
            })

        return CompressedPayload(
            compressed_data=compressed_layers,
            original_shapes=original_shapes,
            original_size_bytes=orig_bytes,
            compressed_size_bytes=comp_bytes,
            compression_type="SignSGD(1-bit + ErrorFeedback)",
        )

    def decompress(self, payload: CompressedPayload) -> List[np.ndarray]:
        decompressed_weights = []

        for layer in payload.compressed_data:
            bit_packed = layer["bit_packed"]
            scale = layer["scale"]
            shape = layer["shape"]
            size = layer["size"]

            # Unpack bits into boolean sign array
            unpacked_bits = np.unpackbits(bit_packed)[:size]
            sign_array = np.where(unpacked_bits > 0, 1.0, -1.0).astype(np.float32)

            w_decomp = (scale * sign_array).reshape(shape)
            decompressed_weights.append(w_decomp)

        return decompressed_weights
