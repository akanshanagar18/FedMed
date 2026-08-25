"""
Module: privacy.decrypt

Purpose:
Secret key decryption of aggregated TenSEAL CKKS ciphertext vectors back into
PyTorch model weight parameters.
"""

import time
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import tenseal as ts

logger = logging.getLogger(__name__)


def decrypt_weights(encrypted_vec: ts.CKKSVector) -> List[float]:
    """
    Helper function to decrypt a TenSEAL CKKSVector.
    """
    return encrypted_vec.decrypt()


def decrypt_weights_vector(

    context: ts.Context,
    ciphertext_bytes_list: List[bytes],
    original_shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Decrypts a list of serialized CKKS ciphertext chunks using secret key context.
    """
    decrypted_chunks = []
    for serialized_chunk in ciphertext_bytes_list:
        ckks_vec = ts.ckks_vector_from(context, serialized_chunk)
        decrypted_vals = ckks_vec.decrypt()
        decrypted_chunks.extend(decrypted_vals)

    total_size = int(np.prod(original_shape))
    flat_arr = np.array(decrypted_chunks[:total_size], dtype=np.float32)
    return flat_arr.reshape(original_shape)


def decrypt_model_parameters(
    context: ts.Context,
    encrypted_chunks_list: List[List[bytes]],
    shapes: List[Tuple[int, ...]],
) -> List[np.ndarray]:
    """
    Decrypts complete list of encrypted model parameter arrays.
    """
    decrypted_params = []
    for chunks, shape in zip(encrypted_chunks_list, shapes):
        arr = decrypt_weights_vector(context, chunks, shape)
        decrypted_params.append(arr)
    return decrypted_params