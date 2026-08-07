"""
Module: privacy.encrypt

Purpose:
Client-side model parameter chunking and TenSEAL CKKS homomorphic vector encryption.
Converts PyTorch model state_dict arrays into encrypted CKKS ciphertext vectors.
"""

import time
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tenseal as ts

logger = logging.getLogger(__name__)


def encrypt_weights(context: ts.Context, weights: Any) -> ts.CKKSVector:
    """
    Helper function to encrypt flat list/array of weights into a single TenSEAL CKKSVector.
    """
    return ts.ckks_vector(context, weights)


def encrypt_weights_vector(

    context: ts.Context,
    weights_vector: np.ndarray,
    chunk_size: int = 4096,
) -> Tuple[List[bytes], float, int]:
    """
    Encrypts a 1D NumPy weight vector using TenSEAL CKKS in chunks.
    
    Returns:
        Tuple of (serialized_ciphertext_bytes_list, encryption_time_ms, total_bytes)
    """
    start_time = time.time()
    flat_weights = weights_vector.flatten().astype(np.float64)
    num_elements = len(flat_weights)

    ciphertext_bytes_list: List[bytes] = []
    total_bytes = 0

    for i in range(0, num_elements, chunk_size):
        chunk = flat_weights[i : i + chunk_size]
        ckks_vec = ts.ckks_vector(context, chunk)
        serialized = ckks_vec.serialize()
        ciphertext_bytes_list.append(serialized)
        total_bytes += len(serialized)

    enc_time_ms = (time.time() - start_time) * 1000.0
    return ciphertext_bytes_list, enc_time_ms, total_bytes


def encrypt_model_parameters(
    context: ts.Context,
    parameters: List[np.ndarray],
    chunk_size: int = 4096,
) -> Dict[str, Any]:
    """
    Encrypts complete list of model parameter NumPy arrays.
    """
    start_time = time.time()
    all_ciphertext_chunks: List[List[bytes]] = []
    total_ciphertext_bytes = 0
    shapes = [p.shape for p in parameters]

    for param_arr in parameters:
        chunks, _, num_bytes = encrypt_weights_vector(context, param_arr, chunk_size=chunk_size)
        all_ciphertext_chunks.append(chunks)
        total_ciphertext_bytes += num_bytes

    enc_time_ms = (time.time() - start_time) * 1000.0

    return {
        "encrypted_chunks": all_ciphertext_chunks,
        "shapes": shapes,
        "encryption_time_ms": enc_time_ms,
        "ciphertext_size_bytes": total_ciphertext_bytes,
    }