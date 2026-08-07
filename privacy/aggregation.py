"""
Module: privacy.aggregation

Purpose:
Homomorphic Ciphertext Aggregation Engine for FedMed v2.0.
Performs weighted ciphertext addition directly on TenSEAL CKKS vectors without server-side decryption.
"""

import time
import logging
from typing import Any, Dict, List, Tuple
import tenseal as ts

logger = logging.getLogger(__name__)


def aggregate_encrypted_updates(
    public_context: ts.Context,
    encrypted_results: List[Tuple[List[List[bytes]], int]],
    shapes: List[Tuple[int, ...]],
) -> Dict[str, Any]:
    """
    Homomorphically aggregates encrypted model weight updates from client silos.
    
    Args:
        public_context: TenSEAL public evaluation context (no secret key needed)
        encrypted_results: List of (encrypted_chunks_per_layer, num_samples) from each hospital silo
        shapes: List of original parameter tensor shapes
        
    Returns:
        Dict containing aggregated_ciphertext_chunks, aggregation_time_ms, and ciphertext_size_bytes
    """
    start_time = time.time()
    total_examples = sum(num_examples for _, num_examples in encrypted_results)

    if total_examples == 0 or len(encrypted_results) == 0:
        raise ValueError("Cannot perform homomorphic aggregation on empty client updates list.")

    num_layers = len(shapes)
    aggregated_layers_chunks: List[List[bytes]] = []
    total_ciphertext_bytes = 0

    # Iterate over each parameter tensor layer
    for layer_idx in range(num_layers):
        num_chunks = len(encrypted_results[0][0][layer_idx])
        aggregated_layer_chunks: List[bytes] = []

        # Iterate over each chunk within the layer
        for chunk_idx in range(num_chunks):
            accumulated_ckks_vec = None

            for client_chunks_list, num_examples in encrypted_results:
                weight = float(num_examples) / float(total_examples)
                serialized_chunk = client_chunks_list[layer_idx][chunk_idx]

                # Deserialize chunk into CKKSVector using public context
                ckks_vec = ts.ckks_vector_from(public_context, serialized_chunk)
                weighted_vec = ckks_vec * weight

                if accumulated_ckks_vec is None:
                    accumulated_ckks_vec = weighted_vec
                else:
                    accumulated_ckks_vec += weighted_vec

            # Serialize accumulated aggregated ciphertext vector
            agg_serialized = accumulated_ckks_vec.serialize()
            aggregated_layer_chunks.append(agg_serialized)
            total_ciphertext_bytes += len(agg_serialized)

        aggregated_layers_chunks.append(aggregated_layer_chunks)

    agg_time_ms = (time.time() - start_time) * 1000.0

    logger.info(
        f"[HE AGGREGATION] Successfully aggregated ciphertext from {len(encrypted_results)} hospital silos "
        f"in {agg_time_ms:.2f}ms — Total Payload: {total_ciphertext_bytes / 1024:.2f} KB"
    )

    return {
        "aggregated_chunks": aggregated_layers_chunks,
        "shapes": shapes,
        "aggregation_time_ms": agg_time_ms,
        "ciphertext_size_bytes": total_ciphertext_bytes,
        "num_clients_aggregated": len(encrypted_results),
    }