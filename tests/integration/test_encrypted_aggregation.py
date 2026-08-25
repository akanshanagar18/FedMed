"""
Integration tests for privacy.aggregation (homomorphic ciphertext aggregation).
"""

import numpy as np
import pytest
from privacy.context import create_ckks_context, get_public_context
from privacy.encrypt import encrypt_model_parameters
from privacy.decrypt import decrypt_model_parameters
from privacy.aggregation import aggregate_encrypted_updates


def test_homomorphic_encrypted_aggregation_two_clients():
    """Verify homomorphic ciphertext aggregation across 2 clients with different sample weights."""
    ctx = create_ckks_context(poly_modulus_degree=8192)
    pub_ctx = get_public_context(ctx)

    # Client 1 updates (weight = 10 samples)
    c1_p1 = np.ones((4, 4), dtype=np.float32) * 2.0
    c1_p2 = np.ones((4,), dtype=np.float32) * 4.0
    enc1 = encrypt_model_parameters(ctx, [c1_p1, c1_p2], chunk_size=16)

    # Client 2 updates (weight = 30 samples)
    c2_p1 = np.ones((4, 4), dtype=np.float32) * 6.0
    c2_p2 = np.ones((4,), dtype=np.float32) * 8.0
    enc2 = encrypt_model_parameters(ctx, [c2_p1, c2_p2], chunk_size=16)

    encrypted_results = [
        (enc1["encrypted_chunks"], 10),
        (enc2["encrypted_chunks"], 30),
    ]

    # Homomorphically aggregate ciphertext using ONLY public evaluation context
    agg_res = aggregate_encrypted_updates(pub_ctx, encrypted_results, enc1["shapes"])

    assert agg_res["ciphertext_size_bytes"] > 0
    assert agg_res["aggregation_time_ms"] > 0.0

    # Decrypt aggregated global parameters using secret key
    decrypted_params = decrypt_model_parameters(ctx, agg_res["aggregated_chunks"], agg_res["shapes"])

    # Expected weighted averages:
    # layer 0: (10/40)*2.0 + (30/40)*6.0 = 0.5 + 4.5 = 5.0
    # layer 1: (10/40)*4.0 + (30/40)*8.0 = 1.0 + 6.0 = 7.0
    expected_p1 = np.full((4, 4), 5.0, dtype=np.float32)
    expected_p2 = np.full((4,), 7.0, dtype=np.float32)

    np.testing.assert_allclose(decrypted_params[0], expected_p1, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(decrypted_params[1], expected_p2, rtol=1e-3, atol=1e-3)
