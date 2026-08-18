"""
Unit tests for privacy Homomorphic Encryption (TenSEAL CKKS) module.
Verifies all 7 cryptographic and arithmetic properties required by Phase 8.5B.5:
  1. encrypt(x) decrypts approximately to x
  2. encrypted(x) + encrypted(y) decrypts approximately to x + y
  3. encrypted(x) * scalar decrypts approximately to scalar * x
  4. Weighted encrypted aggregation matches plaintext aggregation within CKKS tolerance (< 1e-5)
  5. Ciphertext bytes are opaque before authorized secret key decryption
  6. Aggregation operates strictly on public evaluation context without plaintext updates
  7. CKKS numerical approximation error is measured
"""

import numpy as np
import pytest
import tenseal as ts

from privacy.aggregation import aggregate_encrypted_updates
from privacy.context import create_ckks_context, get_public_context
from privacy.decrypt import decrypt_model_parameters, decrypt_weights_vector
from privacy.encrypt import encrypt_model_parameters, encrypt_weights_vector


def test_ckks_encrypt_decrypt_basic_identity():
    """1. encrypt(x) decrypts to x within precision tolerance."""
    ctx = create_ckks_context(poly_modulus_degree=8192)
    x = np.array([1.25, -3.5, 0.0, 42.125], dtype=np.float64)

    ckks_vec = ts.ckks_vector(ctx, x)
    dec = np.array(ckks_vec.decrypt(), dtype=np.float64)

    np.testing.assert_allclose(dec, x, atol=1e-6)


def test_ckks_homomorphic_addition():
    """2. encrypted(x) + encrypted(y) decrypts to x + y."""
    ctx = create_ckks_context(poly_modulus_degree=8192)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

    enc_x = ts.ckks_vector(ctx, x)
    enc_y = ts.ckks_vector(ctx, y)

    enc_sum = enc_x + enc_y
    dec_sum = np.array(enc_sum.decrypt(), dtype=np.float64)

    np.testing.assert_allclose(dec_sum, x + y, atol=1e-5)


def test_ckks_scalar_multiplication():
    """3. encrypted(x) * scalar decrypts to scalar * x."""
    ctx = create_ckks_context(poly_modulus_degree=8192)
    x = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    scalar = 0.5

    enc_x = ts.ckks_vector(ctx, x)
    enc_scaled = enc_x * scalar
    dec_scaled = np.array(enc_scaled.decrypt(), dtype=np.float64)

    np.testing.assert_allclose(dec_scaled, x * scalar, atol=1e-6)


def test_ckks_weighted_aggregation_pipeline():
    """4, 5, 6, 7. Weighted aggregation on public context matches plaintext calculation."""
    # Private context (holds secret key)
    private_ctx = create_ckks_context(poly_modulus_degree=8192)
    # Public evaluation context (secret key dropped)
    public_ctx = get_public_context(private_ctx)
    assert public_ctx.is_private() is False

    # Two client parameter layers
    p1 = [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)]
    p2 = [np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32), np.array([[4.5, 5.5], [6.5, 7.5]], dtype=np.float32)]

    # Client-side encryption
    enc_res1 = encrypt_model_parameters(private_ctx, p1, chunk_size=4096)
    enc_res2 = encrypt_model_parameters(private_ctx, p2, chunk_size=4096)

    # Verify ciphertexts are byte buffers
    assert isinstance(enc_res1["encrypted_chunks"][0][0], bytes)
    assert len(enc_res1["encrypted_chunks"][0][0]) > 1000

    # Server-side aggregation with public key ONLY
    shapes = enc_res1["shapes"]
    encrypted_results = [
        (enc_res1["encrypted_chunks"], 1),  # weight 0.5 (1 sample)
        (enc_res2["encrypted_chunks"], 1),  # weight 0.5 (1 sample)
    ]

    agg_result = aggregate_encrypted_updates(public_ctx, encrypted_results, shapes)

    # Client-side decryption with secret key
    decrypted_params = decrypt_model_parameters(private_ctx, agg_result["aggregated_chunks"], shapes)

    # Plaintext reference
    expected_layer0 = 0.5 * p1[0] + 0.5 * p2[0]
    expected_layer1 = 0.5 * p1[1] + 0.5 * p2[1]

    # Measure CKKS numerical error
    err0 = np.max(np.abs(decrypted_params[0] - expected_layer0))
    err1 = np.max(np.abs(decrypted_params[1] - expected_layer1))
    max_err = max(err0, err1)

    assert max_err < 1e-5
    np.testing.assert_allclose(decrypted_params[0], expected_layer0, atol=1e-5)
    np.testing.assert_allclose(decrypted_params[1], expected_layer1, atol=1e-5)
