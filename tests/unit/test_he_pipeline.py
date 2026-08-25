"""
Unit tests for privacy.context, privacy.encrypt, and privacy.decrypt (TenSEAL CKKS HE pipeline).
"""

import numpy as np
import pytest
from privacy.context import create_ckks_context, get_public_context, serialize_context, deserialize_context
from privacy.encrypt import encrypt_weights_vector, encrypt_model_parameters
from privacy.decrypt import decrypt_weights_vector, decrypt_model_parameters


def test_ckks_context_creation_and_key_isolation():
    """Verify CKKS context creation, public key isolation, and serialization."""
    ctx = create_ckks_context(poly_modulus_degree=8192)
    assert ctx.is_private() is True

    pub_ctx = get_public_context(ctx)
    assert pub_ctx.is_private() is False

    ctx_bytes = serialize_context(pub_ctx, save_secret_key=False)
    deserialized = deserialize_context(ctx_bytes)
    assert deserialized.is_private() is False


def test_ckks_vector_encryption_decryption_accuracy():
    """Verify weight vector chunking, encryption, and decryption numerical accuracy."""
    ctx = create_ckks_context(poly_modulus_degree=8192)
    weights = np.random.randn(100).astype(np.float32)

    chunks, enc_time, num_bytes = encrypt_weights_vector(ctx, weights, chunk_size=32)
    assert len(chunks) == 4
    assert num_bytes > 0

    decrypted = decrypt_weights_vector(ctx, chunks, weights.shape)
    np.testing.assert_allclose(weights, decrypted, rtol=1e-3, atol=1e-3)


def test_encrypt_decrypt_model_parameters():
    """Verify multi-tensor model parameter encryption and decryption."""
    ctx = create_ckks_context(poly_modulus_degree=8192)
    p1 = np.random.randn(10, 5).astype(np.float32)
    p2 = np.random.randn(5).astype(np.float32)
    params = [p1, p2]

    enc_dict = encrypt_model_parameters(ctx, params, chunk_size=16)
    assert "encrypted_chunks" in enc_dict
    assert enc_dict["ciphertext_size_bytes"] > 0

    decrypted_params = decrypt_model_parameters(ctx, enc_dict["encrypted_chunks"], enc_dict["shapes"])
    assert len(decrypted_params) == 2
    np.testing.assert_allclose(p1, decrypted_params[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(p2, decrypted_params[1], rtol=1e-3, atol=1e-3)
