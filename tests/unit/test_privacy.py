"""
Unit tests for privacy package (TenSEAL Homomorphic Encryption & Communication)
"""

import pytest
import tenseal as ts
from privacy.context import create_context
from privacy.encrypt import encrypt_weights
from privacy.decrypt import decrypt_weights
from privacy.communication import serialize_update, deserialize_update


@pytest.mark.unit
def test_tenseal_context_creation():
    """Verify TenSEAL CKKS context initialization."""
    context = create_context()
    assert context is not None
    assert context.is_private() is True
    assert context.global_scale == 2**40


@pytest.mark.unit
def test_homomorphic_encryption_decryption_roundtrip():
    """Verify CKKS vector encryption and decryption accuracy."""
    context = create_context()
    raw_weights = [0.123, -0.456, 0.789, 0.0]

    # Encrypt
    encrypted_vec = encrypt_weights(context, raw_weights)
    assert isinstance(encrypted_vec, ts.CKKSVector)

    # Decrypt
    decrypted_vec = decrypt_weights(encrypted_vec)
    assert len(decrypted_vec) == len(raw_weights)

    # Check homomorphic numerical precision within 1e-3
    for orig, dec in zip(raw_weights, decrypted_vec):
        assert abs(orig - dec) < 1e-3


@pytest.mark.unit
def test_privacy_communication_helpers():
    """Verify privacy communication serialization stubs."""
    dummy_payload = {"weights": [1.0, 2.0, 3.0]}
    serialized = serialize_update(dummy_payload)
    deserialized = deserialize_update(serialized)
    assert deserialized == dummy_payload
