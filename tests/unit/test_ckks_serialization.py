"""
Unit tests for privacy.communication serialization and deserialization helpers.
"""

import numpy as np
import pytest
from privacy.context import create_ckks_context
from privacy.encrypt import encrypt_model_parameters
from privacy.communication import serialize_encrypted_payload, deserialize_encrypted_payload


def test_ckks_payload_serialization_roundtrip():
    """Verify base64 JSON payload serialization and deserialization."""
    ctx = create_ckks_context(poly_modulus_degree=8192)
    p1 = np.random.randn(8, 8).astype(np.float32)
    enc_dict = encrypt_model_parameters(ctx, [p1], chunk_size=32)

    serialized_str = serialize_encrypted_payload(enc_dict)
    assert isinstance(serialized_str, str)
    assert len(serialized_str) > 0

    deserialized_dict = deserialize_encrypted_payload(serialized_str)
    assert "encrypted_chunks" in deserialized_dict
    assert deserialized_dict["shapes"] == enc_dict["shapes"]
    assert len(deserialized_dict["encrypted_chunks"][0]) == len(enc_dict["encrypted_chunks"][0])
