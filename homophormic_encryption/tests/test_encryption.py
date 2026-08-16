import pytest
import torch
from homophormic_encryption import create_ckks_context, encrypt_state_dict, decrypt_state_dict
from homophormic_encryption.serialization import serialize_encrypted_state_dict, deserialize_encrypted_state_dict

def test_encrypt_decrypt_roundtrip():
    context = create_ckks_context()
    original_state = {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([0.5])}
    encrypted = encrypt_state_dict(context, original_state)
    
    # Test serialization
    serialized = serialize_encrypted_state_dict(encrypted)
    deserialized = deserialize_encrypted_state_dict(context, serialized)

    decrypted = decrypt_state_dict(deserialized)
    
    assert torch.allclose(original_state["weight"], decrypted["weight"], atol=1e-2)
    assert torch.allclose(original_state["bias"], decrypted["bias"], atol=1e-2)
