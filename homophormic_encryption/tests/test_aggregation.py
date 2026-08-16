import pytest
import torch
from homophormic_encryption import create_ckks_context, encrypt_state_dict, decrypt_state_dict, aggregate_encrypted_state_dicts, weighted_aggregate_encrypted_state_dicts

def test_aggregation():
    context = create_ckks_context()
    sd1 = {"w": torch.tensor([1.0])}
    sd2 = {"w": torch.tensor([2.0])}
    
    enc1 = encrypt_state_dict(context, sd1)
    enc2 = encrypt_state_dict(context, sd2)
    
    agg = aggregate_encrypted_state_dicts([enc1, enc2])
    dec = decrypt_state_dict(agg)
    
    assert torch.allclose(dec["w"], torch.tensor([3.0]), atol=1e-2)

def test_weighted_aggregation():
    context = create_ckks_context()
    sd1 = {"w": torch.tensor([10.0])}
    sd2 = {"w": torch.tensor([20.0])}
    
    enc1 = encrypt_state_dict(context, sd1)
    enc2 = encrypt_state_dict(context, sd2)
    
    agg = weighted_aggregate_encrypted_state_dicts([enc1, enc2], weights=[1.0, 3.0])
    dec = decrypt_state_dict(agg)
    
    assert torch.allclose(dec["w"], torch.tensor([17.5]), atol=1e-2)
