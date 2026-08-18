import pytest
import torch

try:
    import homomorphic_encryption as he
except Exception:
    he = None

from privacy import communication as privacy_comm
from privacy import aggregation as privacy_agg
from privacy.secure_aggregator import SecureAggregator

pytestmark = pytest.mark.skipif(he is None, reason="homomorphic_encryption not available; integration tests require partner package")

def approx_tensor_eq(a: torch.Tensor, b: torch.Tensor, rtol=1e-3, atol=1e-3) -> bool:
    return torch.allclose(a, b, rtol=rtol, atol=atol)

def make_sample_state_dicts():
    s1 = {"fc.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "fc.bias": torch.tensor([0.1, 0.2])}
    s2 = {"fc.weight": torch.tensor([[0.5, -1.0], [2.0, 0.0]]), "fc.bias": torch.tensor([0.0, 0.3])}
    s3 = {"fc.weight": torch.tensor([[2.0, 1.0], [0.0, -1.0]]), "fc.bias": torch.tensor([0.2, -0.1])}
    return [s1, s2, s3]

def test_encrypt_decrypt_roundtrip():
    ctx = he.context.create_ckks_context()
    state = {"w": torch.tensor([0.1, 0.2, -0.3])}
    encrypted = he.encryptor.encrypt_state_dict(ctx, state)
    serialized = he.serialization.serialize_encrypted_state_dict(encrypted)
    deserialized = he.serialization.deserialize_encrypted_state_dict(ctx, serialized)
    decrypted = he.decryptor.decrypt_state_dict(deserialized)
    assert "w" in decrypted
    assert approx_tensor_eq(decrypted["w"], state["w"])

def test_encrypted_addition():
    ctx = he.context.create_ckks_context()
    s1, s2, _ = make_sample_state_dicts()
    enc1 = he.encryptor.encrypt_state_dict(ctx, s1)
    enc2 = he.encryptor.encrypt_state_dict(ctx, s2)
    aggregated_enc = he.aggregation.aggregate_encrypted_state_dicts([enc1, enc2])
    decrypted = he.decryptor.decrypt_state_dict(aggregated_enc)
    expected = {k: s1[k] + s2[k] for k in s1.keys()}
    for k in expected:
        assert approx_tensor_eq(decrypted[k], expected[k])

def test_three_hospital_weighted_aggregation():
    ctx = he.context.create_ckks_context()
    sd_list = make_sample_state_dicts()
    sample_counts = [10, 30, 60]
    encs = [he.encryptor.encrypt_state_dict(ctx, sd) for sd in sd_list]
    aggregated_enc = he.aggregation.weighted_aggregate_encrypted_state_dicts(encs, sample_counts)
    decrypted = he.decryptor.decrypt_state_dict(aggregated_enc)
    total = sum(sample_counts)
    expected = {}
    for name in sd_list[0].keys():
        acc = sum((n * sd[name] for sd, n in zip(sd_list, sample_counts)))
        expected[name] = acc / float(total)
    for k in expected:
        assert approx_tensor_eq(decrypted[k], expected[k], rtol=1e-2, atol=1e-2)

def test_shape_mismatch_raises():
    ctx = he.context.create_ckks_context()
    s1 = {"w": torch.randn(3)}
    s2 = {"w": torch.randn(4)}
    enc1 = he.encryptor.encrypt_state_dict(ctx, s1)
    enc2 = he.encryptor.encrypt_state_dict(ctx, s2)
    with pytest.raises(Exception):
        he.aggregation.aggregate_encrypted_state_dicts([enc1, enc2])

def test_empty_aggregation_raises_privacy_layer():
    with pytest.raises(ValueError):
        privacy_agg.aggregate_encrypted_updates([])

def test_serialization_roundtrip_partner():
    ctx = he.context.create_ckks_context()
    sd = {"w": torch.tensor([1.0, 2.0, 3.0])}
    enc = he.encryptor.encrypt_state_dict(ctx, sd)
    serialized = he.serialization.serialize_encrypted_state_dict(enc)
    deserialized = he.serialization.deserialize_encrypted_state_dict(ctx, serialized)
    dec = he.decryptor.decrypt_state_dict(deserialized)
    assert approx_tensor_eq(dec["w"], sd["w"])

def test_end_to_end_pipeline_with_secure_aggregator():
    ctx = he.context.create_ckks_context()
    public_ctx = he.context.make_public_context(ctx)
    sd_list = make_sample_state_dicts()
    sample_counts = [5, 3, 2]
    serialized_updates = []
    for sd in sd_list:
        enc = he.encryptor.encrypt_state_dict(ctx, sd)
        serialized_updates.append(he.serialization.serialize_encrypted_state_dict(enc))
    server_agg = SecureAggregator(context=public_ctx)
    serialized_global = server_agg.aggregate(serialized_updates, weights=sample_counts)
    aggregated_enc = he.serialization.deserialize_encrypted_state_dict(ctx, serialized_global)
    global_plain = he.decryptor.decrypt_state_dict(aggregated_enc)
    total = sum(sample_counts)
    expected = {}
    for name in sd_list[0].keys():
        acc = sum((n * sd[name] for sd, n in zip(sd_list, sample_counts)))
        expected[name] = acc / float(total)
    for k in expected:
        assert approx_tensor_eq(global_plain[k], expected[k], rtol=1e-2, atol=1e-2)