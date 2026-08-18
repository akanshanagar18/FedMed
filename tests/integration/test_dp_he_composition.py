"""
Integration Test: tests/integration/test_dp_he_composition.py

Purpose:
Verifies direct composition of Differential Privacy and Homomorphic Encryption:
  1. Forward pass on local client batches
  2. Backward pass computing gradients
  3. DP gradient clipping and Gaussian noise injection
  4. Local Adam optimizer step
  5. CKKS homomorphic vector encryption
  6. Server public-key weighted addition
  7. Secret-key decryption and global model parameter synchronization
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from privacy.aggregation import aggregate_encrypted_updates
from privacy.context import create_ckks_context, get_public_context
from privacy.decrypt import decrypt_model_parameters
from privacy.dp_engine import DifferentialPrivacyEngine
from privacy.encrypt import encrypt_model_parameters


def test_dp_he_end_to_end_composition_pipeline():
    """Tests complete composition pipeline on a multi-layer neural network."""
    # 1. Setup Contexts
    priv_ctx = create_ckks_context(poly_modulus_degree=8192)
    pub_ctx = get_public_context(priv_ctx)

    # 2. Client Models & Optimizers
    m_alpha = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    m_beta = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    m_beta.load_state_dict(m_alpha.state_dict())

    opt_alpha = torch.optim.SGD(m_alpha.parameters(), lr=0.01)
    opt_beta = torch.optim.SGD(m_beta.parameters(), lr=0.01)

    dp_alpha = DifferentialPrivacyEngine(m_alpha, opt_alpha, max_grad_norm=1.0, noise_multiplier=0.5)
    dp_beta = DifferentialPrivacyEngine(m_beta, opt_beta, max_grad_norm=1.0, noise_multiplier=0.5)

    # 3. Local DP Training
    x_a, y_a = torch.randn(4, 8), torch.tensor([0, 1, 0, 1])
    opt_alpha.zero_grad()
    loss_a = nn.CrossEntropyLoss()(m_alpha(x_a), y_a)
    loss_a.backward()
    dp_alpha.apply_gradient_clipping_and_noise(batch_size=4)
    opt_alpha.step()

    x_b, y_b = torch.randn(4, 8), torch.tensor([1, 1, 0, 0])
    opt_beta.zero_grad()
    loss_b = nn.CrossEntropyLoss()(m_beta(x_b), y_b)
    loss_b.backward()
    dp_beta.apply_gradient_clipping_and_noise(batch_size=4)
    opt_beta.step()

    w_alpha = [p.detach().cpu().numpy() for p in m_alpha.parameters()]
    w_beta = [p.detach().cpu().numpy() for p in m_beta.parameters()]

    # 4. Homomorphic Encryption
    enc_alpha = encrypt_model_parameters(priv_ctx, w_alpha, chunk_size=4096)
    enc_beta = encrypt_model_parameters(priv_ctx, w_beta, chunk_size=4096)

    # 5. Public Homomorphic Aggregation
    shapes = enc_alpha["shapes"]
    enc_results = [(enc_alpha["encrypted_chunks"], 1), (enc_beta["encrypted_chunks"], 1)]
    agg_res = aggregate_encrypted_updates(pub_ctx, enc_results, shapes)

    # 6. Decryption
    dec_params = decrypt_model_parameters(priv_ctx, agg_res["aggregated_chunks"], shapes)

    # 7. Plaintext Reference Comparison
    expected_params = [0.5 * a + 0.5 * b for a, b in zip(w_alpha, w_beta)]
    for dec_p, exp_p in zip(dec_params, expected_params):
        np.testing.assert_allclose(dec_p, exp_p, atol=1e-5)

    # 8. Load into Global Model
    m_global = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    state_keys = list(m_global.state_dict().keys())
    m_global.load_state_dict({k: torch.tensor(arr) for k, arr in zip(state_keys, dec_params)})

    # Test forward pass on global model
    out = m_global(torch.randn(2, 8))
    assert out.shape == (2, 2)
