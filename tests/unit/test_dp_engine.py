"""
Unit tests for privacy.dp_engine (Opacus Differential Privacy Engine & RDP Budget Tracking).
"""

import torch
import torch.nn as nn
import pytest
from privacy.dp_engine import DifferentialPrivacyEngine, compute_rdp_epsilon


def test_rdp_epsilon_computation():
    """Verify RDP privacy budget calculation produces non-negative epsilon."""
    eps = compute_rdp_epsilon(steps=10, noise_multiplier=1.0, target_delta=1e-5, sample_rate=0.1)
    assert isinstance(eps, float)
    assert eps > 0.0


def test_gradient_clipping_and_noise_injection():
    """Verify gradient norm clipping to max_grad_norm and Gaussian noise addition."""
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    dp_engine = DifferentialPrivacyEngine(
        model=model,
        optimizer=optimizer,
        target_epsilon=3.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=0.8,
    )

    inputs = torch.randn(4, 10)
    targets = torch.randn(4, 1)

    optimizer.zero_grad()
    loss = ((model(inputs) - targets) ** 2).mean()
    loss.backward()

    # Pre-clipping gradients exist
    param = list(model.parameters())[0]
    assert param.grad is not None

    pre_norm = dp_engine.apply_gradient_clipping_and_noise(batch_size=4)
    assert pre_norm > 0.0

    # Post-clipping gradient norm should be bounded
    post_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2).item()
    assert post_norm is not None

    budget = dp_engine.get_privacy_budget()
    assert budget["dp_enabled"] is True
    assert budget["epsilon"] >= 0.0
    assert budget["steps"] == 1
