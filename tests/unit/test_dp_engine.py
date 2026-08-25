"""
Unit tests for privacy.dp_engine:
1. Exact Analytical Rényi Differential Privacy (Poisson Subsampled Gaussian RDP)
2. Numerical Integration Reference Validation & Cross-Check
3. Monotonicity & Sensitivity Properties across Steps, Sigma, and Delta
4. Calibration of Minimum Noise Multiplier
5. Poisson Batch Sampler Mechanics & Bernoulli Inclusion
6. Gradient Norm Clipping & Noise Scale Verification
"""

import math
import pytest
import torch
import torch.nn as nn

from privacy.dp_engine import (
    DifferentialPrivacyEngine,
    PoissonBatchSampler,
    compute_poisson_rdp_step,
    compute_rdp_epsilon,
    compute_rdp_budget_detailed,
    compute_rdp_reference,
    calibrate_noise_multiplier,
)


def test_rdp_step_analytical_vs_reference_cross_check():
    """Verify analytical log-sum-exp RDP matches independent numerical integration reference."""
    q = 1.0 / 236.0
    sigmas = [0.5, 0.8, 1.0, 1.5, 2.0]
    orders = [2, 3]

    for sigma in sigmas:
        for alpha in orders:
            analytical_val = compute_poisson_rdp_step(alpha=alpha, q=q, sigma=sigma)
            reference_val = compute_rdp_reference(alpha=alpha, q=q, sigma=sigma)
            
            # Verify exact match between analytical discrete formula and numerical integral
            assert pytest.approx(analytical_val, rel=1e-5, abs=1e-10) == reference_val, (
                f"Mismatch for sigma={sigma}, alpha={alpha}: "
                f"analytical={analytical_val}, reference={reference_val}"
            )


def test_rdp_known_baseline_values():
    """Verify exact computed epsilon against validated reference points."""
    q = 1.0 / 236.0
    delta = 1e-5

    # 1. 1-Step at sigma=0.50 (Exact Poisson RDP baseline)
    eps_1step_05 = compute_rdp_epsilon(steps=1, noise_multiplier=0.50, target_delta=delta, sample_rate=q)
    assert pytest.approx(eps_1step_05, abs=1e-3) == 4.5914

    # 2. 4720-Steps at sigma=0.50 (Full 20-round experiment under old uncalibrated sigma)
    eps_4720_05 = compute_rdp_epsilon(steps=4720, noise_multiplier=0.50, target_delta=delta, sample_rate=q)
    assert pytest.approx(eps_4720_05, abs=1e-3) == 16.0530

    # 3. 4720-Steps at calibrated sigma=0.87 (Conservative calibrated noise multiplier)
    budget_087 = compute_rdp_budget_detailed(steps=4720, noise_multiplier=0.87, target_delta=delta, sample_rate=q)
    assert budget_087["optimal_alpha"] == 7
    assert budget_087["epsilon"] < 2.9633
    assert pytest.approx(budget_087["epsilon"], abs=1e-3) == 2.8934


def test_noise_multiplier_calibration():
    """Verify binary search solver finds minimum sigma for epsilon <= 2.9633."""
    q = 1.0 / 236.0
    delta = 1e-5
    steps = 4720
    target_eps = 2.9633

    sigma_min = calibrate_noise_multiplier(
        target_epsilon=target_eps,
        target_delta=delta,
        sample_rate=q,
        steps=steps,
        tol=1e-4,
    )

    # sigma_min should be approx 0.8612
    assert 0.855 <= sigma_min <= 0.865
    eps_min = compute_rdp_epsilon(steps=steps, noise_multiplier=sigma_min, target_delta=delta, sample_rate=q)
    assert eps_min <= target_eps + 1e-3


def test_rdp_monotonicity_properties():
    """Verify fundamental monotonicity properties of differential privacy."""
    q = 1.0 / 236.0
    delta = 1e-5
    sigma = 0.87

    # 1. Epsilon strictly increases with steps T
    eps_100 = compute_rdp_epsilon(steps=100, noise_multiplier=sigma, target_delta=delta, sample_rate=q)
    eps_1000 = compute_rdp_epsilon(steps=1000, noise_multiplier=sigma, target_delta=delta, sample_rate=q)
    eps_4720 = compute_rdp_epsilon(steps=4720, noise_multiplier=sigma, target_delta=delta, sample_rate=q)
    assert eps_100 < eps_1000 < eps_4720

    # 2. Epsilon strictly decreases as sigma increases (more noise = more privacy)
    eps_sig_07 = compute_rdp_epsilon(steps=4720, noise_multiplier=0.70, target_delta=delta, sample_rate=q)
    eps_sig_087 = compute_rdp_epsilon(steps=4720, noise_multiplier=0.87, target_delta=delta, sample_rate=q)
    eps_sig_12 = compute_rdp_epsilon(steps=4720, noise_multiplier=1.20, target_delta=delta, sample_rate=q)
    assert eps_sig_07 > eps_sig_087 > eps_sig_12

    # 3. Epsilon strictly increases as target delta decreases
    eps_delta_5 = compute_rdp_epsilon(steps=4720, noise_multiplier=sigma, target_delta=1e-5, sample_rate=q)
    eps_delta_6 = compute_rdp_epsilon(steps=4720, noise_multiplier=sigma, target_delta=1e-6, sample_rate=q)
    assert eps_delta_6 > eps_delta_5


def test_poisson_batch_sampler_properties():
    """Verify PoissonBatchSampler performs independent Bernoulli trials per record."""
    dataset_size = 236
    q = 1.0 / 236.0
    num_steps = 1000
    sampler = PoissonBatchSampler(dataset_size=dataset_size, sample_rate=q, num_steps=num_steps, seed=42)

    assert len(sampler) == num_steps

    batches = list(sampler)
    assert len(batches) == num_steps

    batch_lens = [len(b) for b in batches]
    mean_batch_size = sum(batch_lens) / len(batch_lens)
    
    # Expected batch size = N * q = 236 * (1/236) = 1.0
    assert pytest.approx(mean_batch_size, rel=0.1) == 1.0

    # Empirical empty batch rate should be near (1 - 1/236)^236 ~ 1/e ~ 0.3678
    empty_rate = batch_lens.count(0) / num_steps
    assert 0.30 <= empty_rate <= 0.44

    # All generated indices must be within [0, dataset_size - 1]
    for b in batches:
        for idx in b:
            assert 0 <= idx < dataset_size


def test_dp_engine_detailed_budget_telemetry():
    """Verify DifferentialPrivacyEngine exposes accurate budget and telemetry."""
    model = nn.Linear(5, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    dp_engine = DifferentialPrivacyEngine(
        model=model,
        optimizer=optimizer,
        target_epsilon=3.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=0.87,
        sample_rate=1.0 / 236.0,
    )

    # 1. Step with synthetic gradient
    x = torch.randn(1, 5)
    y = torch.randn(1, 2)
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).sum()
    loss.backward()

    raw_norm = dp_engine.apply_gradient_clipping_and_noise()
    assert raw_norm > 0.0
    optimizer.step()

    budget = dp_engine.get_privacy_budget()
    assert budget["dp_enabled"] is True
    assert budget["steps"] == 1
    assert budget["noise_multiplier"] == 0.87
    assert budget["optimal_alpha"] >= 2
    assert budget["epsilon"] > 0.0
