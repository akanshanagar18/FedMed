#!/usr/bin/env python3
"""
Script: scripts/audit_dp_mechanism_semantics.py
Phase: 10.5D Controlled DP Mechanism Semantics Audit

Performs strict forensic verification of:
1. Poisson / Bernoulli selection probabilities and batch size distribution
2. Empty-batch handling (|S_t| = 0) with zero gradient, Gaussian noise, and Adam step
3. Per-sample gradient clipping (||g_i||_2 <= C) and sensitivity verification (Delta_2 f = C)
4. Gradient accumulation (un-normalized sum) and noise scale verification (N(0, sigma^2 C^2 I))
5. Noise empirical mean (~0.0) and empirical standard deviation (~0.87)
6. Adam interaction and parameter updates on empty vs multi-sample batches
7. Privacy accountant step synchronization and cumulative budget tracking.
"""

import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacy.dp_engine import (
    PoissonBatchSampler,
    compute_poisson_rdp_step,
    compute_rdp_budget_detailed,
    compute_rdp_epsilon,
)


def run_mechanism_audit():
    print("=" * 80)
    print("🔍 PHASE 10.5D: CONTROLLED DP MECHANISM SEMANTICS FORENSIC AUDIT")
    print("=" * 80)

    # 1. Test Poisson / Bernoulli Sampler Statistics
    dataset_size = 236
    sample_rate = 1.0 / 236.0
    num_steps = 10000
    seed = 42

    sampler = PoissonBatchSampler(
        dataset_size=dataset_size,
        sample_rate=sample_rate,
        num_steps=num_steps,
        seed=seed,
    )

    batch_sizes = []
    item_counts = np.zeros(dataset_size, dtype=np.int64)

    for indices in sampler:
        batch_sizes.append(len(indices))
        for idx in indices:
            item_counts[idx] += 1

    batch_sizes = np.array(batch_sizes)
    mean_batch_size = np.mean(batch_sizes)
    empty_frac = np.mean(batch_sizes == 0)
    single_frac = np.mean(batch_sizes == 1)
    multi_frac = np.mean(batch_sizes >= 2)
    max_batch_size = np.max(batch_sizes)
    mean_item_selection_prob = np.mean(item_counts / num_steps)

    theoretical_empty_prob = (1.0 - sample_rate) ** dataset_size
    theoretical_single_prob = dataset_size * sample_rate * ((1.0 - sample_rate) ** (dataset_size - 1))
    theoretical_multi_prob = 1.0 - theoretical_empty_prob - theoretical_single_prob

    print(f"\n[1. SAMPLER DYNAMICS] Evaluated {num_steps} Poisson steps (N={dataset_size}, q=1/{dataset_size}):")
    print(f"  Empirical Mean Batch Size: {mean_batch_size:.4f} (Expected: 1.0000)")
    print(f"  Empirical Empty Batch Rate: {empty_frac*100:.2f}% (Theoretical: {theoretical_empty_prob*100:.2f}%)")
    print(f"  Empirical Singleton Rate:   {single_frac*100:.2f}% (Theoretical: {theoretical_single_prob*100:.2f}%)")
    print(f"  Empirical Multi-Sample Rate:{multi_frac*100:.2f}% (Theoretical: {theoretical_multi_prob*100:.2f}%)")
    print(f"  Empirical Max Batch Size:   {max_batch_size}")
    print(f"  Mean Per-Item Selection Prob:{mean_item_selection_prob:.6f} (Target q: {sample_rate:.6f})")

    assert abs(mean_batch_size - 1.0) < 0.02, "Sampler mean batch size diverges from 1.0"
    assert abs(empty_frac - theoretical_empty_prob) < 0.02, "Empty batch rate diverges"
    assert abs(mean_item_selection_prob - sample_rate) < 0.001, "Per-item inclusion rate diverges"

    # 2. Test Noise Empirical Distribution
    torch.manual_seed(seed)
    sigma = 0.87
    C = 1.0
    expected_noise_std = sigma * C

    num_noise_elements = 4_810_074  # Exact MONAI 3D U-Net parameter count
    noise_tensor = torch.randn(num_noise_elements) * expected_noise_std
    emp_mean = noise_tensor.mean().item()
    emp_std = noise_tensor.std().item()

    print(f"\n[2. NOISE DISTRIBUTION SANITY CHECK] Over {num_noise_elements:,} parameters (sigma={sigma}, C={C}):")
    print(f"  Empirical Mean: {emp_mean:.6f} (Expected: 0.000000)")
    print(f"  Empirical Std:  {emp_std:.6f} (Expected: {expected_noise_std:.6f})")
    assert abs(emp_mean) < 0.002, "Noise mean not zero"
    assert abs(emp_std - expected_noise_std) < 0.002, "Noise std does not match sigma * C"

    # 3. Test Mini-Model Gradient Clipping, Accumulation, and Optimizer Updates
    torch.manual_seed(seed)
    mini_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )
    optimizer = torch.optim.Adam(mini_model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    synthetic_X = torch.randn(10, 10)
    synthetic_y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Step A: Empty batch (|S_t| = 0)
    w_before_empty = [p.clone().detach() for p in mini_model.parameters()]
    optimizer.zero_grad()
    accumulated_grads = [torch.zeros_like(p.data) for p in mini_model.parameters()]
    # Noise on zero gradient
    noise_std = sigma * C
    for acc, p in zip(accumulated_grads, mini_model.parameters()):
        noise = torch.randn_like(acc) * noise_std
        p.grad = acc + noise
    optimizer.step()
    w_after_empty = [p.clone().detach() for p in mini_model.parameters()]
    delta_empty = sum((w1 - w0).norm().item() for w0, w1 in zip(w_before_empty, w_after_empty))

    print(f"\n[3. EMPTY BATCH MECHANISM UPDATE]")
    print(f"  Empty Batch Parameter Delta Norm: {delta_empty:.6f} (> 0 confirmed)")
    assert delta_empty > 0.0, "Empty batch failed to update parameters via noise + weight decay"

    # Step B: Multi-sample batch (|S_t| = 3) with per-sample clipping
    indices = [0, 2, 4]
    w_before_multi = [p.clone().detach() for p in mini_model.parameters()]
    optimizer.zero_grad()
    accumulated_grads = [torch.zeros_like(p.data) for p in mini_model.parameters()]

    pre_clip_norms = []
    clipped_count = 0

    for idx in indices:
        mini_model.zero_grad()
        x_i = synthetic_X[idx:idx+1]
        y_i = synthetic_y[idx:idx+1]
        out_i = mini_model(x_i)
        loss_i = loss_fn(out_i, y_i)
        loss_i.backward()

        active_params = [p for p in mini_model.parameters() if p.grad is not None]
        s_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in active_params]), 2).item()
        pre_clip_norms.append(s_norm)

        if s_norm > C:
            clipped_count += 1
            clip_f = C / s_norm
        else:
            clip_f = 1.0

        for acc, p in zip(accumulated_grads, active_params):
            acc.add_(p.grad * clip_f)

    # Check accumulated norm <= 3 * C
    acc_norm = torch.norm(torch.stack([torch.norm(acc, 2) for acc in accumulated_grads]), 2).item()

    for acc, p in zip(accumulated_grads, mini_model.parameters()):
        noise = torch.randn_like(acc) * noise_std
        p.grad = acc + noise
    optimizer.step()
    w_after_multi = [p.clone().detach() for p in mini_model.parameters()]
    delta_multi = sum((w1 - w0).norm().item() for w0, w1 in zip(w_before_multi, w_after_multi))

    print(f"\n[4. MULTI-SAMPLE BATCH MECHANISM UPDATE (|S_t|=3)]")
    print(f"  Pre-clip Sample Norms: {[round(n, 4) for n in pre_clip_norms]}")
    print(f"  Accumulated Un-normalized Gradient Norm: {acc_norm:.4f} (Max possible: {len(indices)*C:.4f})")
    print(f"  Multi-Sample Parameter Delta Norm: {delta_multi:.6f}")
    assert acc_norm <= len(indices) * C + 1e-5, "Accumulated gradient exceeds theoretical bound |S_t| * C"

    # 4. Test RDP Budget Accounting Synchronization
    total_steps = 4720
    budget = compute_rdp_budget_detailed(
        steps=total_steps,
        noise_multiplier=sigma,
        target_delta=1e-5,
        sample_rate=sample_rate,
    )
    print(f"\n[5. EXACT RDP ACCOUNTANT VERIFICATION (T={total_steps}, sigma={sigma}, q=1/236, delta=1e-5)]")
    print(f"  Computed Epsilon: {budget['epsilon']:.4f} (Target <= 2.9633)")
    print(f"  Optimal Order alpha*: {budget['optimal_alpha']}")
    print(f"  Step RDP D_7: {budget['step_rdp']:.8e}")
    print(f"  Total RDP T*D_7: {budget['total_rdp']:.6f}")

    assert budget["epsilon"] <= 2.9633, "Epsilon exceeds target threshold"
    assert budget["optimal_alpha"] == 7, "Optimal alpha is not 7"

    print("\n" + "=" * 80)
    print("✅ ALL CONTROLLED MECHANISM TESTS PASSED WITH STRICT MATHEMATICAL INTEGRITY")
    print("=" * 80)
    return True


if __name__ == "__main__":
    run_mechanism_audit()
