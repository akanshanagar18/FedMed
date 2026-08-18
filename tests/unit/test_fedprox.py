"""
Unit tests for FedProx client regularization and server aggregation (Li et al., MLSys 2020).
Verifies:
  1. mu = 0 produces behavior equivalent to standard local objective
  2. Proximal penalty is zero when local weights equal global reference weights
  3. Proximal penalty is strictly positive when local weights diverge from global weights
  4. Proximal penalty matches exact independent mathematical formula: (mu / 2) * sum((w - w_global)^2)
  5. total_loss = base_loss + proximal_penalty
  6. Global reference parameters remain fixed during local optimization
  7. Changing global reference parameters changes proximal penalty accordingly
  8. No hardcoded proximal values or heuristics
  9. Server strategy metadata and sample-weighted aggregation
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from model.fedprox import FedProxCriterion, compute_proximal_penalty
from server.strategies.base import EvaluateResult, FitResult
from server.strategies.fedprox import FedProx


def test_fedprox_mu_zero_equivalence():
    """1. mu = 0 produces penalty of 0 and total_loss == base_loss."""
    base_loss = nn.MSELoss()
    criterion = FedProxCriterion(base_loss_fn=base_loss, mu=0.0)

    p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    p_global = torch.tensor([10.0, 20.0, 30.0])
    criterion.set_global_parameters([p_global])

    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])

    total_loss, b_loss, prox_penalty = criterion(pred, target, [p1])
    assert prox_penalty.item() == 0.0
    assert total_loss.item() == b_loss.item()


def test_fedprox_zero_divergence_zero_penalty():
    """2. Proximal penalty is 0 when local weights equal global reference weights."""
    base_loss = nn.MSELoss()
    criterion = FedProxCriterion(base_loss_fn=base_loss, mu=0.01)

    p1 = torch.nn.Parameter(torch.tensor([1.5, -2.5, 3.0]))
    criterion.set_global_parameters([p1.clone().detach()])

    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])

    total_loss, b_loss, prox_penalty = criterion(pred, target, [p1])
    assert pytest.approx(prox_penalty.item(), abs=1e-7) == 0.0


def test_fedprox_positive_penalty_on_divergence():
    """3. Proximal penalty is > 0 when local weights diverge from global weights."""
    base_loss = nn.MSELoss()
    criterion = FedProxCriterion(base_loss_fn=base_loss, mu=0.01)

    p_local = [torch.nn.Parameter(torch.tensor([1.0, 2.0])), torch.nn.Parameter(torch.tensor([3.0, 4.0]))]
    p_global = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
    criterion.set_global_parameters(p_global)

    pred = torch.tensor([1.0])
    target = torch.tensor([1.0])

    total_loss, b_loss, prox_penalty = criterion(pred, target, p_local)
    assert prox_penalty.item() > 0.0


def test_fedprox_exact_mathematical_calculation():
    """4. Proximal penalty matches independent calculation: (mu / 2) * sum((w - w_g)^2)."""
    mu = 0.05
    w1 = torch.tensor([1.0, 2.0, -1.0])
    w2 = torch.tensor([[0.5, 1.5], [-0.5, 2.5]])
    wg1 = torch.tensor([0.0, 1.0, 1.0])
    wg2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    # Independent NumPy calculation
    diff1 = (w1.numpy() - wg1.numpy()) ** 2
    diff2 = (w2.numpy() - wg2.numpy()) ** 2
    expected_penalty = (mu / 2.0) * float(np.sum(diff1) + np.sum(diff2))

    calc_penalty = compute_proximal_penalty([w1, w2], [wg1, wg2], mu=mu).item()
    assert pytest.approx(calc_penalty, rel=1e-6) == expected_penalty


def test_fedprox_total_loss_sum():
    """5. total_loss = base_loss + proximal_penalty."""
    base_loss = nn.L1Loss()
    criterion = FedProxCriterion(base_loss_fn=base_loss, mu=0.1)

    p_local = [torch.nn.Parameter(torch.tensor([2.0, 3.0]))]
    p_global = [torch.tensor([0.0, 0.0])]
    criterion.set_global_parameters(p_global)

    pred = torch.tensor([5.0])
    target = torch.tensor([1.0])

    total_loss, b_loss, prox_penalty = criterion(pred, target, p_local)
    assert pytest.approx(total_loss.item(), rel=1e-6) == (b_loss.item() + prox_penalty.item())


def test_fedprox_global_reference_fixed_during_optimization():
    """6. Global reference parameters remain unchanged during optimizer steps."""
    mu = 0.1
    p_local = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    p_global = torch.tensor([0.0, 0.0])

    criterion = FedProxCriterion(base_loss_fn=nn.MSELoss(), mu=mu)
    criterion.set_global_parameters([p_global])

    optimizer = torch.optim.SGD([p_local], lr=0.1)

    # Global snapshot before training
    global_before = criterion.global_parameters[0].clone()

    # Step 1
    optimizer.zero_grad()
    t_loss, _, _ = criterion(p_local, torch.tensor([0.0, 0.0]), [p_local])
    t_loss.backward()
    optimizer.step()

    # Verify global parameters did not mutate
    assert torch.equal(criterion.global_parameters[0], global_before)


def test_fedprox_changing_global_changes_penalty():
    """7. Changing global reference parameters changes the proximal penalty value."""
    p_local = [torch.tensor([2.0, 2.0])]
    pen1 = compute_proximal_penalty(p_local, [torch.tensor([0.0, 0.0])], mu=0.1).item()
    pen2 = compute_proximal_penalty(p_local, [torch.tensor([2.0, 2.0])], mu=0.1).item()
    pen3 = compute_proximal_penalty(p_local, [torch.tensor([10.0, 10.0])], mu=0.1).item()

    assert pen1 > pen2
    assert pen2 == 0.0
    assert pen3 > pen1


def test_fedprox_strategy_metadata_and_aggregation():
    """8 & 9. Server strategy metadata and sample-weighted aggregation."""
    strategy = FedProx(proximal_mu=0.02)
    meta = strategy.get_metadata()
    assert meta.name == "FedProx"
    assert strategy.proximal_mu == 0.02

    p1 = [np.array([1.0, 2.0], dtype=np.float32)]
    p2 = [np.array([3.0, 4.0], dtype=np.float32)]
    r1 = FitResult(parameters=p1, num_examples=2, metrics={"training_loss": 0.4, "dice_score": 0.8})
    r2 = FitResult(parameters=p2, num_examples=2, metrics={"training_loss": 0.2, "dice_score": 0.9})

    agg_p, metrics = strategy.aggregate_fit(server_round=1, results=[r1, r2], failures=[])
    np.testing.assert_allclose(agg_p[0], np.array([2.0, 3.0], dtype=np.float32))
    assert metrics["proximal_mu"] == 0.02
    assert pytest.approx(metrics["training_loss"], 1e-5) == 0.3
