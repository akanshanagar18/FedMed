"""
Module: model.fedprox

Purpose:
FedProx proximal loss regularization engine.
Reference: Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
Calculates:
  L_prox(w) = L_local(w) + (mu / 2) * sum_i ||w_i - w_global_i||^2
"""

from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def compute_proximal_penalty(
    model_params: List[torch.Tensor],
    global_params: List[Union[torch.Tensor, Any]],
    mu: float = 0.01,
) -> torch.Tensor:
    """
    Computes exact mathematical proximal term: (mu / 2) * sum ||w - w_global||^2.
    """
    if mu <= 0.0 or not global_params or not model_params:
        first_device = model_params[0].device if model_params else torch.device("cpu")
        return torch.tensor(0.0, device=first_device, dtype=torch.float32)

    penalty = torch.tensor(0.0, device=model_params[0].device, dtype=torch.float32)
    for w, w_g in zip(model_params, global_params):
        if not isinstance(w_g, torch.Tensor):
            w_g_t = torch.tensor(w_g, device=w.device, dtype=w.dtype)
        else:
            w_g_t = w_g.to(device=w.device, dtype=w.dtype)
        penalty = penalty + (w - w_g_t).pow(2).sum()

    return (float(mu) / 2.0) * penalty


class FedProxCriterion(nn.Module):
    """
    Wraps base segmentation loss function with genuine FedProx proximal regularization.
    """

    def __init__(self, base_loss_fn: nn.Module, mu: float = 0.01):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.mu = float(mu)
        self.global_parameters: List[torch.Tensor] = []

    def set_global_parameters(self, global_model_or_params: Any) -> None:
        """
        Freezes global reference model weights at the start of a round.
        """
        if isinstance(global_model_or_params, nn.Module):
            self.global_parameters = [p.detach().clone() for p in global_model_or_params.parameters()]
        elif isinstance(global_model_or_params, list):
            self.global_parameters = [
                p.detach().clone() if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32)
                for p in global_model_or_params
            ]

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        current_model_params: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes (total_loss, base_loss, proximal_penalty).
        """
        base_loss = self.base_loss_fn(predictions, targets)
        prox_penalty = compute_proximal_penalty(current_model_params, self.global_parameters, self.mu)
        total_loss = base_loss + prox_penalty
        return total_loss, base_loss, prox_penalty
