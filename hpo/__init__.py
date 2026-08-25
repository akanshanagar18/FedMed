"""
Module: hpo

Purpose:
Federated Hyperparameter Optimization (FedHPO) Engine for FedMed v2.0.
Implements Federated Successive Halving (Asynchronous Hyperband) and Bayesian Search
for automated hyperparameter tuning across heterogeneous hospital nodes.
"""

from hpo.hpo_engine import FederatedHPOEngine, HyperparameterSpace

__all__ = ["FederatedHPOEngine", "HyperparameterSpace"]
