"""
Module: deployment

Purpose:
Production Deployment Manager supporting Canary, Rolling, Blue-Green, Shadow,
and Emergency Rollback deployment strategies for FedMed v2.0 models.
"""

from deployment.manager import ProductionDeploymentManager, DeploymentStrategy, DeploymentStatus

__all__ = [
    "ProductionDeploymentManager",
    "DeploymentStrategy",
    "DeploymentStatus",
]
