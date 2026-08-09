"""
Module: tests.unit.test_deployment_manager

Purpose:
Unit test suite for ProductionDeploymentManager (Canary/Rolling deployment initiation, promotion, emergency rollback).
"""

import pytest
from deployment.manager import ProductionDeploymentManager, DeploymentStrategy, DeploymentStatus


def test_deployment_manager_initiation_and_promotion():
    mgr = ProductionDeploymentManager()
    cand_metrics = {"mean_dice": 0.88, "hd95": 3.2}
    dep = mgr.initiate_deployment(
        model_id="brats_unet",
        version="v2.1.0",
        strategy=DeploymentStrategy.CANARY,
        candidate_metrics=cand_metrics,
    )
    assert dep["model_id"] == "brats_unet"
    assert dep["status"] == DeploymentStatus.IN_PROGRESS.value
    assert dep["traffic_percentage"] == 10.0

    promo = mgr.promote_deployment(dep["deployment_id"])
    assert promo["success"] is True
    assert promo["deployment"]["status"] == DeploymentStatus.COMPLETED.value
    assert promo["deployment"]["traffic_percentage"] == 100.0


def test_deployment_manager_emergency_rollback():
    mgr = ProductionDeploymentManager()
    dep = mgr.initiate_deployment(
        model_id="brats_unet_flawed",
        version="v2.2.0",
        strategy=DeploymentStrategy.ROLLING,
        candidate_metrics={"mean_dice": 0.70},
    )
    rollback = mgr.emergency_rollback(dep["deployment_id"], "Error rate spike detected")
    assert rollback["success"] is True
    assert rollback["deployment"]["status"] == DeploymentStatus.ROLLED_BACK.value
    assert rollback["deployment"]["traffic_percentage"] == 0.0
