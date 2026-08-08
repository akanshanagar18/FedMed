"""
Module: tests.unit.test_governance_model_registry

Purpose:
Unit test suite for EnterpriseModelGovernance (HMAC signing, signature verification, canary promotion gates).
"""

import pytest
from governance.model_registry import EnterpriseModelGovernance


def test_model_signature_generation_and_verification():
    gov = EnterpriseModelGovernance()
    metrics = {"mean_dice": 0.85, "hd95": 4.1}
    sig = gov.generate_model_signature("brats_unet", "v2.0", metrics)
    assert len(sig) == 64
    assert gov.verify_model_signature("brats_unet", "v2.0", metrics, sig) is True
    assert gov.verify_model_signature("brats_unet", "v2.0", {"mean_dice": 0.80}, sig) is False


def test_canary_promotion_evaluation():
    gov = EnterpriseModelGovernance()
    cand_metrics = {"mean_dice": 0.865, "hd95": 3.8}
    prod_metrics = {"mean_dice": 0.825, "hd95": 4.5}

    eval_res = gov.evaluate_canary_promotion(cand_metrics, prod_metrics)
    assert eval_res["approved_for_promotion"] is True
    assert eval_res["dice_gate_passed"] is True
    assert eval_res["hd95_gate_passed"] is True


def test_stage_promotion_staging_to_production():
    gov = EnterpriseModelGovernance()
    cand_metrics = {"mean_dice": 0.865, "hd95": 3.8}
    prod_metrics = {"mean_dice": 0.825, "hd95": 4.5}
    sig = gov.generate_model_signature("brats_unet", "v2.0", cand_metrics)

    res = gov.promote_stage(
        model_id="brats_unet",
        current_stage="Staging",
        target_stage="Production",
        signature=sig,
        candidate_metrics=cand_metrics,
        production_baseline_metrics=prod_metrics,
    )
    assert res["success"] is True
    assert res["new_stage"] == "Production"
