"""
Module: governance.model_registry

Purpose:
Enterprise Model Lifecycle Governance Registry.
Handles model stage promotion (Candidate -> Staging -> Production -> Archived), cryptographic model signing (HMAC-SHA256),
canary rollout verification, and automated safety rollback gates.
"""

import hmac
import hashlib
import json
import time
from typing import Any, Dict, List, Optional


class EnterpriseModelGovernance:
    """
    Enterprise Governance Registry for managing model lifecycles across federated hospital deployments.
    """

    STAGES = ["Candidate", "Staging", "Production", "Archived"]

    def __init__(self, signing_key: str = "FedMed_Model_Governance_Key_2026"):
        self.signing_key = signing_key

    def generate_model_signature(self, model_id: str, version: str, metrics: Dict[str, float]) -> str:
        """
        Generates HMAC-SHA256 signature for a model checkpoint to guarantee integrity.
        """
        payload = json.dumps({"model_id": model_id, "version": version, "metrics": metrics}, sort_keys=True)
        signature = hmac.new(
            self.signing_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def verify_model_signature(
        self, model_id: str, version: str, metrics: Dict[str, float], signature: str
    ) -> bool:
        """
        Verifies HMAC-SHA256 signature against payload.
        """
        expected_sig = self.generate_model_signature(model_id, version, metrics)
        return hmac.compare_digest(expected_sig, signature)

    def evaluate_canary_promotion(
        self,
        candidate_metrics: Dict[str, float],
        production_baseline_metrics: Dict[str, float],
        min_dice_improvement: float = 0.005,
        max_hd95_degradation: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluates canary rollout metrics of candidate model against current production model.
        Promotes candidate to Production if it exceeds quality thresholds without safety regressions.
        """
        cand_dice = candidate_metrics.get("mean_dice", 0.0)
        prod_dice = production_baseline_metrics.get("mean_dice", 0.0)

        cand_hd95 = candidate_metrics.get("hd95", 999.0)
        prod_hd95 = production_baseline_metrics.get("hd95", 999.0)

        dice_diff = cand_dice - prod_dice
        hd95_diff = cand_hd95 - prod_hd95

        dice_pass = dice_diff >= min_dice_improvement
        hd95_pass = hd95_diff <= max_hd95_degradation

        approved = bool(dice_pass and hd95_pass)

        return {
            "approved_for_promotion": approved,
            "candidate_dice": round(cand_dice, 4),
            "production_baseline_dice": round(prod_dice, 4),
            "dice_improvement": round(dice_diff, 4),
            "dice_gate_passed": dice_pass,
            "candidate_hd95": round(cand_hd95, 2),
            "production_baseline_hd95": round(prod_hd95, 2),
            "hd95_gate_passed": hd95_pass,
            "recommendation": (
                "Promote to Production (Canary verification passed)"
                if approved
                else "Reject promotion: candidate failed quality or safety gate"
            ),
        }

    def promote_stage(
        self,
        model_id: str,
        current_stage: str,
        target_stage: str,
        signature: str,
        candidate_metrics: Dict[str, float],
        production_baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes stage promotion with signature validation and canary gates.
        """
        if current_stage not in self.STAGES or target_stage not in self.STAGES:
            return {
                "success": False,
                "message": f"Invalid stage transition from {current_stage} to {target_stage}",
            }

        # Verify Signature
        if not self.verify_model_signature(model_id, "v2.0", candidate_metrics, signature):
            return {
                "success": False,
                "message": "Invalid HMAC model signature. Promotion aborted.",
            }

        # If promoting from Staging -> Production, run canary gate
        if current_stage == "Staging" and target_stage == "Production":
            baseline = production_baseline_metrics or {"mean_dice": 0.82, "hd95": 4.5}
            canary_eval = self.evaluate_canary_promotion(candidate_metrics, baseline)
            if not canary_eval["approved_for_promotion"]:
                return {
                    "success": False,
                    "message": "Canary rollout failed quality gate",
                    "canary_evaluation": canary_eval,
                }

        return {
            "success": True,
            "model_id": model_id,
            "previous_stage": current_stage,
            "new_stage": target_stage,
            "promoted_at": int(time.time()),
            "signature": signature,
            "message": f"Model {model_id} successfully promoted to {target_stage}",
        }
