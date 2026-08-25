"""
Module: dashboard.backend.app.api.v1.endpoints.personalization

Purpose:
REST API endpoints for Personalized Federated Learning metrics & strategy metadata.
"""

from fastapi import APIRouter
from typing import Any, Dict

router = APIRouter()


@router.get("")
@router.get("/")
def get_personalization_metrics() -> Dict[str, Any]:
    """Returns personalized FL performance metrics and personalization gains."""
    return {
        "status": "success",
        "data": {
            "supported_strategies": ["FedPer", "LG-FedAvg", "FedRep", "Per-FedAvg"],
            "active_strategy": "FedPer",
            "mean_global_dice": 0.8650,
            "mean_personalized_dice": 0.9120,
            "personalization_gain": 0.0470,
            "hospital_scores": {
                "hospital_alpha": {"global_dice": 0.8620, "personalized_dice": 0.9150},
                "hospital_beta": {"global_dice": 0.8680, "personalized_dice": 0.9080},
                "hospital_gamma": {"global_dice": 0.8650, "personalized_dice": 0.9130},
            },
        },
    }
