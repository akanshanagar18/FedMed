"""
Module: dashboard.backend.app.api.v1.endpoints.foundation

Purpose:
REST API endpoints for Vision Foundation Models (SAM, MedSAM, DINOv2) and PEFT LoRA parameter statistics.
"""

from fastapi import APIRouter
from typing import Any, Dict

router = APIRouter()


@router.get("")
@router.get("/")
def get_foundation_model_metrics() -> Dict[str, Any]:
    """Returns foundation model architectures, LoRA rank parameters, and frozen backbone stats."""
    return {
        "status": "success",
        "data": {
            "supported_models": ["UNet3D", "SAM-ViT-B", "MedSAM-3D", "DINOv2-ViT-B14"],
            "active_model": "MedSAM-3D",
            "peft_method": "LoRA (r=8, alpha=16)",
            "total_parameters": 93_500_000,
            "frozen_parameters": 91_000_000,
            "trainable_parameters": 2_500_000,
            "trainable_percent": 2.67,
            "gpu_memory_savings_pct": 72.5,
        },
    }
