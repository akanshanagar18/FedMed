"""
Module: dashboard.backend.app.api.v1.endpoints.inference_api

Purpose:
REST API endpoints for 3D Segmentation Inference Engine (/api/v1/inference/*).
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from inference.pipeline import global_inference_engine
from common.schemas import SuccessResponse

router = APIRouter()


class PredictRequest(BaseModel):
    patient_id: str = Field(default="PATIENT_DEMO_001")
    model_version: str = Field(default="v2.0.0-prod")


@router.post("/predict", response_model=SuccessResponse)
async def run_inference(req: PredictRequest):
    """Executes 3D MONAI sliding window segmentation inference."""
    res = global_inference_engine.predict(patient_id=req.patient_id, model_version=req.model_version)
    return SuccessResponse(
        message=f"Inference completed for patient '{req.patient_id}'",
        data=res,
    )


@router.get("/history", response_model=SuccessResponse)
async def get_inference_history():
    """Returns historical prediction records."""
    history = global_inference_engine.get_history()
    return SuccessResponse(
        message="Inference history retrieved",
        data={"total_predictions": len(history), "history": history},
    )
