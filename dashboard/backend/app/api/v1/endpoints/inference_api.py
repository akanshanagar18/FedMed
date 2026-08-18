"""
Module: dashboard.backend.app.api.v1.endpoints.inference_api

Purpose:
Production REST API endpoints for 3D Brain Tumor Segmentation Inference Engine (/api/v1/inference/*).
Serves real inference from the frozen Differential Privacy model (checkpoints/final/fedmed_dp_final_model.pt),
multi-planar 2D slice overlays, and clinical model metadata.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from inference.pipeline import global_inference_engine
from common.schemas import SuccessResponse

router = APIRouter()


class PredictRequest(BaseModel):
    patient_id: str = Field(default="BraTS-GLI-00005-100", description="Subject ID or demo case")
    model_version: str = Field(default="v2.1.0-dp-prod", description="Deployed model version")
    custom_modalities: Optional[Dict[str, str]] = Field(default=None, description="Optional custom file paths")


@router.get("/model", response_model=SuccessResponse)
async def get_model_info():
    """Returns the frozen production model specification card and DP guarantees."""
    if global_inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is offline. Model checkpoint missing or verification failed.",
        )
    info = global_inference_engine.get_model_info()
    return SuccessResponse(
        message="Production model specification retrieved",
        data=info,
    )


@router.get("/cases", response_model=SuccessResponse)
async def list_available_cases():
    """Lists safe demonstration cases available for inference."""
    demo_cases = [
        {
            "patient_id": "BraTS-GLI-00005-100",
            "cohort": "Training Silo Alpha",
            "description": "Adult Glioma Case with active Tumor Core and necrosis",
            "status": "READY",
        },
        {
            "patient_id": "BraTS-GLI-00006-100",
            "cohort": "Training Silo Alpha",
            "description": "Adult Glioma Case with non-enhancing infiltration",
            "status": "READY",
        },
        {
            "patient_id": "BraTS-GLI-00008-101",
            "cohort": "Training Silo Beta",
            "description": "Adult Glioma Case with multi-focal enhancing tumor",
            "status": "READY",
        },
        {
            "patient_id": "BraTS-GLI-00020-100",
            "cohort": "Validation Cohort",
            "description": "Adult Glioma Case (Validation split)",
            "status": "READY",
        }
    ]
    return SuccessResponse(
        message="Available demo cases retrieved",
        data={"cases": demo_cases, "total_cases": len(demo_cases)},
    )


@router.post("/predict", response_model=SuccessResponse)
async def run_inference(req: PredictRequest):
    """Executes real 3D MONAI U-Net segmentation inference and generates slice overlays."""
    if global_inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine offline.",
        )
    res = global_inference_engine.predict(
        patient_id=req.patient_id,
        model_version=req.model_version,
        custom_modality_paths=req.custom_modalities,
    )
    if res.get("status") != "SUCCESS":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.get("error_message", "Inference execution failed"),
        )
    return SuccessResponse(
        message=f"Inference completed successfully for patient '{req.patient_id}'",
        data=res,
    )


@router.get("/history", response_model=SuccessResponse)
async def get_inference_history():
    """Returns historical prediction records."""
    if global_inference_engine is None:
        return SuccessResponse(message="Engine offline", data={"total_predictions": 0, "history": []})
    history = global_inference_engine.get_history()
    return SuccessResponse(
        message="Inference history retrieved",
        data={"total_predictions": len(history), "history": history},
    )
