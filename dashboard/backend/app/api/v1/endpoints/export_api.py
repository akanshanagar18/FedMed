"""
Module: dashboard.backend.app.api.v1.endpoints.export_api

Purpose:
REST API endpoints for Model Export & Governance Certificates (/api/v1/export/*).
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from deployment.exporter import global_model_exporter
from common.schemas import SuccessResponse

router = APIRouter()


class ExportModelRequest(BaseModel):
    model_name: str = Field(default="brats_monai_3d_unet")
    version: str = Field(default="v2.0.0-prod")


@router.post("/model", response_model=SuccessResponse)
async def export_production_model(req: ExportModelRequest):
    """Exports candidate model to TorchScript and generates HIPAA/GDPR certificate."""
    res = global_model_exporter.export_model(model_name=req.model_name, version=req.version)
    return SuccessResponse(
        message=f"Model '{req.model_name}:{req.version}' exported successfully",
        data=res,
    )
