"""
Module: dashboard.backend.app.api.v1.endpoints.dataset_mgmt

Purpose:
REST API endpoints for Dataset Management (/api/v1/dataset-management/*).
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from dataset.manager import global_dataset_manager
from common.schemas import SuccessResponse

router = APIRouter()


class DatasetRegisterRequest(BaseModel):
    dataset_name: str = Field(default="BraTS2021_Synthetic_Cohort")
    version: str = Field(default="v1.1.0")
    modality: str = Field(default="3D_MRI")
    modalities: Optional[List[str]] = Field(default_factory=lambda: ["T1", "T1ce", "T2", "FLAIR"])
    num_samples: int = Field(default=250, ge=1)
    storage_path: str = Field(default="data/brats2021_v1")


@router.post("/register", response_model=SuccessResponse)
async def register_dataset(req: DatasetRegisterRequest):
    """Registers a dataset version manifest with SHA-256 checksum."""
    ds = global_dataset_manager.register_dataset(
        dataset_name=req.dataset_name,
        version=req.version,
        modality=req.modality,
        modalities=req.modalities,
        num_samples=req.num_samples,
        storage_path=req.storage_path,
    )
    return SuccessResponse(
        message=f"Dataset '{ds['dataset_id']}' registered and validated",
        data=ds,
    )


@router.get("/datasets", response_model=SuccessResponse)
async def list_datasets():
    """Lists registered datasets."""
    datasets = global_dataset_manager.list_datasets()
    return SuccessResponse(
        message="Datasets retrieved",
        data={"total_datasets": len(datasets), "datasets": datasets},
    )


@router.post("/validate/{dataset_id}", response_model=SuccessResponse)
async def validate_dataset_checksum(dataset_id: str):
    """Validates SHA-256 integrity checksum for a dataset."""
    valid = global_dataset_manager.validate_checksum(dataset_id)
    if not valid:
        raise HTTPException(status_code=400, detail=f"Checksum validation failed for dataset '{dataset_id}'")

    return SuccessResponse(
        message=f"SHA-256 checksum validation PASSED for dataset '{dataset_id}'",
        data={"dataset_id": dataset_id, "checksum_valid": True},
    )
