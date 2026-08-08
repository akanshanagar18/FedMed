"""REST API endpoint for dataset versioning and lineage."""

from fastapi import APIRouter
from typing import Any, Dict

router = APIRouter()


@router.get("")
@router.get("/")
def get_dataset_version() -> Dict[str, Any]:
    return {
        "status": "success",
        "data": {
            "dataset_name": "BraTS2021",
            "version": "2.0.0",
            "dataset_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "pipeline_hash": "a1b2c3d4e5f6",
            "partition_strategy": "Dirichlet(alpha=0.5)",
            "lineage_verified": True,
        },
    }
