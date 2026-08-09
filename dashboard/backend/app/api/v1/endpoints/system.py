"""
Module: dashboard.backend.app.api.v1.endpoints.system

Purpose:
REST API endpoints for platform reproducibility metadata, environment parameters, and health proxies.
"""

from fastapi import APIRouter
from common.schemas import SuccessResponse
from utils.reproducibility import get_system_reproducibility_metadata

router = APIRouter()


@router.get("/reproducibility", response_model=SuccessResponse)
async def get_reproducibility_metadata():
    """Returns complete system reproducibility metadata (git, python, torch, monai, flower, hardware)."""
    meta = get_system_reproducibility_metadata()
    return SuccessResponse(
        message="System reproducibility metadata retrieved successfully",
        data=meta,
    )


@router.get("/environment", response_model=SuccessResponse)
async def get_environment_info():
    """Returns platform runtime environment variables and software versions."""
    meta = get_system_reproducibility_metadata()
    return SuccessResponse(
        message="System environment metadata retrieved successfully",
        data={
            "python_version": meta.get("python", {}).get("version"),
            "environment": "production",
            "git": meta.get("git", {}),
            "hardware": meta.get("hardware", {}),
            "metadata": meta,
        },
    )


@router.get("/health", response_model=SuccessResponse)
async def get_system_health_endpoint():
    """Returns full distributed system health status."""
    from app.api.v1.endpoints.health import get_system_health
    from app.database.session import SessionLocal

    db = SessionLocal()
    try:
        return await get_system_health(db=db)
    finally:
        db.close()
