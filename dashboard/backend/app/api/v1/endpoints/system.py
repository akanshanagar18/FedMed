"""
Module: dashboard.backend.app.api.v1.endpoints.system

Purpose:
REST API endpoint routes exposing reproducibility metadata, system architecture info, and hardware context.
"""

from fastapi import APIRouter
from common.schemas import SuccessResponse
from utils.reproducibility import collect_reproducibility_metadata, get_hardware_metadata, get_dependency_versions

router = APIRouter()


@router.get("/reproducibility", response_model=SuccessResponse)
async def get_system_reproducibility():
    """Returns full system reproducibility metadata snapshot."""
    meta = collect_reproducibility_metadata()
    return SuccessResponse(
        message="System reproducibility metadata retrieved successfully",
        data=meta,
    )


@router.get("/environment", response_model=SuccessResponse)
async def get_system_environment():
    """Returns hardware context and library dependency versions."""
    hw = get_hardware_metadata()
    deps = get_dependency_versions()
    return SuccessResponse(
        message="Environment details retrieved successfully",
        data={"hardware": hw, "dependencies": deps},
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
