"""
Module: dashboard.backend.app.api.v1.endpoints.health

Purpose:
Provides a simple health check endpoint to verify backend uptime.
"""

from fastapi import APIRouter
from app.schemas.responses import SuccessResponse
from app.schemas.node import NodeHealth

router = APIRouter()

@router.get("", response_model=SuccessResponse)
async def health_check():
    """Returns the current health status of the monitoring backend."""
    health_status = NodeHealth(
        status="ok",
        active_connections=0, # TODO: Wire to ConnectionManager
        uptime_seconds=0      # TODO: Implement uptime tracker
    )
    return SuccessResponse(
        message="Backend is operational",
        data=health_status.model_dump()
    )
