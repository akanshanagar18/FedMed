"""
Module: dashboard.backend.app.api.v1.endpoints.health

Purpose:
Provides a health check endpoint to verify backend uptime and live connection count.
"""

import time
from fastapi import APIRouter
from app.schemas.responses import SuccessResponse
from app.schemas.node import NodeHealth
from app.websocket.manager import manager

router = APIRouter()

_start_time = time.time()


@router.get("", response_model=SuccessResponse)
async def health_check():
    """Returns the current health status of the monitoring backend."""
    health_status = NodeHealth(
        status="ok",
        active_connections=len(manager.active_connections),
        uptime_seconds=int(time.time() - _start_time),
    )
    return SuccessResponse(
        message="Backend is operational",
        data=health_status.model_dump(),
    )
