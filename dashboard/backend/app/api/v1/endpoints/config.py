"""
Module: dashboard.backend.app.api.v1.endpoints.config

Purpose:
REST API endpoint returning active platform configuration payload.
"""

from fastapi import APIRouter
from configs.loader import load_config
from common.schemas import SuccessResponse

router = APIRouter()


@router.get("", response_model=SuccessResponse)
async def get_active_config():
    """Returns active platform configuration summary."""
    cfg = load_config()
    return SuccessResponse(
        message="Active platform configuration retrieved successfully",
        data=cfg.model_dump(mode="json"),
    )
