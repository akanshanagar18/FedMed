"""
Module: dashboard.backend.app.api.v1.endpoints.checkpoints

Purpose:
REST API endpoint routes for querying Checkpoint Registry, listing state checkpoints, and retrieving best/latest model entries.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from common.schemas import SuccessResponse
from utils.checkpoint_registry import get_checkpoint_registry

router = APIRouter()


@router.get("", response_model=SuccessResponse)
async def list_checkpoints(
    experiment_id: Optional[str] = Query(None, description="Filter by experiment ID"),
    benchmark_id: Optional[str] = Query(None, description="Filter by benchmark ID"),
):
    """Lists registered checkpoints matching optional filter parameters."""
    registry = get_checkpoint_registry()
    checkpoints = registry.list_checkpoints(experiment_id=experiment_id, benchmark_id=benchmark_id)
    return SuccessResponse(
        message="Checkpoints retrieved successfully",
        data=[c.to_dict() for c in checkpoints],
    )


@router.get("/latest", response_model=SuccessResponse)
async def get_latest_checkpoint(experiment_id: Optional[str] = None):
    """Retrieves the most recent checkpoint."""
    registry = get_checkpoint_registry()
    chk = registry.get_latest_checkpoint(experiment_id=experiment_id)
    if not chk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No checkpoints found in registry",
        )
    return SuccessResponse(
        message="Latest checkpoint retrieved successfully",
        data=chk.to_dict(),
    )


@router.get("/best", response_model=SuccessResponse)
async def get_best_checkpoint(
    experiment_id: Optional[str] = None,
    metric: str = Query("dice", description="Ranking metric ('dice' or 'loss')"),
):
    """Retrieves the highest-performing model checkpoint by specified research metric."""
    registry = get_checkpoint_registry()
    chk = registry.get_best_checkpoint(experiment_id=experiment_id, metric=metric)
    if not chk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No checkpoints found in registry",
        )
    return SuccessResponse(
        message="Best checkpoint retrieved successfully",
        data=chk.to_dict(),
    )
