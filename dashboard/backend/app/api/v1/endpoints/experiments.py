"""
Module: dashboard.backend.app.api.v1.endpoints.experiments

Purpose:
REST API endpoints for complete experiment lifecycle management in FedMed OS v2.1.
Supports create, start, pause, resume, cancel, archive, retry, and details querying.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import time
from typing import Any, Dict, List, Optional

from app.database.session import get_db
from app.schemas.responses import SuccessResponse
from server.experiment_manager import global_experiment_manager

router = APIRouter()


@router.get("", response_model=SuccessResponse)
async def list_experiments(db: Session = Depends(get_db)):
    """Retrieves all registered experiments."""
    experiments = global_experiment_manager.list_experiments()
    return SuccessResponse(
        message="Experiments retrieved successfully",
        data=experiments,
    )


@router.get("/{experiment_id}", response_model=SuccessResponse)
async def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Retrieves metadata and status for a single experiment."""
    exp = global_experiment_manager.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found",
        )
    return SuccessResponse(
        message="Experiment retrieved successfully",
        data=exp,
    )


@router.post("", response_model=SuccessResponse)
async def create_experiment(payload: Dict[str, Any], db: Session = Depends(get_db)):
    """Creates a new experiment and attaches an orchestration workflow DAG."""
    name = payload.get("name", f"Experiment_{int(time.time())}" if 'time' in globals() else "New Experiment")
    strategy_name = payload.get("strategy_name", "FedAvg")
    num_clients = int(payload.get("num_clients", 2))
    num_rounds = int(payload.get("num_rounds", 3))
    learning_rate = float(payload.get("learning_rate", 1e-4))
    dp_enabled = bool(payload.get("dp_enabled", False))
    he_enabled = bool(payload.get("he_enabled", False))
    description = payload.get("description", "")
    exp_id = payload.get("experiment_id")
    exp_status = payload.get("status")

    res = global_experiment_manager.create_experiment(
        name=name,
        strategy_name=strategy_name,
        num_clients=num_clients,
        num_rounds=num_rounds,
        learning_rate=learning_rate,
        dp_enabled=dp_enabled,
        he_enabled=he_enabled,
        description=description,
        experiment_id=exp_id,
        status=exp_status,
    )
    return SuccessResponse(
        message="Experiment created successfully",
        data=res,
    )


@router.post("/{experiment_id}/start", response_model=SuccessResponse)
async def start_experiment(experiment_id: str):
    """Starts experiment execution by advancing workflow engine DAG."""
    res = global_experiment_manager.start_experiment(experiment_id)
    if not res.get("success", True):
        raise HTTPException(status_code=400, detail=res.get("message", "Failed to start experiment"))
    return SuccessResponse(message=f"Experiment '{experiment_id}' started successfully", data=res)


@router.post("/{experiment_id}/pause", response_model=SuccessResponse)
async def pause_experiment(experiment_id: str):
    """Pauses a running experiment."""
    res = global_experiment_manager.pause_experiment(experiment_id)
    if not res.get("success", True):
        raise HTTPException(status_code=400, detail=res.get("message", "Failed to pause experiment"))
    return SuccessResponse(message=f"Experiment '{experiment_id}' paused successfully", data=res)


@router.post("/{experiment_id}/resume", response_model=SuccessResponse)
async def resume_experiment(experiment_id: str):
    """Resumes a paused experiment."""
    res = global_experiment_manager.resume_experiment(experiment_id)
    if not res.get("success", True):
        raise HTTPException(status_code=400, detail=res.get("message", "Failed to resume experiment"))
    return SuccessResponse(message=f"Experiment '{experiment_id}' resumed successfully", data=res)


@router.post("/{experiment_id}/cancel", response_model=SuccessResponse)
async def cancel_experiment(experiment_id: str):
    """Cancels an active experiment."""
    res = global_experiment_manager.cancel_experiment(experiment_id)
    if not res.get("success", True):
        raise HTTPException(status_code=400, detail=res.get("message", "Failed to cancel experiment"))
    return SuccessResponse(message=f"Experiment '{experiment_id}' cancelled successfully", data=res)


@router.post("/{experiment_id}/archive", response_model=SuccessResponse)
async def archive_experiment(experiment_id: str):
    """Archives an experiment record."""
    res = global_experiment_manager.archive_experiment(experiment_id)
    if not res.get("success", True):
        raise HTTPException(status_code=400, detail=res.get("message", "Failed to archive experiment"))
    return SuccessResponse(message=f"Experiment '{experiment_id}' archived successfully", data=res)


@router.delete("/{experiment_id}", response_model=SuccessResponse)
async def delete_experiment(experiment_id: str):
    """Deletes an experiment record by ID."""
    success = global_experiment_manager.delete_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return SuccessResponse(message=f"Experiment '{experiment_id}' deleted successfully", data={"experiment_id": experiment_id})
