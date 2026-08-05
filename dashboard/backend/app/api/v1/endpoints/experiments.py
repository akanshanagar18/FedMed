"""
Module: dashboard.backend.app.api.v1.endpoints.experiments

Purpose:
REST API endpoints for creating, querying, updating, and deleting
experiment metadata records in FedMed v2.0.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.database.session import get_db
from app.services.experiment import ExperimentService
from common.schemas import Experiment, SuccessResponse, ErrorResponse

router = APIRouter()


@router.get("", response_model=SuccessResponse)
async def list_experiments(db: Session = Depends(get_db)):
    """Retrieves all registered experiments."""
    experiments = ExperimentService.list_experiments(db)
    return SuccessResponse(
        message="Experiments retrieved successfully",
        data=[e.model_dump(mode="json") for e in experiments],
    )


@router.get("/{experiment_id}", response_model=SuccessResponse)
async def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Retrieves metadata for a specific experiment ID."""
    experiment = ExperimentService.get_experiment(db, experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found",
        )
    return SuccessResponse(
        message="Experiment retrieved successfully",
        data=experiment.model_dump(mode="json"),
    )


@router.post("", response_model=SuccessResponse)
async def create_experiment(exp: Experiment, db: Session = Depends(get_db)):
    """Creates a new experiment metadata entry or updates an existing one."""
    saved_model = ExperimentService.create_or_update_experiment(db, exp)
    saved_schema = ExperimentService.get_experiment(db, saved_model.experiment_id)
    return SuccessResponse(
        message="Experiment registered successfully",
        data=saved_schema.model_dump(mode="json") if saved_schema else exp.model_dump(mode="json"),
    )


@router.delete("/{experiment_id}", response_model=SuccessResponse)
async def delete_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Deletes an experiment record by ID."""
    deleted = ExperimentService.delete_experiment(db, experiment_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found",
        )
    return SuccessResponse(
        message=f"Experiment '{experiment_id}' deleted successfully",
        data={"experiment_id": experiment_id},
    )
