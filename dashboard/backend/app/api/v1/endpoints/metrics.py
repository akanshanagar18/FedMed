"""
Module: dashboard.backend.app.api.v1.endpoints.metrics

Purpose:
API endpoints for ingesting and retrieving training metrics.
Provides the contract for the Flower server to push round updates.
"""

from fastapi import APIRouter, Depends, HTTPException
from app.schemas.metrics import TrainingMetric
from app.schemas.responses import SuccessResponse
from app.services.metrics import MetricsService
from typing import List

router = APIRouter()

@router.post("", response_model=SuccessResponse)
async def submit_metric(metric: TrainingMetric):
    """
    Ingests a new training metric from the Federated Learning server.
    """
    await MetricsService.save_metric(metric)
    return SuccessResponse(message="Metric saved successfully", data=metric.model_dump())

@router.get("/{experiment_id}", response_model=SuccessResponse)
async def get_metrics(experiment_id: str):
    """
    Retrieves all metrics for a given experiment ID.
    """
    metrics = await MetricsService.get_metrics_by_experiment(experiment_id)
    return SuccessResponse(
        message="Metrics retrieved successfully", 
        data=[m.model_dump() for m in metrics]
    )
