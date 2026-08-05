"""
Module: dashboard.backend.app.api.v1.endpoints.metrics

Purpose:
API endpoints for ingesting and retrieving training metrics.
Provides the contract for the Flower server to push round updates.
After saving, broadcasts the metric to all connected WebSocket clients.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.metrics import TrainingMetric
from app.schemas.responses import SuccessResponse
from app.services.metrics import MetricsService
from app.database.session import get_db
from app.websocket.manager import manager
from typing import List

router = APIRouter()


@router.post("", response_model=SuccessResponse)
async def submit_metric(metric: TrainingMetric, db: Session = Depends(get_db)):
    """
    Ingests a new training metric from the Federated Learning server.
    Persists to database and broadcasts to all WebSocket clients.
    """
    MetricsService.save_metric(db, metric)

    # Broadcast to all connected dashboard clients
    await manager.broadcast("metrics_updated", metric.model_dump(mode="json"))

    return SuccessResponse(message="Metric saved successfully", data=metric.model_dump(mode="json"))


@router.get("/{experiment_id}", response_model=SuccessResponse)
async def get_metrics(experiment_id: str, db: Session = Depends(get_db)):
    """
    Retrieves all metrics for a given experiment ID.
    """
    metrics = MetricsService.get_metrics_by_experiment(db, experiment_id)
    return SuccessResponse(
        message="Metrics retrieved successfully",
        data=[m.model_dump(mode="json") for m in metrics],
    )
