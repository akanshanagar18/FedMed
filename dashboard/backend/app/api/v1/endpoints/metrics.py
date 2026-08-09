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
    Persists to SQLite database and broadcasts real-time telemetry event over WebSocket.
    """
    saved_row = MetricsService.save_metric(db, metric)

    # Prepare payload for real-time WebSocket broadcast
    payload = {
        "event": "metrics_updated",
        "data": {
            "experiment_id": saved_row.experiment_id,
            "round_number": saved_row.round_number,
            "round": saved_row.round_number,
            "epoch": saved_row.epoch,
            "training_loss": saved_row.training_loss,
            "validation_loss": saved_row.validation_loss,
            "dice_score": saved_row.dice_score,
            "mean_loss": saved_row.training_loss,
            "mean_dice": saved_row.dice_score,
            "iou": saved_row.iou,
            "hospital_id": saved_row.hospital_id,
            "timestamp": saved_row.timestamp.isoformat() if saved_row.timestamp else None,
        },
    }
    await manager.broadcast(payload)

    return SuccessResponse(
        message="Metric saved successfully",
        data={"metric_id": saved_row.id, "round_number": saved_row.round_number},
    )


@router.get("/default", response_model=SuccessResponse)
async def get_default_experiment_metrics(db: Session = Depends(get_db)):
    """
    Retrieves metrics for the default baseline experiment ('default' or 'exp_default').
    """
    metrics = MetricsService.get_metrics_by_experiment(db, "default")
    if not metrics:
        metrics = MetricsService.get_metrics_by_experiment(db, "exp_default")
    if not metrics:
        from app.models.base import TrainingMetricModel
        rows = db.query(TrainingMetricModel).order_by(TrainingMetricModel.round_number.asc()).all()
        if rows:
            return SuccessResponse(
                message="Metrics retrieved successfully",
                data=[
                    {
                        "experiment_id": r.experiment_id,
                        "round_number": r.round_number,
                        "round": r.round_number,
                        "training_loss": r.training_loss,
                        "mean_loss": r.training_loss,
                        "dice_score": r.dice_score,
                        "mean_dice": r.dice_score,
                        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    }
                    for r in rows
                ],
            )
    return SuccessResponse(
        message="Metrics retrieved successfully",
        data=[m.model_dump() for m in metrics],
    )


@router.get("/{experiment_id}", response_model=SuccessResponse)
async def get_experiment_metrics(experiment_id: str, db: Session = Depends(get_db)):
    """
    Retrieves all recorded round metrics for a given experiment_id.
    """
    metrics = MetricsService.get_metrics_by_experiment(db, experiment_id)
    return SuccessResponse(
        message=f"Metrics for experiment '{experiment_id}' retrieved successfully",
        data=[m.model_dump() for m in metrics],
    )
