"""
Module: dashboard.backend.app.services.metrics

Purpose:
MetricsService handles persistence and retrieval of training metrics.
"""

from sqlalchemy.orm import Session
from app.models.base import TrainingMetricModel
from app.schemas.metrics import TrainingMetric

from typing import List


class MetricsService:
    """CRUD operations for Training Metrics."""

    @classmethod
    def save_metric(cls, db: Session, metric: TrainingMetric) -> TrainingMetricModel:
        """Persists a new metric to the database."""
        row = TrainingMetricModel(
            experiment_id=metric.experiment_id,
            round_number=metric.round_number,
            epoch=metric.epoch,
            training_loss=metric.training_loss,
            validation_loss=metric.validation_loss,
            dice_score=metric.dice_score,
            iou=metric.iou,
            hospital_id=getattr(metric, "hospital_id", None),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    @classmethod
    def get_metrics_by_experiment(cls, db: Session, experiment_id: str) -> List[TrainingMetric]:
        """Retrieves all metrics for a given experiment, ordered by round."""
        rows = (
            db.query(TrainingMetricModel)
            .filter(TrainingMetricModel.experiment_id == experiment_id)
            .order_by(TrainingMetricModel.round_number)
            .all()
        )
        return [
            TrainingMetric(
                experiment_id=r.experiment_id,
                round_number=r.round_number,
                epoch=r.epoch,
                training_loss=r.training_loss,
                validation_loss=r.validation_loss,
                dice_score=r.dice_score,
                iou=r.iou,
                hospital_id=r.hospital_id,
            )
            for r in rows
        ]
