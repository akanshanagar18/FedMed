"""
Module: dashboard.backend.app.services.metrics

Purpose:
Defines the interface for the MetricsService.
Abstracts the storage and retrieval of training metrics away from the API routers.

TODO:
- [ ] Implement CRUD operations with SQLAlchemy session.
"""

from app.schemas.metrics import TrainingMetric
from typing import List


class MetricsService:
    """Interface for handling Training Metrics."""
    
    @classmethod
    async def save_metric(cls, metric: TrainingMetric) -> None:
        """Saves a new metric to the database."""
        pass
        
    @classmethod
    async def get_metrics_by_experiment(cls, experiment_id: str) -> List[TrainingMetric]:
        """Retrieves all metrics for a given experiment."""
        return []
