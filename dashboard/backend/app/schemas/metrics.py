"""
Module: dashboard.backend.app.schemas.metrics

Purpose:
Defines data contracts for telemetry metrics coming from the ML and FL layers.
Ensures consistency in how metrics like loss and dice score are parsed and stored.

TODO:
- [ ] Add specific metric fields based on final model evaluation requirements.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TrainingMetric(BaseModel):
    """Contract for a training metric payload."""
    experiment_id: str
    round_number: int
    epoch: Optional[int] = None
    
    # ML Metrics
    training_loss: Optional[float] = Field(None, description="Average training loss")
    validation_loss: Optional[float] = Field(None, description="Average validation loss")
    dice_score: Optional[float] = Field(None, description="Dice Similarity Coefficient")
    iou: Optional[float] = Field(None, description="Intersection over Union")
    
    # Optional fields for future expansion
    # TODO: Exact values depend on future ML implementation
