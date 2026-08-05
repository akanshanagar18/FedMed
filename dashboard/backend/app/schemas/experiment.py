"""
Module: dashboard.backend.app.schemas.experiment

Purpose:
Contract for Experiment metadata. Tracks hyperparameter setups and run configurations.

TODO:
- [ ] Align hyperparameter fields with actual PyTorch/Flower configs.
"""

from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime


class Experiment(BaseModel):
    experiment_id: str
    name: str
    description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str  # "running", "completed", "aborted"
    hyperparameters: Dict[str, Any]
    encryption_status: str = "active"
