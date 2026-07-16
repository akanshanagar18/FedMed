"""
Module: dashboard.backend.app.services.experiment

Purpose:
Defines the interface for the ExperimentService.

TODO:
- [ ] Implement actual logic.
"""

from app.schemas.experiment import Experiment
from typing import Optional


class ExperimentService:
    """Interface for managing Experiments."""
    
    @classmethod
    async def get_experiment(cls, experiment_id: str) -> Optional[Experiment]:
        """Retrieves an experiment's metadata."""
        return None
