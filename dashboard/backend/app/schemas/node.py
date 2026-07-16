"""
Module: dashboard.backend.app.schemas.node

Purpose:
Contracts for node health, hospital status, and training round orchestration.
Used by the dashboard to show which hospitals are active and training progress.

TODO:
- [ ] Refine statuses based on Flower client states.
"""

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class HospitalStatus(BaseModel):
    hospital_id: str
    name: str
    connection_status: str = "disconnected"  # "connected", "disconnected", "training"
    client_latency_ms: Optional[int] = None
    last_seen: datetime


class NodeHealth(BaseModel):
    status: str
    active_connections: int
    uptime_seconds: int


class TrainingRound(BaseModel):
    round_number: int
    status: str  # "pending", "in_progress", "completed", "failed"
    participating_hospitals: List[str]
    aggregation_time_seconds: Optional[float] = None
    model_version: Optional[str] = None
