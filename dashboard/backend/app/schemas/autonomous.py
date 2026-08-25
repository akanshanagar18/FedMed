"""
Module: dashboard.backend.app.schemas.autonomous

Purpose:
Pydantic Schemas for Autonomous OS endpoints (Orchestrator, Adaptive Strategy, RCA, Recommendations, Experiment Planner, Self-Healing, Deployment Manager, Knowledge Graph, Executive Reports).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class OrchestratorEvaluateRequest(BaseModel):
    current_round: int = Field(default=5, ge=1)
    training_loss: float = Field(default=0.185, ge=0.0)
    validation_dice: float = Field(default=0.862, ge=0.0, le=1.0)
    drift_mmd: float = Field(default=0.042, ge=0.0)
    sla_compliant: bool = Field(default=True)
    active_nodes_count: int = Field(default=2, ge=0)
    total_nodes_count: int = Field(default=2, ge=1)
    privacy_epsilon: float = Field(default=2.5, ge=0.0)
    candidate_dice: Optional[float] = Field(default=0.875)


class AdaptiveStrategyRequest(BaseModel):
    current_strategy: str = Field(default="FedAvg")
    participation_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    avg_latency_ms: float = Field(default=120.0, ge=0.0)
    gradient_divergence: float = Field(default=0.05, ge=0.0)
    drift_mmd: float = Field(default=0.042, ge=0.0)
    privacy_epsilon: float = Field(default=2.5, ge=0.0)
    node_dropouts: int = Field(default=0, ge=0)


class RcaDiagnoseRequest(BaseModel):
    current_loss: float = Field(default=0.25, ge=0.0)
    previous_loss: float = Field(default=0.18, ge=0.0)
    current_dice: float = Field(default=0.79, ge=0.0, le=1.0)
    previous_dice: float = Field(default=0.85, ge=0.0, le=1.0)
    drift_mmd: float = Field(default=0.15, ge=0.0)
    active_clients: int = Field(default=1, ge=0)
    total_clients: int = Field(default=2, ge=1)
    dp_epsilon: float = Field(default=8.5, ge=0.0)
    avg_latency_ms: float = Field(default=2200.0, ge=0.0)


class RecommendationRequest(BaseModel):
    current_dice: float = Field(default=0.81, ge=0.0, le=1.0)
    drift_mmd: float = Field(default=0.12, ge=0.0)
    epsilon_consumed: float = Field(default=6.5, ge=0.0)
    avg_latency_ms: float = Field(default=150.0, ge=0.0)
    candidate_dice: Optional[float] = Field(default=0.865)


class DeploymentInitiateRequest(BaseModel):
    model_id: str = Field(default="brats_monai_3d_unet")
    version: str = Field(default="v2.1.0")
    strategy: str = Field(default="CANARY")
    candidate_metrics: Dict[str, float] = Field(default_factory=lambda: {"mean_dice": 0.875, "hd95": 3.5})
    target_hospitals: Optional[List[str]] = Field(default_factory=lambda: ["hospital_alpha", "hospital_beta"])


class SelfHealingRequest(BaseModel):
    node_id: str = Field(default="hospital_beta")
    failure_reason: str = Field(default="gRPC heartbeat timeout (120s)")
