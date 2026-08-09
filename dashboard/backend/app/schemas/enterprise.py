"""
Module: dashboard.backend.app.schemas.enterprise

Purpose:
Pydantic Schemas for Milestone T REST APIs (Workflows, Simulator, Policies, Scheduler, Digital Twin, Experiment Lifecycle).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkflowCreateRequest(BaseModel):
    workflow_name: str = Field(default="Enterprise_EndToEnd_FL_Pipeline")


class ScenarioTriggerRequest(BaseModel):
    scenario_type: str = Field(default="HOSPITAL_FAILURE")
    target_node: str = Field(default="hospital_beta")
    custom_params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class JobScheduleRequest(BaseModel):
    job_type: str = Field(default="DRIFT_SCAN")
    schedule_type: str = Field(default="PERIODIC")
    interval_seconds: int = Field(default=300, ge=10)
    cron_expression: Optional[str] = Field(default="*/5 * * * *")


class DigitalTwinSimRequest(BaseModel):
    scenario_description: str = Field(default="Hospital Alpha disconnects and latency doubles")
    baseline_dice: float = Field(default=0.865, ge=0.0, le=1.0)
    hospital_dropout_pct: float = Field(default=0.33, ge=0.0, le=1.0)
    latency_multiplier: float = Field(default=2.0, ge=0.1)
    drift_mmd: float = Field(default=0.12, ge=0.0)
    privacy_noise_multiplier: float = Field(default=1.5, ge=0.1)


class ExperimentCreateRequest(BaseModel):
    name: str = Field(default="BraTS 3D U-Net Federated Benchmark")
    owner: str = Field(default="neuro_radiology_team")
    dataset_version: str = Field(default="BraTS2021_v1")


class ExperimentTransitionRequest(BaseModel):
    target_stage: str = Field(default="TRAINING")
    notes: Optional[str] = Field(default="Passed dataset validation checks")
    metrics: Optional[Dict[str, float]] = Field(default_factory=dict)
