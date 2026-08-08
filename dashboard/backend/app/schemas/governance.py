"""
Module: dashboard.backend.app.schemas.governance

Purpose:
Pydantic Schemas for Milestone R Governance, Drift, FedHPO, and SLA Compliance API requests & responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DriftEvaluationRequest(BaseModel):
    node_id: str = Field(default="hospital_alpha", description="Target hospital silo node ID")
    significance_level: float = Field(default=0.05, ge=0.001, le=0.2)
    psi_threshold: float = Field(default=0.2, ge=0.05, le=0.5)


class DriftMetricsData(BaseModel):
    mmd: float
    ks_statistic: float
    ks_p_value: float
    wasserstein_distance: float
    psi: float


class DriftEvaluationResponse(BaseModel):
    node_id: str
    drift_detected: bool
    risk_level: str
    recommended_action: str
    metrics: DriftMetricsData


class SlaAuditRequest(BaseModel):
    run_id: str = Field(default="run_milestone_r_001")
    epsilon_consumed: float = Field(default=2.5, ge=0.0)
    delta_consumed: float = Field(default=1e-5, ge=0.0)
    participating_nodes: List[str] = Field(default_factory=lambda: ["hospital_alpha", "hospital_beta"])
    total_nodes: List[str] = Field(default_factory=lambda: ["hospital_alpha", "hospital_beta", "hospital_gamma"])
    avg_latency_ms: float = Field(default=120.5, ge=0.0)
    encryption_scheme: str = Field(default="TenSEAL_CKKS")


class SlaAuditResponse(BaseModel):
    run_id: str
    status: str
    overall_compliant: bool
    audit_timestamp: int
    certificate_hash: str
    regulatory_frameworks: List[str]
    sla_metrics: Dict[str, Any]
    compliance_checks: Dict[str, bool]
    violations: List[str]


class HpoStudyRequest(BaseModel):
    study_id: str = Field(default="hpo_study_001")
    num_initial_configs: int = Field(default=6, ge=2, le=20)
    max_rounds: int = Field(default=12, ge=3, le=50)


class HpoTrialResult(BaseModel):
    best_config: Dict[str, Any]
    best_dice: float
    total_trials_evaluated: int


class ModelPromotionRequest(BaseModel):
    model_id: str = Field(default="brats_monai_3d_unet")
    current_stage: str = Field(default="Staging")
    target_stage: str = Field(default="Production")
    signature: Optional[str] = None
    candidate_metrics: Dict[str, float] = Field(default_factory=lambda: {"mean_dice": 0.865, "hd95": 3.8})
    production_baseline_metrics: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"mean_dice": 0.825, "hd95": 4.5}
    )


class ModelPromotionResponse(BaseModel):
    success: bool
    model_id: Optional[str] = None
    previous_stage: Optional[str] = None
    new_stage: Optional[str] = None
    signature: Optional[str] = None
    message: str
    canary_evaluation: Optional[Dict[str, Any]] = None
