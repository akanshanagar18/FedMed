"""
Module: common.schemas

Purpose:
Canonical Pydantic schemas shared across the entire FedMed platform.
Every module MUST import from here — never define duplicate schemas locally.

This package sits at the absolute bottom of the dependency graph.
It MUST NOT import from any other FedMed module.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConnectionStatus(str, Enum):
    """Hospital client connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    IDLE = "idle"
    TRAINING = "training"
    ENCRYPTING = "encrypting"
    UPLOADING = "uploading"
    WAITING = "waiting"


class RoundStatus(str, Enum):
    """Training round lifecycle states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentStatus(str, Enum):
    """Experiment lifecycle states for FedMed v2.0."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class BenchmarkStatus(str, Enum):
    """Benchmark suite lifecycle states."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Core Schemas
# ---------------------------------------------------------------------------

class TrainingMetric(BaseModel):
    """Payload for a single training round metric update."""
    experiment_id: str
    round_number: int
    epoch: Optional[int] = None

    # ML Metrics
    training_loss: Optional[float] = Field(None, description="Average training loss")
    validation_loss: Optional[float] = Field(None, description="Average validation loss")
    dice_score: Optional[float] = Field(None, description="Dice Similarity Coefficient")
    iou: Optional[float] = Field(None, description="Intersection over Union")

    # Metadata
    hospital_id: Optional[str] = Field(None, description="Source hospital for this metric")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HospitalStatus(BaseModel):
    """Status of a connected hospital node."""
    hospital_id: str
    name: str
    connection_status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    client_latency_ms: Optional[int] = None
    last_seen: datetime = Field(default_factory=datetime.utcnow)


class TrainingRound(BaseModel):
    """State of a single federated training round."""
    round_number: int
    status: RoundStatus = RoundStatus.PENDING
    participating_hospitals: List[str] = Field(default_factory=list)
    aggregation_time_seconds: Optional[float] = None
    model_version: Optional[str] = None


class NodeHealth(BaseModel):
    """System health snapshot for the monitoring backend."""
    status: str = "ok"
    active_connections: int = 0
    uptime_seconds: int = 0


class Experiment(BaseModel):
    """Metadata contract for a research-grade federated learning experiment."""
    experiment_id: str
    name: str
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.CREATED
    
    # Federated & ML Hyperparameters
    strategy_name: str = "FedAvg"
    num_clients: int = 2
    learning_rate: float = 1e-4
    batch_size: int = 2
    local_epochs: int = 1
    num_rounds: int = 3
    seed: int = 42
    
    # Privacy & Dataset Flags
    dp_enabled: bool = False
    he_enabled: bool = False
    dataset_name: str = "BraTS2021"
    partition_strategy: str = "IID"
    notes: Optional[str] = None
    
    # Checkpoint Metadata
    checkpoint_path: Optional[str] = None
    best_dice_score: Optional[float] = None
    best_round: Optional[int] = None
    
    # Timestamps
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None


class BenchmarkMatrixConfig(BaseModel):
    """Matrix parameters for generating benchmark experiment sweeps."""
    strategies: List[str] = Field(default_factory=lambda: ["FedAvg", "FedProx"])
    partition_strategies: List[str] = Field(
        default_factory=lambda: ["IID", "NonIID(alpha=0.5)", "NonIID(alpha=0.2)"]
    )
    seeds: List[int] = Field(default_factory=lambda: [42, 123, 999])
    num_rounds: int = 3
    num_clients: int = 2


class Benchmark(BaseModel):
    """Metadata contract for a multi-experiment benchmark suite."""
    benchmark_id: str
    name: str
    description: str = ""
    status: BenchmarkStatus = BenchmarkStatus.CREATED
    total_experiments: int = 0
    completed_experiments: int = 0
    best_experiment_id: Optional[str] = None
    best_dice_score: Optional[float] = None
    avg_dice_score: Optional[float] = None
    matrix_config: BenchmarkMatrixConfig = Field(default_factory=BenchmarkMatrixConfig)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None


class LeaderboardEntry(BaseModel):
    """Single row entry in a benchmark leaderboard table."""
    rank: int
    experiment_id: str
    strategy_name: str
    partition_strategy: str
    seed: int
    best_dice: float
    avg_loss: float
    convergence_round: int
    runtime_sec: float
    status: str


# ---------------------------------------------------------------------------
# Response Wrappers
# ---------------------------------------------------------------------------

class SuccessResponse(BaseModel):
    """Standard API success envelope."""
    message: str
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Standard API error envelope."""
    status_code: int = 500
    error_code: str = "UNKNOWN_ERROR"
    message: str = "An unexpected error occurred."
    developer_details: Optional[str] = None
    recovery_suggestions: Optional[str] = None


# ---------------------------------------------------------------------------
# Milestone R: Governance, Drift, FedHPO, & SLA Schemas
# ---------------------------------------------------------------------------

class GovernanceRecordSchema(BaseModel):
    model_id: str
    previous_stage: str = "Candidate"
    new_stage: str = "Staging"
    promoted_by: str = "automated_governance"
    hmac_signature: Optional[str] = None


class DriftMetricSchema(BaseModel):
    node_id: str
    mmd_score: float
    ks_statistic: float
    ks_p_value: float
    wasserstein_distance: float
    psi_score: float
    risk_level: str = "LOW"


class HPOTrialSchema(BaseModel):
    study_id: str
    trial_id: str
    strategy: str
    learning_rate: float
    proximal_mu: float
    ewc_lambda: float
    dp_noise_multiplier: float
    val_dice: float


class SLACertificateSchema(BaseModel):
    run_id: str
    certificate_hash: str
    status: str = "PASSED_COMPLIANT"
    overall_compliant: bool = True
    epsilon_consumed: float
    participation_rate: float
    avg_latency_ms: float
    encryption_scheme: str = "TenSEAL_CKKS"

