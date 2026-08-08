"""
Module: dashboard.backend.app.models.base

Purpose:
SQLAlchemy Declarative Base and ORM table models for the FedMed monitoring database.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class TrainingMetricModel(Base):
    """Persisted training metric row."""
    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, nullable=False, index=True)
    round_number = Column(Integer, nullable=False)
    epoch = Column(Integer, nullable=True)
    training_loss = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    dice_score = Column(Float, nullable=True)
    iou = Column(Float, nullable=True)
    hospital_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class HospitalNodeModel(Base):
    """Persisted hospital node registration."""
    __tablename__ = "hospital_nodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    hospital_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    connection_status = Column(String, default="disconnected")
    client_latency_ms = Column(Integer, nullable=True)
    last_seen = Column(DateTime, default=datetime.utcnow)


class ExperimentModel(Base):
    """Persisted experiment metadata for FedMed v2.0."""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    status = Column(String, default="created")
    
    # Federated & Hyperparameter Attributes
    strategy_name = Column(String, default="FedAvg")
    num_clients = Column(Integer, default=2)
    learning_rate = Column(Float, default=1e-4)
    batch_size = Column(Integer, default=2)
    local_epochs = Column(Integer, default=1)
    num_rounds = Column(Integer, default=3)
    seed = Column(Integer, default=42)
    
    # Flags & Dataset
    dp_enabled = Column(Boolean, default=False)
    he_enabled = Column(Boolean, default=False)
    dataset_name = Column(String, default="BraTS2021")
    partition_strategy = Column(String, default="IID")
    notes = Column(Text, nullable=True)
    
    # Checkpoints
    checkpoint_path = Column(String, nullable=True)
    best_dice_score = Column(Float, nullable=True)
    best_round = Column(Integer, nullable=True)
    
    # Timestamps
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)


class BenchmarkModel(Base):
    """Persisted metadata for multi-experiment benchmark suites."""
    __tablename__ = "benchmarks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    status = Column(String, default="created")
    total_experiments = Column(Integer, default=0)
    completed_experiments = Column(Integer, default=0)
    best_experiment_id = Column(String, nullable=True)
    best_dice_score = Column(Float, nullable=True)
    avg_dice_score = Column(Float, nullable=True)
    matrix_config_json = Column(Text, default="{}")
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)


class BenchmarkExperimentModel(Base):
    """Mapping table linking experiments to a benchmark suite."""
    __tablename__ = "benchmark_experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    benchmark_id = Column(String, nullable=False, index=True)
    experiment_id = Column(String, nullable=False, index=True)
    sequence_order = Column(Integer, default=0)
    status = Column(String, default="pending")


class GovernanceRecordModel(Base):
    """Persisted governance events and model stage transition logs."""
    __tablename__ = "governance_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, nullable=False, index=True)
    previous_stage = Column(String, default="Candidate")
    new_stage = Column(String, default="Staging")
    promoted_by = Column(String, default="automated_governance")
    hmac_signature = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class DriftMetricModel(Base):
    """Persisted feature and concept drift measurements per node."""
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String, nullable=False, index=True)
    mmd_score = Column(Float, nullable=False)
    ks_statistic = Column(Float, nullable=False)
    ks_p_value = Column(Float, nullable=False)
    wasserstein_distance = Column(Float, nullable=False)
    psi_score = Column(Float, nullable=False)
    risk_level = Column(String, default="LOW")
    timestamp = Column(DateTime, default=datetime.utcnow)


class HPOTrialModel(Base):
    """Persisted Federated Hyperparameter Optimization trial results."""
    __tablename__ = "hpo_trials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(String, nullable=False, index=True)
    trial_id = Column(String, nullable=False)
    strategy = Column(String, default="fedavg")
    learning_rate = Column(Float, nullable=False)
    proximal_mu = Column(Float, nullable=False)
    ewc_lambda = Column(Float, nullable=False)
    dp_noise_multiplier = Column(Float, nullable=False)
    val_dice = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class SLACertificateModel(Base):
    """Persisted HIPAA/GDPR institutional SLA compliance certificate attestations."""
    __tablename__ = "sla_certificates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    certificate_hash = Column(String, unique=True, nullable=False)
    status = Column(String, default="PASSED_COMPLIANT")
    overall_compliant = Column(Boolean, default=True)
    epsilon_consumed = Column(Float, nullable=False)
    participation_rate = Column(Float, nullable=False)
    avg_latency_ms = Column(Float, nullable=False)
    encryption_scheme = Column(String, default="TenSEAL_CKKS")
    timestamp = Column(DateTime, default=datetime.utcnow)

