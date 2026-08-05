"""
Module: dashboard.backend.app.models.base

Purpose:
SQLAlchemy Declarative Base and ORM table models for the FedMed monitoring database.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Text
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
    """Persisted experiment metadata."""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="running")
    encryption_status = Column(String, default="active")
