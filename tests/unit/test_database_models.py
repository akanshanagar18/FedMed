"""
Unit tests for SQLAlchemy Database ORM Models
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.base import Base, TrainingMetricModel, HospitalNodeModel, ExperimentModel


@pytest.fixture
def in_memory_db():
    """Fixture providing an isolated in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.mark.unit
def test_training_metric_model_crud(in_memory_db):
    """Verify TrainingMetricModel insertion and retrieval."""
    metric = TrainingMetricModel(
        experiment_id="exp_db_test",
        round_number=1,
        training_loss=0.35,
        dice_score=0.88,
        hospital_id="hospital_a",
    )
    in_memory_db.add(metric)
    in_memory_db.commit()

    saved = in_memory_db.query(TrainingMetricModel).filter_by(experiment_id="exp_db_test").first()
    assert saved is not None
    assert saved.round_number == 1
    assert saved.training_loss == 0.35
    assert saved.dice_score == 0.88
    assert saved.hospital_id == "hospital_a"


@pytest.mark.unit
def test_hospital_node_model_crud(in_memory_db):
    """Verify HospitalNodeModel registration and status tracking."""
    node = HospitalNodeModel(
        hospital_id="hospital_alpha",
        name="General Hospital Alpha",
        connection_status="connected",
        client_latency_ms=15,
    )
    in_memory_db.add(node)
    in_memory_db.commit()

    saved = in_memory_db.query(HospitalNodeModel).filter_by(hospital_id="hospital_alpha").first()
    assert saved is not None
    assert saved.name == "General Hospital Alpha"
    assert saved.connection_status == "connected"


@pytest.mark.unit
def test_experiment_model_crud(in_memory_db):
    """Verify ExperimentModel lifecycle metadata persistence."""
    exp = ExperimentModel(
        experiment_id="exp_001",
        name="FedAvg 3D UNet Benchmark",
        description="Standard 3 round benchmark",
        status="running",
    )
    in_memory_db.add(exp)
    in_memory_db.commit()

    saved = in_memory_db.query(ExperimentModel).filter_by(experiment_id="exp_001").first()
    assert saved is not None
    assert saved.name == "FedAvg 3D UNet Benchmark"
    assert saved.status == "running"
