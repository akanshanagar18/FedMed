"""
Module: dashboard.backend.app.services.experiment

Purpose:
ExperimentService manages database persistence, metadata queries, and lifecycle updates
for research-grade federated learning experiments.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.base import ExperimentModel
from common.schemas import Experiment, ExperimentStatus


class ExperimentService:
    """CRUD operations for Experiment metadata."""

    @classmethod
    def create_or_update_experiment(cls, db: Session, exp: Experiment) -> ExperimentModel:
        """Create a new experiment record or update an existing one."""
        row = (
            db.query(ExperimentModel)
            .filter(ExperimentModel.experiment_id == exp.experiment_id)
            .first()
        )
        if not row:
            row = ExperimentModel(experiment_id=exp.experiment_id, name=exp.name)
            db.add(row)

        row.name = exp.name
        row.description = exp.description
        row.status = exp.status.value if isinstance(exp.status, ExperimentStatus) else str(exp.status)
        row.strategy_name = exp.strategy_name
        row.num_clients = exp.num_clients
        row.learning_rate = exp.learning_rate
        row.batch_size = exp.batch_size
        row.local_epochs = exp.local_epochs
        row.num_rounds = exp.num_rounds
        row.seed = exp.seed
        row.dp_enabled = exp.dp_enabled
        row.he_enabled = exp.he_enabled
        row.dataset_name = exp.dataset_name
        row.partition_strategy = exp.partition_strategy
        row.notes = exp.notes
        row.checkpoint_path = exp.checkpoint_path
        row.best_dice_score = exp.best_dice_score
        row.best_round = exp.best_round
        if exp.end_time:
            row.end_time = exp.end_time

        db.commit()
        db.refresh(row)
        return row

    @classmethod
    def get_experiment(cls, db: Session, experiment_id: str) -> Optional[Experiment]:
        """Retrieves a single experiment's metadata by ID."""
        row = (
            db.query(ExperimentModel)
            .filter(ExperimentModel.experiment_id == experiment_id)
            .first()
        )
        if not row:
            return None
        return cls._row_to_schema(row)

    @classmethod
    def list_experiments(cls, db: Session) -> List[Experiment]:
        """Retrieves all registered experiments ordered by start time."""
        rows = db.query(ExperimentModel).order_by(ExperimentModel.start_time.desc()).all()
        return [cls._row_to_schema(r) for r in rows]

    @classmethod
    def update_status(
        cls,
        db: Session,
        experiment_id: str,
        status: ExperimentStatus,
        best_dice: Optional[float] = None,
        best_round: Optional[int] = None,
    ) -> Optional[ExperimentModel]:
        """Updates the status and optional metrics of an ongoing experiment."""
        row = (
            db.query(ExperimentModel)
            .filter(ExperimentModel.experiment_id == experiment_id)
            .first()
        )
        if not row:
            return None

        row.status = status.value if isinstance(status, ExperimentStatus) else str(status)
        if status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.STOPPED):
            row.end_time = datetime.utcnow()
        if best_dice is not None:
            row.best_dice_score = best_dice
        if best_round is not None:
            row.best_round = best_round

        db.commit()
        db.refresh(row)
        return row

    @classmethod
    def delete_experiment(cls, db: Session, experiment_id: str) -> bool:
        """Deletes an experiment record by ID."""
        row = (
            db.query(ExperimentModel)
            .filter(ExperimentModel.experiment_id == experiment_id)
            .first()
        )
        if not row:
            return False
        db.delete(row)
        db.commit()
        return True

    @staticmethod
    def _row_to_schema(r: ExperimentModel) -> Experiment:
        return Experiment(
            experiment_id=r.experiment_id,
            name=r.name,
            description=r.description or "",
            status=ExperimentStatus(r.status) if r.status in ExperimentStatus._value2member_map_ else ExperimentStatus.RUNNING,
            strategy_name=r.strategy_name or "FedAvg",
            num_clients=r.num_clients or 2,
            learning_rate=r.learning_rate or 1e-4,
            batch_size=r.batch_size or 2,
            local_epochs=r.local_epochs or 1,
            num_rounds=r.num_rounds or 3,
            seed=r.seed or 42,
            dp_enabled=bool(r.dp_enabled),
            he_enabled=bool(r.he_enabled),
            dataset_name=r.dataset_name or "BraTS2021",
            partition_strategy=r.partition_strategy or "IID",
            notes=r.notes,
            checkpoint_path=r.checkpoint_path,
            best_dice_score=r.best_dice_score,
            best_round=r.best_round,
            start_time=r.start_time or datetime.utcnow(),
            end_time=r.end_time,
        )
