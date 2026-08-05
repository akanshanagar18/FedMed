"""
Module: dashboard.backend.app.services.benchmark

Purpose:
BenchmarkService manages multi-experiment benchmark suites, matrix run generation,
leaderboard aggregation, and results reporting.
"""

import json
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.base import BenchmarkModel, BenchmarkExperimentModel, ExperimentModel, TrainingMetricModel
from common.schemas import Benchmark, BenchmarkStatus, BenchmarkMatrixConfig, LeaderboardEntry, Experiment


class BenchmarkService:
    """CRUD and analysis operations for Benchmark runs."""

    @classmethod
    def create_benchmark(cls, db: Session, bench: Benchmark) -> BenchmarkModel:
        """Create a new benchmark suite record."""
        row = (
            db.query(BenchmarkModel)
            .filter(BenchmarkModel.benchmark_id == bench.benchmark_id)
            .first()
        )
        if not row:
            row = BenchmarkModel(
                benchmark_id=bench.benchmark_id,
                name=bench.name,
                description=bench.description,
                status=bench.status.value if isinstance(bench.status, BenchmarkStatus) else str(bench.status),
                total_experiments=bench.total_experiments,
                matrix_config_json=json.dumps(bench.matrix_config.model_dump(mode="json")),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
        return row

    @classmethod
    def get_benchmark(cls, db: Session, benchmark_id: str) -> Optional[Benchmark]:
        """Retrieve benchmark metadata by ID."""
        row = (
            db.query(BenchmarkModel)
            .filter(BenchmarkModel.benchmark_id == benchmark_id)
            .first()
        )
        if not row:
            return None
        return cls._row_to_schema(row)

    @classmethod
    def list_benchmarks(cls, db: Session) -> List[Benchmark]:
        """List all benchmarks ordered by start time."""
        rows = db.query(BenchmarkModel).order_by(BenchmarkModel.start_time.desc()).all()
        return [cls._row_to_schema(r) for r in rows]

    @classmethod
    def update_benchmark_progress(
        cls,
        db: Session,
        benchmark_id: str,
        completed_count: int,
        best_exp_id: Optional[str] = None,
        best_dice: Optional[float] = None,
        avg_dice: Optional[float] = None,
        status: Optional[BenchmarkStatus] = None,
    ) -> Optional[BenchmarkModel]:
        """Update benchmark execution progress and leader statistics."""
        row = (
            db.query(BenchmarkModel)
            .filter(BenchmarkModel.benchmark_id == benchmark_id)
            .first()
        )
        if not row:
            return None

        row.completed_experiments = completed_count
        if best_exp_id:
            row.best_experiment_id = best_exp_id
        if best_dice is not None:
            row.best_dice_score = best_dice
        if avg_dice is not None:
            row.avg_dice_score = avg_dice
        if status:
            row.status = status.value if isinstance(status, BenchmarkStatus) else str(status)
            if status in (BenchmarkStatus.COMPLETED, BenchmarkStatus.FAILED):
                row.end_time = datetime.utcnow()

        db.commit()
        db.refresh(row)
        return row

    @classmethod
    def generate_leaderboard(cls, db: Session, benchmark_id: str) -> List[LeaderboardEntry]:
        """
        Computes leaderboard entries for a benchmark by ranking associated experiments
        by best Dice Similarity Coefficient.
        """
        mappings = (
            db.query(BenchmarkExperimentModel)
            .filter(BenchmarkExperimentModel.benchmark_id == benchmark_id)
            .order_by(BenchmarkExperimentModel.sequence_order)
            .all()
        )
        if not mappings:
            # Fallback: Find all experiments matching prefix
            exp_rows = (
                db.query(ExperimentModel)
                .filter(ExperimentModel.experiment_id.like(f"{benchmark_id}_%"))
                .all()
            )
        else:
            exp_ids = [m.experiment_id for m in mappings]
            exp_rows = (
                db.query(ExperimentModel)
                .filter(ExperimentModel.experiment_id.in_(exp_ids))
                .all()
            )

        leaderboard = []
        for exp in exp_rows:
            # Get metrics for this experiment
            metrics = (
                db.query(TrainingMetricModel)
                .filter(TrainingMetricModel.experiment_id == exp.experiment_id)
                .all()
            )
            best_dice = exp.best_dice_score or max(([m.dice_score for m in metrics if m.dice_score] or [0.0]))
            avg_loss = (sum([m.training_loss for m in metrics if m.training_loss]) / max(len(metrics), 1)) if metrics else 0.0
            conv_round = exp.best_round or (len(metrics) if metrics else 0)
            runtime = (exp.end_time - exp.start_time).total_seconds() if (exp.end_time and exp.start_time) else 0.0

            leaderboard.append(
                LeaderboardEntry(
                    rank=0,
                    experiment_id=exp.experiment_id,
                    strategy_name=exp.strategy_name or "FedAvg",
                    partition_strategy=exp.partition_strategy or "IID",
                    seed=exp.seed or 42,
                    best_dice=round(best_dice, 4),
                    avg_loss=round(avg_loss, 4),
                    convergence_round=conv_round,
                    runtime_sec=round(runtime, 2),
                    status=exp.status,
                )
            )

        # Sort leaderboard by best_dice descending
        leaderboard.sort(key=lambda x: x.best_dice, reverse=True)
        for idx, entry in enumerate(leaderboard, start=1):
            entry.rank = idx

        return leaderboard

    @classmethod
    def delete_benchmark(cls, db: Session, benchmark_id: str) -> bool:
        """Delete benchmark record and mapping rows."""
        row = (
            db.query(BenchmarkModel)
            .filter(BenchmarkModel.benchmark_id == benchmark_id)
            .first()
        )
        if not row:
            return False
        db.query(BenchmarkExperimentModel).filter(
            BenchmarkExperimentModel.benchmark_id == benchmark_id
        ).delete()
        db.delete(row)
        db.commit()
        return True

    @staticmethod
    def _row_to_schema(r: BenchmarkModel) -> Benchmark:
        matrix_cfg = BenchmarkMatrixConfig()
        if r.matrix_config_json:
            try:
                matrix_cfg = BenchmarkMatrixConfig(**json.loads(r.matrix_config_json))
            except Exception:
                pass

        return Benchmark(
            benchmark_id=r.benchmark_id,
            name=r.name,
            description=r.description or "",
            status=BenchmarkStatus(r.status) if r.status in BenchmarkStatus._value2member_map_ else BenchmarkStatus.RUNNING,
            total_experiments=r.total_experiments or 0,
            completed_experiments=r.completed_experiments or 0,
            best_experiment_id=r.best_experiment_id,
            best_dice_score=r.best_dice_score,
            avg_dice_score=r.avg_dice_score,
            matrix_config=matrix_cfg,
            start_time=r.start_time or datetime.utcnow(),
            end_time=r.end_time,
        )
