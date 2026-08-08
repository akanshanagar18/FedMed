"""
Module: dashboard.backend.app.api.v1.router

Purpose:
Aggregates all API v1 endpoints into a single APIRouter.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import health, metrics, telemetry, experiments, benchmarks, config, strategies, dataset, partition, nodes, mlflow, checkpoints, artifacts, tensorboard, system

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["System Health"])
api_router.include_router(config.router, prefix="/config", tags=["Configuration Engine"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["Training Metrics"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["Experiment Management"])
api_router.include_router(benchmarks.router, prefix="/benchmarks", tags=["Benchmark Framework"])
api_router.include_router(telemetry.router, prefix="/telemetry", tags=["Real-time Telemetry"])
api_router.include_router(strategies.router, prefix="/strategies", tags=["Strategy Engine"])
api_router.include_router(dataset.router, prefix="/dataset", tags=["Research Data Engine"])
api_router.include_router(partition.router, prefix="/dataset/partitions", tags=["Non-IID Partition Engine"])
api_router.include_router(nodes.router, prefix="/nodes", tags=["Node Resilience & Fault Tolerance"])
api_router.include_router(mlflow.router, prefix="/mlflow", tags=["MLflow Tracking Platform"])
api_router.include_router(checkpoints.router, prefix="/checkpoints", tags=["Checkpoint Registry Engine"])
api_router.include_router(artifacts.router, prefix="/artifacts", tags=["Research Artifact Manager"])
api_router.include_router(tensorboard.router, prefix="/tensorboard", tags=["TensorBoard Integration"])
api_router.include_router(system.router, prefix="/system", tags=["System Reproducibility Platform"])




