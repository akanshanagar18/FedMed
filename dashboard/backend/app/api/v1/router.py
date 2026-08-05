"""
Module: dashboard.backend.app.api.v1.router

Purpose:
Aggregates all API v1 endpoints into a single APIRouter.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import health, metrics, telemetry, experiments, benchmarks, config

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["System Health"])
api_router.include_router(config.router, prefix="/config", tags=["Configuration Engine"])
api_router.include_router(metrics.router, prefix="/metrics", tags=["Training Metrics"])
api_router.include_router(experiments.router, prefix="/experiments", tags=["Experiment Management"])
api_router.include_router(benchmarks.router, prefix="/benchmarks", tags=["Benchmark Framework"])
api_router.include_router(telemetry.router, prefix="/telemetry", tags=["Real-time Telemetry"])
