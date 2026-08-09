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

from app.api.v1.endpoints import metrics_prometheus, distributed, personalization, continual, foundation, explainability, security, fairness, calibration, clinical, audit
api_router.include_router(metrics_prometheus.router, prefix="/metrics-prometheus", tags=["Prometheus Metrics"])
api_router.include_router(distributed.router, prefix="/distributed-systems", tags=["Distributed Federated Systems"])
api_router.include_router(personalization.router, prefix="/personalization", tags=["Personalized Federated Learning"])
api_router.include_router(continual.router, prefix="/continual", tags=["Continual Federated Learning"])
api_router.include_router(foundation.router, prefix="/foundation", tags=["Vision Foundation Models & PEFT"])
api_router.include_router(explainability.router, prefix="/explainability", tags=["Explainability & Grad-CAM"])
api_router.include_router(security.router, prefix="/security", tags=["Byzantine Security & Attacks"])
api_router.include_router(fairness.router, prefix="/fairness", tags=["Federated Demographic Fairness"])
api_router.include_router(calibration.router, prefix="/calibration", tags=["Uncertainty Calibration & ECE"])
api_router.include_router(clinical.router, prefix="/clinical", tags=["Clinical Validation Suite"])
api_router.include_router(audit.router, prefix="/audit", tags=["Audit Trail & Dataset Lineage"])

from app.api.v1.endpoints import dataset_version, governance, autonomous_platform
api_router.include_router(dataset_version.router, prefix="/dataset-version", tags=["Dataset Versioning & Lineage"])
api_router.include_router(governance.router, prefix="/governance", tags=["Enterprise Governance, Drift & SLA"])
api_router.include_router(autonomous_platform.router, prefix="/autonomous", tags=["Autonomous Federated OS Engine"])






