"""
Module: dashboard.backend.app.api.v1.endpoints.metrics_prometheus

Purpose:
REST API endpoint exposing Prometheus exposition format metrics at GET /metrics and GET /api/v1/metrics/prometheus.
"""

from fastapi import APIRouter, Response
from utils.telemetry import prometheus_registry

router = APIRouter()


@router.get("", include_in_schema=True)
async def get_prometheus_metrics():
    """Returns system, training, federated, and privacy metrics in Prometheus text exposition format."""
    content = prometheus_registry.generate_prometheus_text()
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
