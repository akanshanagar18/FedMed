"""
Module: dashboard.backend.app.api.v1.endpoints.mlflow

Purpose:
REST API endpoint routes exposing MLflow run metadata, tracking URI status, and experiment runs.
"""

import os
from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, status

from common.schemas import SuccessResponse
from utils.mlflow_tracker import MLflowTracker, MLFLOW_AVAILABLE

router = APIRouter()


@router.get("/status", response_model=SuccessResponse)
async def get_mlflow_status():
    """Returns MLflow tracking integration status and tracking URI."""
    tracker = MLflowTracker()
    return SuccessResponse(
        message="MLflow tracking status retrieved successfully",
        data={
            "enabled": tracker.is_active,
            "tracking_uri": tracker.tracking_uri,
            "experiment_name": tracker.experiment_name,
            "mlflow_version_installed": MLFLOW_AVAILABLE,
        },
    )


@router.get("/runs", response_model=SuccessResponse)
async def list_mlflow_runs():
    """Lists recent MLflow tracking runs from local file store or tracking server."""
    runs = []
    mlruns_dir = os.path.abspath("mlruns")

    if os.path.exists(mlruns_dir):
        for root, dirs, files in os.walk(mlruns_dir):
            if "meta.yaml" in files or "meta.json" in files:
                run_id = os.path.basename(root)
                if len(run_id) >= 20:  # standard run id
                    runs.append({
                        "run_id": run_id,
                        "path": root,
                        "status": "FINISHED",
                    })

    return SuccessResponse(
        message="MLflow runs listed successfully",
        data={"total_runs": len(runs), "runs": runs},
    )
