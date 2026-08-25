"""
Module: dashboard.backend.app.api.v1.endpoints.tensorboard

Purpose:
REST API endpoint routes for querying TensorBoard logging status and discovered run directories.
"""

import os
from typing import List, Dict, Any
from fastapi import APIRouter

from common.schemas import SuccessResponse

router = APIRouter()


@router.get("/status", response_model=SuccessResponse)
async def get_tensorboard_status():
    """Returns TensorBoard log directory status and launch command instructions."""
    runs_dir = os.path.abspath("runs")
    has_runs = os.path.exists(runs_dir) and len(os.listdir(runs_dir)) > 0
    return SuccessResponse(
        message="TensorBoard status retrieved successfully",
        data={
            "logdir": runs_dir,
            "has_active_runs": has_runs,
            "launch_command": "tensorboard --logdir runs/",
        },
    )


@router.get("/runs", response_model=SuccessResponse)
async def list_tensorboard_runs():
    """Lists all experiment run directories inside `runs/`."""
    runs_dir = os.path.abspath("runs")
    runs = []
    if os.path.exists(runs_dir):
        for entry in os.listdir(runs_dir):
            full_p = os.path.join(runs_dir, entry)
            if os.path.isdir(full_p):
                events = [f for f in os.listdir(full_p) if "events.out.tfevents" in f]
                runs.append({
                    "experiment_id": entry,
                    "path": full_p,
                    "event_files_count": len(events),
                })

    return SuccessResponse(
        message="TensorBoard runs listed successfully",
        data={"total_runs": len(runs), "runs": runs},
    )
