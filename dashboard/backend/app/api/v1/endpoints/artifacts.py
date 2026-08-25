"""
Module: dashboard.backend.app.api.v1.endpoints.artifacts

Purpose:
REST API endpoint routes for discovering and downloading research artifacts (PDF reports, CSV leaderboards, convergence plots, YAML configs).
"""

import os
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from common.schemas import SuccessResponse

router = APIRouter()


@router.get("", response_model=SuccessResponse)
async def list_artifacts():
    """Discovers all generated research artifacts across `results/` and `checkpoints/` directories."""
    artifacts = []
    scan_dirs = ["results", "checkpoints", "runs"]

    for d in scan_dirs:
        abs_d = os.path.abspath(d)
        if os.path.exists(abs_d):
            for root, dirs, files in os.walk(abs_d):
                for f in files:
                    if f.endswith((".pdf", ".csv", ".json", ".png", ".md", ".pth", ".yaml")):
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, os.getcwd())
                        artifacts.append({
                            "name": f,
                            "path": rel_path,
                            "size_bytes": os.path.getsize(full_path),
                            "extension": f.split(".")[-1],
                            "directory": d,
                        })

    return SuccessResponse(
        message="Artifacts listed successfully",
        data={"total_artifacts": len(artifacts), "artifacts": artifacts},
    )


@router.get("/download", response_class=FileResponse)
async def download_artifact(path: str):
    """Downloads a specific artifact file given its relative or absolute path."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact file '{path}' not found",
        )
    return FileResponse(abs_path, filename=os.path.basename(abs_path))
