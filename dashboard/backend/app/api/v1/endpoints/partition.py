"""
Module: dashboard.backend.app.api.v1.endpoints.partition

Purpose:
REST API endpoint returning Non-IID federated data partition statistics across
hospital silos (Hospital Alpha, Hospital Beta, Hospital Gamma).
"""

from typing import Optional
from fastapi import APIRouter, Query

from common.schemas import SuccessResponse
from configs.loader import load_config
from data.datasets.brats import BraTSDataset
from data.partitioner import get_partitioner

router = APIRouter()


@router.get("", response_model=SuccessResponse)
async def get_dataset_partitions(
    strategy: Optional[str] = Query("dirichlet", description="Partition strategy: iid or dirichlet"),
    alpha: Optional[float] = Query(0.5, description="Dirichlet alpha hyperparameter"),
):
    """Returns patient counts, class distributions, Dirichlet alpha, and hospital partition metadata."""
    cfg = load_config()

    dataset = BraTSDataset(
        data_dir=cfg.data.data_dir,
        modalities=cfg.data.modalities,
        cache_type=cfg.data.cache_type,
        allow_synthetic_fallback=True,
    )

    strat = strategy or cfg.data.partition_strategy
    dir_alpha = alpha if alpha is not None else cfg.data.dirichlet_alpha
    hospitals = ["hospital_alpha", "hospital_beta", "hospital_gamma"]

    partitioner = get_partitioner(strategy=strat, alpha=dir_alpha, seed=cfg.federated.seed)
    stats = partitioner.partition(dataset.patient_metadata, hospital_ids=hospitals)

    return SuccessResponse(
        message=f"Federated data partitions retrieved successfully ({strat.upper()} alpha={dir_alpha})",
        data=stats.model_dump(mode="json"),
    )
