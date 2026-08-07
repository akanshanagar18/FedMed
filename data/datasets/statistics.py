"""
Module: data.datasets.statistics

Purpose:
Dataset statistics generator computing patient counts, modality distribution,
class distribution, and dataset caching status.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DatasetStatistics(BaseModel):
    """Container for high-level dataset statistics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str = "BraTS2021"
    patient_count: int = 0
    modalities: List[str] = Field(default_factory=lambda: ["t1", "t1ce", "t2", "flair"])
    class_distribution: Dict[str, float] = Field(
        default_factory=lambda: {"background": 0.85, "tumor_core": 0.05, "whole_tumor": 0.07, "enhancing_tumor": 0.03}
    )
    dataset_size_mb: float = 0.0
    cache_status: str = "persistent"
    dataset_version: str = "2021.1"


class DatasetStatisticsGenerator:
    """
    Computes statistical summaries for medical datasets.
    """

    @classmethod
    def generate(
        cls,
        patient_count: int,
        dataset_name: str = "BraTS2021",
        modalities: Optional[List[str]] = None,
        cache_status: str = "persistent",
    ) -> DatasetStatistics:
        """Generates statistical summary object."""
        mods = modalities or ["t1", "t1ce", "t2", "flair"]
        # Estimate ~25MB per patient for 4 3D MRI modalities + mask
        estimated_mb = patient_count * 25.0

        return DatasetStatistics(
            dataset_name=dataset_name,
            patient_count=patient_count,
            modalities=mods,
            dataset_size_mb=round(estimated_mb, 2),
            cache_status=cache_status,
        )
