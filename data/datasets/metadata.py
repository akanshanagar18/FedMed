"""
Module: data.datasets.metadata

Purpose:
Patient metadata indexer and manifest generator for medical MRI datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class PatientMetadata(BaseModel):
    """Metadata specification for an individual patient / subject."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    patient_id: str
    dataset_version: str = "BraTS2021"
    modality_paths: Dict[str, str] = Field(default_factory=dict)
    mask_path: Optional[str] = None
    spatial_shape: Optional[List[int]] = None
    voxel_spacing: Optional[List[float]] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class PatientMetadataIndexer:
    """
    Scans dataset path, indexes patient metadata, and builds structural data manifests.
    """

    @classmethod
    def index_directory(
        cls,
        data_dir: Union[str, Path],
        modalities: Optional[List[str]] = None,
    ) -> List[PatientMetadata]:
        """
        Scans data_dir and returns structured PatientMetadata list.
        """
        path = Path(data_dir)
        target_modalities = modalities or ["t1", "t1ce", "t2", "flair"]

        if not path.exists() or not path.is_dir():
            return []

        subject_dirs = sorted([d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")])
        index_list: List[PatientMetadata] = []

        for subj_dir in subject_dirs:
            patient_id = subj_dir.name
            mod_map: Dict[str, str] = {}
            for mod in target_modalities:
                matches = list(subj_dir.glob(f"*{mod}.nii*"))
                if matches:
                    mod_map[mod] = str(matches[0])

            mask_path = None
            mask_matches = list(subj_dir.glob("*seg.nii*"))
            if mask_matches:
                mask_path = str(mask_matches[0])

            if mod_map:
                meta = PatientMetadata(
                    patient_id=patient_id,
                    modality_paths=mod_map,
                    mask_path=mask_path,
                )
                index_list.append(meta)

        return index_list
