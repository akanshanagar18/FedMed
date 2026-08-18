"""
Module: data.datasets.metadata

Purpose:
Patient metadata indexer and manifest generator for medical MRI datasets.
Supports both BraTS 2021 and BraTS-GLI 2024 naming schemes via canonical adapter.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from data.canonical_adapter import ModalityCanonicalizer, DatasetVersion


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

        subject_dirs = []
        for d in sorted(list(path.iterdir())):
            if d.is_dir() and not d.name.startswith("."):
                if d.name.startswith("training_data"):
                    for sub in sorted(list(d.iterdir())):
                        if sub.is_dir() and not sub.name.startswith("."):
                            subject_dirs.append(sub)
                else:
                    subject_dirs.append(d)

        index_list: List[PatientMetadata] = []

        for subj_dir in subject_dirs:
            patient_id = subj_dir.name
            version, mod_paths, missing = ModalityCanonicalizer.discover_subject_modalities(subj_dir)

            mod_map = {m: str(mod_paths[m]) for m in target_modalities if m in mod_paths}
            mask_path = str(mod_paths["seg"]) if "seg" in mod_paths else None

            if len(mod_map) == len(target_modalities) and mask_path:
                index_list.append(PatientMetadata(
                    patient_id=patient_id,
                    dataset_version=version.value,
                    modality_paths=mod_map,
                    mask_path=mask_path,
                ))

        return index_list
