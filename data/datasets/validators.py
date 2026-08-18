"""
Module: data.datasets.validators

Purpose:
Automatic dataset validator for BraTS medical imaging datasets.
Inspects subject directories, verifies modality file existence, reports missing files,
and calculates dataset integrity statistics using the canonical adapter.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

from data.canonical_adapter import ModalityCanonicalizer, DatasetVersion


class ValidationReport(BaseModel):
    """Pydantic report containing dataset inspection and integrity metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_dir: str
    dataset_version: str = "BraTS2021"
    is_valid: bool = True
    total_subjects_found: int = 0
    valid_subjects_count: int = 0
    corrupted_subjects_count: int = 0
    missing_files: Dict[str, List[str]] = Field(default_factory=dict)
    modality_counts: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class DatasetValidator:
    """
    Automatic validator for BraTS 2021 / 2024 NIfTI datasets.
    """

    REQUIRED_MODALITIES = ["t1", "t1ce", "t2", "flair"]
    MASK_MODALITY = "seg"

    @classmethod
    def validate_dataset_directory(
        cls,
        data_dir: Union[str, Path],
        modalities: Optional[List[str]] = None,
        require_mask: bool = True,
    ) -> ValidationReport:
        """
        Scans data_dir for subject directories and validates required NIfTI modality files.
        """
        path = Path(data_dir)
        target_modalities = modalities or cls.REQUIRED_MODALITIES
        report = ValidationReport(dataset_dir=str(path))

        if not path.exists() or not path.is_dir():
            report.is_valid = False
            report.warnings.append(f"Dataset directory '{path}' does not exist or is not a directory.")
            return report

        # Locate subject subdirectories (supports direct or nested training_data1_v2/ folders)
        subject_dirs = []
        for d in sorted(list(path.iterdir())):
            if d.is_dir() and not d.name.startswith("."):
                if d.name.startswith("training_data"):
                    for sub in sorted(list(d.iterdir())):
                        if sub.is_dir() and not sub.name.startswith("."):
                            subject_dirs.append(sub)
                else:
                    subject_dirs.append(d)

        if not subject_dirs:
            report.warnings.append(f"No subject subdirectories found in '{path}'.")
            report.is_valid = False
            return report

        report.total_subjects_found = len(subject_dirs)
        missing_files_map: Dict[str, List[str]] = {}
        modality_counters: Dict[str, int] = {m: 0 for m in target_modalities}
        if require_mask:
            modality_counters[cls.MASK_MODALITY] = 0

        valid_count = 0
        detected_version = "BraTS2021"

        for subj_dir in subject_dirs:
            subj_id = subj_dir.name
            version, mod_paths, missing = ModalityCanonicalizer.discover_subject_modalities(subj_dir)
            detected_version = version.value

            for mod in target_modalities:
                if mod in mod_paths:
                    modality_counters[mod] += 1

            if require_mask and "seg" in mod_paths:
                modality_counters[cls.MASK_MODALITY] += 1

            if missing:
                missing_files_map[subj_id] = missing
            else:
                valid_count += 1

        report.dataset_version = detected_version
        report.valid_subjects_count = valid_count
        report.corrupted_subjects_count = report.total_subjects_found - valid_count
        report.missing_files = missing_files_map
        report.modality_counts = modality_counters
        report.is_valid = (valid_count > 0) and (report.corrupted_subjects_count == 0)

        if report.corrupted_subjects_count > 0:
            report.warnings.append(
                f"Found {report.corrupted_subjects_count} subjects with missing modality files."
            )

        return report
