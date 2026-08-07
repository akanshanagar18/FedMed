"""
Module: data.datasets.validators

Purpose:
Automatic dataset validator for BraTS medical imaging datasets.
Inspects subject directories, verifies modality file existence, reports missing files,
and calculates dataset integrity statistics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field


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
    Automatic validator for BraTS 2021 / 2023 NIfTI datasets.
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

        # Locate subject subdirectories (e.g. BraTS2021_00001)
        subject_dirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not subject_dirs:
            # Check if files directly exist inside path
            report.warnings.append(f"No subject subdirectories found in '{path}'.")
            report.is_valid = False
            return report

        report.total_subjects_found = len(subject_dirs)
        missing_files_map: Dict[str, List[str]] = {}
        modality_counters: Dict[str, int] = {m: 0 for m in target_modalities}
        if require_mask:
            modality_counters[cls.MASK_MODALITY] = 0

        valid_count = 0
        for subj_dir in subject_dirs:
            subj_id = subj_dir.name
            subj_missing: List[str] = []

            # Check each modality file
            for mod in target_modalities:
                matching_files = list(subj_dir.glob(f"*{mod}.nii*"))
                if matching_files:
                    modality_counters[mod] += 1
                else:
                    subj_missing.append(f"{mod}.nii.gz")

            if require_mask:
                mask_files = list(subj_dir.glob(f"*{cls.MASK_MODALITY}.nii*"))
                if mask_files:
                    modality_counters[cls.MASK_MODALITY] += 1
                else:
                    subj_missing.append(f"{cls.MASK_MODALITY}.nii.gz")

            if subj_missing:
                missing_files_map[subj_id] = subj_missing
            else:
                valid_count += 1

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
