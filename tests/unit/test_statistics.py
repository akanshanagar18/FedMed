"""
Unit tests for DatasetValidator, PatientMetadataIndexer, and DatasetStatisticsGenerator.
"""

from data.datasets.brats import generate_synthetic_brats_nifti
from data.datasets.metadata import PatientMetadataIndexer
from data.datasets.statistics import DatasetStatisticsGenerator
from data.datasets.validators import DatasetValidator


def test_dataset_validator_and_metadata_indexer(tmp_path):
    """Verify validator scans directory and indexes patient metadata."""
    generate_synthetic_brats_nifti(tmp_path, num_subjects=3, spatial_shape=(16, 16, 16))

    report = DatasetValidator.validate_dataset_directory(tmp_path)
    assert report.is_valid is True
    assert report.total_subjects_found == 3
    assert report.valid_subjects_count == 3
    assert report.corrupted_subjects_count == 0

    metadata = PatientMetadataIndexer.index_directory(tmp_path)
    assert len(metadata) == 3
    assert metadata[0].patient_id.startswith("BraTS2021_")
    assert len(metadata[0].modality_paths) == 4
    assert metadata[0].mask_path is not None


def test_dataset_statistics_generator():
    """Verify statistical metric generator."""
    stats = DatasetStatisticsGenerator.generate(patient_count=10, dataset_name="BraTS2021")
    assert stats.patient_count == 10
    assert stats.dataset_name == "BraTS2021"
    assert stats.dataset_size_mb == 250.0
    assert "t1" in stats.modalities
    assert "flair" in stats.modalities
