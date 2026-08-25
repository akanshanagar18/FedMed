"""
Unit tests for data.partitioner Non-IID Federated Data Partitioning Engine.
"""

import pytest
from data.datasets.metadata import PatientMetadata
from data.partitioner import DirichletPartitioner, IIDPartitioner, get_partitioner


@pytest.fixture
def sample_patients():
    """Generates synthetic patient metadata objects for testing partitioners."""
    patients = []
    for i in range(1, 13):
        p_id = f"BraTS2021_{i:05d}"
        patients.append(
            PatientMetadata(
                patient_id=p_id,
                modality_paths={
                    "t1": f"/path/{p_id}_t1.nii.gz",
                    "t1ce": f"/path/{p_id}_t1ce.nii.gz",
                    "t2": f"/path/{p_id}_t2.nii.gz",
                    "flair": f"/path/{p_id}_flair.nii.gz",
                },
                mask_path=f"/path/{p_id}_seg.nii.gz",
            )
        )
    return patients


def test_iid_partitioner_disjoint_allocation(sample_patients):
    """Verify IIDPartitioner splits patients equally with zero overlap."""
    hospitals = ["hospital_alpha", "hospital_beta", "hospital_gamma"]
    partitioner = IIDPartitioner(seed=42)

    stats = partitioner.partition(sample_patients, hospitals)

    assert stats.partition_strategy == "IID"
    assert stats.num_hospitals == 3
    assert stats.disjoint_verified is True

    # Check each hospital received 4 patients (12 / 3)
    for h_id in hospitals:
        part = stats.partitions[h_id]
        assert part.total_patients == 4
        assert len(part.patient_ids) == 4

    # Verify zero overlap
    all_assigned = []
    for part in stats.partitions.values():
        all_assigned.extend(part.patient_ids)
    assert len(all_assigned) == len(set(all_assigned))


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.5, 1.0])
def test_dirichlet_partitioner_alphas(sample_patients, alpha):
    """Verify DirichletPartitioner with various alpha hyperparameters."""
    hospitals = ["hospital_alpha", "hospital_beta", "hospital_gamma"]
    partitioner = DirichletPartitioner(alpha=alpha, seed=42)

    stats = partitioner.partition(sample_patients, hospitals)

    assert stats.partition_strategy == "Dirichlet"
    assert stats.dirichlet_alpha == alpha
    assert stats.disjoint_verified is True
    assert stats.total_patients == 12

    # Total assigned patients across hospitals equals total patients
    total_assigned_count = sum(p.total_patients for p in stats.partitions.values())
    assert total_assigned_count == 12


def test_partitioner_seed_determinism(sample_patients):
    """Verify partitioners yield identical results given identical seed."""
    hospitals = ["hospital_alpha", "hospital_beta", "hospital_gamma"]

    part1 = DirichletPartitioner(alpha=0.5, seed=123).partition(sample_patients, hospitals)
    part2 = DirichletPartitioner(alpha=0.5, seed=123).partition(sample_patients, hospitals)

    for h_id in hospitals:
        assert part1.partitions[h_id].patient_ids == part2.partitions[h_id].patient_ids


def test_get_partitioner_factory():
    """Verify get_partitioner returns correct instance type."""
    p_iid = get_partitioner("iid")
    assert isinstance(p_iid, IIDPartitioner)

    p_dir = get_partitioner("dirichlet", alpha=0.5)
    assert isinstance(p_dir, DirichletPartitioner)
