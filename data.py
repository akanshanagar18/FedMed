"""Dataset loading and deterministic hospital partitioning utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


IMAGE_SIZE = 128
HOSPITALS = ("A", "B", "C")


def _transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _partition(dataset: Dataset, hospital_id: str) -> Subset:
    """Split a local development dataset into three non-overlapping partitions.

    In production, point each hospital at its own local data root instead. This
    partitioning is only for one-machine demonstrations and tests.
    """
    hospital_id = hospital_id.upper()
    if hospital_id not in HOSPITALS:
        raise ValueError(f"hospital_id must be one of {HOSPITALS}, got {hospital_id!r}")
    shard = HOSPITALS.index(hospital_id)
    return Subset(dataset, list(range(shard, len(dataset), len(HOSPITALS))))


def load_data(
    data_root: str,
    hospital_id: str,
    batch_size: int,
    synthetic: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Return private train/test loaders for a single Flower client.

    Expected BRISC layout: <data_root>/classification_task/train/<class>/*.jpg
    and the matching test directory. A single hospital may use a data root that
    contains only the samples it is permitted to access.
    """
    transform = _transform()
    if synthetic:
        train_set = datasets.FakeData(
            size=90, image_size=(3, IMAGE_SIZE, IMAGE_SIZE), num_classes=4,
            transform=transform,
        )
        test_set = datasets.FakeData(
            size=30, image_size=(3, IMAGE_SIZE, IMAGE_SIZE), num_classes=4,
            transform=transform,
        )
    else:
        root = Path(data_root)
        train_path = root / "classification_task" / "train"
        test_path = root / "classification_task" / "test"
        if not train_path.is_dir() or not test_path.is_dir():
            raise FileNotFoundError(
                "BRISC classification directories were not found. Expected "
                f"{train_path} and {test_path}. Extract the archive, set "
                "--data-root, or use --synthetic for a smoke test."
            )
        train_set = datasets.ImageFolder(train_path, transform=transform)
        test_set = datasets.ImageFolder(test_path, transform=transform)
        if len(train_set.classes) != 4:
            raise ValueError(f"Expected 4 class folders, found {train_set.classes}")

    return (
        DataLoader(_partition(train_set, hospital_id), batch_size=batch_size, shuffle=True),
        DataLoader(_partition(test_set, hospital_id), batch_size=batch_size, shuffle=False),
    )
