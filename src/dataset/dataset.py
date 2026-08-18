"""
FedMed - Brain Tumour Dataset Utilities

Handles discovery of image/label pairs from the
Medical Segmentation Decathlon Task01_BrainTumour dataset.
"""

from pathlib import Path


def get_dataset_files(dataset_root):
    """
    Find all training MRI volumes and their corresponding
    segmentation masks.

    Expected structure:

    Task01_BrainTumour/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    └── dataset.json

    Returns:
        list[dict]: List containing image/label pairs.
    """

    dataset_root = Path(dataset_root)

    images_dir = dataset_root / "imagesTr"
    labels_dir = dataset_root / "labelsTr"

    # --------------------------------------------------------
    # Validate directories
    # --------------------------------------------------------

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Training image directory not found: {images_dir}"
        )

    if not labels_dir.exists():
        raise FileNotFoundError(
            f"Training label directory not found: {labels_dir}"
        )

    # --------------------------------------------------------
    # Find MRI files
    # --------------------------------------------------------

    image_files = sorted(
        images_dir.glob("*.nii.gz")
    )

    if not image_files:
        raise RuntimeError(
            f"No .nii.gz training images found in {images_dir}"
        )

    # --------------------------------------------------------
    # Create image-label pairs
    # --------------------------------------------------------

    dataset_files = []

    for image_path in image_files:

        label_path = labels_dir / image_path.name

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for image: {image_path.name}"
            )

        dataset_files.append(
            {
                "image": str(image_path),
                "label": str(label_path),
            }
        )

    return dataset_files


def validate_dataset(dataset_files):
    """
    Validate that all image/label pairs exist.

    Args:
        dataset_files: List returned by get_dataset_files().

    Returns:
        bool: True if dataset is valid.
    """

    if not dataset_files:
        raise ValueError("Dataset is empty.")

    for sample in dataset_files:

        image_path = Path(sample["image"])
        label_path = Path(sample["label"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label does not exist: {label_path}"
            )

    return True