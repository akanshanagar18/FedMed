"""
FedMed - Training and Validation DataLoaders

Creates training and validation DataLoaders for the
MSD Task01 Brain Tumour dataset.
"""

from sklearn.model_selection import train_test_split

from monai.data import Dataset, DataLoader

from configs.config import (
    DATASET_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    PATCH_SIZE,
    PIN_MEMORY,
    VALIDATION_RATIO,
    SEED,
)

from src.dataset.dataset import get_dataset_files

from src.dataset.transforms import (
    get_train_transforms,
    get_validation_transforms,
)


def create_dataloaders():
    """
    Create training and validation DataLoaders.

    Returns:
        train_loader: PyTorch/MONAI training DataLoader
        validation_loader: PyTorch/MONAI validation DataLoader
    """

    # --------------------------------------------------------
    # 1. Find all image/label pairs
    # --------------------------------------------------------

    all_files = get_dataset_files(
        DATASET_PATH
    )

    # --------------------------------------------------------
    # 2. Split into training and validation sets
    # --------------------------------------------------------

    train_files, validation_files = train_test_split(
        all_files,
        test_size=VALIDATION_RATIO,
        random_state=SEED,
        shuffle=True,
    )

    print(
        f"Total samples      : {len(all_files)}"
    )

    print(
        f"Training samples   : {len(train_files)}"
    )

    print(
        f"Validation samples : {len(validation_files)}"
    )

    # --------------------------------------------------------
    # 3. Create MONAI datasets
    # --------------------------------------------------------

    train_dataset = Dataset(
        data=train_files,
        transform=get_train_transforms(
            PATCH_SIZE
        ),
    )

    validation_dataset = Dataset(
        data=validation_files,
        transform=get_validation_transforms(
            PATCH_SIZE
        ),
    )

    # --------------------------------------------------------
    # 4. Create training DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # --------------------------------------------------------
    # 5. Create validation DataLoader
    # --------------------------------------------------------

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, validation_loader


# ------------------------------------------------------------
# Backward-compatible function
# ------------------------------------------------------------

def create_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=0,
):
    """
    Backward-compatible generic DataLoader creator.

    This preserves the older function used elsewhere in
    the project while create_dataloaders() is used for
    the complete medical dataset pipeline.
    """

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )