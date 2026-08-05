"""
DataLoader Interface
"""

from torch.utils.data import DataLoader


def create_dataloader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=2
):
    """
    Creates a PyTorch DataLoader.
    """

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )