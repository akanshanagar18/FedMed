"""
FedMed - Segmentation Postprocessing

Converts 3D U-Net output logits into a
discrete brain tumour segmentation mask.
"""

import torch


def logits_to_segmentation(
    logits,
):
    """
    Convert model logits into a class segmentation mask.

    Parameters
    ----------
    logits : torch.Tensor
        Model output.

        Expected shape:
            [B, 4, H, W, D]

    Returns
    -------
    torch.Tensor
        Segmentation mask.

        Shape:
            [B, H, W, D]

        Values:
            0, 1, 2, 3
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(
        logits,
        torch.Tensor,
    ):
        raise TypeError(
            "Logits must be a torch.Tensor."
        )

    if logits.ndim != 5:
        raise ValueError(
            "Expected logits with shape "
            "[B, C, H, W, D]. "
            f"Received: {tuple(logits.shape)}"
        )

    # --------------------------------------------------------
    # Check number of output classes
    # --------------------------------------------------------

    if logits.shape[1] != 4:
        raise ValueError(
            "Expected 4 output classes. "
            f"Received {logits.shape[1]}."
        )

    # --------------------------------------------------------
    # Select the class with the highest score
    # --------------------------------------------------------

    segmentation = torch.argmax(
        logits,
        dim=1,
    )

    return segmentation


def remove_batch_dimension(
    segmentation,
):
    """
    Remove the batch dimension from a
    single-volume segmentation.

    Parameters
    ----------
    segmentation : torch.Tensor
        Shape:
            [1, H, W, D]

    Returns
    -------
    torch.Tensor
        Shape:
            [H, W, D]
    """

    if segmentation.ndim != 4:
        raise ValueError(
            "Expected segmentation shape "
            "[B, H, W, D]. "
            f"Received: {tuple(segmentation.shape)}"
        )

    if segmentation.shape[0] != 1:
        raise ValueError(
            "This function expects a single-volume "
            f"batch. Received batch size {segmentation.shape[0]}."
        )

    return segmentation.squeeze(0)