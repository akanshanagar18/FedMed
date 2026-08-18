"""
FedMed - Sliding Window Inference

Runs the trained 3D U-Net over a full MRI volume
using overlapping 3D windows.
"""

import torch

from monai.inferers import sliding_window_inference


def run_sliding_window_inference(
    model,
    image,
    roi_size=(96, 96, 96),
    sw_batch_size=1,
    overlap=0.25,
    device="cpu",
):
    """
    Run sliding-window inference on a full MRI volume.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation model.

    image : torch.Tensor
        Preprocessed MRI tensor.

        Expected shape:
            [B, 4, H, W, D]

    roi_size : tuple
        Size of each 3D inference window.

    sw_batch_size : int
        Number of windows processed at once.

    overlap : float
        Fraction of overlap between neighboring windows.

    device : str or torch.device
        Device used for inference.

    Returns
    -------
    torch.Tensor
        Full-volume model prediction.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(
        image,
        torch.Tensor,
    ):
        raise TypeError(
            "Image must be a torch.Tensor."
        )

    if image.ndim != 5:
        raise ValueError(
            "Expected image shape "
            "[B, C, H, W, D]. "
            f"Received: {tuple(image.shape)}"
        )

    if image.shape[1] != 4:
        raise ValueError(
            "Expected 4 MRI channels. "
            f"Received {image.shape[1]}."
        )

    # --------------------------------------------------------
    # Move image to inference device
    # --------------------------------------------------------

    image = image.to(device)

    # --------------------------------------------------------
    # Put model in evaluation mode
    # --------------------------------------------------------

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Sliding-window inference
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )

    return prediction