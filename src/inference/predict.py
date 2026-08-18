"""
FedMed - Prediction Pipeline

Provides a reusable interface for running the trained
3D U-Net on an already-prepared MRI tensor.
"""

import torch


class Predictor:
    """
    Handles model inference.

    This class expects an input tensor that has already
    been preprocessed and has the expected channel and
    spatial dimensions.
    """

    def __init__(
        self,
        model,
        device,
    ):
        """
        Initialize the predictor.

        Parameters
        ----------
        model : torch.nn.Module
            Trained medical segmentation model.

        device : str or torch.device
            Device used for inference.
        """

        self.model = model
        self.device = device

        # Move model to selected device
        self.model.to(
            self.device
        )

        # Put model in evaluation mode
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image,
    ):
        """
        Run inference on a preprocessed MRI tensor.

        Parameters
        ----------
        image : torch.Tensor
            Preprocessed MRI tensor.

            Expected shape:
                [B, 4, H, W, D]

        Returns
        -------
        torch.Tensor
            Raw model output/logits.
        """

        # ----------------------------------------------------
        # Validate input type
        # ----------------------------------------------------

        if not isinstance(
            image,
            torch.Tensor,
        ):
            raise TypeError(
                "Image input must be a torch.Tensor."
            )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        if image.ndim != 5:
            raise ValueError(
                "Expected a 5-dimensional tensor "
                "with shape [B, C, H, W, D]. "
                f"Received shape: {tuple(image.shape)}"
            )

        # ----------------------------------------------------
        # Validate MRI channels
        # ----------------------------------------------------

        if image.shape[1] != 4:
            raise ValueError(
                "Expected 4 MRI input channels. "
                f"Received {image.shape[1]}."
            )

        # ----------------------------------------------------
        # Move input to device
        # ----------------------------------------------------

        image = image.to(
            self.device
        )

        # ----------------------------------------------------
        # Run model
        # ----------------------------------------------------

        prediction = self.model(
            image
        )

        return prediction


def predict(
    model,
    image,
):
    """
    Simple prediction function retained for
    compatibility with existing code.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    image : torch.Tensor
        Preprocessed MRI tensor.

    Returns
    -------
    torch.Tensor
        Model prediction.
    """

    model.eval()

    with torch.no_grad():

        prediction = model(
            image
        )

    return prediction