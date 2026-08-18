"""
FedMed - Medical Model Loader

Loads the trained 3D U-Net checkpoint
for inference.
"""

import torch

from src.model.unet3d import UNet3D

from configs.config import (
    DEVICE,
    BEST_MODEL_PATH,
)


def load_model(
    checkpoint_path=BEST_MODEL_PATH,
    device=DEVICE,
):
    """
    Load the trained 3D U-Net model.

    Parameters
    ----------
    checkpoint_path : str or Path
        Location of the trained checkpoint.

    device : str or torch.device
        Device used for inference.

    Returns
    -------
    model : UNet3D
        Loaded model in evaluation mode.
    """

    print("Loading trained model...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Create model architecture
    # --------------------------------------------------------

    model = UNet3D().to(device)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Set inference mode
    # --------------------------------------------------------

    model.eval()

    print("Model loaded successfully.")

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'N/A')}"
    )

    print(
        f"Validation Dice: "
        f"{checkpoint.get('dice_score', 'N/A')}"
    )

    return model