"""
FedMed - 3D U-Net Training Pipeline

Trains the medical image segmentation model using
the Task01_BrainTumour dataset.
"""

import json
import torch

from monai.metrics import DiceMetric

from src.model.unet3d import UNet3D
from src.model.losses import BrainTumourLoss
from src.training.dataloader import create_dataloaders

from configs.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
    METRICS_FILE,
    create_output_directories,
)
QUICK_TEST = False
MAX_TRAIN_BATCHES = 3
MAX_VALIDATION_BATCHES = 2

def train_one_epoch(
    model,
    train_loader,
    loss_function,
    optimizer,
    device,
):
    """
    Train the model for one epoch.

    When QUICK_TEST is enabled, only a small number of
    batches are processed. This is useful for testing the
    complete training pipeline on CPU before starting
    a long training run.
    """

    model.train()

    running_loss = 0.0

    total_batches = len(train_loader)

    # --------------------------------------------------------
    # Determine how many batches should actually be processed
    # --------------------------------------------------------

    if QUICK_TEST:
        batches_to_process = min(
            total_batches,
            MAX_TRAIN_BATCHES,
        )
    else:
        batches_to_process = total_batches

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for batch_index, batch in enumerate(train_loader):

        # Stop after the requested number of batches
        if batch_index >= batches_to_process:
            break

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Clear gradients from previous iteration
        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        predictions = model(images)

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = loss_function(
            predictions,
            labels,
        )

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update model parameters
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Track loss
        # ----------------------------------------------------

        running_loss += loss.item()

        print(
            f"    Batch "
            f"{batch_index + 1}/{batches_to_process} "
            f"- Loss: {loss.item():.4f}"
        )

    # --------------------------------------------------------
    # Calculate average loss
    # --------------------------------------------------------

    average_loss = (
        running_loss / batches_to_process
    )

    return average_loss

def validate(
    model,
    validation_loader,
    loss_function,
    device,
):
    """
    Evaluate the model on the validation dataset.

    When QUICK_TEST is enabled, only a small number of
    validation batches are processed.
    """

    model.eval()

    validation_loss = 0.0

    # --------------------------------------------------------
    # Dice metric
    # --------------------------------------------------------

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
    )

    total_batches = len(validation_loader)

    # --------------------------------------------------------
    # Determine number of validation batches
    # --------------------------------------------------------

    if QUICK_TEST:
        batches_to_process = min(
            total_batches,
            MAX_VALIDATION_BATCHES,
        )
    else:
        batches_to_process = total_batches

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_index, batch in enumerate(
            validation_loader
        ):

            if batch_index >= batches_to_process:
                break

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            predictions = model(images)

            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            loss = loss_function(
                predictions,
                labels,
            )

            validation_loss += loss.item()

            # ------------------------------------------------
            # Convert predictions to class labels
            # ------------------------------------------------

            predicted_classes = torch.argmax(
                predictions,
                dim=1,
                keepdim=True,
            )

            # ------------------------------------------------
            # Calculate Dice score
            # ------------------------------------------------

            dice_metric(
                y_pred=predicted_classes,
                y=labels,
            )

            print(
                f"    Validation batch "
                f"{batch_index + 1}/{batches_to_process}"
            )

    # --------------------------------------------------------
    # Average validation loss
    # --------------------------------------------------------

    average_loss = (
        validation_loss / batches_to_process
    )

    # --------------------------------------------------------
    # Calculate Dice score
    # --------------------------------------------------------

    dice_score = dice_metric.aggregate().item()

    dice_metric.reset()

    return average_loss, dice_score

def main():
    """
    Main training function.
    """

    print("=" * 70)
    print("FedMed - Brain Tumour Segmentation Training")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    create_output_directories()

    # --------------------------------------------------------
    # Display device
    # --------------------------------------------------------

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    print("\nLoading dataset...")

    train_loader, validation_loader = (
        create_dataloaders()
    )

    print("\nDataset loaded successfully.")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating 3D U-Net...")

    model = UNet3D().to(DEVICE)

    # --------------------------------------------------------
    # Create loss
    # --------------------------------------------------------

    loss_function = BrainTumourLoss()

    # --------------------------------------------------------
    # Create optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_dice = -1.0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, NUM_EPOCHS + 1):

        print("\n" + "=" * 70)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        print("\nTraining...")

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=DEVICE,
        )

        print(
            f"\nAverage Training Loss: "
            f"{train_loss:.4f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print("\nValidation...")

        validation_loss, dice_score = validate(
            model=model,
            validation_loader=validation_loader,
            loss_function=loss_function,
            device=DEVICE,
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Dice: "
            f"{dice_score:.4f}"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "dice_score": dice_score,
        }

        history.append(epoch_result)

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if dice_score > best_dice:

            best_dice = dice_score

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "dice_score": dice_score,
                },
                BEST_MODEL_PATH,
            )

            print(
                f"\n✓ Best model saved:"
                f"\n  {BEST_MODEL_PATH}"
            )

        # ----------------------------------------------------
        # Save latest model
        # ----------------------------------------------------

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "dice_score": dice_score,
            },
            LAST_MODEL_PATH,
        )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    with open(
        METRICS_FILE,
        "w",
    ) as file:

        json.dump(
            history,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)

    print(
        f"Best Dice Score: "
        f"{best_dice:.4f}"
    )

    print(
        f"Best Model: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Last Model: "
        f"{LAST_MODEL_PATH}"
    )

    print(
        f"Metrics: "
        f"{METRICS_FILE}"
    )


if __name__ == "__main__":
    main()