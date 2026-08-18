"""
FedMed - End-to-End Medical Image Prediction

Complete inference pipeline:

MRI
    ↓
Input validation
    ↓
Preprocessing
    ↓
Model loading
    ↓
Sliding-window inference
    ↓
Segmentation postprocessing
    ↓
NIfTI output
"""

from pathlib import Path

import torch

from configs.config import DEVICE

from src.inference.input_validator import (
    validate_mri_file,
)

from src.inference.preprocess import (
    get_inference_transform,
)

from src.inference.model_loader import (
    load_model,
)

from src.inference.sliding_window import (
    run_sliding_window_inference,
)

from src.inference.postprocess import (
    logits_to_segmentation,
    remove_batch_dimension,
)

from src.inference.output import (
    get_prediction_path,
    save_segmentation_nifti,
)


class Predictor:
    """
    End-to-end MRI prediction pipeline.

    The predictor validates, preprocesses, runs inference,
    postprocesses the model output, and saves the final
    segmentation as a NIfTI file.
    """

    def __init__(
        self,
        model=None,
        device=DEVICE,
    ):
        """
        Initialize the predictor.

        Parameters
        ----------
        model : torch.nn.Module, optional
            Trained model.

            If None, the trained model is loaded
            automatically.

        device : str or torch.device
            Inference device.
        """

        self.device = device

        # ----------------------------------------------------
        # Load model if one was not provided
        # ----------------------------------------------------

        if model is None:
            self.model = load_model(
                device=self.device
            )
        else:
            self.model = model.to(
                self.device
            )
            self.model.eval()

        # ----------------------------------------------------
        # Create preprocessing transform
        # ----------------------------------------------------

        self.transform = (
            get_inference_transform()
        )

    def predict(
        self,
        image_path,
        output_path=None,
    ):
        """
        Run the complete prediction pipeline.

        Parameters
        ----------
        image_path : str or Path
            Input MRI NIfTI file.

        output_path : str or Path, optional
            Destination prediction path.

            If None, a standard output path is
            automatically generated.

        Returns
        -------
        Path
            Path to the saved segmentation.
        """

        image_path = Path(
            image_path
        )

        # ----------------------------------------------------
        # Step 1: Validate input
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("FedMed Medical Image Inference")
        print("=" * 60)

        print()
        print("Step 1/6 - Validating MRI...")

        validation = validate_mri_file(
            image_path
        )

        print(
            f"  Valid MRI: {validation['valid']}"
        )

        print(
            f"  Shape: {validation['shape']}"
        )

        print(
            f"  Modalities: "
            f"{validation['num_modalities']}"
        )

        # ----------------------------------------------------
        # Step 2: Preprocess
        # ----------------------------------------------------

        print()
        print("Step 2/6 - Preprocessing MRI...")

        image = self.transform(
            str(image_path)
        )

        print(
            f"  Preprocessed shape: "
            f"{tuple(image.shape)}"
        )

        # Add batch dimension
        image = image.unsqueeze(
            0
        )

        print(
            f"  Model input shape: "
            f"{tuple(image.shape)}"
        )

        # ----------------------------------------------------
        # Step 3: Sliding-window inference
        # ----------------------------------------------------

        print()
        print(
            "Step 3/6 - Running sliding-window inference..."
        )

        logits = (
            run_sliding_window_inference(
                model=self.model,
                image=image,
                roi_size=(
                    96,
                    96,
                    96,
                ),
                sw_batch_size=1,
                overlap=0.25,
                device=self.device,
            )
        )

        print(
            f"  Model output shape: "
            f"{tuple(logits.shape)}"
        )

        # ----------------------------------------------------
        # Step 4: Convert logits to segmentation
        # ----------------------------------------------------

        print()
        print(
            "Step 4/6 - Creating segmentation mask..."
        )

        segmentation = (
            logits_to_segmentation(
                logits
            )
        )

        mask = (
            remove_batch_dimension(
                segmentation
            )
        )

        print(
            f"  Segmentation shape: "
            f"{tuple(mask.shape)}"
        )

        print(
            f"  Segmentation dtype: "
            f"{mask.dtype}"
        )

        print(
            f"  Predicted classes: "
            f"{torch.unique(mask).tolist()}"
        )

        # ----------------------------------------------------
        # Step 5: Determine output path
        # ----------------------------------------------------

        print()
        print(
            "Step 5/6 - Preparing output..."
        )

        if output_path is None:

            output_path = (
                get_prediction_path(
                    image_path
                )
            )

        else:

            output_path = Path(
                output_path
            )

        print(
            f"  Output path: "
            f"{output_path}"
        )

        # ----------------------------------------------------
        # Step 6: Save NIfTI segmentation
        # ----------------------------------------------------

        print()
        print(
            "Step 6/6 - Saving segmentation..."
        )

        saved_path = (
            save_segmentation_nifti(
                segmentation=mask,
                reference_image_path=image_path,
                output_path=output_path,
            )
        )

        print(
            f"  Saved: {saved_path}"
        )

        print()
        print("=" * 60)
        print("Inference completed successfully.")
        print("=" * 60)

        return saved_path


def predict(
    model,
    image,
    device="cpu",
):
    """
    Compatibility function for inference on an
    already-prepared tensor.

    This preserves the simple API used by earlier code.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    image : torch.Tensor
        Preprocessed image tensor.

    device : str or torch.device
        Inference device.

    Returns
    -------
    torch.Tensor
        Raw model prediction.
    """

    model = model.to(
        device
    )

    model.eval()

    image = image.to(
        device
    )

    with torch.no_grad():

        prediction = model(
            image
        )

    return prediction


def run_prediction(
    image_path,
    output_path=None,
    device=DEVICE,
):
    """
    Convenience function for running the complete
    end-to-end inference pipeline.

    Parameters
    ----------
    image_path : str or Path
        Input MRI file.

    output_path : str or Path, optional
        Output segmentation file.

    device : str or torch.device
        Inference device.

    Returns
    -------
    Path
        Saved prediction path.
    """

    predictor = Predictor(
        device=device
    )

    return predictor.predict(
        image_path=image_path,
        output_path=output_path,
    )


if __name__ == "__main__":

    # --------------------------------------------------------
    # Default local test MRI
    # --------------------------------------------------------

    input_path = Path(
        "inference_data"
    ) / "BRATS_001.nii.gz"

    run_prediction(
        image_path=input_path,
        device=DEVICE,
    )