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

import argparse
from pathlib import Path

import torch

from configs.config import DEVICE

from configs.inference_config import (
    DEFAULT_INPUT_FILE,
    ROI_SIZE,
    SW_BATCH_SIZE,
    SW_OVERLAP,
)

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
    """

    def __init__(
        self,
        model=None,
        device=DEVICE,
    ):
        """
        Initialize the predictor.
        """

        self.device = device

        if model is None:
            self.model = load_model(
                device=self.device
            )
        else:
            self.model = model.to(
                self.device
            )
            self.model.eval()

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

        Returns
        -------
        Path
            Path to the saved segmentation.
        """

        image_path = Path(
            image_path
        )

        print()
        print("=" * 60)
        print("FedMed Medical Image Inference")
        print("=" * 60)

        # ----------------------------------------------------
        # Step 1: Validate input
        # ----------------------------------------------------

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
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                overlap=SW_OVERLAP,
                device=self.device,
            )
        )

        print(
            f"  Model output shape: "
            f"{tuple(logits.shape)}"
        )

        # ----------------------------------------------------
        # Step 4: Postprocessing
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
        # Step 5: Output path
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
        # Step 6: Save NIfTI
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
    Run the complete inference pipeline.
    """

    predictor = Predictor(
        device=device
    )

    return predictor.predict(
        image_path=image_path,
        output_path=output_path,
    )


def parse_arguments():
    """
    Parse command-line inference arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "FedMed brain tumour "
            "segmentation inference"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(
            DEFAULT_INPUT_FILE
        ),
        help=(
            "Path to input MRI NIfTI file. "
            "Defaults to the configured "
            "sample MRI."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional output prediction path. "
            "If omitted, the standard output "
            "path is generated automatically."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    run_prediction(
        image_path=args.input,
        output_path=args.output,
        device=DEVICE,
    )