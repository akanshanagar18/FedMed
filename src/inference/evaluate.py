"""
FedMed - Evaluation Command

Runs segmentation evaluation for a prediction and its
corresponding ground-truth NIfTI file.
"""

import json
from pathlib import Path

from src.inference.evaluation import (
    evaluate_nifti,
)


DEFAULT_PREDICTION = Path(
    "outputs"
) / "predictions" / "BRATS_001_prediction.nii.gz"

DEFAULT_TARGET = Path(
    "evaluation_data"
) / "BRATS_001.nii.gz"

DEFAULT_OUTPUT = Path(
    "outputs"
) / "evaluation" / "BRATS_001_metrics.json"


def run_evaluation(
    prediction_path=DEFAULT_PREDICTION,
    target_path=DEFAULT_TARGET,
    output_path=DEFAULT_OUTPUT,
):
    """
    Run segmentation evaluation and save the results.

    Parameters
    ----------
    prediction_path : str or Path
        Predicted segmentation NIfTI.

    target_path : str or Path
        Ground-truth segmentation NIfTI.

    output_path : str or Path
        JSON output path.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    prediction_path = Path(
        prediction_path
    )

    target_path = Path(
        target_path
    )

    output_path = Path(
        output_path
    )

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction not found: "
            f"{prediction_path}"
        )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: "
            f"{target_path}"
        )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FedMed Segmentation Evaluation")
    print("=" * 60)

    print()
    print("Prediction:")
    print(f"  {prediction_path}")

    print()
    print("Ground truth:")
    print(f"  {target_path}")

    print()
    print("Calculating metrics...")

    results = evaluate_nifti(
        prediction_path,
        target_path,
    )

    # --------------------------------------------------------
    # Save JSON report
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("-" * 60)
    print("Evaluation Results")
    print("-" * 60)

    print(
        f"Mean Dice:          "
        f"{results['mean_dice']:.4f}"
    )

    print(
        f"Mean IoU:           "
        f"{results['mean_iou']:.4f}"
    )

    print(
        f"Whole Tumour Dice:  "
        f"{results['whole_tumour_dice']:.4f}"
    )

    print(
        f"Whole Tumour IoU:   "
        f"{results['whole_tumour_iou']:.4f}"
    )

    print()
    print("Class Metrics:")

    for class_name, metrics in (
        results["class_metrics"].items()
    ):

        print(
            f"  {class_name}:"
        )

        print(
            f"    Dice: "
            f"{metrics['dice']:.4f}"
        )

        print(
            f"    IoU:  "
            f"{metrics['iou']:.4f}"
        )

    print()
    print(
        f"Metrics saved to:"
    )

    print(
        f"  {output_path}"
    )

    print()
    print("=" * 60)
    print("Evaluation completed successfully.")
    print("=" * 60)

    return results


if __name__ == "__main__":

    run_evaluation()