"""
FedMed - Segmentation Evaluation

Computes quantitative metrics between a predicted
brain tumour segmentation and a ground-truth mask.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


def load_segmentation(
    image_path,
):
    """
    Load a segmentation NIfTI file.

    Parameters
    ----------
    image_path : str or Path
        Path to segmentation NIfTI file.

    Returns
    -------
    numpy.ndarray
        3D segmentation array.
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Segmentation file not found: "
            f"{image_path}"
        )

    image = nib.load(
        str(image_path)
    )

    data = image.get_fdata()

    if data.ndim != 3:
        raise ValueError(
            "Expected a 3D segmentation volume. "
            f"Received shape: {data.shape}"
        )

    return data.astype(
        np.int32
    )


def calculate_dice(
    prediction,
    target,
):
    """
    Calculate Dice similarity coefficient.

    Dice = 2 * intersection / (prediction + target)

    Parameters
    ----------
    prediction : numpy.ndarray
        Predicted binary mask.

    target : numpy.ndarray
        Ground-truth binary mask.

    Returns
    -------
    float
        Dice score.
    """

    prediction = np.asarray(
        prediction
    ).astype(bool)

    target = np.asarray(
        target
    ).astype(bool)

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes do not match. "
            f"Prediction: {prediction.shape}, "
            f"Target: {target.shape}"
        )

    prediction_sum = prediction.sum()
    target_sum = target.sum()

    # Both masks are empty
    if prediction_sum == 0 and target_sum == 0:
        return 1.0

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    dice = (
        2.0 * intersection
        / (prediction_sum + target_sum)
    )

    return float(
        dice
    )


def calculate_iou(
    prediction,
    target,
):
    """
    Calculate Intersection over Union.

    IoU = intersection / union

    Parameters
    ----------
    prediction : numpy.ndarray
        Predicted binary mask.

    target : numpy.ndarray
        Ground-truth binary mask.

    Returns
    -------
    float
        IoU score.
    """

    prediction = np.asarray(
        prediction
    ).astype(bool)

    target = np.asarray(
        target
    ).astype(bool)

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes do not match. "
            f"Prediction: {prediction.shape}, "
            f"Target: {target.shape}"
        )

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    union = np.logical_or(
        prediction,
        target,
    ).sum()

    # Both masks are empty
    if union == 0:
        return 1.0

    iou = (
        intersection
        / union
    )

    return float(
        iou
    )


def evaluate_segmentation(
    prediction,
    target,
):
    """
    Calculate Dice and IoU for a multiclass segmentation.

    Background class 0 is excluded from the primary
    tumour metrics.

    Parameters
    ----------
    prediction : numpy.ndarray
        Predicted class mask.

    target : numpy.ndarray
        Ground-truth class mask.

    Returns
    -------
    dict
        Evaluation results.
    """

    prediction = np.asarray(
        prediction
    ).astype(np.int32)

    target = np.asarray(
        target
    ).astype(np.int32)

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes do not match. "
            f"Prediction: {prediction.shape}, "
            f"Target: {target.shape}"
        )

    class_results = {}

    # --------------------------------------------------------
    # Evaluate each tumour class
    # --------------------------------------------------------

    for class_id in [1, 2, 3]:

        prediction_mask = (
            prediction == class_id
        )

        target_mask = (
            target == class_id
        )

        dice = calculate_dice(
            prediction_mask,
            target_mask,
        )

        iou = calculate_iou(
            prediction_mask,
            target_mask,
        )

        class_results[
            f"class_{class_id}"
        ] = {
            "dice": dice,
            "iou": iou,
            "predicted_voxels": int(
                prediction_mask.sum()
            ),
            "target_voxels": int(
                target_mask.sum()
            ),
        }

    # --------------------------------------------------------
    # Whole tumour evaluation
    #
    # Any non-background label is considered tumour.
    # --------------------------------------------------------

    prediction_tumour = (
        prediction > 0
    )

    target_tumour = (
        target > 0
    )

    whole_tumour_dice = calculate_dice(
        prediction_tumour,
        target_tumour,
    )

    whole_tumour_iou = calculate_iou(
        prediction_tumour,
        target_tumour,
    )

    # --------------------------------------------------------
    # Average tumour metrics
    # --------------------------------------------------------

    mean_dice = float(
        np.mean(
            [
                class_results[
                    f"class_{class_id}"
                ]["dice"]
                for class_id in [1, 2, 3]
            ]
        )
    )

    mean_iou = float(
        np.mean(
            [
                class_results[
                    f"class_{class_id}"
                ]["iou"]
                for class_id in [1, 2, 3]
            ]
        )
    )

    return {
        "class_metrics": class_results,
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "whole_tumour_dice": whole_tumour_dice,
        "whole_tumour_iou": whole_tumour_iou,
    }


def evaluate_nifti(
    prediction_path,
    target_path,
):
    """
    Evaluate two NIfTI segmentation files.

    Parameters
    ----------
    prediction_path : str or Path
        Predicted segmentation.

    target_path : str or Path
        Ground-truth segmentation.

    Returns
    -------
    dict
        Evaluation results.
    """

    prediction = load_segmentation(
        prediction_path
    )

    target = load_segmentation(
        target_path
    )

    return evaluate_segmentation(
        prediction,
        target,
    )