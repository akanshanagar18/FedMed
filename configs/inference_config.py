"""
FedMed - Inference Configuration

Central configuration for the medical image
inference pipeline.
"""

from pathlib import Path


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INFERENCE_DATA_DIR = (
    PROJECT_ROOT / "inference_data"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

VISUALIZATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "visualizations"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)


# ------------------------------------------------------------
# Default inference input
# ------------------------------------------------------------

DEFAULT_INPUT_FILE = (
    INFERENCE_DATA_DIR
    / "BRATS_001.nii.gz"
)


# ------------------------------------------------------------
# Model inference settings
# ------------------------------------------------------------

ROI_SIZE = (
    96,
    96,
    96,
)

SW_BATCH_SIZE = 1

SW_OVERLAP = 0.25


# ------------------------------------------------------------
# Segmentation settings
# ------------------------------------------------------------

NUM_CLASSES = 4

BACKGROUND_CLASS = 0

TUMOUR_CLASSES = (
    1,
    2,
    3,
)


# ------------------------------------------------------------
# Output settings
# ------------------------------------------------------------

PREDICTION_SUFFIX = (
    "_prediction.nii.gz"
)

VISUALIZATION_SUFFIX = (
    "_visualization.png"
)

METRICS_SUFFIX = (
    "_metrics.json"
)