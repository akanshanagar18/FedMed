"""
FedMed - Centralized Computer Vision Configuration

This file contains the common configuration used by the
medical image segmentation pipeline.

Dataset:
    Medical Segmentation Decathlon - Task01_BrainTumour

Model:
    3D U-Net

Framework:
    PyTorch + MONAI
"""

from pathlib import Path
import torch


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

# Root directory of the FedMed project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset directory
DATA_DIR = PROJECT_ROOT / "data"

# Task01 Brain Tumour dataset
DATASET_PATH = DATA_DIR / "Task01_BrainTumour"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Model checkpoints
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Prediction outputs
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# Evaluation results
EVALUATION_DIR = OUTPUT_DIR / "evaluation"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_NAME = "Task01_BrainTumour"

# Training image directory
IMAGES_TRAIN_DIR = DATASET_PATH / "imagesTr"

# Training segmentation directory
LABELS_TRAIN_DIR = DATASET_PATH / "labelsTr"

# Test image directory
IMAGES_TEST_DIR = DATASET_PATH / "imagesTs"

# Dataset metadata
DATASET_JSON = DATASET_PATH / "dataset.json"


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

# The MSD Brain Tumour images contain four MRI modalities:
#
# Channel 0 -> MRI modality 1
# Channel 1 -> MRI modality 2
# Channel 2 -> MRI modality 3
# Channel 3 -> MRI modality 4
#
# Therefore, the model receives 4 input channels.

IN_CHANNELS = 4

# Segmentation classes:
#
# 0 -> Background
# 1 -> Tumour class 1
# 2 -> Tumour class 2
# 3 -> Tumour class 3

OUT_CHANNELS = 4

# Spatial patch used during training.
#
# Full MRI volume:
# approximately 240 x 240 x 155
#
# Training patch:
# 96 x 96 x 96
#
# Patch-based training reduces GPU memory usage.

PATCH_SIZE = (96, 96, 96)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "3D U-Net"

# Number of channels at each U-Net level
CHANNELS = (
    16,
    32,
    64,
    128,
)

# Downsampling factors
STRIDES = (
    2,
    2,
    2,
)

# Number of residual units
NUM_RES_UNITS = 2


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Batch size.
#
# Keep this at 1 initially because 3D MRI data requires
# significant GPU memory.

BATCH_SIZE = 1

# Number of epochs for the initial centralized baseline.
#
# For today's final-submission build, start with 3 epochs.
# Increase later if hardware/time allows.

NUM_EPOCHS = 3

# Learning rate
LEARNING_RATE = 1e-4

# Weight decay for regularization
WEIGHT_DECAY = 1e-5

# Validation frequency
VALIDATION_INTERVAL = 1

# Random seed for reproducibility
SEED = 42


# ============================================================
# DATA SPLIT CONFIGURATION
# ============================================================

# Percentage of data used for validation

VALIDATION_RATIO = 0.20


# ============================================================
# DATALOADER CONFIGURATION
# ============================================================

# Number of samples loaded per batch
# This is intentionally kept at 1 for 3D MRI training.

NUM_WORKERS = 0

# Shuffle training data
SHUFFLE_TRAIN = True

# Do not shuffle validation data
SHUFFLE_VALIDATION = False

# Pin memory when CUDA is available
PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

# Automatically use GPU if available.
# Otherwise use CPU.

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Display the selected device
DEVICE_NAME = (
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)


# ============================================================
# OPTIMIZER CONFIGURATION
# ============================================================

OPTIMIZER_NAME = "AdamW"

# Scheduler configuration
USE_SCHEDULER = True

SCHEDULER_NAME = "CosineAnnealingLR"


# ============================================================
# LOSS CONFIGURATION
# ============================================================

LOSS_NAME = "DiceCE"


# ============================================================
# CHECKPOINT CONFIGURATION
# ============================================================

BEST_MODEL_NAME = "best_model.pth"

LAST_MODEL_NAME = "last_model.pth"

BEST_MODEL_PATH = CHECKPOINT_DIR / BEST_MODEL_NAME

LAST_MODEL_PATH = CHECKPOINT_DIR / LAST_MODEL_NAME


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

METRICS_FILE = EVALUATION_DIR / "metrics.json"

PREDICTION_METRICS_FILE = EVALUATION_DIR / "prediction_metrics.json"


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

LOG_INTERVAL = 10


# ============================================================
# REPRODUCIBILITY
# ============================================================

DETERMINISTIC = True


# ============================================================
# HELPER FUNCTION
# ============================================================

def create_output_directories():
    """
    Create required output directories if they do not exist.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREDICTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

def print_config():
    """
    Print the most important configuration values.
    """

    print("=" * 60)
    print("FedMed - Computer Vision Configuration")
    print("=" * 60)

    print(f"Project Root     : {PROJECT_ROOT}")
    print(f"Dataset          : {DATASET_PATH}")
    print(f"Dataset Name     : {DATASET_NAME}")

    print("-" * 60)

    print(f"Input Channels   : {IN_CHANNELS}")
    print(f"Output Channels  : {OUT_CHANNELS}")
    print(f"Patch Size       : {PATCH_SIZE}")

    print("-" * 60)

    print(f"Model            : {MODEL_NAME}")
    print(f"Channels         : {CHANNELS}")
    print(f"Strides          : {STRIDES}")

    print("-" * 60)

    print(f"Batch Size       : {BATCH_SIZE}")
    print(f"Epochs           : {NUM_EPOCHS}")
    print(f"Learning Rate    : {LEARNING_RATE}")
    print(f"Weight Decay     : {WEIGHT_DECAY}")

    print("-" * 60)

    print(f"Device            : {DEVICE}")
    print(f"Device Name       : {DEVICE_NAME}")
    print(f"Workers           : {NUM_WORKERS}")

    print("=" * 60)