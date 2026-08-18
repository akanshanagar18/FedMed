"""
=========================================================
FedMed Configuration File
=========================================================
This file stores every project configuration.

Instead of hardcoding values inside different files,
the entire project imports variables from here.

Example:

from configs.config import BATCH_SIZE
=========================================================
"""

from pathlib import Path
import torch

# =========================================================
# Project Directories
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

# =========================================================
# Dataset Configuration
# =========================================================

DATASET_NAME = "BraTS2021"

DATASET_PATH = DATA_DIR / DATASET_NAME

IMAGE_SIZE = (128, 128, 128)

IN_CHANNELS = 4

OUT_CHANNELS = 3

# =========================================================
# Model Configuration
# =========================================================

MODEL_NAME = "3D U-Net"

CHANNELS = (16, 32, 64, 128, 256)

STRIDES = (2, 2, 2, 2)

NUM_RES_UNITS = 2

# =========================================================
# Training Configuration
# =========================================================

BATCH_SIZE = 2

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

NUM_WORKERS = 2

VALIDATION_INTERVAL = 1

# =========================================================
# Hardware
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Random Seed
# =========================================================

SEED = 42

# =========================================================
# Model Saving
# =========================================================

BEST_MODEL_NAME = "best_model.pth"

LAST_MODEL_NAME = "last_model.pth"

# =========================================================
# Logging
# =========================================================

LOG_INTERVAL = 10