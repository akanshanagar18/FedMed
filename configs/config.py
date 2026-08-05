"""
Module: configs.config

Purpose:
Backward compatibility wrapper re-exporting project settings and constants.
Prefer importing directly from `configs.settings` or `configs.constants`.
"""

import torch
from configs.settings import settings
from configs.constants import (
    PROJECT_ROOT,
    DATA_DIR,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    PREDICTION_DIR,
)

# Dataset Config
DATASET_NAME = settings.DATASET_NAME
DATASET_PATH = DATA_DIR / DATASET_NAME
IMAGE_SIZE = settings.IMAGE_SIZE
IN_CHANNELS = settings.IN_CHANNELS
OUT_CHANNELS = settings.OUT_CHANNELS

# Model Config
MODEL_NAME = "3D U-Net"
CHANNELS = settings.CHANNELS
STRIDES = settings.STRIDES
NUM_RES_UNITS = settings.NUM_RES_UNITS

# Training Config
BATCH_SIZE = settings.BATCH_SIZE
NUM_EPOCHS = settings.NUM_EPOCHS
LEARNING_RATE = settings.LEARNING_RATE
WEIGHT_DECAY = settings.WEIGHT_DECAY
NUM_WORKERS = settings.NUM_WORKERS
VALIDATION_INTERVAL = settings.VALIDATION_INTERVAL

# Hardware & Seed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = settings.SEED

# Model Checkpoints & Logging
BEST_MODEL_NAME = "best_model.pth"
LAST_MODEL_NAME = "last_model.pth"
LOG_INTERVAL = 10