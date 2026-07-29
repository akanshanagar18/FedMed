"""
Project Configuration
FedMed - Medical Image Segmentation
"""

# Dataset

DATASET_NAME = "BraTS"

DATASET_PATH = "data/BraTS"

IMAGE_SIZE = (128, 128, 128)

# Training

BATCH_SIZE = 2

NUM_EPOCHS = 50

LEARNING_RATE = 0.001

# Model

MODEL_NAME = "3D U-Net"

IN_CHANNELS = 4

OUT_CHANNELS = 3

# Hardware

DEVICE = "cuda"

# Output

CHECKPOINT_DIR = "outputs/checkpoints"

PREDICTION_DIR = "outputs/predictions"