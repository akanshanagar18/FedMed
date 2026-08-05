"""
Module: configs.constants

Purpose:
Project-wide constants including static paths, standard error messages, 
and default directory structures.
"""

from pathlib import Path

# Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
LOGS_DIR = PROJECT_ROOT / "logs"

# Error Codes
ERROR_VALIDATION = "VALIDATION_ERROR"
ERROR_AUTH_FAILED = "AUTH_FAILED"
ERROR_NODE_OFFLINE = "NODE_OFFLINE"
ERROR_AGGREGATION_FAILED = "AGGREGATION_FAILED"
ERROR_ENCRYPTION_FAILED = "ENCRYPTION_FAILED"
ERROR_TRAINING_FAILED = "TRAINING_FAILED"
ERROR_UNKNOWN = "UNKNOWN_ERROR"

