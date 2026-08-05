"""
Module: configs.settings

Purpose:
Centralized configuration management using Pydantic BaseSettings.
Loads environment variables from .env for easy access across the project.
"""

from pathlib import Path
from typing import Tuple
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # System Configs
    PROJECT_NAME: str = "FedMed Monitoring API"
    API_V1_STR: str = "/api/v1"
    
    # Database & Server
    DATABASE_URL: str = "sqlite:///./fedmed.db"
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Dataset & Model Parameters
    DATASET_NAME: str = "BraTS2021"
    IMAGE_SIZE: Tuple[int, int, int] = (128, 128, 128)
    IN_CHANNELS: int = 4
    OUT_CHANNELS: int = 3
    CHANNELS: Tuple[int, ...] = (16, 32, 64, 128, 256)
    STRIDES: Tuple[int, ...] = (2, 2, 2, 2)
    NUM_RES_UNITS: int = 2
    
    # Training Parameters
    BATCH_SIZE: int = 2
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    NUM_WORKERS: int = 2
    VALIDATION_INTERVAL: int = 1
    SEED: int = 42
    
    # Federated Learning Configs
    MAX_HOSPITALS: int = 10

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

