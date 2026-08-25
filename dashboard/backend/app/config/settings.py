"""
Module: dashboard.backend.app.config.settings

Purpose:
Manages backend configurations utilizing Pydantic BaseSettings.
Acts as the central source of truth for environment variables, standardizing
database URLs, API versions, and CORS origins.

TODO:
- [ ] Define precise CORS allowed origins based on future React deployment.
- [ ] Incorporate encryption key paths for TenSEAL initialization if needed.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "FedMed Monitoring API"
    API_V1_STR: str = "/api/v1"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Database
    DATABASE_URL: str = "sqlite:///./fedmed.db"
    
    # Federated Learning Configs
    MAX_HOSPITALS: int = 10
    
    class Config:
        env_file = ".env"
        extra = "ignore"



settings = Settings()
