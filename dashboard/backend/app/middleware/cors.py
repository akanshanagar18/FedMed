"""
Module: dashboard.backend.app.middleware.cors

Purpose:
Configures Cross-Origin Resource Sharing (CORS) middleware.
Allows the React frontend to securely communicate with the FastAPI backend.

TODO:
- [ ] Restrict origins in production based on deployment environments.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings


def register_cors(app: FastAPI) -> None:
    """Registers CORS middleware using settings."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
