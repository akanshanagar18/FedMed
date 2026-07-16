"""
Module: dashboard.backend.app.main

Purpose:
FastAPI application entrypoint.
Bootstraps configuration, middleware, error handlers, and routers.
"""

from fastapi import FastAPI
from app.config.settings import settings
from app.core.logging import setup_logging
from app.core.exceptions import register_exception_handlers
from app.middleware.cors import register_cors
from app.api.v1.router import api_router
import logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Application factory for FastAPI."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.1.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        description="Monitoring backend API contract for FedMed."
    )
    
    # Register components
    register_cors(app)
    register_exception_handlers(app)
    
    # Include main API router
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    return app

app = create_app()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FedMed Backend...")
    # TODO: Initialize database connection pools or Redis here.

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down FedMed Backend...")
    # TODO: Clean up database connections and WebSocket active pools.
