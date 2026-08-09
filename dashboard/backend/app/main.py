"""
Module: dashboard.backend.app.main

Purpose:
FastAPI application entrypoint.
Bootstraps configuration, middleware, error handlers, routers,
lifespan events, and static file serving for the React dashboard frontend.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config.settings import settings
from app.core.logging import setup_logging
from app.core.exceptions import register_exception_handlers
from app.middleware.cors import register_cors
from app.api.v1.router import api_router
from app.database.session import init_db

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager replacing deprecated @app.on_event handlers."""
    logger.info("Starting up FedMed Backend...")
    init_db()
    logger.info("Database tables initialized.")
    yield
    logger.info("Shutting down FedMed Backend...")


def create_app() -> FastAPI:
    """Application factory for FastAPI."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="2.0.0-rc1",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        description="Monitoring backend API contract for FedMed.",
        lifespan=lifespan,
    )

    # Register core middleware and exception handlers
    register_cors(app)
    register_exception_handlers(app)

    # Include main API router
    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/metrics", include_in_schema=True)
    async def prometheus_metrics():
        from utils.telemetry import prometheus_registry
        return Response(
            content=prometheus_registry.generate_prometheus_text(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    # Static files setup for frontend React dashboard
    frontend_dist = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
    )
    if os.path.exists(frontend_dist):
        logger.info(f"Mounting static frontend dashboard from {frontend_dist}")
        app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

        @app.get("/", include_in_schema=False)
        async def serve_dashboard_root():
            return FileResponse(os.path.join(frontend_dist, "index.html"))

    return app


app = create_app()
