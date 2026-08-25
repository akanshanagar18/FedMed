"""
Module: dashboard.backend.app.core.exceptions

Purpose:
Registers custom global exception handlers for the FastAPI application.
Ensures that all errors returned to the React frontend match a strictly typed
ErrorResponse schema, avoiding unstructured tracebacks.

TODO:
- [ ] Add handlers for SQLAlchemy Integrity errors.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.schemas.responses import ErrorResponse




class FedMedException(Exception):
    """Base exception for custom FedMed business logic errors."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code


def register_exception_handlers(app: FastAPI) -> None:
    """Attaches custom exception handlers to the FastAPI app."""
    
    @app.exception_handler(FedMedException)
    async def fedmed_exception_handler(request: Request, exc: FedMedException):
        response_data = ErrorResponse(
            error="FedMedError",
            message=exc.message
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=response_data.model_dump()
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        # In production, log the traceback here.
        response_data = ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred."
        )
        return JSONResponse(
            status_code=500,
            content=response_data.model_dump()
        )
