"""
Module: dashboard.backend.app.schemas.responses

Purpose:
Standardized success and error response schemas.
Ensures the frontend has a predictable contract for parsing API replies.

TODO:
- [ ] Enhance with pagination metadata schema if needed.
"""

from pydantic import BaseModel
from typing import Any, Optional


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Any] = None


class SuccessResponse(BaseModel):
    message: str
    data: Optional[Any] = None
