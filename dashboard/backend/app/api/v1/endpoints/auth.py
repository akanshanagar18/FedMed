"""
Module: dashboard.backend.app.api.v1.endpoints.auth

Purpose:
REST API endpoints for Authentication & RBAC (/api/v1/auth/login, /me, /users, /audit-logs).
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Header, status
from pydantic import BaseModel, Field

from security.auth import global_auth_manager, Role
from common.schemas import SuccessResponse

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(default="admin")
    password: str = Field(default="admin123")


class RegisterUserRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = Field(default="Viewer")
    hospital_id: Optional[str] = None


@router.post("/login", response_model=SuccessResponse)
async def login(req: LoginRequest):
    """Authenticates user credentials and issues JWT access token."""
    user = global_auth_manager.authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = global_auth_manager.create_access_token(user["username"], user["role"])
    return SuccessResponse(
        message="Authentication successful",
        data={
            "access_token": token,
            "token_type": "bearer",
            "username": user["username"],
            "role": user["role"],
            "hospital_id": user.get("hospital_id"),
        },
    )


@router.get("/me", response_model=SuccessResponse)
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Returns currently authenticated user profile."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1]
    payload = global_auth_manager.decode_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Token expired or invalid signature")

    return SuccessResponse(
        message="User profile retrieved",
        data=payload,
    )


@router.post("/users", response_model=SuccessResponse)
async def register_user(req: RegisterUserRequest):
    """Registers a new enterprise user."""
    try:
        r_enum = Role(req.role)
    except ValueError:
        r_enum = Role.VIEWER

    res = global_auth_manager.register_user(req.username, req.email, req.password, r_enum, req.hospital_id)
    return SuccessResponse(
        message=f"User '{req.username}' registered successfully",
        data=res,
    )


@router.get("/audit-logs", response_model=SuccessResponse)
async def get_audit_logs():
    """Returns security audit logs."""
    logs = global_auth_manager.audit_logs
    return SuccessResponse(
        message="Audit logs retrieved",
        data={"total_logs": len(logs), "logs": logs},
    )
