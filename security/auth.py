"""
Module: security.auth

Purpose:
Enterprise Authentication & Role-Based Access Control (RBAC) Engine for FedMed v2.0.
Provides JWT token creation/decoding, password hashing, role permissions
(Administrator, Hospital Administrator, Research Scientist, Auditor, Viewer), and audit logging.
"""

from enum import Enum
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    ADMINISTRATOR = "Administrator"
    HOSPITAL_ADMINISTRATOR = "Hospital Administrator"
    RESEARCH_SCIENTIST = "Research Scientist"
    AUDITOR = "Auditor"
    VIEWER = "Viewer"


class AuthManager:
    """
    Enterprise Authentication and RBAC Manager.
    """

    SECRET_KEY = "FedMed_Enterprise_Secret_Key_2026"
    ALGORITHM = "HS256"
    TOKEN_EXPIRE_SECONDS = 3600

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuthManager, cls).__new__(cls)
            cls._instance.users: Dict[str, Dict[str, Any]] = {}
            cls._instance.audit_logs: List[Dict[str, Any]] = []
            cls._instance._seed_default_users()
        return cls._instance

    def _seed_default_users(self):
        """Seeds default enterprise users."""
        self.register_user("admin", "admin@fedmed.org", "admin123", Role.ADMINISTRATOR)
        self.register_user("hospital_admin_alpha", "alpha@hospital.org", "alpha123", Role.HOSPITAL_ADMINISTRATOR, "hospital_alpha")
        self.register_user("researcher", "scientist@fedmed.org", "sci123", Role.RESEARCH_SCIENTIST)
        self.register_user("auditor", "auditor@hipaa.gov", "audit123", Role.AUDITOR)

    def hash_password(self, password: str) -> str:
        """Returns SHA256 hashed password string."""
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.hash_password(plain_password) == hashed_password

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        role: Role = Role.VIEWER,
        hospital_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Registers a new user record."""
        user = {
            "username": username,
            "email": email,
            "hashed_password": self.hash_password(password),
            "role": role.value if isinstance(role, Role) else str(role),
            "hospital_id": hospital_id,
            "created_at": time.time(),
        }
        self.users[username] = user
        return {"username": username, "email": email, "role": user["role"]}

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticates user credentials."""
        user = self.users.get(username)
        if user and self.verify_password(password, user["hashed_password"]):
            self.log_audit(username, "AUTHENTICATE", "auth/login", "SUCCESS")
            return user
        self.log_audit(username, "AUTHENTICATE", "auth/login", "FAILED")
        return None

    def create_access_token(self, username: str, role: str) -> str:
        """Generates a signed pseudo-JWT token."""
        payload = {
            "sub": username,
            "role": role,
            "exp": int(time.time() + self.TOKEN_EXPIRE_SECONDS),
        }
        payload_str = json.dumps(payload)
        sig = hmac.new(self.SECRET_KEY.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload_str}.{sig}"

    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decodes and validates signed access token."""
        try:
            parts = token.rsplit(".", 1)
            if len(parts) != 2:
                return None
            payload_str, sig = parts[0], parts[1]
            expected_sig = hmac.new(self.SECRET_KEY.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(sig, expected_sig):
                return None
            payload = json.loads(payload_str)
            if payload.get("exp", 0) < time.time():
                return None
            return payload
        except Exception:
            return None


    def check_permission(self, role: str, required_permission: str) -> bool:
        """Checks if a role has the required permission."""
        permissions_matrix = {
            Role.ADMINISTRATOR.value: ["read", "write", "deploy", "audit", "admin"],
            Role.HOSPITAL_ADMINISTRATOR.value: ["read", "write", "hospital_manage"],
            Role.RESEARCH_SCIENTIST.value: ["read", "write", "train", "hpo"],
            Role.AUDITOR.value: ["read", "audit"],
            Role.VIEWER.value: ["read"],
        }
        allowed = permissions_matrix.get(role, ["read"])
        return required_permission in allowed or "admin" in allowed

    def log_audit(self, username: str, action: str, resource: str, status: str = "SUCCESS") -> None:
        """Appends security audit log."""
        self.audit_logs.append({
            "username": username,
            "action": action,
            "resource": resource,
            "status": status,
            "timestamp": time.time(),
        })


global_auth_manager = AuthManager()
