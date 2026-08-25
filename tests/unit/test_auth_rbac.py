"""
Module: tests.unit.test_auth_rbac

Purpose:
Unit test suite for AuthManager (registration, password hashing, JWT encoding/decoding, RBAC role permissions).
"""

import pytest
from security.auth import AuthManager, Role, global_auth_manager


def test_auth_manager_registration_and_authentication():
    mgr = global_auth_manager
    reg = mgr.register_user("testuser", "test@fedmed.org", "pass123", Role.RESEARCH_SCIENTIST)
    assert reg["username"] == "testuser"
    assert reg["role"] == Role.RESEARCH_SCIENTIST.value

    auth_user = mgr.authenticate_user("testuser", "pass123")
    assert auth_user is not None
    assert auth_user["username"] == "testuser"

    invalid_user = mgr.authenticate_user("testuser", "wrongpass")
    assert invalid_user is None


def test_jwt_token_issuance_and_decoding():
    mgr = global_auth_manager
    token = mgr.create_access_token("testuser", Role.RESEARCH_SCIENTIST.value)
    assert isinstance(token, str)

    payload = mgr.decode_token(token)
    assert payload is not None
    assert payload["sub"] == "testuser"
    assert payload["role"] == Role.RESEARCH_SCIENTIST.value


def test_rbac_permissions():
    mgr = global_auth_manager
    assert mgr.check_permission(Role.ADMINISTRATOR.value, "deploy") is True
    assert mgr.check_permission(Role.VIEWER.value, "deploy") is False
