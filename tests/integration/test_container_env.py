"""
Integration tests for container environment variable parsing and configuration.
"""

import os
import pytest


def test_dotenv_file_parsing():
    """Verify .env file exists and contains essential production environment variables."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_path = os.path.join(project_root, ".env")
    assert os.path.exists(env_path)

    with open(env_path, "r") as f:
        content = f.read()

    assert "DATABASE_URL=" in content
    assert "FLOWER_SERVER=" in content
    assert "BACKEND_URL=" in content
    assert "CHECKPOINT_DIR=" in content
    assert "CERT_DIR=" in content
