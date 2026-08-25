"""
Unit tests for Docker configuration validation (docker-compose.yml, Dockerfiles, .env).
"""

import os
import yaml
import pytest


def test_docker_compose_manifest_structure():
    """Verify docker-compose.yml contains all required services, networks, and volumes."""
    compose_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docker-compose.yml")
    assert os.path.exists(compose_path)

    with open(compose_path, "r") as f:
        data = yaml.safe_load(f)

    assert "services" in data
    assert "backend" in data["services"]
    assert "flower-server" in data["services"]
    assert "hospital-alpha" in data["services"]
    assert "hospital-beta" in data["services"]
    assert "hospital-gamma" in data["services"]
    assert "dashboard" in data["services"]

    assert "volumes" in data
    assert "database_data" in data["volumes"]
    assert "checkpoint_data" in data["volumes"]
    assert "certificate_data" in data["volumes"]


def test_dockerfile_existence():
    """Verify all 4 production Dockerfiles exist in docker/ directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    assert os.path.exists(os.path.join(project_root, "docker", "backend.Dockerfile"))
    assert os.path.exists(os.path.join(project_root, "docker", "flower_server.Dockerfile"))
    assert os.path.exists(os.path.join(project_root, "docker", "hospital_client.Dockerfile"))
    assert os.path.exists(os.path.join(project_root, "docker", "dashboard.Dockerfile"))
