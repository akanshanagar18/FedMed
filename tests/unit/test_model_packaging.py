"""
Unit tests for scripts/package_production_model.py and artifacts/fedmed_model/.
Verifies model package file completeness, cryptographic hash integrity,
and clinical regulatory non-claim designations.
"""

import hashlib
import json
from pathlib import Path
import pytest

from scripts.package_production_model import package_production_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "fedmed_model"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_model_packaging_integrity():
    res = package_production_model()
    assert res["status"] == "PASS"

    assert (ARTIFACTS_DIR / "model.pt").exists()
    assert (ARTIFACTS_DIR / "model_manifest.json").exists()
    assert (ARTIFACTS_DIR / "architecture.json").exists()
    assert (ARTIFACTS_DIR / "preprocessing.yaml").exists()
    assert (ARTIFACTS_DIR / "README.md").exists()

    with open(ARTIFACTS_DIR / "model_manifest.json", "r") as f:
        manifest = json.load(f)

    # 1. Verify Hashes
    assert manifest["model_hash"] == compute_sha256(ARTIFACTS_DIR / "model.pt")
    assert manifest["architecture_hash"] == compute_sha256(ARTIFACTS_DIR / "architecture.json")
    assert manifest["preprocessing_hash"] == compute_sha256(ARTIFACTS_DIR / "preprocessing.yaml")

    # 2. Verify Classification & Non-Claim Labels
    assert manifest["artifact_classification"] == "DEVELOPMENT_VALIDATION_ARTIFACT"
    assert manifest["training_provenance"]["training_dataset_mode"] == "DEVELOPMENT_SYNTHETIC"

    # 3. Verify Architecture
    with open(ARTIFACTS_DIR / "architecture.json", "r") as f:
        arch = json.load(f)
    assert arch["parameter_count"] == 4810074
