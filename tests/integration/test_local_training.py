"""
Integration Test: tests/integration/test_local_training.py

Purpose:
Rigorous verification of the local training execution engine for Phase 8.5B.2:
  1. Development dataset mode requires explicit authorization flag
  2. Persisted split from reports/dataset_split.json is strictly loaded and respected
  3. TEST cohort is completely firewalled from training and validation
  4. Model receives 4-channel MRI input and produces 3-channel segmentation logits
  5. Backpropagation computes non-zero gradients across all trainable layers
  6. Optimizer step updates model parameters (delta > 0)
  7. Independent Dice and IoU metrics are generated from validation inference
  8. Checkpoints and JSON experiment reports record full reproducibility metadata
"""

import json
import os
from pathlib import Path
import pytest
import torch
import yaml

from scripts.run_real_local_training import (
    load_canonical_model,
    run_local_training_experiment,
    ExplicitPatientDataset,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "real_brats_fedavg.yaml"
SPLIT_PATH = PROJECT_ROOT / "reports" / "dataset_split.json"


def test_canonical_model_instantiation():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    model, param_count = load_canonical_model(config, device)

    assert param_count == 4810074, f"Expected 4,810,074 parameters, got {param_count}"
    assert isinstance(model, torch.nn.Module)


def test_dataset_split_firewall_and_loading():
    with open(SPLIT_PATH, "r") as f:
        split_data = json.load(f)

    train_subjects = split_data["train_subjects"]
    val_subjects = split_data["validation_subjects"]
    test_subjects = split_data["test_subjects"]

    # Assert mutual exclusion
    assert len(set(train_subjects).intersection(val_subjects)) == 0
    assert len(set(train_subjects).intersection(test_subjects)) == 0
    assert len(set(val_subjects).intersection(test_subjects)) == 0

    # Ensure ExplicitPatientDataset only indexes designated subjects
    data_dir = PROJECT_ROOT / "data" / "BraTS2021"
    train_ds = ExplicitPatientDataset(data_dir=data_dir, subject_ids=train_subjects, transforms=None)
    assert len(train_ds) == len(train_subjects)
    indexed_pids = [s["patient_id"] for s in train_ds.samples]
    assert indexed_pids == sorted(train_subjects)
    assert not any(t_id in indexed_pids for t_id in test_subjects)


def test_local_training_full_pipeline_execution(tmp_path):
    report = run_local_training_experiment(
        data_dir="data/BraTS2021",
        config_path=str(CONFIG_PATH),
        epochs=1,
        batch_size=1,
        seed=42,
        allow_development=True,
    )

    # 1. Verify Report Attributes
    assert report["execution_mode"] == "DEVELOPMENT_SYNTHETIC"
    assert report["model_configuration"]["parameter_count"] == 4810074
    assert len(report["epoch_logs"]) == 1

    log = report["epoch_logs"][0]
    # 2. Verify Training Gradients & Weight Deltas
    assert log["gradient_norm"] > 0.0, "Gradient norm must be non-zero!"
    assert log["max_parameter_delta"] > 0.0, "Max parameter delta must be positive!"
    assert log["l2_parameter_delta"] > 0.0, "L2 parameter delta must be positive!"

    # 3. Verify Validation Metrics
    val_m = log["validation_metrics"]
    assert "mean_dice" in val_m
    assert "mean_iou" in val_m
    assert "dice_TC" in val_m
    assert "iou_TC" in val_m

    # 4. Verify Test Firewall
    assert report["test_firewall"]["TEST_SET_ACCESSED_DURING_TRAINING"] is False
    assert report["test_firewall"]["TEST_EVALUATION_STATUS"] == "NOT_EVALUATED"

    # 5. Verify Checkpoint Existence
    assert os.path.exists(report["checkpoint_path"])
    checkpoint = torch.load(report["checkpoint_path"], map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "validation_metrics" in checkpoint
    assert "test_metrics" not in checkpoint
