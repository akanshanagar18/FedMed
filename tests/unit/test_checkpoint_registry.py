"""
Module: tests.unit.test_checkpoint_registry

Purpose:
Unit test suite for CheckpointRegistry module.
"""

import os
import torch
import pytest
from utils.checkpoint_registry import CheckpointRegistry, compute_sha256


def test_checkpoint_registry_save_and_list(tmp_path):
    registry_dir = str(tmp_path / "checkpoints")
    registry = CheckpointRegistry(registry_dir=registry_dir)

    dummy_state = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    meta1 = registry.save_checkpoint(
        model_state_dict=dummy_state,
        experiment_id="exp_01",
        filename="model_r1.pth",
        strategy="FedAvg",
        round=1,
        dice=0.82,
        loss=0.25,
    )

    assert meta1 is not None
    assert os.path.exists(meta1.file_path)
    assert meta1.file_hash != ""

    # Save second checkpoint with better dice
    meta2 = registry.save_checkpoint(
        model_state_dict=dummy_state,
        experiment_id="exp_01",
        filename="model_r2.pth",
        strategy="FedAvg",
        round=2,
        dice=0.89,
        loss=0.18,
    )

    checkpoints = registry.list_checkpoints(experiment_id="exp_01")
    assert len(checkpoints) == 2

    latest = registry.get_latest_checkpoint(experiment_id="exp_01")
    assert latest.round == 2

    best = registry.get_best_checkpoint(experiment_id="exp_01", metric="dice")
    assert best.dice == 0.89
    assert best.round == 2


def test_checkpoint_resume(tmp_path):
    registry_dir = str(tmp_path / "checkpoints")
    registry = CheckpointRegistry(registry_dir=registry_dir)

    dummy_state = {"param": torch.tensor([10.0])}
    saved_meta = registry.save_checkpoint(
        model_state_dict=dummy_state,
        experiment_id="resume_exp",
        filename="resume.pth",
        epoch=5,
        dice=0.91,
    )

    payload, meta = registry.load_checkpoint_for_resume(saved_meta.checkpoint_id)
    assert "model_state_dict" in payload
    assert torch.equal(payload["model_state_dict"]["param"], torch.tensor([10.0]))
    assert meta.dice == 0.91
