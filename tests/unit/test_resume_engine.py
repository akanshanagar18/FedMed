"""
Module: tests.unit.test_resume_engine

Purpose:
Unit test suite for automatic experiment resume engine (utils/resume_engine.py).
"""

import os
import pytest
from utils.checkpoint_registry import CheckpointRegistry
from utils.resume_engine import ExperimentResumeEngine


def test_resume_engine_no_previous_checkpoint(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    resume_engine = ExperimentResumeEngine(checkpoint_dir=ckpt_dir)

    status = resume_engine.check_resume_status("exp_non_existent")
    assert status["should_resume"] is False
    assert status["resume_round"] == 1


def test_resume_engine_with_saved_checkpoint(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    registry = CheckpointRegistry(registry_dir=ckpt_dir)

    # Save mock round 2 checkpoint
    registry.save_checkpoint(
        model_state_dict={"weight": [1.0, 2.0]},
        experiment_id="exp_resume_test",
        filename="fl_round_2.pth",
        round=2,
        dice=0.88,
    )

    resume_engine = ExperimentResumeEngine(checkpoint_dir=ckpt_dir)
    status = resume_engine.check_resume_status("exp_resume_test")

    assert status["should_resume"] is True
    assert status["resume_round"] == 3
    assert status["last_dice"] == 0.88
