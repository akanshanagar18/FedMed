"""
Module: utils.resume_engine

Purpose:
Production Experiment Resume Engine for FedMed v2.0.
Inspects CheckpointRegistry to automatically detect interrupted runs and resume from the latest FL round.
"""

import os
import logging
from typing import Any, Dict, Optional, Tuple

from utils.checkpoint_registry import CheckpointRegistry, CheckpointMetadata

logger = logging.getLogger("resume_engine")


class ExperimentResumeEngine:
    """
    Manages automatic detection and resumption of interrupted training runs.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.registry = CheckpointRegistry(registry_dir=checkpoint_dir)

    def check_resume_status(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Inspects registry for the latest saved checkpoint for the given experiment ID (or global latest).
        
        Returns:
            Dict containing:
                "should_resume": bool,
                "resume_round": int,
                "latest_checkpoint_id": str,
                "latest_checkpoint_path": str,
                "last_dice": float,
                "last_loss": float,
        """
        latest_meta = self.registry.get_latest_checkpoint(experiment_id=experiment_id)
        if not latest_meta or not os.path.exists(latest_meta.file_path):
            return {
                "should_resume": False,
                "resume_round": 1,
                "latest_checkpoint_id": None,
                "latest_checkpoint_path": None,
                "last_dice": 0.0,
                "last_loss": 0.0,
            }

        last_round = latest_meta.round or latest_meta.epoch or 0
        resume_round = last_round + 1

        logger.info(
            f"[RESUME ENGINE] Found valid checkpoint '{latest_meta.checkpoint_id}' "
            f"(Round {last_round}, Dice={latest_meta.dice:.4f}). Next round: {resume_round}."
        )

        return {
            "should_resume": True,
            "resume_round": resume_round,
            "latest_checkpoint_id": latest_meta.checkpoint_id,
            "latest_checkpoint_path": latest_meta.file_path,
            "last_dice": latest_meta.dice,
            "last_loss": latest_meta.loss,
        }

    def load_checkpoint_for_resume(self, checkpoint_id_or_path: str) -> Tuple[Dict[str, Any], CheckpointMetadata]:
        """Loads PyTorch checkpoint weights and metadata for training continuation."""
        return self.registry.load_checkpoint_for_resume(checkpoint_id_or_path)
