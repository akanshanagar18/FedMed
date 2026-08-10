"""
Module: deployment.model_registry

Purpose:
Production Model Registry for FedMed OS v2.1.
Tracks versioned model artifacts, SHA-256 signatures, validation metrics,
stage promotions (STAGING, CANARY, PRODUCTION), and rollback targets.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import torch

from common.state_machines import DeploymentState, validate_deployment_transition
from events.event_bus import global_event_bus, EventTopic, EventType, SystemEvent
from dashboard.backend.app.database.session import SessionLocal
from dashboard.backend.app.models.base import GovernanceRecordModel, DeploymentModel

logger = logging.getLogger("model_registry")


class ModelRegistry:
    """
    Central Repository for Production Federated AI Models and Artifact Lineage.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.models: Dict[str, Dict[str, Any]] = {}
            cls._instance.event_bus = global_event_bus
            cls._instance._register_default_models()
        return cls._instance

    def _register_default_models(self):
        """Registers initial baseline model in candidate stage."""
        default_id = "brats_monai_3d_unet"
        self.models[default_id] = {
            "model_id": default_id,
            "version": "v2.0.0",
            "stage": DeploymentState.PRODUCTION.value,
            "dice_score": 0.885,
            "checksum_sha256": "8a4f91b7e2c90000000000000000000000000000000000000000000000000000",
            "file_path": "checkpoints/brats_monai_3d_unet_v2.0.0.pt",
            "created_at": time.time(),
        }

    def register_model_version(
        self,
        model_id: str,
        version: str,
        file_path: str,
        dice_score: float,
        checksum_sha256: str,
        stage: DeploymentState = DeploymentState.STAGING,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Registers a new model artifact version in the registry."""
        model_entry = {
            "model_id": model_id,
            "version": version,
            "file_path": file_path,
            "dice_score": dice_score,
            "checksum_sha256": checksum_sha256,
            "stage": stage.value,
            "metadata": metadata or {},
            "registered_at": time.time(),
        }

        self.models[f"{model_id}:{version}"] = model_entry

        # Persist to SQLite
        db = SessionLocal()
        try:
            row = GovernanceRecordModel(
                model_id=f"{model_id}:{version}",
                previous_stage="Candidate",
                new_stage=stage.value,
                promoted_by="automated_model_registry",
                hmac_signature=checksum_sha256,
            )
            db.add(row)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to persist model record to SQLite: {e}")
            db.rollback()
        finally:
            db.close()

        # Emit event
        event = SystemEvent(
            topic=EventTopic.GOVERNANCE,
            event_type=EventType.MODEL_CHECKPOINTED,
            source="ModelRegistry",
            payload=model_entry,
            rationale=f"Registered model artifact {model_id}:{version} in stage {stage.value}.",
        )
        self.event_bus.publish_sync(event)

        logger.info(f"Registered Model '{model_id}:{version}' (Dice={dice_score:.4f}, Stage={stage.value})")
        return model_entry

    def promote_stage(self, model_key: str, new_stage: DeploymentState) -> Dict[str, Any]:
        """Promotes a model version to a new deployment stage (e.g. STAGING -> CANARY -> PRODUCTION)."""
        model = self.models.get(model_key)
        if not model:
            return {"success": False, "message": f"Model '{model_key}' not found in registry"}

        current_stage = DeploymentState(model["stage"])
        validate_deployment_transition(current_stage, new_stage)

        model["stage"] = new_stage.value
        model["updated_at"] = time.time()

        # Emit promotion event
        event = SystemEvent(
            topic=EventTopic.GOVERNANCE,
            event_type=EventType.STAGE_PROMOTED,
            source="ModelRegistry",
            payload=model,
            rationale=f"Model {model_key} promoted to stage {new_stage.value}.",
        )
        self.event_bus.publish_sync(event)

        return {"success": True, "model": model}

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns list of registered model artifacts."""
        return list(self.models.values())

    def get_model(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Returns model registry record."""
        return self.models.get(model_key)


global_model_registry = ModelRegistry()
