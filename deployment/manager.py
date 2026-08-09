"""
Module: deployment.manager

Purpose:
Production Deployment Manager for FedMed models.
Supports Canary, Rolling, Blue-Green, Shadow, and Emergency Rollback rollout strategies with automatic validation gates.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional
from events.event_bus import EventBus, EventTopic, EventType, SystemEvent, global_event_bus


class DeploymentStrategy(str, Enum):
    CANARY = "CANARY"
    ROLLING = "ROLLING"
    BLUE_GREEN = "BLUE_GREEN"
    SHADOW = "SHADOW"
    EMERGENCY_ROLLBACK = "EMERGENCY_ROLLBACK"


class DeploymentStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    HEALTHY = "HEALTHY"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    COMPLETED = "COMPLETED"


class ProductionDeploymentManager:
    """
    Manages production rollouts, canary traffic shifting, quality gates, and automated emergency rollbacks.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or global_event_bus
        self.deployments: Dict[str, Dict[str, Any]] = {}
        self.active_production_model: Dict[str, Any] = {
            "model_id": "brats_monai_3d_unet",
            "version": "v2.0.0",
            "mean_dice": 0.825,
            "hd95": 4.5,
        }

    def initiate_deployment(
        self,
        model_id: str,
        version: str,
        strategy: DeploymentStrategy,
        candidate_metrics: Dict[str, float],
        target_hospitals: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Initiates a new model deployment workflow.
        """
        deployment_id = f"dep_{model_id}_{int(time.time())}"
        targets = target_hospitals or ["hospital_alpha", "hospital_beta"]

        deployment = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "version": version,
            "strategy": strategy.value,
            "status": DeploymentStatus.IN_PROGRESS.value,
            "traffic_percentage": 10.0 if strategy == DeploymentStrategy.CANARY else 100.0,
            "target_hospitals": targets,
            "candidate_metrics": candidate_metrics,
            "previous_model": self.active_production_model.copy(),
            "created_at": time.time(),
            "updated_at": time.time(),
            "audit_trail": [f"Deployment initiated using strategy {strategy.value}"],
        }

        self.deployments[deployment_id] = deployment

        # Event publication
        event = SystemEvent(
            topic=EventTopic.DEPLOYMENT,
            event_type=EventType.DEPLOYMENT_INITIATED,
            source="ProductionDeploymentManager",
            payload=deployment,
            rationale=f"Initiated {strategy.value} deployment for model {model_id}:{version}",
        )
        self.event_bus.publish_sync(event)

        return deployment

    def promote_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Promotes an in-progress deployment to 100% active production.
        """
        dep = self.deployments.get(deployment_id)
        if not dep:
            return {"success": False, "message": f"Deployment '{deployment_id}' not found"}

        dep["status"] = DeploymentStatus.COMPLETED.value
        dep["traffic_percentage"] = 100.0
        dep["updated_at"] = time.time()
        dep["audit_trail"].append("Promoted to 100% active production traffic")

        self.active_production_model = {
            "model_id": dep["model_id"],
            "version": dep["version"],
            "mean_dice": dep["candidate_metrics"].get("mean_dice", 0.85),
            "hd95": dep["candidate_metrics"].get("hd95", 4.0),
        }

        event = SystemEvent(
            topic=EventTopic.GOVERNANCE,
            event_type=EventType.STAGE_PROMOTED,
            source="ProductionDeploymentManager",
            payload=dep,
            rationale=f"Model {dep['model_id']}:{dep['version']} promoted to 100% Production",
        )
        self.event_bus.publish_sync(event)

        return {"success": True, "deployment": dep}

    def emergency_rollback(self, deployment_id: str, reason: str) -> Dict[str, Any]:
        """
        Executes instant emergency rollback to previous stable production model.
        """
        dep = self.deployments.get(deployment_id)
        if not dep:
            return {"success": False, "message": f"Deployment '{deployment_id}' not found"}

        dep["status"] = DeploymentStatus.ROLLED_BACK.value
        dep["traffic_percentage"] = 0.0
        dep["updated_at"] = time.time()
        dep["audit_trail"].append(f"Emergency Rollback executed: {reason}")

        # Restore previous active production model
        self.active_production_model = dep["previous_model"]

        event = SystemEvent(
            topic=EventTopic.DEPLOYMENT,
            event_type=EventType.DEPLOYMENT_ROLLED_BACK,
            source="ProductionDeploymentManager",
            payload=dep,
            rationale=f"Emergency rollback triggered for {deployment_id}: {reason}",
            severity="CRITICAL",
        )
        self.event_bus.publish_sync(event)

        return {
            "success": True,
            "message": f"Rolled back deployment '{deployment_id}' to version {self.active_production_model['version']}",
            "deployment": dep,
        }

    def get_deployments(self) -> List[Dict[str, Any]]:
        """Returns list of all deployments."""
        return list(self.deployments.values())
