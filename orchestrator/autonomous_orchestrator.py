"""
Module: orchestrator.autonomous_orchestrator

Purpose:
Autonomous Federated Orchestrator for FedMed v2.0 Operating System.
Evaluates platform telemetry across all subsystems and makes explainable autonomous decisions:
CONTINUE, PAUSE, RESUME, TRIGGER_RETRAINING, ABORT_ROUND, PROMOTE_MODEL, ROLLBACK_DEPLOYMENT, LAUNCH_HPO, NOTIFY_ADMIN.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional
from events.event_bus import EventBus, EventTopic, EventType, SystemEvent, global_event_bus


class OrchestratorDecision(str, Enum):
    CONTINUE_TRAINING = "CONTINUE_TRAINING"
    PAUSE_TRAINING = "PAUSE_TRAINING"
    RESUME_TRAINING = "RESUME_TRAINING"
    TRIGGER_RETRAINING = "TRIGGER_RETRAINING"
    ABORT_ROUND = "ABORT_ROUND"
    PROMOTE_MODEL = "PROMOTE_MODEL"
    ROLLBACK_DEPLOYMENT = "ROLLBACK_DEPLOYMENT"
    LAUNCH_HPO = "LAUNCH_HPO"
    NOTIFY_ADMIN = "NOTIFY_ADMIN"


class AutonomousOrchestrator:
    """
    Intelligent Autonomous Orchestrator Engine.
    Continuously evaluates system telemetry across Drift, SLA, Governance, Metrics, Privacy, and Health.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or global_event_bus
        self.decision_history: List[Dict[str, Any]] = []

    def evaluate_system_state_and_decide(
        self,
        current_round: int,
        training_loss: float,
        validation_dice: float,
        drift_mmd: float,
        sla_compliant: bool,
        active_nodes_count: int,
        total_nodes_count: int,
        privacy_epsilon: float,
        candidate_dice: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates system telemetry and makes an explainable autonomous decision.
        """
        decision = OrchestratorDecision.CONTINUE_TRAINING
        rationale = "System metrics stable; continuing federated training round."
        severity = "INFO"

        if not sla_compliant or privacy_epsilon > 10.0:
            decision = OrchestratorDecision.PAUSE_TRAINING
            rationale = "HIPAA/GDPR SLA compliance check failed or privacy budget exhausted. Training paused automatically."
            severity = "CRITICAL"
        elif active_nodes_count == 0:
            decision = OrchestratorDecision.ABORT_ROUND
            rationale = "Zero hospital nodes connected. Current round aborted automatically."
            severity = "HIGH"
        elif drift_mmd > 0.30:
            decision = OrchestratorDecision.TRIGGER_RETRAINING
            rationale = f"CRITICAL feature drift detected (MMD={drift_mmd:.4f} > 0.30). Retraining local hospital cohorts."
            severity = "WARNING"
        elif validation_dice < 0.80:
            decision = OrchestratorDecision.LAUNCH_HPO
            rationale = f"Validation Dice score ({validation_dice:.4f}) below threshold. Triggering Federated Hyperparameter Optimization."
            severity = "WARNING"
        elif candidate_dice and candidate_dice >= validation_dice + 0.015:
            decision = OrchestratorDecision.PROMOTE_MODEL
            rationale = f"Candidate model Dice ({candidate_dice:.4f}) exceeds active model ({validation_dice:.4f}). Triggering stage promotion."
            severity = "INFO"

        result = {
            "decision_id": f"dec_{current_round}_{int(time.time())}",
            "decision": decision.value,
            "current_round": current_round,
            "rationale": rationale,
            "severity": severity,
            "timestamp": time.time(),
            "telemetry": {
                "training_loss": round(training_loss, 4),
                "validation_dice": round(validation_dice, 4),
                "drift_mmd": round(drift_mmd, 4),
                "sla_compliant": sla_compliant,
                "active_nodes": f"{active_nodes_count}/{total_nodes_count}",
                "privacy_epsilon": round(privacy_epsilon, 2),
            },
        }

        self.decision_history.append(result)

        # Publish event
        event = SystemEvent(
            topic=EventTopic.ORCHESTRATOR,
            event_type=EventType.AUTONOMOUS_DECISION,
            source="AutonomousOrchestrator",
            payload=result,
            rationale=rationale,
            severity=severity,
        )
        self.event_bus.publish_sync(event)

        return result

    def get_decisions(self) -> List[Dict[str, Any]]:
        """Returns log of all executed autonomous decisions."""
        return self.decision_history
