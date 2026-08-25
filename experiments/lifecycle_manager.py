"""
Module: experiments.lifecycle_manager

Purpose:
Experiment Lifecycle Manager for FedMed v2.0.
Orchestrates stage transitions across the full lifecycle:
DATASET_PREPARATION -> TRAINING -> VALIDATION -> GOVERNANCE -> DEPLOYMENT -> MONITORING -> RETIREMENT -> ARCHIVE.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional
from events.event_bus import EventBus, EventTopic, EventType, SystemEvent, global_event_bus


class ExperimentStage(str, Enum):
    DATASET_PREPARATION = "DATASET_PREPARATION"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    GOVERNANCE = "GOVERNANCE"
    DEPLOYMENT = "DEPLOYMENT"
    MONITORING = "MONITORING"
    RETIREMENT = "RETIREMENT"
    ARCHIVE = "ARCHIVE"


class ExperimentLifecycleManager:
    """
    Manages experiment metadata, stage transitions, lineage records, and archival.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExperimentLifecycleManager, cls).__new__(cls)
            cls._instance.experiments: Dict[str, Dict[str, Any]] = {}
            cls._instance.event_bus = global_event_bus
        return cls._instance

    def create_experiment(
        self,
        name: str,
        owner: str = "research_team",
        dataset_version: str = "BraTS2021_v1",
    ) -> Dict[str, Any]:
        """Creates a new experiment lifecycle instance."""
        exp_id = f"exp_{name.lower().replace(' ', '_')}_{int(time.time())}"
        record = {
            "experiment_id": exp_id,
            "name": name,
            "owner": owner,
            "stage": ExperimentStage.DATASET_PREPARATION.value,
            "dataset_version": dataset_version,
            "created_at": time.time(),
            "updated_at": time.time(),
            "metrics_summary": {},
            "stage_history": [{
                "stage": ExperimentStage.DATASET_PREPARATION.value,
                "timestamp": time.time(),
                "notes": "Experiment initialized",
            }],
        }
        self.experiments[exp_id] = record
        return record

    def transition_stage(
        self,
        experiment_id: str,
        target_stage: ExperimentStage,
        notes: str = "",
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Transitions an experiment to a target lifecycle stage.
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            return {"success": False, "message": f"Experiment '{experiment_id}' not found"}

        old_stage = exp["stage"]
        exp["stage"] = target_stage.value
        exp["updated_at"] = time.time()
        if metrics:
            exp["metrics_summary"].update(metrics)

        exp["stage_history"].append({
            "stage": target_stage.value,
            "timestamp": time.time(),
            "notes": notes or f"Transitioned from {old_stage} to {target_stage.value}",
        })

        # Event publication
        event = SystemEvent(
            topic=EventTopic.GOVERNANCE,
            event_type=EventType.STAGE_PROMOTED,
            source="ExperimentLifecycleManager",
            payload=exp,
            rationale=f"Experiment '{experiment_id}' transitioned to stage {target_stage.value}.",
        )
        self.event_bus.publish_sync(event)

        return {"success": True, "experiment": exp}

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves experiment details."""
        return self.experiments.get(experiment_id)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Lists all experiments."""
        return list(self.experiments.values())
