"""
Module: server.experiment_manager

Purpose:
Enterprise Experiment Lifecycle Manager for FedMed OS v2.1.
Responsible for creating, tracking, persisting, pausing, resuming, cancelling, retrying,
and archiving federated learning experiments driven by WorkflowEngine DAGs and EventBus.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from common.state_machines import ExperimentState, validate_experiment_transition
from events.event_bus import global_event_bus, EventTopic, EventType, SystemEvent
from workflows.workflow_engine import global_workflow_engine, WorkflowInstance
from dashboard.backend.app.database.session import SessionLocal
from dashboard.backend.app.models.base import ExperimentModel, TrainingMetricModel

logger = logging.getLogger("experiment_manager")


class ExperimentManager:
    """
    Manages complete experiment lifecycle and runtime execution.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExperimentManager, cls).__new__(cls)
            cls._instance.active_experiments: Dict[str, Dict[str, Any]] = {}
            cls._instance.workflow_engine = global_workflow_engine
            cls._instance.event_bus = global_event_bus
        return cls._instance

    def create_experiment(
        self,
        name: str,
        strategy_name: str = "FedAvg",
        num_clients: int = 2,
        num_rounds: int = 3,
        learning_rate: float = 1e-4,
        dp_enabled: bool = False,
        he_enabled: bool = False,
        description: str = "",
        experiment_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Creates a new experiment and attaches a workflow DAG.
        """
        exp_id = experiment_id or f"exp_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        init_status = status or ExperimentState.CREATED.value
        
        # Instantiate workflow DAG
        wf_inst = self.workflow_engine.instantiate_workflow("Enterprise_EndToEnd_FL_Pipeline")

        exp_data = {
            "experiment_id": exp_id,
            "name": name,
            "description": description,
            "strategy_name": strategy_name,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "learning_rate": learning_rate,
            "dp_enabled": dp_enabled,
            "he_enabled": he_enabled,
            "state": init_status,
            "status": init_status,
            "workflow_instance_id": wf_inst.instance_id,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        self.active_experiments[exp_id] = exp_data

        # Persist to SQLite
        db = SessionLocal()
        try:
            row = ExperimentModel(
                experiment_id=exp_id,
                name=name,
                description=description,
                strategy_name=strategy_name,
                num_clients=num_clients,
                num_rounds=num_rounds,
                learning_rate=learning_rate,
                dp_enabled=dp_enabled,
                he_enabled=he_enabled,
                status=ExperimentState.CREATED.value,
            )
            db.add(row)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to persist experiment '{exp_id}' to SQLite: {e}")
            db.rollback()
        finally:
            db.close()

        # Publish Event
        event = SystemEvent(
            topic=EventTopic.EXPERIMENT,
            event_type=EventType.EXPERIMENT_CREATED,
            source="ExperimentManager",
            payload=exp_data,
            rationale=f"Experiment '{name}' ({exp_id}) created cleanly.",
        )
        self.event_bus.publish_sync(event)

        logger.info(f"Created Experiment '{exp_id}' ({name}). State=CREATED.")
        return exp_data

    def start_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Starts experiment execution by triggering workflow engine execution."""
        exp = self.active_experiments.get(experiment_id)
        if not exp:
            exp = self._load_from_db(experiment_id)
            if not exp:
                return {"success": False, "message": f"Experiment '{experiment_id}' not found"}

        current_state = ExperimentState(exp["state"])
        validate_experiment_transition(current_state, ExperimentState.RUNNING)

        exp["state"] = ExperimentState.RUNNING.value
        exp["updated_at"] = time.time()

        # Update SQLite DB
        self._update_db_status(experiment_id, ExperimentState.RUNNING.value)

        # Trigger workflow DAG execution
        wf_id = exp.get("workflow_instance_id")
        if wf_id:
            self.workflow_engine.execute_workflow_step(wf_id)

        event = SystemEvent(
            topic=EventTopic.EXPERIMENT,
            event_type=EventType.EXPERIMENT_STARTED,
            source="ExperimentManager",
            payload=exp,
            rationale=f"Experiment '{experiment_id}' started execution.",
        )
        self.event_bus.publish_sync(event)

        return {"success": True, "experiment": exp}

    def pause_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Pauses a running experiment."""
        exp = self.active_experiments.get(experiment_id)
        if not exp:
            return {"success": False, "message": f"Experiment '{experiment_id}' not found"}

        current_state = ExperimentState(exp["state"])
        validate_experiment_transition(current_state, ExperimentState.PAUSED)

        exp["state"] = ExperimentState.PAUSED.value
        exp["updated_at"] = time.time()

        self._update_db_status(experiment_id, ExperimentState.PAUSED.value)
        
        wf_id = exp.get("workflow_instance_id")
        if wf_id:
            self.workflow_engine.pause_workflow(wf_id)

        return {"success": True, "experiment": exp}

    def resume_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Resumes a paused experiment."""
        exp = self.active_experiments.get(experiment_id)
        if not exp:
            return {"success": False, "message": f"Experiment '{experiment_id}' not found"}

        current_state = ExperimentState(exp["state"])
        validate_experiment_transition(current_state, ExperimentState.RUNNING)

        exp["state"] = ExperimentState.RUNNING.value
        exp["updated_at"] = time.time()

        self._update_db_status(experiment_id, ExperimentState.RUNNING.value)
        
        wf_id = exp.get("workflow_instance_id")
        if wf_id:
            self.workflow_engine.resume_workflow(wf_id)

        return {"success": True, "experiment": exp}

    def cancel_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Cancels an experiment."""
        exp = self.active_experiments.get(experiment_id)
        if not exp:
            return {"success": False, "message": f"Experiment '{experiment_id}' not found"}

        exp["state"] = ExperimentState.FAILED.value
        exp["updated_at"] = time.time()

        self._update_db_status(experiment_id, ExperimentState.FAILED.value)

        wf_id = exp.get("workflow_instance_id")
        if wf_id:
            self.workflow_engine.cancel_workflow(wf_id)

        return {"success": True, "experiment": exp}

    def archive_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Archives an experiment."""
        exp = self.active_experiments.get(experiment_id)
        if not exp:
            return {"success": False, "message": f"Experiment '{experiment_id}' not found"}

        current_state = ExperimentState(exp["state"])
        validate_experiment_transition(current_state, ExperimentState.ARCHIVED)

        exp["state"] = ExperimentState.ARCHIVED.value
        exp["updated_at"] = time.time()

        self._update_db_status(experiment_id, ExperimentState.ARCHIVED.value)
        return {"success": True, "experiment": exp}

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Returns experiment details dictionary."""
        exp = self.active_experiments.get(experiment_id)
        if not exp:
            exp = self._load_from_db(experiment_id)
        return exp

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Lists all active and persisted experiments."""
        db = SessionLocal()
        try:
            rows = db.query(ExperimentModel).all()
            results = []
            for r in rows:
                results.append({
                    "experiment_id": r.experiment_id,
                    "name": r.name,
                    "description": r.description,
                    "strategy_name": r.strategy_name,
                    "num_clients": r.num_clients,
                    "num_rounds": r.num_rounds,
                    "status": r.status,
                    "best_dice_score": r.best_dice_score,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                })
            return results
        finally:
            db.close()

    def _update_db_status(self, experiment_id: str, status: str) -> None:
        db = SessionLocal()
        try:
            row = db.query(ExperimentModel).filter(ExperimentModel.experiment_id == experiment_id).first()
            if row:
                row.status = status
                db.commit()
        except Exception as e:
            logger.error(f"Error updating DB status for '{experiment_id}': {e}")
            db.rollback()
        finally:
            db.close()

    def delete_experiment(self, experiment_id: str) -> bool:
        """Deletes an experiment record by ID."""
        if experiment_id in self.active_experiments:
            del self.active_experiments[experiment_id]
        db = SessionLocal()
        try:
            row = db.query(ExperimentModel).filter(ExperimentModel.experiment_id == experiment_id).first()
            if row:
                db.delete(row)
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting experiment '{experiment_id}': {e}")
            db.rollback()
            return False
    def _load_from_db(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        db = SessionLocal()
        try:
            row = db.query(ExperimentModel).filter(ExperimentModel.experiment_id == experiment_id).first()
            if row:
                data = {
                    "experiment_id": row.experiment_id,
                    "name": row.name,
                    "description": row.description,
                    "strategy_name": row.strategy_name,
                    "num_clients": row.num_clients,
                    "num_rounds": row.num_rounds,
                    "learning_rate": row.learning_rate,
                    "dp_enabled": row.dp_enabled,
                    "he_enabled": row.he_enabled,
                    "state": row.status,
                    "status": row.status,
                    "created_at": row.start_time.timestamp() if row.start_time else time.time(),
                }
                self.active_experiments[experiment_id] = data
                return data
            return None
        finally:
            db.close()


global_experiment_manager = ExperimentManager()
