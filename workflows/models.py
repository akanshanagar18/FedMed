"""
Module: workflows.models

Purpose:
Data models and Enums for the Unified Workflow Engine in FedMed v2.0.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional
import uuid


class WorkflowStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass
class WorkflowStep:
    step_id: str
    name: str
    action_target: str  # Function or service handler name
    depends_on: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.CREATED
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "action_target": self.action_target,
            "depends_on": self.depends_on,
            "parameters": self.parameters,
            "status": self.status.value,
            "result_data": self.result_data,
            "error_message": self.error_message,
        }


@dataclass
class WorkflowDefinition:
    workflow_name: str
    description: str
    steps: List[WorkflowStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class WorkflowInstance:
    instance_id: str = field(default_factory=lambda: f"wf_inst_{uuid.uuid4().hex[:8]}")
    workflow_name: str = "Standard_FL_Pipeline"
    status: WorkflowStatus = WorkflowStatus.CREATED
    current_step_index: int = 0
    steps: List[WorkflowStep] = field(default_factory=list)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "steps": [s.to_dict() for s in self.steps],
            "execution_context": self.execution_context,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
