"""
Module: common.state_machines

Purpose:
Explicit, deterministic State Machines and Transition Validation Engine for FedMed OS v2.1.
Enforces strict valid transition rules across Experiments, Hospitals, Training Rounds, and Deployments.
"""

from enum import Enum
import logging
from typing import Dict, Set

logger = logging.getLogger("state_machines")


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal state transition is attempted."""
    pass


class ExperimentState(str, Enum):
    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    RESUMING = "RESUMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class HospitalState(str, Enum):
    REGISTERING = "REGISTERING"
    CONNECTED = "CONNECTED"
    READY = "READY"
    TRAINING = "TRAINING"
    WAITING = "WAITING"
    OFFLINE = "OFFLINE"
    RECOVERING = "RECOVERING"
    ONLINE = "ONLINE"


class RoundState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AGGREGATING = "AGGREGATING"
    VALIDATING = "VALIDATING"
    CHECKPOINTING = "CHECKPOINTING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class DeploymentState(str, Enum):
    CREATED = "CREATED"
    STAGING = "STAGING"
    CANARY = "CANARY"
    PRODUCTION = "PRODUCTION"
    ROLLBACK = "ROLLBACK"
    RETIRED = "RETIRED"


# Valid Transition Rules Matrix
EXPERIMENT_TRANSITIONS: Dict[ExperimentState, Set[ExperimentState]] = {
    ExperimentState.CREATED: {ExperimentState.INITIALIZING, ExperimentState.READY, ExperimentState.RUNNING, ExperimentState.FAILED, ExperimentState.ARCHIVED},
    ExperimentState.INITIALIZING: {ExperimentState.READY, ExperimentState.FAILED},
    ExperimentState.READY: {ExperimentState.RUNNING, ExperimentState.PAUSED, ExperimentState.FAILED, ExperimentState.ARCHIVED},
    ExperimentState.RUNNING: {ExperimentState.PAUSED, ExperimentState.COMPLETED, ExperimentState.FAILED, ExperimentState.ARCHIVED},
    ExperimentState.PAUSED: {ExperimentState.RESUMING, ExperimentState.RUNNING, ExperimentState.FAILED, ExperimentState.ARCHIVED},
    ExperimentState.RESUMING: {ExperimentState.RUNNING, ExperimentState.FAILED},
    ExperimentState.COMPLETED: {ExperimentState.ARCHIVED},
    ExperimentState.FAILED: {ExperimentState.READY, ExperimentState.ARCHIVED},
    ExperimentState.ARCHIVED: set(),
}

HOSPITAL_TRANSITIONS: Dict[HospitalState, Set[HospitalState]] = {
    HospitalState.REGISTERING: {HospitalState.CONNECTED, HospitalState.OFFLINE},
    HospitalState.CONNECTED: {HospitalState.READY, HospitalState.OFFLINE},
    HospitalState.READY: {HospitalState.TRAINING, HospitalState.WAITING, HospitalState.OFFLINE},
    HospitalState.TRAINING: {HospitalState.WAITING, HospitalState.READY, HospitalState.OFFLINE},
    HospitalState.WAITING: {HospitalState.TRAINING, HospitalState.READY, HospitalState.OFFLINE},
    HospitalState.OFFLINE: {HospitalState.RECOVERING, HospitalState.CONNECTED},
    HospitalState.RECOVERING: {HospitalState.ONLINE, HospitalState.CONNECTED, HospitalState.OFFLINE},
    HospitalState.ONLINE: {HospitalState.READY, HospitalState.TRAINING, HospitalState.OFFLINE},
}

ROUND_TRANSITIONS: Dict[RoundState, Set[RoundState]] = {
    RoundState.QUEUED: {RoundState.RUNNING, RoundState.FAILED},
    RoundState.RUNNING: {RoundState.AGGREGATING, RoundState.FAILED},
    RoundState.AGGREGATING: {RoundState.VALIDATING, RoundState.FAILED},
    RoundState.VALIDATING: {RoundState.CHECKPOINTING, RoundState.FAILED},
    RoundState.CHECKPOINTING: {RoundState.FINISHED, RoundState.FAILED},
    RoundState.FINISHED: set(),
    RoundState.FAILED: {RoundState.QUEUED},
}

DEPLOYMENT_TRANSITIONS: Dict[DeploymentState, Set[DeploymentState]] = {
    DeploymentState.CREATED: {DeploymentState.STAGING, DeploymentState.RETIRED},
    DeploymentState.STAGING: {DeploymentState.CANARY, DeploymentState.PRODUCTION, DeploymentState.RETIRED},
    DeploymentState.CANARY: {DeploymentState.PRODUCTION, DeploymentState.ROLLBACK, DeploymentState.RETIRED},
    DeploymentState.PRODUCTION: {DeploymentState.ROLLBACK, DeploymentState.RETIRED},
    DeploymentState.ROLLBACK: {DeploymentState.STAGING, DeploymentState.RETIRED},
    DeploymentState.RETIRED: set(),
}


def validate_experiment_transition(current: ExperimentState, next_state: ExperimentState) -> bool:
    """Validates state transition for Experiment lifecycle."""
    if current == next_state:
        return True
    allowed = EXPERIMENT_TRANSITIONS.get(current, set())
    if next_state not in allowed:
        msg = f"Invalid Experiment state transition: '{current.value}' -> '{next_state.value}'"
        logger.error(msg)
        raise InvalidStateTransitionError(msg)
    return True


def validate_hospital_transition(current: HospitalState, next_state: HospitalState) -> bool:
    """Validates state transition for Hospital node lifecycle."""
    if current == next_state:
        return True
    allowed = HOSPITAL_TRANSITIONS.get(current, set())
    if next_state not in allowed:
        msg = f"Invalid Hospital state transition: '{current.value}' -> '{next_state.value}'"
        logger.error(msg)
        raise InvalidStateTransitionError(msg)
    return True


def validate_round_transition(current: RoundState, next_state: RoundState) -> bool:
    """Validates state transition for Training Round lifecycle."""
    if current == next_state:
        return True
    allowed = ROUND_TRANSITIONS.get(current, set())
    if next_state not in allowed:
        msg = f"Invalid Training Round state transition: '{current.value}' -> '{next_state.value}'"
        logger.error(msg)
        raise InvalidStateTransitionError(msg)
    return True


def validate_deployment_transition(current: DeploymentState, next_state: DeploymentState) -> bool:
    """Validates state transition for Deployment lifecycle."""
    if current == next_state:
        return True
    allowed = DEPLOYMENT_TRANSITIONS.get(current, set())
    if next_state not in allowed:
        msg = f"Invalid Deployment state transition: '{current.value}' -> '{next_state.value}'"
        logger.error(msg)
        raise InvalidStateTransitionError(msg)
    return True
