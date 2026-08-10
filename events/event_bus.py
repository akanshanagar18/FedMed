"""
Module: events.event_bus

Purpose:
Asynchronous Pub/Sub Event Bus for FedMed v2.0 Autonomous Operating System.
Facilitates real-time event-driven communication between Drift, SLA, Governance, Orchestrator,
Adaptive Strategy, Deployment Manager, Self-Healing, and WebSockets telemetry.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("event_bus")


class EventTopic(str, Enum):
    METRICS = "metrics"
    DRIFT = "drift"
    SLA = "sla"
    GOVERNANCE = "governance"
    HEALTH = "health"
    PRIVACY = "privacy"
    ORCHESTRATOR = "orchestrator"
    STRATEGY = "strategy"
    DEPLOYMENT = "deployment"
    SELF_HEALING = "self_healing"
    RECOMMENDATION = "recommendation"
    EXPERIMENT = "experiment"
    WORKFLOW = "workflow"


class EventType(str, Enum):
    METRIC_UPDATED = "METRIC_UPDATED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    SLA_VIOLATED = "SLA_VIOLATED"
    SLA_PASSED = "SLA_PASSED"
    STAGE_PROMOTED = "STAGE_PROMOTED"
    NODE_FAILURE = "NODE_FAILURE"
    NODE_RECONNECTED = "NODE_RECONNECTED"
    PRIVACY_EXHAUSTED = "PRIVACY_EXHAUSTED"
    AUTONOMOUS_DECISION = "AUTONOMOUS_DECISION"
    STRATEGY_ADAPTED = "STRATEGY_ADAPTED"
    DEPLOYMENT_INITIATED = "DEPLOYMENT_INITIATED"
    DEPLOYMENT_ROLLED_BACK = "DEPLOYMENT_ROLLED_BACK"
    SELF_HEALING_TRIGGERED = "SELF_HEALING_TRIGGERED"
    RECOMMENDATION_ISSUED = "RECOMMENDATION_ISSUED"
    EXPERIMENT_CREATED = "EXPERIMENT_CREATED"
    EXPERIMENT_STARTED = "EXPERIMENT_STARTED"
    MODEL_CHECKPOINTED = "MODEL_CHECKPOINTED"
    WORKFLOW_EXECUTED = "WORKFLOW_EXECUTED"


@dataclass
class SystemEvent:
    topic: EventTopic
    event_type: EventType
    source: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    rationale: Optional[str] = None
    severity: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "topic": self.topic.value,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "datetime_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "payload": self.payload,
            "rationale": self.rationale,
            "severity": self.severity,
        }


class EventBus:
    """
    Asynchronous Pub/Sub Event Bus with topic subscriptions, event history buffer, and subscriber dispatching.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance.subscribers: Dict[EventTopic, List[Callable]] = {t: [] for t in EventTopic}
            cls._instance.history: List[SystemEvent] = []
            cls._instance.max_history: int = 1000
        return cls._instance

    def subscribe(self, topic: EventTopic, callback: Callable[[SystemEvent], Any]) -> None:
        """Subscribes a callback handler to a specific EventTopic."""
        if callback not in self.subscribers[topic]:
            self.subscribers[topic].append(callback)
            logger.info(f"Subscribed callback {callback.__name__} to topic '{topic.value}'")

    def publish_sync(self, event: SystemEvent) -> None:
        """Publishes an event synchronously to all topic subscribers and appends to history."""
        self.history.append(event)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        for callback in self.subscribers.get(event.topic, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber {callback.__name__}: {e}")

    async def publish(self, event: SystemEvent) -> None:
        """Asynchronously publishes an event to subscribers."""
        self.publish_sync(event)

    def get_history(self, topic: Optional[EventTopic] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent event history filtered by topic."""
        events = self.history
        if topic:
            events = [e for e in events if e.topic == topic]
        return [e.to_dict() for e in events[-limit:]]


global_event_bus = EventBus()
