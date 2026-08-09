"""
Module: simulation.scenario_engine

Purpose:
Enterprise Scenario Simulation Engine for FedMed v2.0.
Simulates 20 distinct enterprise failure, attack, load, and topology change scenarios,
emitting event streams to trigger automatic reactions across all operating system subsystems.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional
from events.event_bus import EventBus, EventTopic, EventType, SystemEvent, global_event_bus


class ScenarioType(str, Enum):
    HOSPITAL_FAILURE = "HOSPITAL_FAILURE"
    HOSPITAL_DISCONNECT = "HOSPITAL_DISCONNECT"
    HIGH_LATENCY = "HIGH_LATENCY"
    MODEL_POISONING = "MODEL_POISONING"
    GRADIENT_EXPLOSION = "GRADIENT_EXPLOSION"
    PRIVACY_BUDGET_EXHAUSTION = "PRIVACY_BUDGET_EXHAUSTION"
    EXTREME_CONCEPT_DRIFT = "EXTREME_CONCEPT_DRIFT"
    FEATURE_DRIFT = "FEATURE_DRIFT"
    COMMUNICATION_FAILURE = "COMMUNICATION_FAILURE"
    AGGREGATION_FAILURE = "AGGREGATION_FAILURE"
    SERVER_CRASH = "SERVER_CRASH"
    CHECKPOINT_CORRUPTION = "CHECKPOINT_CORRUPTION"
    DISK_FAILURE = "DISK_FAILURE"
    GPU_FAILURE = "GPU_FAILURE"
    MEMORY_EXHAUSTION = "MEMORY_EXHAUSTION"
    SLOW_CLIENT = "SLOW_CLIENT"
    MALICIOUS_CLIENT = "MALICIOUS_CLIENT"
    HOSPITAL_JOIN = "HOSPITAL_JOIN"
    HOSPITAL_LEAVE = "HOSPITAL_LEAVE"
    FEDERATED_SCALEUP = "FEDERATED_SCALEUP"


class ScenarioSimulationEngine:
    """
    Simulates enterprise operational incidents and scale events to test system resilience.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or global_event_bus
        self.simulation_history: List[Dict[str, Any]] = []

    def trigger_scenario(self, scenario_type: ScenarioType, target_node: str = "hospital_beta", custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a scenario simulation, emits event stream, and records simulation run.
        """
        params = custom_params or {}
        timestamp = time.time()
        sim_id = f"sim_{scenario_type.value.lower()}_{int(timestamp)}"

        # Map scenario to event topic & type
        if scenario_type in [ScenarioType.HOSPITAL_FAILURE, ScenarioType.HOSPITAL_DISCONNECT, ScenarioType.SERVER_CRASH]:
            topic = EventTopic.HEALTH
            event_type = EventType.NODE_FAILURE
            severity = "CRITICAL"
        elif scenario_type in [ScenarioType.FEATURE_DRIFT, ScenarioType.EXTREME_CONCEPT_DRIFT]:
            topic = EventTopic.DRIFT
            event_type = EventType.DRIFT_DETECTED
            severity = "WARNING"
        elif scenario_type in [ScenarioType.PRIVACY_BUDGET_EXHAUSTION]:
            topic = EventTopic.PRIVACY
            event_type = EventType.PRIVACY_EXHAUSTED
            severity = "CRITICAL"
        else:
            topic = EventTopic.HEALTH
            event_type = EventType.NODE_FAILURE
            severity = "WARNING"

        payload = {
            "simulation_id": sim_id,
            "scenario_type": scenario_type.value,
            "target_node": target_node,
            "params": params,
            "timestamp": timestamp,
            "affected_subsystems": ["Resilience", "RCA", "AdaptiveStrategy", "Orchestrator", "KnowledgeGraph"],
        }

        # Emit event to EventBus
        event = SystemEvent(
            topic=topic,
            event_type=event_type,
            source="ScenarioSimulationEngine",
            payload=payload,
            rationale=f"Simulated incident '{scenario_type.value}' triggered on '{target_node}'.",
            severity=severity,
        )
        self.event_bus.publish_sync(event)

        result = {
            "simulation_id": sim_id,
            "scenario_type": scenario_type.value,
            "status": "EXECUTED",
            "events_emitted": 1,
            "payload": payload,
        }

        self.simulation_history.append(result)
        return result

    def get_history(self) -> List[Dict[str, Any]]:

        return self.simulation_history
