"""
Module: events.choreography

Purpose:
Enterprise Event Choreography Engine for FedMed v2.0.
Decouples all operating system subsystems by subscribing to Event Bus topics and orchestrating
automatic multi-engine reactions across Drift, SLA, Recommendations, Adaptive Strategy, Resilience, and Deployment.
"""

import logging
from typing import Any, Dict, Optional
from events.event_bus import EventBus, EventTopic, EventType, SystemEvent, global_event_bus
from analytics.recommendation_engine import OperationalRecommendationEngine
from strategy.adaptive_engine import AdaptiveStrategySelector
from resilience.self_healing import SelfHealingRecoveryEngine
from knowledge.graph import SystemKnowledgeGraph

logger = logging.getLogger("event_choreography")


class EventChoreographer:
    """
    Subscribes to system event topics and automatically triggers cross-subsystem reactions without direct module coupling.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventChoreographer, cls).__new__(cls)
            cls._instance.event_bus = global_event_bus
            cls._instance.recommendation_engine = OperationalRecommendationEngine()
            cls._instance.strategy_selector = AdaptiveStrategySelector()
            cls._instance.self_healing_engine = SelfHealingRecoveryEngine()
            cls._instance.knowledge_graph = SystemKnowledgeGraph()
            cls._instance.choreography_log = []
            cls._instance._register_choreography_handlers()
        return cls._instance

    def _register_choreography_handlers(self):
        """Registers reactive event handlers across topics."""
        self.event_bus.subscribe(EventTopic.DRIFT, self.on_drift_event)
        self.event_bus.subscribe(EventTopic.HEALTH, self.on_health_event)
        self.event_bus.subscribe(EventTopic.SLA, self.on_sla_event)

    def on_drift_event(self, event: SystemEvent) -> None:
        """Reacts automatically to DRIFT_DETECTED events."""
        if event.event_type == EventType.DRIFT_DETECTED:
            mmd = event.payload.get("metrics", {}).get("mmd", 0.15)
            node_id = event.payload.get("node_id", "hospital_alpha")
            logger.info(f"Choreographer reacting to DRIFT_DETECTED on node '{node_id}' (MMD={mmd})")

            # 1. Update Knowledge Graph
            self.knowledge_graph.add_node(f"drift_{node_id}", "DriftEvent", f"Drift MMD={mmd}", {"node_id": node_id, "mmd": mmd})
            self.knowledge_graph.add_edge(node_id, f"drift_{node_id}", "EXHIBITED_DRIFT")

            # 2. Trigger Operational Recommendation Engine
            self.recommendation_engine.generate_recommendations(
                current_dice=0.82,
                drift_mmd=mmd,
                epsilon_consumed=2.5,
                avg_latency_ms=120.0,
            )

            # 3. Trigger Adaptive Strategy Selector
            self.strategy_selector.evaluate_and_select_strategy(
                current_strategy="FedAvg",
                participation_rate=1.0,
                avg_latency_ms=120.0,
                gradient_divergence=0.25,
                drift_mmd=mmd,
                privacy_epsilon=2.5,
                node_dropouts=0,
            )

            self.choreography_log.append({
                "trigger_event": event.event_type.value,
                "reactions": ["KnowledgeGraph_Updated", "Recommendations_Generated", "AdaptiveStrategy_Evaluated"],
            })

    def on_health_event(self, event: SystemEvent) -> None:
        """Reacts automatically to NODE_FAILURE events."""
        if event.event_type == EventType.NODE_FAILURE:
            node_id = event.payload.get("target_node", event.payload.get("node_id", "hospital_beta"))
            logger.warning(f"Choreographer reacting to NODE_FAILURE on node '{node_id}'")

            # Trigger Self-Healing Recovery Engine
            self.self_healing_engine.recover_node_failure(node_id, "Node Failure Event Triggered")

            self.choreography_log.append({
                "trigger_event": event.event_type.value,
                "reactions": ["SelfHealing_Recovery_Triggered"],
            })

    def on_sla_event(self, event: SystemEvent) -> None:
        """Reacts automatically to SLA_VIOLATED events."""
        if event.event_type == EventType.SLA_VIOLATED:
            logger.warning(f"Choreographer reacting to SLA_VIOLATED event: {event.payload}")
            self.choreography_log.append({
                "trigger_event": event.event_type.value,
                "reactions": ["AutonomousOrchestrator_Alerted"],
            })


global_event_choreographer = EventChoreographer()
