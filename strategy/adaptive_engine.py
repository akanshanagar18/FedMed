"""
Module: strategy.adaptive_engine

Purpose:
Adaptive Strategy Engine for dynamic FL algorithm switching (FedAvg, FedProx, FedNova, FedAdam, FedYogi, Scaffold, Mime).
Evaluates client participation, latency, gradient divergence, feature drift, privacy budget, and stability.
"""

import time
from typing import Any, Dict, List, Optional
from events.event_bus import EventBus, EventTopic, EventType, SystemEvent, global_event_bus


class AdaptiveStrategySelector:
    """
    Intelligent Adaptive Strategy Selector.
    Evaluates real-time training telemetry and automatically selects the optimal strategy.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or global_event_bus
        self.adaptation_history: List[Dict[str, Any]] = []

    def select_strategy(self, current_strategy: str = "FedAvg", dropout_rate: float = 0.0, drift_detected: bool = False, latency_variance: float = 0.0, participation_rate: float = 1.0, avg_latency_ms: float = 120.0, gradient_divergence: float = 0.05, drift_mmd: float = 0.042, privacy_epsilon: float = 2.5, node_dropouts: int = 0) -> tuple:
        """Alias for evaluate_and_select_strategy."""
        res = self.evaluate_and_select_strategy(
            current_strategy=current_strategy,
            participation_rate=participation_rate,
            avg_latency_ms=avg_latency_ms,
            gradient_divergence=gradient_divergence,
            drift_mmd=drift_mmd,
            privacy_epsilon=privacy_epsilon,
            node_dropouts=node_dropouts,
        )
        return res["recommended_strategy"], res["rationale"]


    def evaluate_and_select_strategy(

        self,
        current_strategy: str,
        participation_rate: float,
        avg_latency_ms: float,
        gradient_divergence: float,
        drift_mmd: float,
        privacy_epsilon: float,
        node_dropouts: int,
    ) -> Dict[str, Any]:
        """
        Evaluates system metrics and recommends/selects the optimal strategy.
        """
        recommended_strategy = current_strategy
        rationale = "Current strategy remains optimal based on telemetry."

        # Rule evaluation
        if node_dropouts > 0 or participation_rate < 0.70:
            recommended_strategy = "FedNova"
            rationale = "High client dropout or unequal local steps detected. FedNova compensates for heterogeneous updates."
        elif drift_mmd > 0.15 or gradient_divergence > 0.25:
            recommended_strategy = "Scaffold"
            rationale = "High feature drift / client gradient variance detected. Scaffold applies control variate variance reduction."
        elif avg_latency_ms > 1000.0:
            recommended_strategy = "FedProx"
            rationale = "High network latency observed. FedProx allows partial local work with proximal regularization."
        elif privacy_epsilon > 8.0:
            recommended_strategy = "FedAdam"
            rationale = "Heavy differential privacy noise present. Server-side adaptive momentum (FedAdam) stabilizes noisy gradients."
        else:
            recommended_strategy = "FedAvg"
            rationale = "System state is stable and IID-homogenous. Plain FedAvg minimizes communication overhead."

        strategy_changed = (recommended_strategy != current_strategy)

        result = {
            "current_strategy": current_strategy,
            "recommended_strategy": recommended_strategy,
            "strategy_changed": strategy_changed,
            "rationale": rationale,
            "timestamp": time.time(),
            "telemetry_evaluated": {
                "participation_rate": round(participation_rate, 4),
                "avg_latency_ms": round(avg_latency_ms, 2),
                "gradient_divergence": round(gradient_divergence, 4),
                "drift_mmd": round(drift_mmd, 4),
                "privacy_epsilon": round(privacy_epsilon, 2),
                "node_dropouts": node_dropouts,
            },
        }

        if strategy_changed:
            self.adaptation_history.append(result)
            event = SystemEvent(
                topic=EventTopic.STRATEGY,
                event_type=EventType.STRATEGY_ADAPTED,
                source="AdaptiveStrategySelector",
                payload=result,
                rationale=rationale,
            )
            self.event_bus.publish_sync(event)

        return result
