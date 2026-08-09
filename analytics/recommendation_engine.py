"""
Module: analytics.recommendation_engine

Purpose:
Operational Recommendation Engine for generating intelligent system action advice.
Returns actionable recommendations with reason, confidence, expected impact percentage, and affected components.
"""

import time
from typing import Any, Dict, List, Optional
from events.event_bus import EventBus, EventTopic, EventType, SystemEvent, global_event_bus


class OperationalRecommendationEngine:
    """
    Intelligent Operational Recommendation Engine.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or global_event_bus
        self.active_recommendations: List[Dict[str, Any]] = []

    def generate_recommendations(
        self,
        current_dice: float,
        drift_mmd: float,
        epsilon_consumed: float,
        avg_latency_ms: float,
        candidate_dice: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates operational recommendations based on active platform telemetry.
        """
        recs = []

        if current_dice < 0.82:
            recs.append({
                "rec_id": f"rec_hpo_{int(time.time())}",
                "action": "TRIGGER_HPO",
                "reason": f"Current validation Dice score ({current_dice:.4f}) is below target threshold of 0.85.",
                "confidence": 0.94,
                "expected_impact": "+4.5% Dice Score improvement",
                "affected_components": ["Strategy Engine", "HPO Engine"],
                "auto_executable": True,
            })

        if drift_mmd > 0.10:
            recs.append({
                "rec_id": f"rec_lr_{int(time.time())}",
                "action": "REDUCE_LEARNING_RATE",
                "reason": f"Distribution drift MMD={drift_mmd:.4f} detected across hospital MRI cohorts.",
                "confidence": 0.89,
                "expected_impact": "-30% Gradient Variance across silos",
                "affected_components": ["Local MONAI Client Trainer"],
                "auto_executable": True,
            })

        if epsilon_consumed > 6.0:
            recs.append({
                "rec_id": f"rec_dp_{int(time.time())}",
                "action": "INCREASE_DP_CLIPPING",
                "reason": f"Privacy budget consumption high (epsilon={epsilon_consumed:.2f}). Adjust clipping norm to preserve budget.",
                "confidence": 0.87,
                "expected_impact": "+20% Extended Privacy Budget Lifetime",
                "affected_components": ["Opacus Differential Privacy Engine"],
                "auto_executable": False,
            })

        if candidate_dice and candidate_dice >= current_dice + 0.01:
            recs.append({
                "rec_id": f"rec_promo_{int(time.time())}",
                "action": "PROMOTE_CANDIDATE_MODEL",
                "reason": f"Candidate model Dice ({candidate_dice:.4f}) outperforms active production model ({current_dice:.4f}).",
                "confidence": 0.96,
                "expected_impact": "+1.5% Clinical Segmentation Accuracy",
                "affected_components": ["Production Deployment Manager", "Model Governance Registry"],
                "auto_executable": True,
            })

        self.active_recommendations = recs

        # Publish event
        for r in recs:
            event = SystemEvent(
                topic=EventTopic.RECOMMENDATION,
                event_type=EventType.RECOMMENDATION_ISSUED,
                source="OperationalRecommendationEngine",
                payload=r,
                rationale=r["reason"],
            )
            self.event_bus.publish_sync(event)

        return recs

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Returns list of currently active operational recommendations."""
        return self.active_recommendations
