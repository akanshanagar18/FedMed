"""
Module: tests.e2e.test_autonomous_os_e2e

Purpose:
End-to-End Simulation Test for FedMed Autonomous Federated Learning Operating System.
Verifies complete lifecycle: Drift $\rightarrow$ SLA $\rightarrow$ Recommendations $\rightarrow$ Orchestration $\rightarrow$ Adaptive Strategy $\rightarrow$ Deployment $\rightarrow$ Knowledge Graph $\rightarrow$ Executive Report.
"""

import pytest
import numpy as np
from events.event_bus import global_event_bus, EventTopic
from governance.drift import FederatedDriftDetector
from governance.sla import InstitutionalSLAAuditor
from analytics.rca_engine import RootCauseAnalysisEngine
from analytics.recommendation_engine import OperationalRecommendationEngine
from analytics.experiment_planner import AutonomousExperimentPlanner
from analytics.executive_reports import ExecutiveAnalyticsEngine
from strategy.adaptive_engine import AdaptiveStrategySelector
from orchestrator.autonomous_orchestrator import AutonomousOrchestrator, OrchestratorDecision
from resilience.self_healing import SelfHealingRecoveryEngine
from deployment.manager import ProductionDeploymentManager, DeploymentStrategy
from knowledge.graph import SystemKnowledgeGraph


def test_autonomous_os_e2e_lifecycle():
    # 1. Feature Drift Evaluation
    np.random.seed(42)
    drift_detector = FederatedDriftDetector()
    ref_data = np.random.normal(0, 1, size=(100, 32))
    cur_data = np.random.normal(0.02, 1.01, size=(100, 32))
    drift_res = drift_detector.detect_drift(ref_data, cur_data, node_id="hospital_alpha")
    assert "metrics" in drift_res

    # 2. SLA Audit
    sla_auditor = InstitutionalSLAAuditor()
    sla_res = sla_auditor.audit_compliance(
        run_id="e2e_auto_run_001",
        epsilon_consumed=2.1,
        delta_consumed=1e-5,
        participating_nodes=["hospital_alpha", "hospital_beta"],
        total_nodes=["hospital_alpha", "hospital_beta"],
        avg_latency_ms=115.0,
    )
    assert sla_res["overall_compliant"] is True

    # 3. Operational Recommendation Engine
    rec_engine = OperationalRecommendationEngine()
    recs = rec_engine.generate_recommendations(
        current_dice=0.86,
        drift_mmd=drift_res["metrics"]["mmd"],
        epsilon_consumed=2.1,
        avg_latency_ms=115.0,
        candidate_dice=0.88,
    )
    assert len(recs) >= 1

    # 4. Autonomous Orchestrator Decision
    orchestrator = AutonomousOrchestrator()
    orch_res = orchestrator.evaluate_system_state_and_decide(
        current_round=5,
        training_loss=0.17,
        validation_dice=0.86,
        drift_mmd=drift_res["metrics"]["mmd"],
        sla_compliant=sla_res["overall_compliant"],
        active_nodes_count=2,
        total_nodes_count=2,
        privacy_epsilon=2.1,
        candidate_dice=0.88,
    )
    assert orch_res["decision"] in [OrchestratorDecision.CONTINUE_TRAINING.value, OrchestratorDecision.PROMOTE_MODEL.value]

    # 5. Adaptive Strategy Selector
    strat_selector = AdaptiveStrategySelector()
    strat_res = strat_selector.evaluate_and_select_strategy(
        current_strategy="FedAvg",
        participation_rate=1.0,
        avg_latency_ms=115.0,
        gradient_divergence=0.04,
        drift_mmd=drift_res["metrics"]["mmd"],
        privacy_epsilon=2.1,
        node_dropouts=0,
    )
    assert strat_res["recommended_strategy"] in ["FedAvg", "Scaffold"]

    # 6. Production Deployment Manager
    dep_manager = ProductionDeploymentManager()
    dep = dep_manager.initiate_deployment(
        model_id="brats_monai_3d_unet",
        version="v2.1.0",
        strategy=DeploymentStrategy.CANARY,
        candidate_metrics={"mean_dice": 0.88, "hd95": 3.4},
    )
    promo = dep_manager.promote_deployment(dep["deployment_id"])
    assert promo["success"] is True

    # 7. Knowledge Graph Inspection
    kg = SystemKnowledgeGraph()
    kg.add_node("model_v2_1", "Model", "BraTS MONAI UNet v2.1.0", {"mean_dice": 0.88})
    kg.add_edge("model_v2_1", dep["deployment_id"], "DEPLOYED_VIA")
    summary = kg.get_summary()
    assert summary["total_nodes"] >= 6

    # 8. Executive Report Generation
    exec_engine = ExecutiveAnalyticsEngine()
    report = exec_engine.generate_executive_summary()
    assert "kpis" in report
    assert report["kpis"]["mean_segmentation_dice"] > 0.80

    # 9. Event Bus History Verification
    history = global_event_bus.get_history(limit=50)
    assert len(history) >= 3
