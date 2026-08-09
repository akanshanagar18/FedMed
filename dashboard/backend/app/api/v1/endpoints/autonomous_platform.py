"""
Module: dashboard.backend.app.api.v1.endpoints.autonomous_platform

Purpose:
REST API endpoints for FedMed Autonomous Operating System (Orchestrator, Adaptive Strategy, RCA,
Recommendations, Experiment Planner, Self-Healing, Deployment Manager, Knowledge Graph, Executive Reports).
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status

from app.schemas.autonomous import (
    OrchestratorEvaluateRequest,
    AdaptiveStrategyRequest,
    RcaDiagnoseRequest,
    RecommendationRequest,
    DeploymentInitiateRequest,
    SelfHealingRequest,
)
from common.schemas import SuccessResponse
from orchestrator.autonomous_orchestrator import AutonomousOrchestrator
from strategy.adaptive_engine import AdaptiveStrategySelector
from analytics.rca_engine import RootCauseAnalysisEngine
from analytics.recommendation_engine import OperationalRecommendationEngine
from analytics.experiment_planner import AutonomousExperimentPlanner
from analytics.executive_reports import ExecutiveAnalyticsEngine
from resilience.self_healing import SelfHealingRecoveryEngine
from deployment.manager import ProductionDeploymentManager, DeploymentStrategy
from knowledge.graph import SystemKnowledgeGraph
from events.event_bus import global_event_bus

router = APIRouter()

# Singletons
orchestrator = AutonomousOrchestrator()
adaptive_strategy_selector = AdaptiveStrategySelector()
rca_engine = RootCauseAnalysisEngine()
recommendation_engine = OperationalRecommendationEngine()
experiment_planner = AutonomousExperimentPlanner()
executive_engine = ExecutiveAnalyticsEngine()
self_healing_engine = SelfHealingRecoveryEngine()
deployment_manager = ProductionDeploymentManager()
knowledge_graph = SystemKnowledgeGraph()


@router.get("/decisions", response_model=SuccessResponse)
async def get_autonomous_decisions():
    """Returns log of all executed autonomous orchestrator decisions."""
    decisions = orchestrator.get_decisions()
    return SuccessResponse(
        message="Autonomous decisions retrieved successfully",
        data={"total_decisions": len(decisions), "decisions": decisions},
    )


@router.post("/orchestrate", response_model=SuccessResponse)
async def evaluate_orchestration(req: OrchestratorEvaluateRequest):
    """Evaluates multi-subsystem platform telemetry and triggers autonomous decision."""
    res = orchestrator.evaluate_system_state_and_decide(
        current_round=req.current_round,
        training_loss=req.training_loss,
        validation_dice=req.validation_dice,
        drift_mmd=req.drift_mmd,
        sla_compliant=req.sla_compliant,
        active_nodes_count=req.active_nodes_count,
        total_nodes_count=req.total_nodes_count,
        privacy_epsilon=req.privacy_epsilon,
        candidate_dice=req.candidate_dice,
    )
    return SuccessResponse(
        message="Autonomous orchestration evaluation completed",
        data=res,
    )


@router.post("/adaptive-strategy/recommend", response_model=SuccessResponse)
async def recommend_adaptive_strategy(req: AdaptiveStrategyRequest):
    """Evaluates telemetry and dynamically selects optimal FL strategy (FedAvg, FedProx, Scaffold, FedNova, FedAdam)."""
    res = adaptive_strategy_selector.evaluate_and_select_strategy(
        current_strategy=req.current_strategy,
        participation_rate=req.participation_rate,
        avg_latency_ms=req.avg_latency_ms,
        gradient_divergence=req.gradient_divergence,
        drift_mmd=req.drift_mmd,
        privacy_epsilon=req.privacy_epsilon,
        node_dropouts=req.node_dropouts,
    )
    return SuccessResponse(
        message="Adaptive strategy recommendation computed",
        data=res,
    )


@router.post("/rca/diagnose", response_model=SuccessResponse)
async def diagnose_root_cause(req: RcaDiagnoseRequest):
    """Diagnoses probable root causes of training degradation or loss spikes."""
    res = rca_engine.diagnose_degradation(
        current_loss=req.current_loss,
        previous_loss=req.previous_loss,
        current_dice=req.current_dice,
        previous_dice=req.previous_dice,
        drift_mmd=req.drift_mmd,
        active_clients=req.active_clients,
        total_clients=req.total_clients,
        dp_epsilon=req.dp_epsilon,
        avg_latency_ms=req.avg_latency_ms,
    )
    return SuccessResponse(
        message="Root cause analysis diagnosis completed",
        data=res,
    )


@router.post("/recommendations", response_model=SuccessResponse)
async def generate_recommendations(req: RecommendationRequest):
    """Generates intelligent operational recommendations."""
    recs = recommendation_engine.generate_recommendations(
        current_dice=req.current_dice,
        drift_mmd=req.drift_mmd,
        epsilon_consumed=req.epsilon_consumed,
        avg_latency_ms=req.avg_latency_ms,
        candidate_dice=req.candidate_dice,
    )
    return SuccessResponse(
        message="Operational recommendations generated",
        data={"total_recommendations": len(recs), "recommendations": recs},
    )


@router.get("/recommendations", response_model=SuccessResponse)
async def get_active_recommendations():
    """Lists currently active operational recommendations."""
    recs = recommendation_engine.get_recommendations()
    return SuccessResponse(
        message="Active recommendations retrieved",
        data={"total_recommendations": len(recs), "recommendations": recs},
    )


@router.get("/experiment-planner/propose", response_model=SuccessResponse)
async def propose_experiments():
    """Generates autonomous future experiment proposals."""
    plan = experiment_planner.propose_next_experiments()
    return SuccessResponse(
        message="Autonomous experiment proposals generated",
        data=plan,
    )


@router.post("/self-healing/recover", response_model=SuccessResponse)
async def trigger_self_healing(req: SelfHealingRequest):
    """Triggers automated self-healing recovery workflow for node failure."""
    res = self_healing_engine.recover_node_failure(
        node_id=req.node_id,
        failure_reason=req.failure_reason,
    )
    return SuccessResponse(
        message=f"Self-healing workflow executed for node '{req.node_id}'",
        data=res,
    )


@router.get("/deployments", response_model=SuccessResponse)
async def list_deployments():
    """Lists all model deployments."""
    deps = deployment_manager.get_deployments()
    return SuccessResponse(
        message="Deployments retrieved successfully",
        data={"total_deployments": len(deps), "deployments": deps},
    )


@router.post("/deployments/initiate", response_model=SuccessResponse)
async def initiate_deployment(req: DeploymentInitiateRequest):
    """Initiates a production model deployment (Canary, Rolling, Blue-Green, Shadow)."""
    try:
        strat = DeploymentStrategy(req.strategy.upper())
    except ValueError:
        strat = DeploymentStrategy.CANARY

    dep = deployment_manager.initiate_deployment(
        model_id=req.model_id,
        version=req.version,
        strategy=strat,
        candidate_metrics=req.candidate_metrics,
        target_hospitals=req.target_hospitals,
    )
    return SuccessResponse(
        message=f"Deployment '{dep['deployment_id']}' initiated successfully",
        data=dep,
    )


@router.post("/deployments/{deployment_id}/promote", response_model=SuccessResponse)
async def promote_deployment(deployment_id: str):
    """Promotes a deployment to 100% active production traffic."""
    res = deployment_manager.promote_deployment(deployment_id)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return SuccessResponse(
        message=f"Deployment '{deployment_id}' promoted to 100% Production",
        data=res["deployment"],
    )


@router.post("/deployments/{deployment_id}/rollback", response_model=SuccessResponse)
async def rollback_deployment(deployment_id: str, reason: str = "Automated quality gate rollback"):
    """Triggers instant emergency rollback of deployment to previous stable version."""
    res = deployment_manager.emergency_rollback(deployment_id, reason)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return SuccessResponse(
        message=res["message"],
        data=res["deployment"],
    )


@router.get("/knowledge-graph/summary", response_model=SuccessResponse)
async def get_knowledge_graph_summary():
    """Returns System Knowledge Graph summary metrics and sample node/edge topology."""
    summary = knowledge_graph.get_summary()
    return SuccessResponse(
        message="Knowledge Graph summary retrieved",
        data=summary,
    )


@router.get("/knowledge-graph/node/{node_id}", response_model=SuccessResponse)
async def query_knowledge_graph_node(node_id: str):
    """Queries connected relationships and metadata for a specific Knowledge Graph node."""
    res = knowledge_graph.query_relationships(node_id)
    return SuccessResponse(
        message=f"Relationships for node '{node_id}' retrieved",
        data=res,
    )


@router.get("/executive-reports/generate", response_model=SuccessResponse)
async def generate_executive_report():
    """Generates comprehensive C-level executive report summary."""
    report = executive_engine.generate_executive_summary()
    return SuccessResponse(
        message="Executive analytics report generated successfully",
        data=report,
    )


@router.get("/events/history", response_model=SuccessResponse)
async def get_event_bus_history(limit: int = 50):
    """Returns recent Event Bus published events history."""
    events = global_event_bus.get_history(limit=limit)
    return SuccessResponse(
        message="Event Bus history retrieved",
        data={"total_events": len(events), "events": events},
    )
