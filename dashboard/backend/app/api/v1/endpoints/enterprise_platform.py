"""
Module: dashboard.backend.app.api.v1.endpoints.enterprise_platform

Purpose:
REST API Endpoints for Milestone T Enterprise Platform
(/api/v1/workflows/*, /api/v1/simulator/*, /api/v1/policies/*, /api/v1/scheduler/*, /api/v1/digital-twin/*, /api/v1/lifecycle/*, /api/v1/persistent-graph/*).
"""

import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status

from app.schemas.enterprise import (
    WorkflowCreateRequest,
    ScenarioTriggerRequest,
    JobScheduleRequest,
    DigitalTwinSimRequest,
    ExperimentCreateRequest,
    ExperimentTransitionRequest,
)
from common.schemas import SuccessResponse
from workflows.workflow_engine import global_workflow_engine
from simulation.scenario_engine import ScenarioSimulationEngine, ScenarioType
from configs.policy_engine import global_policy_engine
from scheduler.scheduler_engine import EnterpriseBackgroundScheduler, JobType, ScheduleType
from digital_twin.twin_engine import DigitalTwinSimulationEngine
from experiments.lifecycle_manager import ExperimentLifecycleManager, ExperimentStage
from knowledge.persistent_graph import PersistentKnowledgeGraph

router = APIRouter()

# Singletons
scenario_engine = ScenarioSimulationEngine()
scheduler_engine = EnterpriseBackgroundScheduler()
digital_twin_engine = DigitalTwinSimulationEngine()
lifecycle_manager = ExperimentLifecycleManager()
persistent_kg = PersistentKnowledgeGraph()


# --- WORKFLOW ENDPOINTS ---

@router.post("/workflows/instantiate", response_model=SuccessResponse)
async def instantiate_workflow(req: WorkflowCreateRequest):
    """Instantiates a new unified enterprise workflow DAG."""
    inst = global_workflow_engine.instantiate_workflow(req.workflow_name)
    return SuccessResponse(
        message=f"Workflow instance '{inst.instance_id}' created successfully",
        data=inst.to_dict(),
    )


@router.post("/workflows/{instance_id}/step", response_model=SuccessResponse)
async def step_workflow(instance_id: str):
    """Executes the next pending step in a workflow instance."""
    res = global_workflow_engine.execute_workflow_step(instance_id)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return SuccessResponse(
        message=f"Workflow step executed successfully",
        data=res,
    )


@router.post("/workflows/{instance_id}/run", response_model=SuccessResponse)
async def run_workflow(instance_id: str):
    """Runs a workflow instance completely to completion."""
    res = global_workflow_engine.run_entire_workflow(instance_id)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return SuccessResponse(
        message=f"Workflow '{instance_id}' executed to completion",
        data=res["instance"],
    )


@router.post("/workflows/{instance_id}/pause", response_model=SuccessResponse)
async def pause_workflow(instance_id: str):
    """Pauses an in-progress workflow."""
    res = global_workflow_engine.pause_workflow(instance_id)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return SuccessResponse(message="Workflow paused", data=res)


@router.post("/workflows/{instance_id}/resume", response_model=SuccessResponse)
async def resume_workflow(instance_id: str):
    """Resumes a paused workflow."""
    res = global_workflow_engine.resume_workflow(instance_id)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return SuccessResponse(message="Workflow resumed", data=res)


@router.get("/workflows/instances", response_model=SuccessResponse)
async def list_workflows():
    """Lists all workflow instances."""
    instances = global_workflow_engine.list_instances()
    return SuccessResponse(
        message="Workflow instances retrieved successfully",
        data={"total_instances": len(instances), "instances": instances},
    )


# --- SIMULATOR ENDPOINTS ---

@router.post("/simulator/trigger", response_model=SuccessResponse)
async def trigger_scenario(req: ScenarioTriggerRequest):
    """Triggers an operational incident scenario simulation."""
    try:
        stype = ScenarioType(req.scenario_type.upper())
    except ValueError:
        stype = ScenarioType.HOSPITAL_FAILURE

    res = scenario_engine.trigger_scenario(stype, req.target_node, req.custom_params)
    return SuccessResponse(
        message=f"Scenario '{stype.value}' triggered successfully",
        data=res,
    )


@router.get("/simulator/history", response_model=SuccessResponse)
async def get_simulation_history():
    """Returns history of executed scenario simulations."""
    history = scenario_engine.get_history()
    return SuccessResponse(
        message="Simulation history retrieved",
        data={"total_simulations": len(history), "history": history},
    )


# --- POLICY ENDPOINTS ---

@router.get("/policies", response_model=SuccessResponse)
async def get_policies():
    """Returns active enterprise policy configurations."""
    policies = global_policy_engine.policies
    return SuccessResponse(
        message="Enterprise policies retrieved",
        data={"reload_count": global_policy_engine.reload_count, "policies": policies},
    )


@router.post("/policies/reload", response_model=SuccessResponse)
async def reload_policies():
    """Hot-reloads policy configurations from disk."""
    policies = global_policy_engine.reload_policies()
    return SuccessResponse(
        message="Enterprise policies reloaded from disk",
        data={"reload_count": global_policy_engine.reload_count, "policies": policies},
    )


# --- SCHEDULER ENDPOINTS ---

@router.post("/scheduler/jobs", response_model=SuccessResponse)
async def schedule_job(req: JobScheduleRequest):
    """Schedules a new background job."""
    try:
        jtype = JobType(req.job_type.upper())
    except ValueError:
        jtype = JobType.DRIFT_SCAN

    try:
        stype = ScheduleType(req.schedule_type.upper())
    except ValueError:
        stype = ScheduleType.PERIODIC

    job = scheduler_engine.schedule_job(jtype, stype, req.interval_seconds, req.cron_expression)
    return SuccessResponse(
        message=f"Scheduled background job '{job.job_id}' successfully",
        data=job.to_dict(),
    )


@router.get("/scheduler/jobs", response_model=SuccessResponse)
async def list_scheduled_jobs():
    """Lists all registered background jobs."""
    jobs = scheduler_engine.list_jobs()
    return SuccessResponse(
        message="Scheduled jobs retrieved",
        data={"total_jobs": len(jobs), "jobs": jobs},
    )


@router.post("/scheduler/jobs/{job_id}/execute", response_model=SuccessResponse)
async def execute_job(job_id: str):
    """Manually triggers execution of a scheduled job."""
    res = scheduler_engine.execute_job(job_id)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return SuccessResponse(
        message=f"Job '{job_id}' executed successfully",
        data=res["execution"],
    )


# --- DIGITAL TWIN ENDPOINTS ---

@router.post("/digital-twin/predict", response_model=SuccessResponse)
async def predict_digital_twin(req: DigitalTwinSimRequest):
    """Executes a predictive Digital Twin 'What-If' simulation."""
    res = digital_twin_engine.simulate_what_if(
        scenario_description=req.scenario_description,
        baseline_dice=req.baseline_dice,
        hospital_dropout_pct=req.hospital_dropout_pct,
        latency_multiplier=req.latency_multiplier,
        drift_mmd=req.drift_mmd,
        privacy_noise_multiplier=req.privacy_noise_multiplier,
    )
    return SuccessResponse(
        message="Digital Twin prediction computed successfully",
        data=res,
    )


# --- LIFECYCLE ENDPOINTS ---

@router.post("/lifecycle/create", response_model=SuccessResponse)
async def create_experiment_lifecycle(req: ExperimentCreateRequest):
    """Creates a new experiment lifecycle instance."""
    record = lifecycle_manager.create_experiment(req.name, req.owner, req.dataset_version)
    return SuccessResponse(
        message=f"Experiment '{record['experiment_id']}' created",
        data=record,
    )


@router.post("/lifecycle/{experiment_id}/transition", response_model=SuccessResponse)
async def transition_experiment_stage(experiment_id: str, req: ExperimentTransitionRequest):
    """Transitions an experiment to a target lifecycle stage."""
    try:
        tstage = ExperimentStage(req.target_stage.upper())
    except ValueError:
        tstage = ExperimentStage.TRAINING

    res = lifecycle_manager.transition_stage(experiment_id, tstage, req.notes, req.metrics)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return SuccessResponse(
        message=f"Experiment transitioned to '{tstage.value}'",
        data=res["experiment"],
    )


@router.get("/lifecycle/experiments", response_model=SuccessResponse)
async def list_experiment_lifecycles():
    """Lists all experiment lifecycle records."""
    exps = lifecycle_manager.list_experiments()
    return SuccessResponse(
        message="Experiment lifecycles retrieved",
        data={"total_experiments": len(exps), "experiments": exps},
    )


# --- PERSISTENT GRAPH TIME-TRAVEL ENDPOINT ---

@router.get("/persistent-graph/time-travel", response_model=SuccessResponse)
async def query_time_travel_graph(timestamp: Optional[float] = None):
    """Queries persistent Knowledge Graph state as of a historical timestamp."""
    ts = timestamp or time.time()
    res = persistent_kg.query_as_of(ts)
    return SuccessResponse(
        message=f"Persistent Knowledge Graph time-travel query completed for timestamp {ts}",
        data=res,
    )
