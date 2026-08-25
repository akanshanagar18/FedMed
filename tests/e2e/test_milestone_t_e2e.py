"""
Module: tests.e2e.test_milestone_t_e2e

Purpose:
End-to-End Simulation Test for Milestone T Enterprise Workflow Integration Platform.
Simulates canonical 14-step workflow lifecycle, scenario incident triggering, policy evaluation,
scheduler execution, persistent graph time-travel, digital twin prediction, and event choreography.
"""

import time
import pytest
from workflows.workflow_engine import WorkflowEngine, WorkflowStatus
from simulation.scenario_engine import ScenarioSimulationEngine, ScenarioType
from configs.policy_engine import EnterprisePolicyEngine
from scheduler.scheduler_engine import EnterpriseBackgroundScheduler, JobType
from digital_twin.twin_engine import DigitalTwinSimulationEngine
from experiments.lifecycle_manager import ExperimentLifecycleManager, ExperimentStage
from knowledge.persistent_graph import PersistentKnowledgeGraph
from events.choreography import global_event_choreographer


def test_milestone_t_e2e_full_platform_lifecycle():
    # 1. Instantiate 14-step Enterprise DAG Workflow
    wf_engine = WorkflowEngine()
    inst = wf_engine.instantiate_workflow("Enterprise_EndToEnd_FL_Pipeline")
    assert len(inst.steps) == 14

    # Run entire workflow to completion
    run_res = wf_engine.run_entire_workflow(inst.instance_id)
    assert run_res["success"] is True
    assert run_res["instance"]["status"] == WorkflowStatus.COMPLETED.value

    # 2. Trigger Scenario Incident (Hospital Failure)
    sim_engine = ScenarioSimulationEngine()
    sim_res = sim_engine.trigger_scenario(ScenarioType.HOSPITAL_FAILURE, "hospital_beta")
    assert sim_res["status"] == "EXECUTED"

    # Verify event choreography reaction recorded log
    choreography_log = global_event_choreographer.choreography_log
    assert len(choreography_log) >= 1

    # 3. Enterprise Policy Engine Lookup & Hot-Reload
    policy_engine = EnterprisePolicyEngine()
    policies = policy_engine.reload_policies()
    assert "governance" in policies
    assert policy_engine.get_policy("sla", "max_epsilon") == 10.0

    # 4. Background Scheduler Execution
    sched_engine = EnterpriseBackgroundScheduler()
    job = sched_engine.schedule_job(JobType.DRIFT_SCAN, interval_seconds=300)
    exec_res = sched_engine.execute_job(job.job_id)
    assert exec_res["success"] is True

    # 5. Experiment Lifecycle Manager Stage Transitions
    exp_mgr = ExperimentLifecycleManager()
    exp = exp_mgr.create_experiment("Milestone T 3D BraTS E2E Run")
    exp_id = exp["experiment_id"]
    exp_mgr.transition_stage(exp_id, ExperimentStage.TRAINING, "Started 10 rounds")
    exp_mgr.transition_stage(exp_id, ExperimentStage.VALIDATION, "Validation Dice=0.875")
    exp_mgr.transition_stage(exp_id, ExperimentStage.DEPLOYMENT, "Promoted to Canary")
    final_exp = exp_mgr.get_experiment(exp_id)
    assert final_exp["stage"] == ExperimentStage.DEPLOYMENT.value

    # 6. Persistent Knowledge Graph & Time-Travel Query
    pkg = PersistentKnowledgeGraph()
    pkg.add_node("exp_milestone_t", "Experiment", "Milestone T E2E Run", {"dice": 0.875})
    query_res = pkg.query_as_of(time.time())
    assert query_res["total_nodes"] >= 3

    # 7. Digital Twin Simulation Prediction
    twin_engine = DigitalTwinSimulationEngine()
    twin_res = twin_engine.simulate_what_if(
        scenario_description="Hospital Beta disconnects and latency doubles",
        baseline_dice=0.875,
        hospital_dropout_pct=0.33,
        latency_multiplier=2.0,
    )
    assert "predicted_metrics" in twin_res
    assert twin_res["predicted_metrics"]["expected_dice"] < 0.875
