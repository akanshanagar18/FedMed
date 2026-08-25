"""
Module: tests.unit.test_workflow_engine

Purpose:
Unit test suite for Unified Workflow Engine (DAG instantiation, step execution, pause/resume, cancel).
"""

import pytest
from workflows.workflow_engine import WorkflowEngine
from workflows.models import WorkflowStatus


def test_workflow_instantiation_and_step_execution():
    engine = WorkflowEngine()
    inst = engine.instantiate_workflow("Enterprise_EndToEnd_FL_Pipeline")
    assert inst.status == WorkflowStatus.CREATED
    assert len(inst.steps) == 14

    res = engine.execute_workflow_step(inst.instance_id)
    assert res["success"] is True
    assert res["workflow_status"] == WorkflowStatus.RUNNING.value
    assert res["step_executed"] == "Hospital Registration"


def test_workflow_pause_resume_cancel():
    engine = WorkflowEngine()
    inst = engine.instantiate_workflow("Enterprise_EndToEnd_FL_Pipeline")
    engine.execute_workflow_step(inst.instance_id)

    pause_res = engine.pause_workflow(inst.instance_id)
    assert pause_res["success"] is True
    assert pause_res["status"] == WorkflowStatus.PAUSED.value

    resume_res = engine.resume_workflow(inst.instance_id)
    assert resume_res["success"] is True
    assert resume_res["status"] == WorkflowStatus.RUNNING.value

    cancel_res = engine.cancel_workflow(inst.instance_id)
    assert cancel_res["success"] is True
    assert cancel_res["status"] == WorkflowStatus.CANCELLED.value
