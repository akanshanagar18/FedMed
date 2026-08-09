"""
Module: tests.unit.test_scheduler_engine

Purpose:
Unit test suite for EnterpriseBackgroundScheduler (scheduling jobs, manual execution, job listing).
"""

import pytest
from scheduler.scheduler_engine import EnterpriseBackgroundScheduler, JobType, ScheduleType


def test_background_scheduler_operations():
    scheduler = EnterpriseBackgroundScheduler()
    job = scheduler.schedule_job(JobType.RETRAINING, ScheduleType.PERIODIC, 600)
    assert job.job_type == JobType.RETRAINING
    assert job.interval_seconds == 600

    exec_res = scheduler.execute_job(job.job_id)
    assert exec_res["success"] is True
    assert exec_res["execution"]["status"] == "SUCCESS"

    jobs = scheduler.list_jobs()
    assert len(jobs) >= 11
