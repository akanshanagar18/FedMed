"""
Module: tests.e2e.test_production_demo_e2e

Purpose:
End-to-End Production Demo Test verifying complete zero-setup execution lifecycle.
"""

import pytest
from scripts.run_production_simulation import run_live_fl_simulation
from security.auth import global_auth_manager, Role
from dataset.manager import global_dataset_manager
from evaluation.benchmark_runner import global_benchmark_runner


def test_production_demo_e2e_execution():
    # 1. Live FL Simulation
    sim_res = run_live_fl_simulation(num_rounds=2)
    assert sim_res["success"] is True
    assert len(sim_res["round_summaries"]) == 2

    # 2. Authentication & Authorization
    user = global_auth_manager.authenticate_user("admin", "admin123")
    assert user is not None
    token = global_auth_manager.create_access_token(user["username"], user["role"])
    assert token is not None

    # 3. Dataset Management
    ds = global_dataset_manager.register_dataset("E2E_Test_Dataset", "v1.0", "3D_MRI", ["T1"], 50)
    assert global_dataset_manager.validate_checksum(ds["dataset_id"]) is True

    # 4. Benchmark Execution
    bm = global_benchmark_runner.run_benchmark_matrix(num_rounds=2, algorithms=["FedAvg", "FedProx"])
    assert bm["algorithms_evaluated"] == 2
