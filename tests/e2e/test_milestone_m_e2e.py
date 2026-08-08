"""
Module: tests.e2e.test_milestone_m_e2e

Purpose:
End-to-End verification test for Milestone M Production Operations, Observability, and Resilience Platform.
"""

import os
import pytest
from utils.logger import setup_structured_logger
from utils.telemetry import prometheus_registry, get_system_resource_metrics
from utils.resume_engine import ExperimentResumeEngine
from utils.checkpoint_registry import CheckpointRegistry
from utils.export_engine import ExportEngine


def test_milestone_m_e2e_operations_pipeline(tmp_path):
    log_dir = str(tmp_path / "logs")
    ckpt_dir = str(tmp_path / "checkpoints")
    results_dir = str(tmp_path / "results")

    # 1. Test Structured Logger
    logger = setup_structured_logger(name="e2e_logger", log_dir=log_dir)
    logger.info("Milestone M E2E Test Log Record", extra={"subsystem": "E2E", "experiment_id": "exp_m_001"})

    assert os.path.exists(os.path.join(log_dir, "fedmed.log"))

    # 2. Test Telemetry & Prometheus Exposition
    res = get_system_resource_metrics()
    assert res["cpu_percent"] >= 0.0
    prometheus_registry.set_metric("fedmed_e2e_test_gauge", 99.9)
    prom_str = prometheus_registry.generate_prometheus_text()
    assert "fedmed_e2e_test_gauge 99.9" in prom_str

    # 3. Test Checkpoint & Automatic Resume Engine
    registry = CheckpointRegistry(registry_dir=ckpt_dir)
    registry.save_checkpoint(
        model_state_dict={"param": [0.1, 0.2]},
        experiment_id="exp_m_001",
        filename="fl_round_1.pth",
        round=1,
        dice=0.85,
    )

    resume_engine = ExperimentResumeEngine(checkpoint_dir=ckpt_dir)
    resume_status = resume_engine.check_resume_status("exp_m_001")

    assert resume_status["should_resume"] is True
    assert resume_status["resume_round"] == 2
    assert resume_status["last_dice"] == 0.85

    # 4. Test ExportEngine Resource CSV Generation
    exporter = ExportEngine(output_dir=results_dir)
    files = exporter.export_benchmark_suite(
        benchmark_id="bm_m_e2e",
        name="Milestone M E2E Benchmark",
        results=[{"experiment_id": "exp_m_001", "strategy_name": "SCAFFOLD", "best_dice": 0.85}],
    )

    assert os.path.exists(files["resource_usage_csv"])
    assert os.path.exists(files["latency_csv"])
    assert os.path.exists(files["gpu_usage_csv"])
    assert os.path.exists(files["memory_usage_csv"])
