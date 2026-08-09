"""
Demo Script: demo.py

Purpose:
Single-Command Automated Zero-Setup Enterprise Demo Launcher for FedMed v2.0.
Usage: python demo.py  (or  make demo)
Starts backend, launches hospital node services, executes live FL rounds, triggers feature drift,
demonstrates self-healing recovery, promotes best model, and generates executive C-level report.
"""

import os
import sys
import time
import logging

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from scripts.run_production_simulation import run_live_fl_simulation
from simulation.scenario_engine import ScenarioSimulationEngine, ScenarioType
from resilience.self_healing import SelfHealingRecoveryEngine
from deployment.manager import ProductionDeploymentManager, DeploymentStrategy
from analytics.executive_reports import ExecutiveAnalyticsEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("demo_launcher")


def main():
    print("=" * 80)
    print("🏥 FEDMED v2.0 — ENTERPRISE FEDERATED LEARNING OPERATING SYSTEM DEMO")
    print("=" * 80)

    # 1. Execute Live FL Rounds
    logger.info("\n[1/5] Launching Live Multi-Hospital Federated Learning Rounds...")
    sim_res = run_live_fl_simulation(num_rounds=3)
    logger.info(f"FL Simulation Finished! Instance ID: {sim_res['workflow_instance_id']}")

    # 2. Simulate Feature Drift Incident
    logger.info("\n[2/5] Simulating Cross-Silo Feature Drift Incident (MMD=0.18)...")
    scen_engine = ScenarioSimulationEngine()
    scen_res = scen_engine.trigger_scenario(ScenarioType.FEATURE_DRIFT, "hospital_alpha", {"mmd": 0.18})
    logger.info(f"Scenario Triggered: {scen_res['scenario_type']}")

    # 3. Trigger Self-Healing Recovery
    logger.info("\n[3/5] Triggering Self-Healing Recovery Engine...")
    sh_engine = SelfHealingRecoveryEngine()
    sh_res = sh_engine.recover_node_failure("hospital_alpha", "Feature Drift Compensation")
    logger.info(f"Self-Healing Message: {sh_res['message']}")

    # 4. Model Promotion
    logger.info("\n[4/5] Promoting Candidate Model via Production Deployment Manager...")
    dep_mgr = ProductionDeploymentManager()
    dep = dep_mgr.initiate_deployment(
        model_id="brats_monai_3d_unet",
        version="v2.1.0-demo",
        strategy=DeploymentStrategy.CANARY,
        candidate_metrics={"mean_dice": 0.885, "hd95": 3.2},
    )
    promo = dep_mgr.promote_deployment(dep["deployment_id"])
    logger.info(f"Model Promotion Result: {promo['message']}")

    # 5. Executive Report Generation
    logger.info("\n[5/5] Generating C-Level Executive Analytics Report...")
    exec_engine = ExecutiveAnalyticsEngine()
    report = exec_engine.generate_executive_summary(avg_dice_score=0.885)
    print("\n" + "=" * 80)
    print(report["markdown_report"])
    print("=" * 80)
    print("\n🎉 DEMO COMPLETED SUCCESSFULLY WITH ZERO MANUAL INTERVENTION!")


if __name__ == "__main__":
    main()
