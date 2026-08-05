"""
Module: scripts.run_simulation

Purpose:
One-command End-to-End Orchestrator for the FedMed platform.
Starts:
  1. FastAPI Backend Server (Port 8000)
  2. Flower Central Server (Port 8080)
  3. Hospital Client A (Flower NumPyClient)
  4. Hospital Client B (Flower NumPyClient)

Usage:
  python scripts/run_simulation.py
"""

import os
import sys
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [SIMULATION] - %(message)s")
logger = logging.getLogger("simulation")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "dashboard", "backend")


def main():
    logger.info("==========================================================")
    logger.info(" Starting FedMed End-to-End Simulation Pipeline")
    logger.info("==========================================================")

    processes = []

    try:
        # 1. Start FastAPI Monitoring Backend
        logger.info("1. Bootstrapping FastAPI Monitoring Backend on port 8000...")
        backend_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--app-dir",
                BACKEND_DIR,
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=PROJECT_ROOT,
        )
        processes.append(backend_proc)
        time.sleep(3)  # Wait for DB init and server startup

        # 2. Start Flower Server
        logger.info("2. Launching Flower Central Aggregation Server on port 8080...")
        server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "server.flower_server",
                "--address",
                "127.0.0.1:8080",
                "--rounds",
                "3",
                "--min-clients",
                "2",
            ],
            cwd=PROJECT_ROOT,
        )
        processes.append(server_proc)
        time.sleep(2)

        # 3. Start Hospital Client A
        logger.info("3. Connecting Hospital Node Alpha (hospital_a)...")
        client_a_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "client.flower_client",
                "--server",
                "127.0.0.1:8080",
                "--hospital-id",
                "hospital_a",
            ],
            cwd=PROJECT_ROOT,
        )
        processes.append(client_a_proc)

        # 4. Start Hospital Client B
        logger.info("4. Connecting Hospital Node Beta (hospital_b)...")
        client_b_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "client.flower_client",
                "--server",
                "127.0.0.1:8080",
                "--hospital-id",
                "hospital_b",
            ],
            cwd=PROJECT_ROOT,
        )
        processes.append(client_b_proc)

        logger.info("==========================================================")
        logger.info(" Simulation Pipeline Active! Running 3 FL Rounds...")
        logger.info(" Dashboard API: http://127.0.0.1:8000/docs")
        logger.info(" WebSocket Stream: ws://127.0.0.1:8000/api/v1/telemetry/ws")
        logger.info("==========================================================")

        # Wait for Flower server to complete 3 rounds
        server_proc.wait()
        logger.info("SUCCESS: Federated Learning Simulation completed 3 rounds!")

    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    finally:
        logger.info("Cleaning up background processes...")
        for p in processes:
            if p.poll() is None:
                p.terminate()
        logger.info("All simulation processes terminated.")


if __name__ == "__main__":
    main()
