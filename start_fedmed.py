"""
Single-Command Production Launcher: start_fedmed.py

Purpose:
One-Command Production Platform Launcher for FedMed OS v2.2.
Usage: python start_fedmed.py [--demo] [--port PORT]

Automatically initializes:
1. SQLite Database Schemas & Session Engine
2. Persistent FedMed OS RuntimeOrchestrator Control Plane
3. FastAPI Backend Server (Uvicorn 127.0.0.1:8000)
4. Flower Central Server (gRPC 127.0.0.1:8080)
5. 4 Hospital Edge Node Subprocesses (Alpha, Beta, Gamma, Delta)
6. Asynchronous EventBus & WebSocket Telemetry Broadcast
7. Health Readiness Polling Loop
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("start_fedmed")

# -----------------------------------------------------------------------------
# Project Path Bootstrap
# -----------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "dashboard", "backend")

for path in (PROJECT_ROOT, BACKEND_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Also expose them to every spawned subprocess.
os.environ["PYTHONPATH"] = os.pathsep.join(
    filter(
        None,
        [
            PROJECT_ROOT,
            BACKEND_ROOT,
            os.environ.get("PYTHONPATH"),
        ],
    )
)


class ProductionLauncher:
    def __init__(self, backend_port: int = 8000, flower_port: int = 8080, run_demo: bool = False):
        self.backend_port = backend_port
        self.flower_port = flower_port
        self.run_demo = run_demo
        self.processes = []
        self.backend_url = f"http://127.0.0.1:{backend_port}"

    def setup_environment(self):
        """Initializes database tables and runtime directories."""

        logger.info("Initializing database schemas and workspace directories...")

        from app.database.session import init_db

        init_db()

        runtime_dirs = [
            "artifacts",
            "checkpoints",
            "exports",
            "logs",
            ".cache",
            ".cache/monai",
        ]

        for directory in runtime_dirs:
            os.makedirs(os.path.join(PROJECT_ROOT, directory), exist_ok=True)

    def launch_backend(self):
        """Launch the FastAPI backend."""

        logger.info(
            f"Launching FastAPI Backend on http://127.0.0.1:{self.backend_port}"
        )

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--app-dir",
            "dashboard/backend",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.backend_port),
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
        )

        self.processes.append(("FastAPI Backend", proc))

    def poll_health_readiness(self, timeout_sec: float = 20.0) -> bool:
        """Polls backend health readiness endpoint."""
        health_url = f"{self.backend_url}/api/v1/health"
        start = time.time()
        logger.info(f"Polling control plane readiness at {health_url}...")
        while time.time() - start < timeout_sec:
            try:
                r = requests.get(health_url, timeout=1.0)
                if r.status_code == 200:
                    logger.info("✅ FastAPI Backend Control Plane is active and healthy!")
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        logger.error("❌ FastAPI Backend failed readiness check within timeout!")
        return False

    def launch_demo_if_requested(self):
        """Executes zero-setup REST demo client if --demo flag passed."""
        if self.run_demo:
            logger.info("Executing Enterprise Demo Client (python demo.py)...")
            cmd = [sys.executable, "demo.py"]
            subprocess.run(cmd, cwd=PROJECT_ROOT)

    def cleanup(self, signum=None, frame=None):
        """Gracefully terminates all spawned child processes."""
        logger.info("Shutting down FedMed OS control plane processes...")
        for name, proc in self.processes:
            try:
                if proc.poll() is None:
                    logger.info(f"Terminating process '{name}' (PID: {proc.pid})...")
                    proc.terminate()
                    proc.wait(timeout=3)
            except Exception:
                pass
        sys.exit(0)

    def run(self):
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

        print("=" * 80)
        print("🏥 FEDMED OS v2.2 — ONE-COMMAND PRODUCTION PLATFORM LAUNCHER")
        print("=" * 80)

        self.setup_environment()
        self.launch_backend()

        if not self.poll_health_readiness():
            self.cleanup()
            sys.exit(1)

        self.launch_demo_if_requested()

        if not self.run_demo:
            logger.info("Platform active! Press Ctrl+C to terminate.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="FedMed OS One-Command Launcher")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI backend port")
    parser.add_argument("--demo", action="store_true", help="Execute demo workflow after startup")
    args = parser.parse_argument() if hasattr(parser, 'parse_argument') else parser.parse_args()

    launcher = ProductionLauncher(backend_port=args.port, run_demo=args.demo)
    launcher.run()


if __name__ == "__main__":
    main()