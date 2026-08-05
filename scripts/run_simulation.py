"""
Module: scripts.run_simulation

Purpose:
Production-Grade End-to-End Orchestrator for the FedMed platform.
Handles process bootstrapping, active readiness polling, port collision detection,
structured logging, cross-platform signal handling, and clean shutdown.

Usage:
  python scripts/run_simulation.py
"""

import dataclasses
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    """Production configuration parameters for the orchestrator."""
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    flower_host: str = "127.0.0.1"
    flower_port: int = 8080
    num_rounds: int = 3
    min_clients: int = 2
    startup_timeout_sec: float = 15.0
    poll_interval_sec: float = 0.5
    shutdown_timeout_sec: float = 3.0

    @property
    def backend_health_url(self) -> str:
        return f"http://{self.backend_host}:{self.backend_port}/api/v1/health"

    @property
    def flower_address(self) -> str:
        return f"{self.flower_host}:{self.flower_port}"


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "dashboard", "backend")


# ---------------------------------------------------------------------------
# Structured Logging System
# ---------------------------------------------------------------------------

class SubsystemLogFormatter(logging.Formatter):
    """Custom logging formatter adding clean ISO timestamps and colored subsystem tags."""
    
    def format(self, record: logging.LogRecord) -> str:
        tag = getattr(record, "subsystem", "ORCHESTRATOR")
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        return f"{timestamp} - [{tag}] - {record.getMessage()}"


def setup_orchestrator_logger() -> logging.Logger:
    logger = logging.getLogger("fedmed_orchestrator")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SubsystemLogFormatter())
        logger.addHandler(handler)
    return logger


logger = setup_orchestrator_logger()


def log_subsystem(subsystem: str, message: str, level: int = logging.INFO) -> None:
    """Log a message tagged with a specific subsystem identifier."""
    logger.log(level, message, extra={"subsystem": subsystem})


# ---------------------------------------------------------------------------
# Network & Readiness Helpers
# ---------------------------------------------------------------------------

def is_port_in_use(host: str, port: int) -> bool:
    """Check if a TCP port is currently occupied on the host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def find_occupying_process(port: int) -> Optional[Tuple[int, str]]:
    """
    Cross-platform attempt to locate PID and command name occupying a port.
    Returns (pid, command_str) if found, else None.
    """
    if sys.platform in ("linux", "darwin"):
        try:
            output = subprocess.check_output(
                ["lsof", "-i", f":{port}", "-t", "-sTCP:LISTEN"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if output:
                pids = output.split("\n")
                pid = int(pids[0])
                cmd = subprocess.check_output(
                    ["ps", "-p", str(pid), "-o", "comm="],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                return pid, cmd
        except Exception:
            pass
    return None


def verify_ports_available(config: SimulationConfig) -> None:
    """Verify backend and Flower ports are available before spawning subprocesses."""
    ports_to_check = [
        ("BACKEND", config.backend_host, config.backend_port),
        ("FLOWER", config.flower_host, config.flower_port),
    ]

    collisions = []
    for subsystem, host, port in ports_to_check:
        if is_port_in_use(host, port):
            proc_info = find_occupying_process(port)
            if proc_info:
                pid, cmd = proc_info
                collisions.append(f"{subsystem} port {port} is occupied by PID {pid} ('{cmd}'). Fix: kill -9 {pid}")
            else:
                collisions.append(f"{subsystem} port {port} is already occupied on {host}.")

    if collisions:
        log_subsystem("ORCHESTRATOR", "CRITICAL: Port collisions detected before startup!", logging.ERROR)
        for msg in collisions:
            log_subsystem("ORCHESTRATOR", f"  ↳ {msg}", logging.ERROR)
        sys.exit(1)


def poll_backend_health(health_url: str, timeout_sec: float, interval_sec: float, proc: subprocess.Popen) -> bool:
    """Active HTTP readiness polling for FastAPI backend GET /api/v1/health."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if proc.poll() is not None:
            log_subsystem("BACKEND", f"Process exited prematurely with code {proc.returncode}!", logging.ERROR)
            return False
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "FedMedOrchestrator"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read().decode("utf-8"))
                    if body.get("data", {}).get("status") == "ok":
                        return True
        except Exception:
            pass
        time.sleep(interval_sec)
    return False


def poll_tcp_socket_ready(host: str, port: int, timeout_sec: float, interval_sec: float, proc: subprocess.Popen) -> bool:
    """Active TCP socket readiness polling for Flower central server."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if proc.poll() is not None:
            log_subsystem("FLOWER", f"Process exited prematurely with code {proc.returncode}!", logging.ERROR)
            return False
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (OSError, ConnectionRefusedError):
            pass
        time.sleep(interval_sec)
    return False


# ---------------------------------------------------------------------------
# Process Lifecycle & Signal Management
# ---------------------------------------------------------------------------

class ProcessTracker:
    """Manages active subprocesses and guarantees clean termination."""

    def __init__(self, shutdown_timeout: float = 3.0):
        self.processes: List[Tuple[str, subprocess.Popen]] = []
        self.shutdown_timeout = shutdown_timeout

    def register(self, name: str, proc: subprocess.Popen) -> None:
        self.processes.append((name, proc))

    def terminate_all(self) -> None:
        if not self.processes:
            return

        log_subsystem("ORCHESTRATOR", "Cleaning up background process tree...")
        
        # Step 1: Send SIGTERM to all running subprocesses
        for name, proc in self.processes:
            if proc.poll() is None:
                log_subsystem("ORCHESTRATOR", f"Sending SIGTERM to {name} (PID {proc.pid})...")
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass

        # Step 2: Wait for graceful exit
        start = time.time()
        for name, proc in self.processes:
            remaining = max(0.1, self.shutdown_timeout - (time.time() - start))
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                # Step 3: Escalate to SIGKILL if timeout expires
                if proc.poll() is None:
                    log_subsystem("ORCHESTRATOR", f"Process {name} (PID {proc.pid}) timed out. Escalating to SIGKILL...", logging.WARNING)
                    try:
                        proc.kill()
                        proc.wait()
                    except ProcessLookupError:
                        pass

        log_subsystem("ORCHESTRATOR", "All managed subprocesses terminated cleanly.")


# ---------------------------------------------------------------------------
# Main Orchestration Workflow
# ---------------------------------------------------------------------------

def main():
    config = SimulationConfig()
    tracker = ProcessTracker(shutdown_timeout=config.shutdown_timeout_sec)

    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        log_subsystem("ORCHESTRATOR", f"Received signal {sig_name}. Triggering graceful shutdown...", logging.WARNING)
        tracker.terminate_all()
        sys.exit(128 + signum)

    # Register cross-platform signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log_subsystem("ORCHESTRATOR", "==========================================================")
    log_subsystem("ORCHESTRATOR", " Starting FedMed End-to-End Production Simulation Pipeline")
    log_subsystem("ORCHESTRATOR", "==========================================================")

    try:
        # Pre-flight: Check port availability
        log_subsystem("ORCHESTRATOR", "Performing pre-flight port collision checks...")
        verify_ports_available(config)
        log_subsystem("ORCHESTRATOR", "Port availability confirmed [8000, 8080].")

        # Step 1: Bootstrap FastAPI Monitoring Backend
        log_subsystem("BACKEND", f"Bootstrapping FastAPI Backend on {config.backend_host}:{config.backend_port}...")
        backend_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--app-dir",
                BACKEND_DIR,
                "--host",
                config.backend_host,
                "--port",
                str(config.backend_port),
            ],
            cwd=PROJECT_ROOT,
        )
        tracker.register("FastAPI-Backend", backend_proc)

        log_subsystem("BACKEND", f"Polling health endpoint '{config.backend_health_url}'...")
        if not poll_backend_health(config.backend_health_url, config.startup_timeout_sec, config.poll_interval_sec, backend_proc):
            log_subsystem("BACKEND", "CRITICAL: FastAPI Backend failed readiness check!", logging.ERROR)
            raise RuntimeError("Backend startup failed.")
        log_subsystem("BACKEND", "FastAPI Backend active & healthy! Database tables initialized.")

        # Step 2: Launch Flower Central Aggregation Server
        log_subsystem("FLOWER", f"Launching Flower Server on {config.flower_address} ({config.num_rounds} rounds)...")
        server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "server.flower_server",
                "--address",
                config.flower_address,
                "--rounds",
                str(config.num_rounds),
                "--min-clients",
                str(config.min_clients),
            ],
            cwd=PROJECT_ROOT,
        )
        tracker.register("Flower-Server", server_proc)

        log_subsystem("FLOWER", f"Polling TCP socket '{config.flower_address}'...")
        if not poll_tcp_socket_ready(config.flower_host, config.flower_port, config.startup_timeout_sec, config.poll_interval_sec, server_proc):
            log_subsystem("FLOWER", "CRITICAL: Flower Server failed socket readiness check!", logging.ERROR)
            raise RuntimeError("Flower server startup failed.")
        log_subsystem("FLOWER", "Flower Central Aggregation Server socket listening!")

        # Step 3: Connect Hospital Node Alpha (hospital_a)
        log_subsystem("CLIENT-A", "Connecting Hospital Node Alpha (hospital_a)...")
        client_a_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "client.flower_client",
                "--server",
                config.flower_address,
                "--hospital-id",
                "hospital_a",
            ],
            cwd=PROJECT_ROOT,
        )
        tracker.register("Client-Hospital-A", client_a_proc)

        # Step 4: Connect Hospital Node Beta (hospital_b)
        log_subsystem("CLIENT-B", "Connecting Hospital Node Beta (hospital_b)...")
        client_b_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "client.flower_client",
                "--server",
                config.flower_address,
                "--hospital-id",
                "hospital_b",
            ],
            cwd=PROJECT_ROOT,
        )
        tracker.register("Client-Hospital-B", client_b_proc)

        log_subsystem("ORCHESTRATOR", "==========================================================")
        log_subsystem("ORCHESTRATOR", f" Simulation Pipeline Active! Executing {config.num_rounds} FL Rounds...")
        log_subsystem("ORCHESTRATOR", f" Dashboard API Docs: http://{config.backend_host}:{config.backend_port}/docs")
        log_subsystem("ORCHESTRATOR", f" Telemetry WebSocket: ws://{config.backend_host}:{config.backend_port}/api/v1/telemetry/ws")
        log_subsystem("ORCHESTRATOR", "==========================================================")

        # Wait for Flower server process to complete training rounds
        server_exit_code = server_proc.wait()
        
        if server_exit_code == 0:
            log_subsystem("ORCHESTRATOR", f"SUCCESS: Federated Learning Simulation completed {config.num_rounds} rounds cleanly!")
        else:
            log_subsystem("ORCHESTRATOR", f"FAILURE: Flower Server exited with non-zero code {server_exit_code}!", logging.ERROR)
            sys.exit(server_exit_code)

    except Exception as exc:
        log_subsystem("ORCHESTRATOR", f"Simulation failed with exception: {exc}", logging.ERROR)
        sys.exit(1)
    finally:
        tracker.terminate_all()


if __name__ == "__main__":
    main()
