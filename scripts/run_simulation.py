"""
Module: scripts.run_simulation

Purpose:
Production-Grade Federated Learning Process Orchestrator for FedMed v2.0.
Supports modular YAML configuration loading via --config flag, port collision detection,
active health polling, and clean process tree termination.

Usage:
    python scripts/run_simulation.py --config configs/default.yaml
"""

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from configs.loader import load_config, AppConfig, ConfigValidationError

# Configure structured color-coded logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fedmed_orchestrator")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log_subsystem(tag: str, msg: str, level: int = logging.INFO) -> None:
    """Format and output logs tagged by subsystem."""
    logger.log(level, f"[{tag:^14}] {msg}")


@dataclass
class SimulationConfig:
    """Configuration specification for the Orchestrator."""
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    flower_host: str = "127.0.0.1"
    flower_port: int = 8080
    num_rounds: int = 3
    min_clients: int = 2
    experiment_id: str = "default"
    partition_strategy: str = "dirichlet"
    dirichlet_alpha: float = 0.5
    client_ids: List[str] = field(default_factory=lambda: ["hospital_alpha", "hospital_beta", "hospital_gamma"])
    api_url: str = field(init=False)
    flower_address: str = field(init=False)

    def __post_init__(self):
        self.api_url = f"http://{self.backend_host}:{self.backend_port}"
        self.flower_address = f"{self.flower_host}:{self.flower_port}"



def is_port_in_use(host: str, port: int) -> bool:
    """Check if a TCP port is open and accepting socket connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_occupying_process(port: int) -> Optional[Tuple[int, str]]:
    """Determine PID and process command occupying a specific port on macOS/Linux."""
    try:
        cmd = ["lsof", "-ti", f":{port}"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        if output:
            pids = output.split("\n")
            pid = int(pids[0])
            cmd_name = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "comm="],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return pid, cmd_name
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


class ProcessTracker:
    """Manages active subprocesses and guarantees clean termination."""

    def __init__(self):
        self.processes: List[Tuple[str, subprocess.Popen]] = []

    def register(self, tag: str, proc: subprocess.Popen):
        self.processes.append((tag, proc))

    def cleanup_all(self, timeout_per_proc: float = 3.0):
        log_subsystem("ORCHESTRATOR", "Initiating graceful process tree shutdown...")
        for tag, proc in reversed(self.processes):
            if proc.poll() is None:
                log_subsystem("ORCHESTRATOR", f"Sending SIGTERM to {tag} (PID {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=timeout_per_proc)
                    log_subsystem("ORCHESTRATOR", f"{tag} terminated cleanly (exit code {proc.returncode}).")
                except subprocess.TimeoutExpired:
                    log_subsystem("ORCHESTRATOR", f"{tag} unresponsive to SIGTERM. Escalating to SIGKILL...", logging.WARNING)
                    proc.kill()
                    proc.wait()
                    log_subsystem("ORCHESTRATOR", f"{tag} forcibly killed.")


def run_simulation(
    yaml_config_path: Optional[str] = None,
    partition_strategy: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    enable_he: bool = False,
    enable_dp: bool = False,
):
    """Orchestrates the entire execution pipeline."""
    # Load modular YAML config
    try:
        app_cfg: AppConfig = load_config(yaml_config_path)
        log_subsystem("ORCHESTRATOR", f"Loaded configuration cleanly: {yaml_config_path or 'configs/default.yaml'}")
    except ConfigValidationError as e:
        log_subsystem("ORCHESTRATOR", f"CRITICAL: Configuration error:\n{e}", logging.ERROR)
        sys.exit(1)

    p_strat = partition_strategy or app_cfg.data.partition_strategy
    p_alpha = dirichlet_alpha if dirichlet_alpha is not None else app_cfg.data.dirichlet_alpha
    he_active = enable_he or app_cfg.privacy.he_enabled
    dp_active = enable_dp or app_cfg.privacy.dp_enabled

    sim_config = SimulationConfig(
        backend_host=app_cfg.server.host,
        backend_port=app_cfg.server.port,
        flower_host=app_cfg.server.fl_server_address.split(":")[0],
        flower_port=int(app_cfg.server.fl_server_address.split(":")[1]),
        num_rounds=app_cfg.federated.num_rounds,
        min_clients=app_cfg.federated.min_clients,
        partition_strategy=p_strat,
        dirichlet_alpha=p_alpha,
    )

    tracker = ProcessTracker()

    def signal_handler(signum, frame):
        log_subsystem("ORCHESTRATOR", f"Interrupted by signal {signum}. Shutting down...", logging.WARNING)
        tracker.cleanup_all()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        verify_ports_available(sim_config)

        # 1. Spawn FastAPI Backend
        log_subsystem("BACKEND", f"Launching FastAPI server at {sim_config.api_url}...")
        backend_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--app-dir",
            "dashboard/backend",
            "--host",
            sim_config.backend_host,
            "--port",
            str(sim_config.backend_port),
        ]
        backend_proc = subprocess.Popen(backend_cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tracker.register("BACKEND", backend_proc)

        # Readiness polling for backend
        health_url = f"{sim_config.api_url}/api/v1/health"
        log_subsystem("BACKEND", f"Polling health readiness at {health_url}...")
        if not poll_backend_health(health_url, timeout_sec=15.0, interval_sec=0.5, proc=backend_proc):
            log_subsystem("BACKEND", "FastAPI server failed readiness health check!", logging.ERROR)
            tracker.cleanup_all()
            sys.exit(1)
        log_subsystem("BACKEND", "FastAPI Backend active & healthy!")

        # 2. Spawn Flower Server
        log_subsystem("FLOWER", f"Launching Flower Server at {sim_config.flower_address} for {sim_config.num_rounds} rounds...")
        server_cmd = [
            sys.executable,
            "-m",
            "server.flower_server",
            "--address",
            sim_config.flower_address,
            "--rounds",
            str(sim_config.num_rounds),
            "--min-clients",
            str(sim_config.min_clients),
            "--api-url",
            sim_config.api_url,
            "--experiment-id",
            sim_config.experiment_id,
        ]
        if yaml_config_path:
            server_cmd.extend(["--config", yaml_config_path])

        server_proc = subprocess.Popen(server_cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tracker.register("FLOWER", server_proc)
        time.sleep(2.0)

        # 3. Spawn Hospital Clients (Hospital Alpha, Hospital Beta, Hospital Gamma)
        for client_id in sim_config.client_ids:
            he_status_str = "TEN_SEAL CKKS ENCRYPTED" if he_active else "PLAINTEXT"
            dp_status_str = "OPACUS DIFFERENTIAL PRIVACY" if dp_active else "NO_DP"
            log_subsystem(client_id.upper(), f"Launching Hospital Client ({sim_config.partition_strategy.upper()} alpha={sim_config.dirichlet_alpha} Mode={he_status_str} DP={dp_status_str}) connecting to {sim_config.flower_address}...")
            client_cmd = [
                sys.executable,
                "-m",
                "client.flower_client",
                "--server-address",
                sim_config.flower_address,
                "--hospital-id",
                client_id,
                "--api-url",
                sim_config.api_url,
                "--partition-strategy",
                sim_config.partition_strategy,
                "--dirichlet-alpha",
                str(sim_config.dirichlet_alpha),
            ]
            if he_active:
                client_cmd.append("--enable-he")
            if dp_active:
                client_cmd.append("--enable-dp")
            if yaml_config_path:
                client_cmd.extend(["--config", yaml_config_path])

            client_proc = subprocess.Popen(client_cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            tracker.register(client_id.upper(), client_proc)

        # 4. Monitor Flower Server Completion
        log_subsystem("ORCHESTRATOR", "All subprocesses spawned successfully. Monitoring training execution...")
        server_proc.wait()

        log_subsystem("ORCHESTRATOR", "SUCCESS: Federated Learning Simulation completed cleanly!")
        time.sleep(1.0)
        tracker.cleanup_all()

    except Exception as e:
        log_subsystem("ORCHESTRATOR", f"Unhandled exception in orchestrator: {e}", logging.ERROR)
        tracker.cleanup_all()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedMed Simulation Orchestrator")
    parser.add_argument("--config", type=str, default=None, help="Path to modular YAML config file")
    parser.add_argument("--partition", type=str, default="dirichlet", help="Partition strategy (iid / dirichlet)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha value")
    parser.add_argument("--enable-he", action="store_true", help="Enable TenSEAL Homomorphic Encryption")
    parser.add_argument("--enable-dp", action="store_true", help="Enable Opacus Differential Privacy")
    args = parser.parse_args()

    run_simulation(
        yaml_config_path=args.config,
        partition_strategy=args.partition,
        dirichlet_alpha=args.alpha,
        enable_he=args.enable_he,
        enable_dp=args.enable_dp,
    )



