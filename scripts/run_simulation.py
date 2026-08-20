"""
Module: scripts.run_simulation

Purpose:
Production Master Simulation Orchestrator for FedMed v2.0.
Spawns and manages the FastAPI backend, central Flower server, and 3 hospital clients
(Hospital Alpha, Hospital Beta, Hospital Gamma) on Non-IID Dirichlet patient data partitions.
Supports TenSEAL CKKS Homomorphic Encryption, Opacus Differential Privacy, Production TLS gRPC,
and Fault-Tolerant Node Failure Simulation (`--simulate-failure hospital_beta`).
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
backend_path = os.path.join(PROJECT_ROOT, "dashboard", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from configs.loader import AppConfig, ConfigValidationError, load_config
from privacy.tls_cert_gen import ensure_tls_certificates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fedmed_orchestrator")


def log_subsystem(subsystem: str, message: str, level: int = logging.INFO):
    """Outputs structured, subsystem-prefixed log messages."""
    ts = time.strftime("%H:%M:%S")
    sub_str = f"[{subsystem.center(14)}]"
    out_msg = f"{ts} {sub_str} {message}"
    print(out_msg, flush=True)
    if level == logging.ERROR:
        logger.error(out_msg)
    elif level == logging.WARNING:
        logger.warning(out_msg)
    else:
        logger.info(out_msg)


class ProcessTracker:
    """Tracks spawned subprocesses and provides clean shutdown handling."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}

    def register(self, name: str, proc: subprocess.Popen):
        self.processes[name] = proc

    def cleanup_all(self):
        log_subsystem("ORCHESTRATOR", "Initiating graceful process tree shutdown...")
        for name, proc in self.processes.items():
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()


class SimulationConfig:
    """Resolved runtime configuration for simulation orchestration."""

    def __init__(
        self,
        backend_host: str = "127.0.0.1",
        backend_port: int = 8000,
        flower_host: str = "127.0.0.1",
        flower_port: int = 8080,
        num_rounds: int = 3,
        min_clients: int = 2,
        client_ids: Optional[List[str]] = None,
        partition_strategy: str = "dirichlet",
        dirichlet_alpha: float = 0.5,
        experiment_id: str = "default",
    ):
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.flower_host = flower_host
        self.flower_port = flower_port
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.client_ids = client_ids or ["hospital_alpha", "hospital_beta", "hospital_gamma"]
        self.partition_strategy = partition_strategy
        self.dirichlet_alpha = dirichlet_alpha
        self.api_url = f"http://{backend_host}:{backend_port}"
        self.flower_address = f"{flower_host}:{flower_port}"
        self.experiment_id = experiment_id or "default"


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks whether a local TCP port is currently bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_free_port(start_port: int, host: str = "127.0.0.1") -> int:
    """Finds an available TCP port starting from start_port."""
    for port in range(start_port, start_port + 50):
        if not is_port_in_use(port, host):
            return port
    return start_port


def verify_ports_available(cfg: SimulationConfig):
    """Ensures required network ports are free prior to launching subprocesses."""
    pass


def poll_backend_health(health_url: str, timeout_sec: float = 15.0, interval_sec: float = 0.5, proc: Optional[subprocess.Popen] = None) -> bool:
    """Polls FastAPI GET /api/v1/health endpoint until HTTP 200 is returned."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if proc and proc.poll() is not None:
            return False
        try:
            resp = requests.get(health_url, timeout=1.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval_sec)
    return False


def run_simulation(
    yaml_config_path: Optional[str] = None,
    partition_strategy: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    enable_he: bool = False,
    enable_dp: bool = False,
    enable_tls: bool = False,
    simulate_failure: Optional[str] = None,
    experiment_id: str = "default",
):
    """Orchestrates the entire execution pipeline."""
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
    tls_active = enable_tls or (hasattr(app_cfg, "tls") and app_cfg.tls.enabled)

    if tls_active:
        cert_dir = getattr(app_cfg.tls, "cert_dir", "certs") if hasattr(app_cfg, "tls") else "certs"
        ensure_tls_certificates(cert_dir)
        log_subsystem("ORCHESTRATOR", f"Production TLS certificate suite verified in '{cert_dir}'.")

    sim_config = SimulationConfig(
        backend_host=app_cfg.server.host,
        backend_port=app_cfg.server.port,
        flower_host=app_cfg.server.fl_server_address.split(":")[0],
        flower_port=int(app_cfg.server.fl_server_address.split(":")[1]),
        num_rounds=app_cfg.federated.num_rounds,
        min_clients=app_cfg.federated.min_clients,
        partition_strategy=p_strat,
        dirichlet_alpha=p_alpha,
        experiment_id=experiment_id,
    )

    tracker = ProcessTracker()

    def signal_handler(signum, frame):
        log_subsystem("ORCHESTRATOR", f"Interrupted by signal {signum}. Shutting down...", logging.WARNING)
        tracker.cleanup_all()
        sys.exit(128 + signum)

    try:
        health_url = f"{sim_config.api_url}/api/v1/health"

        backend_already_running = False
        try:
            r_health = requests.get(health_url, timeout=1.0)
            if r_health.status_code == 200:
                backend_already_running = True
                log_subsystem("BACKEND", "FastAPI Backend is already running and healthy!")
        except Exception:
            pass

        child_env = os.environ.copy()
        pythonpath_entries = [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "dashboard", "backend")]
        if child_env.get("PYTHONPATH"):
            pythonpath_entries.append(child_env["PYTHONPATH"])
        child_env["PYTHONPATH"] = os.path.pathsep.join(pythonpath_entries)

        child_env["OMP_NUM_THREADS"] = "1"
        child_env["MKL_NUM_THREADS"] = "1"
        child_env["OPENBLAS_NUM_THREADS"] = "1"
        child_env["VECLIB_MAXIMUM_THREADS"] = "1"
        child_env["NUMEXPR_NUM_THREADS"] = "1"
        child_env["PYTHONUNBUFFERED"] = "1"

        if not backend_already_running:
            if is_port_in_use(sim_config.backend_port, sim_config.backend_host):
                sim_config.backend_port = find_free_port(sim_config.backend_port + 1, sim_config.backend_host)
                sim_config.api_url = f"http://{sim_config.backend_host}:{sim_config.backend_port}"
                health_url = f"{sim_config.api_url}/api/v1/health"

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
            backend_proc = subprocess.Popen(backend_cmd, cwd=PROJECT_ROOT, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            tracker.register("BACKEND", backend_proc)

            log_subsystem("BACKEND", f"Polling health readiness at {health_url}...")
            if not poll_backend_health(health_url, timeout_sec=20.0, interval_sec=0.5, proc=backend_proc):
                out = backend_proc.stdout.read() if backend_proc.stdout else ""
                log_subsystem("BACKEND", f"FastAPI server failed readiness health check! Code: {backend_proc.poll()} Output:\n{out}", logging.ERROR)
                sys.exit(1)
            log_subsystem("BACKEND", "FastAPI Backend active & healthy!")


        # 2. Spawn Flower Server
        if is_port_in_use(sim_config.flower_port, sim_config.flower_host):
            sim_config.flower_port = find_free_port(sim_config.flower_port + 1, sim_config.flower_host)
            sim_config.flower_address = f"{sim_config.flower_host}:{sim_config.flower_port}"


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
        if tls_active:
            server_cmd.append("--enable-tls")
        if yaml_config_path:
            server_cmd.extend(["--config", yaml_config_path])

        os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
        server_log_path = os.path.join(PROJECT_ROOT, "logs", "simulation_server.log")
        server_log_file = open(server_log_path, "w")

        server_proc = subprocess.Popen(server_cmd, cwd=PROJECT_ROOT, env=child_env, stdout=server_log_file, stderr=subprocess.STDOUT)
        tracker.register("FLOWER", server_proc)

        log_subsystem("FLOWER", f"Waiting for Flower server to bind gRPC port at {sim_config.flower_address}...")
        flower_ready = False
        start_wait = time.time()
        while time.time() - start_wait < 30.0:
            if server_proc.poll() is not None:
                break
            if is_port_in_use(sim_config.flower_port, sim_config.flower_host):
                flower_ready = True
                break
            time.sleep(0.2)

        if not flower_ready:
            err_details = ""
            if os.path.exists(server_log_path):
                with open(server_log_path) as sf:
                    err_details = sf.read()
            log_subsystem("FLOWER", f"Flower Server failed to bind gRPC port within 30s! Code: {server_proc.poll()}\nLogs:\n{err_details}", logging.ERROR)
            sys.exit(1)

        log_subsystem("FLOWER", "Flower Server active & listening on gRPC port!")

        # 3. Spawn Participating Hospital Clients
        enc_mode_label = "TEN_SEAL" if he_active else "PLAINTEXT"
        dp_mode_label = "OPACUS" if dp_active else "NO_DP"
        transport_label = "TLS" if tls_active else "INSECURE"
        for client_id in sim_config.client_ids:
            log_subsystem(client_id.upper(), f"Launching Hospital Client ({sim_config.partition_strategy.upper()} alpha={sim_config.dirichlet_alpha} Mode={enc_mode_label} DP={dp_mode_label} Transport={transport_label}) connecting to {sim_config.flower_address}...")
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
                "--partition",
                sim_config.partition_strategy,
                "--alpha",
                str(sim_config.dirichlet_alpha),
                "--experiment-id",
                sim_config.experiment_id,
            ]
            if he_active:
                client_cmd.append("--enable-he")
            if dp_active:
                client_cmd.append("--enable-dp")
            if tls_active:
                client_cmd.append("--enable-tls")
            if yaml_config_path:
                client_cmd.extend(["--config", yaml_config_path])

            client_log_path = os.path.join(PROJECT_ROOT, "logs", f"simulation_{client_id}.log")
            client_log_file = open(client_log_path, "w")
            client_proc = subprocess.Popen(client_cmd, cwd=PROJECT_ROOT, env=child_env, stdout=client_log_file, stderr=subprocess.STDOUT)
            tracker.register(client_id.upper(), client_proc)

        # 4. Handle Simulated Node Failure if specified
        if simulate_failure:
            target_key = simulate_failure.strip().upper()
            if target_key not in tracker.processes:
                target_key = f"HOSPITAL_{target_key}"

            def failure_worker():
                log_subsystem("RESILIENCE", f"Failure simulation active for '{simulate_failure}'. Scheduled SIGTERM termination in 12 seconds during Round 2...")
                time.sleep(12.0)
                if target_key in tracker.processes:
                    proc_to_kill = tracker.processes[target_key]
                    if proc_to_kill.poll() is None:
                        log_subsystem("RESILIENCE", f"SIMULATED FAILURE: Terminating hospital node '{simulate_failure}' process (SIGTERM) during FL training execution!", logging.WARNING)
                        proc_to_kill.terminate()
                        try:
                            proc_to_kill.wait(timeout=2)
                        except Exception:
                            proc_to_kill.kill()
                        # Update REST API node status to FAILED
                        try:
                            requests.post(
                                f"{sim_config.api_url}/api/v1/nodes/heartbeat",
                                json={"hospital_id": simulate_failure.strip().lower(), "status": "FAILED", "active_round": 2, "training_state": "failed"},
                                timeout=2,
                            )
                        except Exception:
                            pass

            fail_thread = threading.Thread(target=failure_worker, daemon=True)
            fail_thread.start()

        # 5. Monitor Flower Server Completion
        log_subsystem("ORCHESTRATOR", "All subprocesses spawned successfully. Monitoring training execution...")
        ret_code = server_proc.wait()

        if ret_code != 0:
            err_details = ""
            if os.path.exists(server_log_path):
                with open(server_log_path) as sf:
                    err_details = sf.read()
            client_logs = []
            for client_id in sim_config.client_ids:
                c_log_path = os.path.join(PROJECT_ROOT, "logs", f"simulation_{client_id}.log")
                if os.path.exists(c_log_path):
                    with open(c_log_path) as cf:
                        client_logs.append(f"=== {client_id.upper()} LOGS ===\n" + cf.read())
            full_err = err_details + "\n" + "\n".join(client_logs)
            log_subsystem("FLOWER", f"Flower Server exited abnormally with code {ret_code}!\nLogs:\n{full_err}", logging.ERROR)
            sys.exit(ret_code)

        log_subsystem("ORCHESTRATOR", "SUCCESS: Federated Learning Simulation completed cleanly!")
        time.sleep(1.0)

    except Exception as e:
        log_subsystem("ORCHESTRATOR", f"Unhandled exception in orchestrator: {e}", logging.ERROR)
        sys.exit(1)
    finally:
        tracker.cleanup_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedMed Simulation Orchestrator")
    parser.add_argument("--config", type=str, default=None, help="Path to modular YAML config file")
    parser.add_argument("--partition", type=str, default="dirichlet", help="Partition strategy (iid / dirichlet)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha value")
    parser.add_argument("--enable-he", action="store_true", help="Enable TenSEAL Homomorphic Encryption")
    parser.add_argument("--enable-dp", action="store_true", help="Enable Opacus Differential Privacy")
    parser.add_argument("--enable-tls", "--tls", action="store_true", help="Enable production TLS gRPC transport")
    parser.add_argument("--simulate-failure", type=str, default=None, help="Simulate node failure during FL training (e.g. hospital_beta)")
    parser.add_argument("--experiment-id", "-e", type=str, default="default", help="Experiment identifier (default: 'default')")
    args = parser.parse_args()

    run_simulation(
        yaml_config_path=args.config,
        partition_strategy=args.partition,
        dirichlet_alpha=args.alpha,
        enable_he=args.enable_he,
        enable_dp=args.enable_dp,
        enable_tls=args.enable_tls,
        simulate_failure=args.simulate_failure,
        experiment_id=args.experiment_id,
    )
