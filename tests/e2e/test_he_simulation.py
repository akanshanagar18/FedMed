"""
End-to-End Homomorphic Encrypted FL Simulation Test.
Executes scripts/run_simulation.py --config configs/privacy.yaml --enable-he with 3 hospital silos.
"""

import os
import sys
import subprocess
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIMULATION_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_simulation.py")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "privacy.yaml")


@pytest.mark.e2e
def test_homomorphic_encryption_fl_simulation_execution():
    """Smoke test running scripts/run_simulation.py with TenSEAL CKKS Homomorphic Encryption."""
    assert os.path.exists(SIMULATION_SCRIPT)

    # Clear port collisions beforehand
    subprocess.run("lsof -ti:8000 -ti:8080 | xargs kill -9 2>/dev/null || true", shell=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    proc = subprocess.Popen(
        [sys.executable, SIMULATION_SCRIPT, "--config", CONFIG_PATH, "--enable-he"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        stdout, _ = proc.communicate(timeout=150)
        exit_code = proc.returncode

        assert exit_code == 0, f"Encrypted simulation failed with exit code {exit_code}.\nOutput: {stdout}"
        assert "SUCCESS: Federated Learning Simulation completed cleanly!" in stdout

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
