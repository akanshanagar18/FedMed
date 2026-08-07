"""
End-to-End Dirichlet Non-IID Partitioning Simulation Test.
Executes scripts/run_simulation.py --partition dirichlet --alpha 0.5 with 3 hospital silos.
"""

import os
import sys
import subprocess
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIMULATION_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_simulation.py")


@pytest.mark.e2e
def test_dirichlet_non_iid_simulation_execution():
    """Smoke test running scripts/run_simulation.py with Dirichlet partitioning."""
    assert os.path.exists(SIMULATION_SCRIPT)

    # Clear port collisions beforehand
    subprocess.run("lsof -ti:8000 -ti:8080 | xargs kill -9 2>/dev/null || true", shell=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    proc = subprocess.Popen(
        [sys.executable, SIMULATION_SCRIPT, "--partition", "dirichlet", "--alpha", "0.5"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        stdout, _ = proc.communicate(timeout=120)
        exit_code = proc.returncode

        assert exit_code == 0, f"Dirichlet simulation failed with exit code {exit_code}.\nOutput: {stdout}"
        assert "SUCCESS: Federated Learning Simulation completed cleanly!" in stdout

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
