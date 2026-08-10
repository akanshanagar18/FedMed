"""
End-to-End Simulation Smoke Test
Executes scripts/run_simulation.py and validates complete 3-round execution,
SQLite metrics persistence, and process termination.
"""

import os
import sys
import sqlite3
import subprocess
import time
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIMULATION_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "run_simulation.py")
DB_PATH = os.path.join(PROJECT_ROOT, "fedmed.db")


@pytest.mark.e2e
def test_full_simulation_pipeline_execution():
    """
    Smoke test running scripts/run_simulation.py end-to-end.
    Verifies:
      1. Process executes cleanly and returns exit code 0.
      2. 3 rounds of FL metrics are written to SQLite 'training_metrics' table.
    """
    assert os.path.exists(SIMULATION_SCRIPT), f"Simulation script not found at {SIMULATION_SCRIPT}"

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    proc = subprocess.Popen(
        [sys.executable, SIMULATION_SCRIPT],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


    try:
        # Wait up to 300 seconds for full 3-round simulation to finish
        stdout, stderr = proc.communicate(timeout=300)
        exit_code = proc.returncode


        assert exit_code == 0, f"Simulation failed with exit code {exit_code}.\nStderr: {stderr}\nStdout: {stdout}"
        assert "SUCCESS: Federated Learning Simulation completed" in stdout


        # Verify SQLite database persistence
        assert os.path.exists(DB_PATH), f"SQLite database file missing at {DB_PATH}"

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT round_number, training_loss, dice_score FROM training_metrics WHERE experiment_id='default' ORDER BY round_number")
        rows = cursor.fetchall()
        conn.close()

        # Must have at least 3 FL rounds persisted
        assert len(rows) >= 3, f"Expected at least 3 metric rows in fedmed.db, found {len(rows)}"
        for round_num, loss, dice in rows[:3]:
            assert isinstance(round_num, int)
            assert isinstance(loss, float)
            assert isinstance(dice, float)
            assert loss > 0.0
            assert 0.0 <= dice <= 1.0

    finally:
        # Guarantee cleanup if timeout expired
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
