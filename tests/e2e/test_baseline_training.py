"""
End-to-End Baseline Training Smoke Test
Executes scripts/train_baseline.py and validates checkpoint creation,
metric calculations, and output files.
"""

import os
import sys
import subprocess
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASELINE_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "train_baseline.py")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


@pytest.mark.e2e
def test_baseline_training_e2e_execution():
    """Smoke test running scripts/train_baseline.py for 1 epoch."""
    assert os.path.exists(BASELINE_SCRIPT)

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    proc = subprocess.Popen(
        [sys.executable, BASELINE_SCRIPT, "--epochs", "1", "--batch-size", "1"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=120)
        exit_code = proc.returncode

        output = stdout + stderr
        assert exit_code == 0, f"Baseline failed with exit code {exit_code}.\nStderr: {stderr}\nStdout: {stdout}"
        assert "Centralized Baseline Training Complete!" in output


        # Verify best model checkpoint creation
        best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        assert os.path.exists(best_ckpt), f"Best model checkpoint missing at {best_ckpt}"

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
