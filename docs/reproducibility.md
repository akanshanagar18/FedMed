# FedMed v2.0 — Scientific Reproducibility Engine

## Overview
The `utils.reproducibility` module enforces 100% deterministic experiment reproducibility by capturing complete environment, hardware, seed, git, and library dependency snapshots.

## Captured Metadata Schema
Every experiment run automatically saves a JSON reproducibility report containing:

```json
{
  "timestamp": "2026-08-08T17:00:00Z",
  "random_seed": 42,
  "git": {
    "git_branch": "feature/integration-green",
    "git_commit": "e8f7a1b...",
    "git_dirty": "False"
  },
  "hardware": {
    "os_platform": "macOS-14.5",
    "cpu_count": 10,
    "cpu_arch": "arm64",
    "ram_gb": 32.0,
    "cuda_available": false,
    "gpu_count": 0,
    "gpu_name": "N/A"
  },
  "dependencies": {
    "python_version": "3.9.6",
    "torch_version": "2.3.0",
    "monai_version": "1.3.1",
    "flower_version": "1.7.0",
    "mlflow_version": "3.1.4"
  }
}
```

## REST API Integration
```http
GET /api/v1/system/reproducibility
GET /api/v1/system/environment
```
