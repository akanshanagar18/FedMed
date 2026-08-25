# FedMed v2.0 — End-to-End Research Workflow Guide

## Overview
This guide documents the end-to-end research lifecycle for running simulations, benchmarking strategy configurations, tracking experiment metrics, exploring research artifacts, and exporting publication-grade reports.

## End-to-End Research Pipeline
```mermaid
sequenceDiagram
    autonumber
    actor Researcher as Research Engineer
    participant Script as Benchmark Script / Simulation Orchestrator
    participant MLflow as MLflow Tracker
    participant TB as TensorBoard SummaryWriter
    participant Reg as Checkpoint Registry
    participant Export as Export Engine (PDF/CSV/JSON/MD)
    participant Dash as React Dashboard & REST API

    Researcher->>Script: Run matrix sweep (run_benchmarks.py)
    loop Each Experiment in Sweep
        Script->>MLflow: Start Run & Log Hyperparams
        Script->>TB: Stream Round Metrics
        Script->>Reg: Register Model Checkpoints + SHA256
        Script->>MLflow: Upload Model Checkpoints & Plots
        Script->>MLflow: End Run
    end
    Script->>Export: Export Report Suite (report.pdf, leaderboard.csv, comparison.md)
    Researcher->>Dash: Explore runs, checkpoints, & download PDF reports
```

## Quick Start Commands
1. **Centralized Baseline Training:**
   ```bash
   python scripts/train_baseline.py --config configs/default.yaml --epochs 5
   ```
2. **Federated Simulation Execution:**
   ```bash
   python scripts/run_simulation.py --partition dirichlet --alpha 0.5 --enable-tls
   ```
3. **Automated Matrix Sweep Benchmarking:**
   ```bash
   python scripts/run_benchmarks.py --quick
   ```
4. **Launch TensorBoard UI:**
   ```bash
   tensorboard --logdir runs/
   ```
5. **Access Research Dashboard:**
   Open browser at `http://127.0.0.1:8000/` or inspect `/api/v1/openapi.json`.
