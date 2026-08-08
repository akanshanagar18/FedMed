# FedMed v2.0 — Production MLflow Tracking Architecture

## Overview
FedMed v2.0 integrates MLflow tracking into both centralized baseline training (`train_baseline.py`) and federated Flower executions (`server.flower_server`).

## Architecture & Integration Flow
```mermaid
sequenceDiagram
    autonumber
    participant Orchestrator as Simulation Orchestrator
    participant Server as Flower Server / Trainer
    participant Adapter as FlowerStrategyAdapter
    participant Tracker as MLflowTracker
    participant MLflow as MLflow Tracking Store (mlruns)

    Orchestrator->>Server: Launch Execution
    Server->>Tracker: start_run(run_name, tags)
    Tracker->>MLflow: Create Run ID
    Server->>Tracker: log_params(hyperparameters)
    loop Each FL Round / Epoch
        Adapter->>Tracker: log_metrics(dice, loss, iou, runtime)
        Tracker->>MLflow: Append Metric Points
    end
    Server->>Tracker: log_artifact(best_model.pth, report.pdf, convergence.png)
    Tracker->>MLflow: Upload Model Weights & Artifacts
    Server->>Tracker: end_run(status="FINISHED")
```

## Tracked Parameters & Metrics
- **Hyperparameters:** Strategy (`FedAvg`, `FedProx`), Dataset (`BraTS2021`), Partition Strategy (`Dirichlet`, `IID`), Dirichlet Alpha (`0.5`), Learning Rate (`1e-4`), Optimizer (`Adam`), Epochs (`1`), Batch Size (`2`), Random Seed (`42`), DP Enabled (`bool`), HE Enabled (`bool`), TLS Enabled (`bool`), Rounds (`3`), Clients (`3`).
- **Metrics:** MONAI Dice Score, Mean IoU, 95th Percentile Hausdorff Distance, Precision, Recall, Training Loss, Validation Loss, Aggregation Runtime, GPU Memory MB.
- **Uploaded Artifacts:** `best_model.pth`, `last_model.pth`, `baseline_convergence.png`, `summary.json`, `leaderboard.csv`, `report.pdf`.

## Usage & API Integration
Access tracking data programmatically via REST API:
```http
GET /api/v1/mlflow/status
GET /api/v1/mlflow/runs
```
