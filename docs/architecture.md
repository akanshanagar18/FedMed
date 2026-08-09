# FedMed Architecture Specification

## 1. Subsystem Overview

FedMed is structured into 6 decoupled core subsystems designed for fault isolation and horizontal scaling.

```mermaid
sequenceDiagram
    autonumber
    actor User as Demo Viewer / Engineer
    participant Script as Orchestrator Script
    participant Backend as FastAPI Backend (8000)
    participant DB as SQLite (fedmed.db)
    participant Flower as Flower Server (8080)
    participant ClientA as Hospital Client A
    participant ClientB as Hospital Client B
    participant React as React Dashboard UI

    User->>Script: python3 scripts/run_simulation.py
    Script->>Backend: Bootstrap Uvicorn (--app-dir dashboard/backend)
    Backend->>DB: init_db() -> Base.metadata.create_all()
    Script->>Backend: GET /api/v1/health polling (Active readiness)
    Backend-->>Script: HTTP 200 OK {"status": "ok"}
    Script->>Flower: Bootstrap Flower Server (--rounds 3)
    Script->>Flower: Socket connection readiness poll
    Flower-->>Script: Socket listening on 8080
    Script->>ClientA: Connect hospital_a
    Script->>ClientB: Connect hospital_b
    ClientA->>Flower: gRPC Register NumPyClient
    ClientB->>Flower: gRPC Register NumPyClient

    loop Federated Round (1 to 3)
        Flower->>ClientA: Send global parameters
        Flower->>ClientB: Send global parameters
        ClientA->>ClientA: Local PyTorch train_one_epoch()
        ClientB->>ClientB: Local PyTorch train_one_epoch()
        ClientA-->>Flower: Return updated weights & metrics
        ClientB-->>Flower: Return updated weights & metrics
        Flower->>Flower: FedAvg Weight Aggregation
        Flower->>Backend: POST /api/v1/metrics (Loss & Dice)
        Backend->>DB: Save TrainingMetricModel row
        Backend->>React: WebSocket broadcast 'metrics_updated'
        React->>React: Update Recharts curves & KPI state
    end

    Flower-->>Script: 3 Rounds Complete (Exit 0)
    Script->>ClientA: Graceful SIGTERM
    Script->>ClientB: Graceful SIGTERM
    Script->>Backend: Graceful SIGTERM
    Script-->>User: SUCCESS: All processes cleaned up cleanly!
```

---

## 2. Component Technical Specifications

### A. Medical Machine Learning Layer (`model/`)
* **Architecture:** 3D U-Net configured for 4-channel MRI input (`T1`, `T1ce`, `T2`, `FLAIR`) and 3-channel binary mask output (`Tumor Core`, `Whole Tumor`, `Enhancing Tumor`).
* **Trainer:** `model/trainer.py` executes local PyTorch gradient updates using `BCEWithLogitsLoss` and computes Dice Similarity Coefficients (DSC).

### B. Federated Learning Layer (`client/` & `server/`)
* **Server:** `server/flower_server.py` extends Flower's `FedAvg` with a custom `MetricsReporterStrategy` that issues asynchronous HTTP POST calls to the FastAPI backend upon round completion.
* **Client:** `client/flower_client.py` extends `flwr.client.NumPyClient`, training locally on synthetic BraTS tensors before serializing parameters.

### C. Homomorphic Privacy Layer (`privacy/`)
* **Scheme:** TenSEAL CKKS (Cheon-Kim-Kim-Song) scheme supporting approximate vector arithmetic over encrypted floats.
* **Context:** Configured with `poly_modulus_degree=8192` and scaling factor `2**40`.

### D. Monitoring Backend (`dashboard/backend/app/`)
* **Framework:** FastAPI with Uvicorn ASGI server.
* **Endpoints:** REST endpoints for system health (`/health`) and metrics history (`/metrics`), plus WebSocket endpoint (`/telemetry/ws`) for live event streaming.

### E. Persistence Layer (`dashboard/backend/app/database/`)
* **ORM:** SQLAlchemy declarative models managing SQLite storage (`fedmed.db`).

### F. Frontend Dashboard (`dashboard/frontend/`)
* **Framework:** React 18 SPA built with Vite and Recharts, styled with custom dark glassmorphic CSS. Hosted automatically by FastAPI static file mounting.

---

## 3. Autonomous Operating System Architecture

FedMed operates as an **Autonomous Federated Learning Operating System** comprising:
* **Async Event Bus (`events/event_bus.py`)**: Asynchronous Pub/Sub engine connecting training, metrics, drift, SLA, recommendations, orchestrator, deployment, and WebSockets.
* **Autonomous Orchestrator (`orchestrator/autonomous_orchestrator.py`)**: Evaluates system state across Drift, SLA, Governance, Metrics, Privacy, and Health to make explainable decisions.
* **Adaptive Strategy Selector (`strategy/adaptive_engine.py`)**: Dynamically switches FL strategies (FedAvg, FedProx, Scaffold, FedNova, FedAdam) based on telemetry.
* **Root Cause Analysis (RCA) Engine (`analytics/rca_engine.py`)**: Diagnoses performance degradation causes (client dropout, feature drift, DP noise, latency) with confidence scores and mitigations.
* **Operational Recommendation Engine (`analytics/recommendation_engine.py`)**: Emits actionable advice (trigger HPO, reduce LR, promote model).
* **Autonomous Experiment Planner (`analytics/experiment_planner.py`)**: Proposes future benchmark matrix sweeps.
* **Self-Healing Recovery Engine (`resilience/self_healing.py`)**: Executes automated recovery workflows (node reconnection, checkpoint restore, strategy fallback).
* **Production Deployment Manager (`deployment/manager.py`)**: Manages Canary, Rolling, Blue-Green, Shadow, and Emergency Rollback deployments with validation gates.
* **System Knowledge Graph (`knowledge/graph.py` & `knowledge/persistent_graph.py`)**: SQLite-backed persistent graph (`fedmed_kg.db`) with Neo4j driver abstraction, node/edge versioning, and time-travel query API (`query_as_of`).

---

## 4. Milestone T — Enterprise Workflow Integration Platform

Milestone T unifies all subsystems into a single autonomous workflow-driven operating platform:
* **Unified Workflow Engine (`workflows/workflow_engine.py`)**: Manages 14-step canonical DAG workflows, execution tracing, retries, pause/resume, cancel, and state transitions (`CREATED`, `RUNNING`, `WAITING`, `PAUSED`, `FAILED`, `RETRYING`, `COMPLETED`, `CANCELLED`).
* **Scenario Simulation Engine (`simulation/scenario_engine.py`)**: Simulates 20 enterprise failure, disconnect, attack, scale-up, and resource exhaustion scenarios, publishing event streams over the Event Bus.
* **Enterprise Policy Engine (`configs/policy_engine.py` & `configs/policy.yaml`)**: Central dynamic policy configuration engine supporting live YAML reloads and zero hardcoded magic numbers.
* **Background Scheduler (`scheduler/scheduler_engine.py`)**: Cron, periodic, and event-triggered background job execution for retraining, drift scans, governance audits, privacy audits, checkpoint cleanup, HPO, and health monitoring.
* **Experiment Lifecycle Manager (`experiments/lifecycle_manager.py`)**: Manages stage transitions: `DATASET_PREPARATION` $\rightarrow$ `TRAINING` $\rightarrow$ `VALIDATION` $\rightarrow$ `GOVERNANCE` $\rightarrow$ `DEPLOYMENT` $\rightarrow$ `MONITORING` $\rightarrow$ `RETIREMENT` $\rightarrow$ `ARCHIVE`.
* **Enterprise Event Choreography (`events/choreography.py`)**: Subscribes to Event Bus topics so signals automatically propagate across recommendations, adaptive strategy, orchestrator, deployment manager, and WebSockets without module coupling.
* **Digital Twin Predictive Simulation (`digital_twin/twin_engine.py`)**: Answers "What happens if..." hypothetical impact questions, estimating expected Dice, convergence, communication overhead, privacy impact, governance status, and deployment recommendations.


