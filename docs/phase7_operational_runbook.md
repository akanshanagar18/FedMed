# FedMed OS v2.1 — Autonomous FLOS Architecture & Operational Runbook

---

## 1. Runtime Architecture Diagram

```mermaid
graph TD
    Client["React Operations Console (Browser)"] -->|REST APIs & WebSockets| Backend["FastAPI Backend Control Plane"]
    Backend -->|Manages| RuntimeOrchestrator["RuntimeOrchestrator (Persistent Singleton)"]
    
    subgraph ControlPlane["FedMed OS v2.1 Control Plane"]
        RuntimeOrchestrator --> StateMachines["State Machine Validator Engine"]
        RuntimeOrchestrator --> WorkflowEngine["Unified Workflow Engine (DAG)"]
        RuntimeOrchestrator --> SchedulerEngine["Enterprise Scheduler"]
        RuntimeOrchestrator --> EventBus["Asynchronous EventBus (Pub/Sub)"]
        RuntimeOrchestrator --> HospitalManager["Hospital Runtime Manager"]
        RuntimeOrchestrator --> ModelRegistry["Production Model Registry"]
    end

    WorkflowEngine -->|Orchestrates| FlowerServer["Flower Central Server (gRPC 8080)"]
    FlowerServer -->|Coordinates| NodeAlpha["Hospital Alpha Node"]
    FlowerServer -->|Coordinates| NodeBeta["Hospital Beta Node"]
    FlowerServer -->|Coordinates| NodeGamma["Hospital Gamma Node"]
    FlowerServer -->|Coordinates| NodeDelta["Hospital Delta Node"]

    EventBus -->|Persists| SQLite["SQLite Persistence (fedmed.db)"]
    EventBus -->|Broadcasts| WSServer["WebSocket Telemetry Manager"]
    WSServer -->|Live Telemetry| Client
```

---

## 2. Sequence Diagram: REST-Driven Experiment Execution

```mermaid
sequenceDiagram
    autonumber
    actor Operator as Operator / Client
    participant API as FastAPI REST API
    participant Mgr as ExperimentManager
    participant WF as WorkflowEngine
    participant FL as Flower FL Server
    participant DB as SQLite DB

    Operator->>API: POST /api/v1/experiments (Create)
    API->>Mgr: create_experiment(params)
    Mgr->>WF: instantiate_workflow("Enterprise_FL_Pipeline")
    Mgr->>DB: INSERT into experiments table
    Mgr-->>Operator: 200 OK (experiment_id, status=CREATED)

    Operator->>API: POST /api/v1/experiments/{id}/start
    API->>Mgr: start_experiment(id)
    Mgr->>WF: execute_workflow_step()
    WF->>FL: Start FL Training Rounds
    FL-->>WF: Round Metrics & Loss
    WF->>DB: Update state to RUNNING & Save Metrics
    Mgr-->>Operator: 200 OK (status=RUNNING)
```

---

## 3. Deterministic State Transition Matrix

```mermaid
stateDiagram-v2
    [*] --> CREATED
    CREATED --> INITIALIZING
    CREATED --> READY
    CREATED --> RUNNING
    INITIALIZING --> READY
    READY --> RUNNING
    RUNNING --> PAUSED
    PAUSED --> RESUMING
    RESUMING --> RUNNING
    RUNNING --> COMPLETED
    RUNNING --> FAILED
    COMPLETED --> ARCHIVED
    FAILED --> ARCHIVED
```

---

## 4. Class Diagram: FLOS Control Plane Subsystems

```mermaid
classDiagram
    class RuntimeOrchestrator {
        +bool is_running
        +float start_time
        +start() void
        +stop() void
        +restore_state_from_db() void
        +get_runtime_health() Dict
    }

    class ExperimentManager {
        +create_experiment(name, strategy) Dict
        +start_experiment(id) Dict
        +pause_experiment(id) Dict
        +resume_experiment(id) Dict
        +cancel_experiment(id) Dict
    }

    class WorkflowEngine {
        +instantiate_workflow(name) WorkflowInstance
        +execute_workflow_step(instance_id) Dict
        +run_entire_workflow(instance_id) Dict
    }

    class ModelRegistry {
        +register_model_version(id, ver, path) Dict
        +promote_stage(key, stage) Dict
    }

    RuntimeOrchestrator --> ExperimentManager
    RuntimeOrchestrator --> WorkflowEngine
    RuntimeOrchestrator --> ModelRegistry
```

---

## 5. REST API & WebSocket Event Catalogs

### Key REST Endpoints

| Method | Path | Subsystem | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/runtime/health` | Runtime | Returns CPU, RAM, active experiments, connected nodes, WS count |
| `GET` | `/api/v1/runtime/status` | Runtime | Returns OS state, workflows, scheduled jobs, hospital nodes |
| `POST` | `/api/v1/runtime/restart` | Runtime | Triggers control plane state reload and orchestrator restart |
| `POST` | `/api/v1/experiments` | Experiment | Creates a new FL experiment and attaches workflow DAG |
| `POST` | `/api/v1/experiments/{id}/start` | Experiment | Starts workflow execution |
| `POST` | `/api/v1/experiments/{id}/pause` | Experiment | Pauses running experiment |
| `POST` | `/api/v1/experiments/{id}/resume` | Experiment | Resumes paused experiment |
| `POST` | `/api/v1/experiments/{id}/cancel` | Experiment | Cancels running experiment |

### WebSocket Event Stream (`/api/v1/telemetry/ws`)

| Event Type | Payload Attributes | Description |
| :--- | :--- | :--- |
| `metrics_updated` | `round_number`, `training_loss`, `dice_score` | Live training round metrics stream |
| `node_status_changed` | `hospital_id`, `status`, `timestamp` | Hospital connection state change |
| `autonomous_decision` | `decision`, `rationale`, `severity` | Autonomous OS decision recommendation |
| `stage_promoted` | `model_id`, `version`, `new_stage` | Production model promotion notification |

---

## 6. Operational Runbook & Troubleshooting

### Operational Command Quick Reference

```bash
# 1. Start Persistent Control Plane Backend
uvicorn dashboard.backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# 2. Run Zero-Setup Autonomous REST Client Demo
python demo.py

# 3. Run Full Negative-Path & Chaos Test Suite
./venv/bin/pytest tests/integration/test_runtime_recovery_and_negative_paths.py -v

# 4. Run Complete Platform Test Suite (258+ Tests)
./venv/bin/pytest -v
```

### Common Failure Modes & Resolution Procedures

1. **Hospital Heartbeat Missed (>30s)**:
   - *Diagnostic*: RuntimeOrchestrator logs `Hospital 'hospital_alpha' missed heartbeat (>30s)`.
   - *Automated Recovery*: SelfHealingRecoveryEngine resets connection state and re-establishes node registration.
   - *Manual Action*: Trigger recovery via REST: `POST /api/v1/autonomous/self-healing/recover` with `{"node_id": "hospital_alpha", "failure_type": "Manual Reconnect"}`.

2. **Backend Process Restart**:
   - *Diagnostic*: Backend process killed or restarted.
   - *Automated Recovery*: `RuntimeOrchestrator.restore_state_from_db()` automatically queries SQLite `experiments` and `hospital_nodes` tables to rebuild active state in memory without losing experiment history.
