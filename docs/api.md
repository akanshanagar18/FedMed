# FedMed API Specification

The FedMed Monitoring Backend provides REST endpoints and a real-time WebSocket stream for monitoring federated learning training progress.

**Base URL:** `http://127.0.0.1:8000/api/v1`  
**WebSocket URL:** `ws://127.0.0.1:8000/api/v1/telemetry/ws`  
**Interactive Docs:** `http://127.0.0.1:8000/docs`  

---

## 1. REST Endpoints

### A. Health Check
* **Endpoint:** `GET /health`
* **Description:** Returns current backend operational state and connection statistics.
* **Request:** None
* **Response (200 OK):**
```json
{
  "message": "Backend is operational",
  "data": {
    "status": "ok",
    "active_connections": 2,
    "uptime_seconds": 145
  }
}
```

---

### B. Submit Training Metric
* **Endpoint:** `POST /metrics`
* **Description:** Ingests a new training metric update from the Flower central aggregator, persists it to SQLite, and broadcasts a `metrics_updated` event to connected WebSocket clients.
* **Headers:** `Content-Type: application/json`
* **Request Body:**
```json
{
  "experiment_id": "default",
  "round_number": 1,
  "epoch": 1,
  "training_loss": 0.4521,
  "dice_score": 0.8124,
  "hospital_id": "hospital_a"
}
```
* **Response (200 OK):**
```json
{
  "message": "Metric saved successfully",
  "data": {
    "experiment_id": "default",
    "round_number": 1,
    "epoch": 1,
    "training_loss": 0.4521,
    "validation_loss": null,
    "dice_score": 0.8124,
    "iou": null,
    "hospital_id": "hospital_a"
  }
}
```

---

### C. Retrieve Metrics by Experiment
* **Endpoint:** `GET /metrics/{experiment_id}`
* **Description:** Returns all historical metrics for the given experiment ID, ordered by round number.
* **Path Parameters:** `experiment_id` (string, e.g. `default`)
* **Response (200 OK):**
```json
{
  "message": "Metrics retrieved successfully",
  "data": [
    {
      "experiment_id": "default",
      "round_number": 1,
      "training_loss": 0.4521,
      "dice_score": 0.8124
    },
    {
      "experiment_id": "default",
      "round_number": 2,
      "training_loss": 0.3891,
      "dice_score": 0.8412
    }
  ]
}
```

---

## 2. WebSocket Telemetry Stream

### WebSocket Endpoint: `ws://127.0.0.1:8000/api/v1/telemetry/ws`

Upon connecting, the client receives JSON text frames whenever new metrics are submitted by the Flower server.

#### Telemetry Frame Event Format:
```json
{
  "event": "metrics_updated",
  "data": {
    "experiment_id": "default",
    "round_number": 3,
    "training_loss": 0.3214,
    "dice_score": 0.8752,
    "hospital_id": "hospital_b"
  }
}
```

---

## 3. Autonomous Operating System REST APIs

* **`GET /api/v1/autonomous/decisions`**: Retrieves log of all executed autonomous orchestrator decisions.
* **`POST /api/v1/autonomous/orchestrate`**: Evaluates platform state across Drift, SLA, Governance, Metrics, Privacy, and Health to emit an autonomous decision (`CONTINUE`, `PAUSE`, `TRIGGER_RETRAINING`, `ABORT_ROUND`, `PROMOTE_MODEL`, `ROLLBACK_DEPLOYMENT`, `LAUNCH_HPO`).
* **`POST /api/v1/autonomous/adaptive-strategy/recommend`**: Evaluates telemetry and selects optimal FL algorithm (`FedAvg`, `FedProx`, `Scaffold`, `FedNova`, `FedAdam`).
* **`POST /api/v1/autonomous/rca/diagnose`**: Diagnoses probable root causes of training degradation with confidence scores and mitigations.
* **`GET /api/v1/autonomous/recommendations`**: Lists active operational recommendations.
* **`GET /api/v1/autonomous/experiment-planner/propose`**: Generates future experiment proposals.
* **`POST /api/v1/autonomous/self-healing/recover`**: Triggers automated recovery workflow for node failure.
* **`POST /api/v1/autonomous/deployments/initiate`**: Initiates production deployment (`CANARY`, `ROLLING`, `BLUE_GREEN`, `SHADOW`).
* **`POST /api/v1/autonomous/deployments/{id}/promote`**: Promotes candidate model to 100% active production.
* **`POST /api/v1/autonomous/deployments/{id}/rollback`**: Triggers emergency instant rollback to previous stable production version.
* **`GET /api/v1/autonomous/knowledge-graph/summary`**: Returns System Knowledge Graph summary and node/edge topology.
* **`GET /api/v1/autonomous/executive-reports/generate`**: Generates C-level executive report summary.

---

## 4. Milestone T — Enterprise Workflow Platform REST APIs

* **`POST /api/v1/enterprise/workflows/instantiate`**: Instantiates a new 14-step canonical DAG workflow.
* **`POST /api/v1/enterprise/workflows/{id}/step`**: Advances workflow execution by one step.
* **`POST /api/v1/enterprise/workflows/{id}/run`**: Executes entire workflow instance to completion.
* **`POST /api/v1/enterprise/workflows/{id}/pause`**: Pauses an in-progress workflow.
* **`POST /api/v1/enterprise/workflows/{id}/resume`**: Resumes a paused workflow.
* **`GET /api/v1/enterprise/workflows/instances`**: Lists all workflow instances and status states.
* **`POST /api/v1/enterprise/simulator/trigger`**: Triggers scenario simulation (hospital failure, latency spike, poisoning, etc.).
* **`GET /api/v1/enterprise/simulator/history`**: Lists history of executed scenario simulations.
* **`GET /api/v1/enterprise/policies`**: Returns active policy configuration thresholds.
* **`POST /api/v1/enterprise/policies/reload`**: Hot-reloads policy configuration YAMLs from disk.
* **`POST /api/v1/enterprise/scheduler/jobs`**: Schedules a new background job (retraining, drift scan, HPO, etc.).
* **`GET /api/v1/enterprise/scheduler/jobs`**: Lists all scheduled background jobs.
* **`POST /api/v1/enterprise/scheduler/jobs/{id}/execute`**: Manually executes a scheduled job.
* **`POST /api/v1/enterprise/digital-twin/predict`**: Executes predictive "What-If" digital twin impact simulation.
* **`POST /api/v1/enterprise/lifecycle/create`**: Creates a new experiment lifecycle instance.
* **`POST /api/v1/enterprise/lifecycle/{id}/transition`**: Transitions experiment stage (Dataset $\rightarrow$ Training $\rightarrow$ Deployment $\rightarrow$ Archive).
* **`GET /api/v1/enterprise/lifecycle/experiments`**: Lists experiment lifecycle records.
* **`GET /api/v1/enterprise/persistent-graph/time-travel`**: Queries persistent Knowledge Graph state as of a historical timestamp.


