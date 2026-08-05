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
