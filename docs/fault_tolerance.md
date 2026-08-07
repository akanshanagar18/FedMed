# FedMed v2.0 Node Resilience & Fault-Tolerant Federated Runtime

## 1. Node Resilience Architecture

FedMed v2.0 provides fault-tolerant distributed federated learning capable of surviving hospital node failures during active training rounds.

```
                   +--------------------------------+
                   |     Central Flower Server      |
                   |  min_fit_clients = 2           |
                   |  accept_failures = True        |
                   +--------------------------------+
                                   ▲
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
+-----------------+       +-----------------+       +-----------------+
| Hospital Alpha  |       | Hospital Beta   |       | Hospital Gamma  |
| State: ONLINE   |       | State: FAILED   |       | State: ONLINE   |
| (Active Participant)    | (Killed Round 2)|       | (Active Participant)
+-----------------+       +-----------------+       +-----------------+
```

---

## 2. Node Lifecycle & State Machine

```
              +-------------+
              |   OFFLINE   |
              +-------------+
                     | (Start Client)
                     v
              +-------------+
              |   ONLINE    | <─────────────┐
              +-------------+               │ (Auto-Reconnect)
                     | (Fit Round)          │
                     v                      │
              +-------------+        +---------------+
              |   ACTIVE    | ─────> | RECONNECTING  |
              +-------------+ (Drop) +---------------+
                     |                      
                     | (Process Kill / Crash)
                     v
              +-------------+
              |   FAILED    |
              +-------------+
```

---

## 3. Failure Handling & Round Continuation

- **Strategy Configuration**: Configured with `min_fit_clients = 2`, `min_available_clients = 2`, `accept_failures = True`.
- **Fault Recovery**: If `Hospital Beta` dies (SIGTERM / crash) during Round 2, the Flower server accepts the failure, aggregates results from `Hospital Alpha` and `Hospital Gamma`, and successfully completes Round 2 and Round 3.
- **REST Status Endpoint**: Exposes `GET /api/v1/nodes` and `POST /api/v1/nodes/heartbeat` for real-time dashboard state tracking (`ONLINE`, `OFFLINE`, `RECONNECTING`, `FAILED`, `ACTIVE`).
