# FedMed v2.0 Production Container Architecture Specification

## 1. Container Topology Overview

FedMed v2.0 is containerized into 5 independent microservice containers orchestrated via Docker Compose over an isolated bridge network (`fedmed-network`).

```
+---------------------------------------------------------------------------------------------------+
|                                      fedmed-network (Bridge)                                      |
|                                                                                                   |
|  +-------------------+       +--------------------+       +------------------------------------+  |
|  |  fedmed-backend   |       |  fedmed-dashboard  |       |        fedmed-flower-server         |  |
|  |  (FastAPI Port    | <---> |  (Nginx Port 80/   | <---> |        (Flower gRPC Port 8080)      |  |
|  |   8000)           |       |   3000)            |       |                                    |  |
|  +-------------------+       +--------------------+       +------------------------------------+  |
|            ^                                                                ^                     |
|            |                                                                |                     |
|            +─────────────────────────────────┬──────────────────────────────┤                     |
|                                              |                              |                     |
|                                              v                              v                     |
|                                  +-----------------------+      +-----------------------+         |
|                                  | fedmed-hospital-alpha |      | fedmed-hospital-beta  | ...     |
|                                  | (Dirichlet Silo 1)    |      | (Dirichlet Silo 2)    |         |
|                                  +-----------------------+      +-----------------------+         |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Service Definitions & Exposed Ports

| Service Name | Container Name | Image Dockerfile | Internal Port | Host Port | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **backend** | `fedmed-backend` | `docker/backend.Dockerfile` | `8000` | `8000` | FastAPI REST API & SQLite DB |
| **flower-server** | `fedmed-flower-server` | `docker/flower_server.Dockerfile` | `8080` | `8080` | Central Flower gRPC Server |
| **hospital-alpha** | `fedmed-hospital-alpha` | `docker/hospital_client.Dockerfile` | - | - | Hospital Silo Alpha Client |
| **hospital-beta** | `fedmed-hospital-beta` | `docker/hospital_client.Dockerfile` | - | - | Hospital Silo Beta Client |
| **hospital-gamma** | `fedmed-hospital-gamma` | `docker/hospital_client.Dockerfile` | - | - | Hospital Silo Gamma Client |
| **dashboard** | `fedmed-dashboard` | `docker/dashboard.Dockerfile` | `80` | `3000` | React Production Nginx Web Server |

---

## 3. Persistent Volumes

| Volume Name | Target Container Path | Purpose |
| :--- | :--- | :--- |
| `fedmed_database_data` | `/app/data_db` | SQLite database persistence (`fedmed.db`) |
| `fedmed_checkpoint_data` | `/app/checkpoints` | PyTorch 3D UNet model weight checkpoints |
| `fedmed_certificate_data` | `/app/certs` | Production X.509 TLS certificates & keys |
| `fedmed_benchmark_results` | `/app/benchmarks` | Benchmark JSON export persistence |
| `fedmed_logs_data` | `/app/logs` | Subsystem execution logs |
