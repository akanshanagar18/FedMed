# FedMed v2.0 — Enterprise Federated AI Platform for Medical Image Segmentation

[![FedMed Release v2.0.0](https://img.shields.io/badge/Release-v2.0.0--Production-emerald.svg)](https://github.com/akanshanagar18/FedMed)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Test Suite Status](https://img.shields.io/badge/tests-258%2F258%20PASSED-success.svg)](https://github.com/akanshanagar18/FedMed)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FedMed v2.0 is a production-grade, cross-silo **Enterprise Federated AI Operating System** engineered for privacy-preserving 3D Brain Tumor MRI Segmentation (**BraTS**). It empowers healthcare institutions and medical research facilities to collaboratively train MONAI 3D U-Net models without exposing sensitive Patient Health Information (PHI).

---

## 🌟 Executive Overview & Problem Statement

Medical imaging datasets are severely fragmented across hospitals due to strict data privacy regulations (HIPAA, GDPR). Centralizing patient MRI scans for model training is legally and ethically impossible. 

**FedMed solves this dilemma by bringing model training to the data:**
* **Decentralized Local Training:** Hospital client nodes train PyTorch MONAI models on private local MRI scans.
* **Homomorphic Encryption & Differential Privacy:** Model parameters are protected using **TenSEAL CKKS** homomorphic encryption and **Opacus DP** before egress.
* **Central Aggregation & Adaptive Strategy Engine:** The Flower FL server aggregates encrypted weight updates via **FedAvg, FedProx, SCAFFOLD, FedAdam, FedYogi** with adaptive strategy selection.
* **Autonomous Operating System:** Live feature drift detection (MMD), HIPAA/GDPR SLA auditing, self-healing node recovery, and automated candidate model promotion.
* **Real-time Telemetry:** A FastAPI backend and React Grafana-style monitoring dashboard stream live training loss, Dice similarity scores, and node health over WebSockets.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Hospital Silo Alpha
        CA[Hospital Node Alpha] -->|PyTorch local fit| MA[MONAI 3D U-Net]
        CA -->|TenSEAL CKKS| EA[Encrypted Weights A]
    end

    subgraph Hospital Silo Beta
        CB[Hospital Node Beta] -->|PyTorch local fit| MB[MONAI 3D U-Net]
        CB -->|TenSEAL CKKS| EB[Encrypted Weights B]
    end

    subgraph Central Server Infrastructure
        EA -->|gRPC Port 8080| FS[Flower Central Aggregator]
        EB -->|gRPC Port 8080| FS
        FS -->|FedAvg / Adaptive Aggregation| GM[Global Model Parameters]
        FS -->|HTTP POST /api/v1/metrics| BE[FastAPI Backend Server]
    end

    subgraph Autonomous OS & Observability
        BE -->|SQLAlchemy ORM| DB[(SQLite fedmed.db)]
        BE -->|EventBus / Choreographer| Auto[Autonomous Orchestrator]
        BE -->|WebSocket Broadcast| WS[ws://127.0.0.1:8000/api/v1/telemetry/ws]
        WS -->|Live Telemetry| UI[React Grafana Dashboard]
    end
```

---

## 🛠️ Technology Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | PyTorch 2.x, MONAI 1.3+ | 3D U-Net Architecture & BraTS MRI Tensor Pipelines |
| **Federated Engine** | Flower (`flwr`) 1.7+ | Cross-silo gRPC Client-Server Orchestration & Pluggable Strategies |
| **Privacy Preserving**| TenSEAL 0.3+, Opacus | CKKS Scheme Homomorphic Encryption & Differential Privacy |
| **Autonomous OS** | EventBus, Choreographer | Drift Detection, SLA Auditing, Self-Healing, HPO & Promotion |
| **Backend API** | FastAPI, Uvicorn, SQLAlchemy | REST API, SQLite Persistence & WebSocket Streaming |
| **Frontend UI** | React 18, Recharts, Vite | Grafana/Prometheus-style Dark Telemetry Dashboard |
| **Testing & CI** | Pytest, GitHub Actions | Automated 258-Test Regression Suite & Multi-Job CI/CD Pipeline |

---

## 📁 Repository Folder Structure

```text
FedMed/
├── analytics/              # Executive reports, RCA, recommendations, experiment planner
├── client/                 # Flower NumPyClient implementation for hospital nodes
├── common/                 # Canonical Pydantic schemas and shared type definitions
├── configs/                # Centralized environment settings, policy engine & YAML configs
├── dashboard/              # Full-stack monitoring platform
│   ├── backend/            # FastAPI REST & WebSocket server
│   └── frontend/           # React + Recharts dashboard UI
├── data/                   # Dataset directory for MRI volumes
├── deployment/             # Production deployment manager (Canary, Rolling, Blue-Green)
├── digital_twin/           # Federated Digital Twin simulation engine
├── docker/                 # Production Dockerfiles (Backend, Dashboard, Flower, Hospital)
├── docs/                   # Platform documentation (Architecture, API, Demo, Dev)
├── events/                 # EventBus Pub/Sub and Event Choreographer engine
├── governance/             # Federated drift detection & HIPAA/GDPR SLA auditing
├── knowledge/              # Persistent Knowledge Graph & Time-Travel engine
├── orchestrator/           # Autonomous Orchestrator Engine
├── privacy/                # TenSEAL CKKS homomorphic encryption & TLS certificates
├── resilience/             # Self-healing node recovery engine
├── scheduler/              # Enterprise background job scheduler
├── scripts/                # Production simulation orchestrator script
├── server/                 # Flower central server with Strategy adapters
├── tests/                  # Automated pytest suite (188 unit, 64 integration, 6 e2e)
├── workflows/              # Declarative workflow engine
├── docker-compose.yml      # Multi-container production deployment stack
├── Makefile                # Standardized CLI commands
├── pyproject.toml          # Packaging setup and tool configurations
├── pytest.ini              # Pytest framework settings
└── requirements.txt        # Production dependency manifest
```

---

## 🚀 Standardized Execution Entrypoints

All platform operations are standardized via `Makefile` from the repository root:

```bash
# 1. Zero-Setup Enterprise Demo (Single Command)
make demo             # Or: python3 demo.py

# 2. Launch FastAPI Backend Server (Port 8000)
make backend          # Or: uvicorn app.main:app --app-dir dashboard/backend --port 8000

# 3. Launch React Frontend Dev Server (Port 3000)
make frontend         # Or: cd dashboard/frontend && npm run dev

# 4. Execute Multi-Hospital Federated Learning Simulation
make simulation       # Or: python3 scripts/run_simulation.py

# 5. Launch Containerized Platform Stack via Docker Compose
make compose          # Or: docker compose up --build -d
```

### 🌐 Accessing System Services
Once launched, open your browser:
* **Live Dashboard UI:** `http://127.0.0.1:8000/` (or `http://localhost:3000` in dev)
* **Swagger API Docs:** `http://127.0.0.1:8000/docs`
* **WebSocket Telemetry Stream:** `ws://127.0.0.1:8000/api/v1/telemetry/ws`
* **MLflow Tracking UI:** `http://127.0.0.1:5000` (`make mlflow`)
* **TensorBoard UI:** `http://127.0.0.1:6006` (`make tensorboard`)

---

## 🧪 Automated Testing & CI Pipeline

Run the automated test suite from repository root:

```bash
# Run complete test suite (258 Tests across Unit, Integration, E2E)
make test             # Or: pytest

# Run specific test suites
make test-unit        # Or: pytest tests/unit
make test-integration # Or: pytest tests/integration
make test-e2e         # Or: pytest tests/e2e
```

### GitHub Actions Integration
Every push to `main` or `release/*` automatically triggers `.github/workflows/ci.yml`, executing backend tests, OpenAPI spec validation, frontend build, and Docker image validation.

---

## 📚 Detailed Documentation

* [Architecture Specification](docs/architecture.md) — Subsystem protocols & sequence diagrams
* [REST & WebSocket API Guide](docs/api.md) — Endpoint specifications & JSON contracts
* [Developer & Contributor Guide](docs/developer_guide.md) — Adding models, nodes, and debugging
* [Live Demonstration Script](docs/demo.md) — Step-by-step judge demonstration guide

---

## 📄 License & Legal

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
