# FedMed — Cross-Silo Federated Learning Engine for Medical Image Segmentation

[![FedMed CI Pipeline](https://github.com/akanshanagar18/FedMed/actions/workflows/ci.yml/badge.svg)](https://github.com/akanshanagar18/FedMed/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

FedMed is a production-grade, cross-silo **Federated Learning (FL)** platform engineered for privacy-preserving 3D Brain Tumor MRI Segmentation (**BraTS**). It empowers healthcare institutions and medical research facilities to collaboratively train deep learning models (MONAI 3D U-Net) without sharing sensitive patient health information (PHI).

---

## 🌟 Executive Overview & Problem Statement

Medical imaging datasets are severely fragmented across hospitals due to strict data privacy regulations (HIPAA, GDPR). Centralizing patient MRI scans for model training is legally and ethically impossible. 

**FedMed solves this dilemma by bringing model training to the data:**
* **Decentralized Local Training:** Hospital client nodes train PyTorch MONAI models on private local MRI scans.
* **Homomorphic Encryption:** Model parameters are protected using **TenSEAL CKKS** homomorphic encryption before egress.
* **Central Aggregation:** The Flower FL server aggregates encrypted weight updates via **FedAvg** without accessing raw patient data.
* **Real-time Telemetry:** A FastAPI backend and React Grafana-style monitoring dashboard stream live training loss and Dice similarity score metrics over WebSockets.

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
        FS -->|FedAvg Weight Aggregation| GM[Global Model Parameters]
        FS -->|HTTP POST /api/v1/metrics| BE[FastAPI Backend Server]
    end

    subgraph Persistence & Observability
        BE -->|SQLAlchemy ORM| DB[(SQLite fedmed.db)]
        BE -->|WebSocket Broadcast| WS[ws://127.0.0.1:8000/ws]
        WS -->|Live Telemetry| UI[React Grafana Dashboard]
    end
```

---

## 🛠️ Technology Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning** | PyTorch 2.x, MONAI 1.3+ | 3D U-Net Architecture & BraTS MRI Tensor Pipelines |
| **Federated Engine** | Flower (`flwr`) 1.7+ | Cross-silo gRPC Client-Server Orchestration & FedAvg Strategy |
| **Privacy Preserving**| TenSEAL 0.3+ | CKKS Scheme Homomorphic Vector Encryption |
| **Backend API** | FastAPI, Uvicorn, SQLAlchemy | REST API, SQLite Persistence & WebSocket Broadcasting |
| **Frontend UI** | React 18, Recharts, Vite | Grafana/Prometheus-style Dark Telemetry Dashboard |
| **Testing & CI** | Pytest, GitHub Actions | Automated Regression Suite & Continuous Integration |

---

## 📁 Repository Folder Structure

```text
FedMed/
├── client/                 # Flower NumPyClient implementation for hospital nodes
├── common/                 # Canonical Pydantic schemas and shared type definitions
├── configs/                # Centralized environment settings & system constants
├── dashboard/              # Full-stack monitoring platform
│   ├── backend/            # FastAPI REST & WebSocket server
│   └── frontend/           # React + Recharts dashboard UI
├── data/                   # Dataset directory for MRI volumes
├── docs/                   # Platform documentation (Architecture, API, Demo, Dev)
├── model/                  # PyTorch MONAI 3D U-Net model and trainer
├── privacy/                # TenSEAL CKKS homomorphic encryption utilities
├── scripts/                # Production simulation orchestrator script
├── server/                 # Flower central server with custom MetricsReporter strategy
├── tests/                  # Automated pytest suite (unit, integration, e2e)
├── .github/                # CI workflows, issue templates, and PR guidance
├── pyproject.toml          # Packaging setup and tool configurations
├── pytest.ini              # Pytest framework settings
└── requirements.txt        # Production dependency manifest
```

---

## 🚀 Quick Start: Running the Simulation

Execute the single-command production orchestrator from the repository root:

```bash
# 1. Clone repository
git clone https://github.com/akanshanagar18/FedMed.git
cd FedMed

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Launch End-to-End Simulation
python3 scripts/run_simulation.py
```

### 🌐 Accessing the Live System
Once launched, open your web browser to access:
* **Live Dashboard UI:** `http://127.0.0.1:8000/`
* **Swagger API Docs:** `http://127.0.0.1:8000/docs`
* **WebSocket Telemetry Stream:** `ws://127.0.0.1:8000/api/v1/telemetry/ws`

---

## 🧪 Automated Testing & CI Pipeline

Run the full automated test suite locally:

```bash
# Run unit and integration tests
pytest tests/unit tests/integration

# Run end-to-end simulation smoke test
pytest tests/e2e/test_simulation_e2e.py
```

### GitHub Actions Integration
Every push to `main` or feature branches automatically triggers `.github/workflows/ci.yml`, running unit, integration, and e2e smoke tests on Ubuntu environments.

---

## 📚 Detailed Documentation

* [Architecture Specification](docs/architecture.md) — Sequence diagrams & subsystem protocols
* [REST & WebSocket API Guide](docs/api.md) — Endpoint specifications & JSON contracts
* [Developer & Contributor Guide](docs/developer_guide.md) — Adding models, nodes, and local debugging
* [3-Minute Live Demonstration Script](docs/demo.md) — Judge walkthrough guide

---

## 📄 License & Legal

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
