# FedMed v2.0 — Quickstart Guide

## 🚀 1-Minute Zero-Setup Demo Execution

FedMed v2.0 can be launched out of the box with zero configuration:

```bash
# Clone and enter directory
git clone https://github.com/akanshanagar18/FedMed.git
cd FedMed

# Run single-command demo launcher
python demo.py
# or
make demo
```

What `python demo.py` performs automatically:
1. Bootstraps multi-hospital edge runtime (Hospital Alpha, Beta, Gamma, Delta).
2. Executes live round-by-round federated training with PyTorch MONAI 3D U-Net models.
3. Simulates cross-silo feature drift incident ($MMD = 0.18$).
4. Triggers Self-Healing Recovery engine to adapt local hyperparameters.
5. Promotes best candidate model to production via Canary deployment gate.
6. Generates a C-level executive summary report.

---

## 🐳 Docker Compose Launcher

To run the entire containerized stack:

```bash
docker compose up --build -d
```

Services exposed:
* **React Dashboard UI:** `http://localhost:3000`
* **FastAPI Backend:** `http://localhost:8000/docs`
* **Flower gRPC Server:** `127.0.0.1:8080`
* **Grafana Observability:** `http://localhost:3001`
* **Prometheus Metrics:** `http://localhost:9090`
