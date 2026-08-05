# FedMed Developer & Contributor Guide

Welcome to the FedMed developer guide! This document provides instructions for extending the platform, adding new hospital client nodes, introducing custom neural network architectures, and running local debugging workflows.

---

## 1. Development Environment Setup

Ensure Python 3.10+ and Node.js 18+ are installed on your machine.

```bash
# Clone the repository
git clone https://github.com/akanshanagar18/FedMed.git
cd FedMed

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements and editable package
pip install -r requirements.txt
pip install -e .
pip install pytest httpx

# Build frontend static assets (optional, pre-built dist included)
cd dashboard/frontend
npm install
npm run build
cd ../..
```

---

## 2. Adding a New Hospital Client Node

To connect a third hospital node (e.g., `hospital_c`), launch a new instance of `client.flower_client`:

```bash
python3 -m client.flower_client --server 127.0.0.1:8080 --hospital-id hospital_c
```

To permanently add `hospital_c` to the automated simulation orchestrator, update `scripts/run_simulation.py`:

```python
# Add Hospital Client C in scripts/run_simulation.py
client_c_proc = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "client.flower_client",
        "--server",
        config.flower_address,
        "--hospital-id",
        "hospital_c",
    ],
    cwd=PROJECT_ROOT,
)
tracker.register("Client-Hospital-C", client_c_proc)
```

---

## 3. Adding Custom Neural Network Architectures

1. Place your PyTorch model implementation in `model/your_model.py`.
2. Ensure your model extends `torch.nn.Module` and accepts 3D tensor inputs `(B, C, D, H, W)`.
3. Update `client/flower_client.py`:

```python
from model.your_model import YourCustomModel

class FedMedClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: str, device: str = "cpu"):
        self.hospital_id = hospital_id
        self.device = device
        self.model = YourCustomModel()
        self.model.to(self.device)
```

---

## 4. Code Formatting & Quality Standards

* **PEP 8 Compliance:** All code must pass `black` and `flake8` checks.
* **Type Hints:** Use explicit type hints for function parameters and return values.
* **Docstrings:** All modules, classes, and functions must include standard Google-style docstrings.
* **Pydantic Schemas:** All network payloads must be declared in [common/schemas.py](file:///Users/siddhant_patil/Projects/FedMed/common/schemas.py). Do not define duplicate data schemas locally.
