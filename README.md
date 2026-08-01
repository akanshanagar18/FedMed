# Federated Brain Tumor Classification

This module implements synchronous **Federated Averaging (FedAvg)** for the BRISC MRI classification dataset. The central Flower server coordinates rounds and aggregates client model updates weighted by local sample count. It never receives or stores raw MRI images, labels, or local test sets.

## Files

- `server.py` — Flower/gRPC coordinator.
- `client.py` — one hospital participant, run separately by Hospital A, B, and C.
- `strategy.py` — FedAvg configuration and weighted metric aggregation.
- `model.py` — shared PyTorch brain-tumor CNN.
- `data.py` — local BRISC loading and demo-only three-way partitioning.

## Setup

Create a virtual environment and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Extract `archive (4).zip` so that the dataset root is `brisc2025`. It should contain `classification_task/train` and `classification_task/test`. In a real deployment each hospital instead places only its authorized images on its own machine and passes its own `--data-root`.

## Smoke test (no MRI data required)

Open four terminals. In terminal one:

```powershell
python server.py --rounds 2
```

Then start these three clients, one per terminal:

```powershell
python client.py --hospital A --synthetic
python client.py --hospital B --synthetic
python client.py --hospital C --synthetic
```

The server waits for all three clients, broadcasts the global model, receives only trained weights and example counts, applies FedAvg, and broadcasts the updated global model for the next round.

## Train with BRISC data

```powershell
python server.py --rounds 10
python client.py --hospital A --data-root .\brisc2025
python client.py --hospital B --data-root .\brisc2025
python client.py --hospital C --data-root .\brisc2025
```

Using the same root on all three processes is solely a local demonstration: `data.py` assigns disjoint deterministic shards. Do **not** do this with confidential hospital data. In deployment, run a client at each hospital and configure each client with its own local, access-controlled data path.

## Notes for review

- Transport is Flower's gRPC transport. For any network beyond a trusted lab, deploy it behind TLS and authenticated network controls.
- `min_available_clients=3` ensures a round cannot start until Hospitals A, B, and C are registered.
- The baseline CNN can be replaced by a pretrained or custom CV architecture as long as every client uses exactly the same model definition.
