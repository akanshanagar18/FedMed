# Federated Learning Module Guide

**Owner:** Vishnu
**Folder:** `client/` and `server/`

## Objective
Orchestrate the decentralized training process using Flower. The client triggers local PyTorch training and passes weights for encryption. The server orchestrates rounds, performs FedAvg on ciphertexts, and pushes telemetry to the Monitoring Platform.

## Dependencies
- **Medical Model (`model/`)**: For instantiating the 3D U-Net and executing local training.
- **Privacy (`privacy/`)**: For encrypting weights before network transmission.
- **Shared Contracts (`common/contracts/`)**: For standardizing API payloads.

## Integration Points
- **Expected Inputs:** PyTorch Tensors (from Model), TenSEAL Ciphertexts (from Privacy).
- **Expected Outputs:** JSON payloads pushed to FastAPI (`POST /api/v1/metrics`, `POST /api/v1/round/status`).

## Development Checklist
- [ ] Implement `FlowerClient` wrapper.
- [ ] Implement Custom `Strategy` (FedAvg) that handles TenSEAL operations.
- [ ] Wire server lifecycle hooks to push REST updates.

## Common Mistakes to Avoid
- **Do not** hardcode neural network shapes in the client. Accept the model via dependency injection.
- **Do not** skip schema validation. Always use Pydantic models from `common/contracts/` before pushing data.
