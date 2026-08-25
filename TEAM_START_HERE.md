# Welcome to FedMed

Welcome to the **Cross-Silo Federated Learning Engine for Medical Image Segmentation**.

This document is your starting point as a contributor to FedMed. Please read it entirely before writing any code.

## ⚠️ Architecture Freeze Notice
The architecture is **100% FROZEN (v1.1)**. 
We are now entering the Implementation Phase (Sprint 1). 
- **DO NOT** change API contracts.
- **DO NOT** change module ownership.
- **DO NOT** change shared schemas.
If you believe a change is absolutely necessary, you must submit a formal RFC and request a review from the Principal Architect.

## Project Overview
FedMed allows multiple hospitals to train a 3D U-Net on private BraTS MRI datasets collaboratively using Federated Learning (via Flower) and Homomorphic Encryption (via TenSEAL), coordinated by a central Monitoring Platform (via FastAPI/React).

## Who Owns What?
- **Federated Learning (`client/`, `server/`)**: Vishnu
- **Medical Model (`model/`)**: Mohit & Shreya
- **Privacy (`privacy/`)**: Akansha & Rakshit
- **Monitoring Platform (`dashboard/`)**: Siddhant
- **Shared Contracts (`common/contracts/`)**: Collective Ownership

## What You Must Read Before Coding
You must read the guide specific to your module. It contains your inputs, outputs, and checklist:
- [Federated Learning Guide](docs/modules/federated_learning.md)
- [Medical Model Guide](docs/modules/medical_model.md)
- [Privacy Guide](docs/modules/privacy.md)
- [Monitoring Platform Guide](docs/modules/monitoring_platform.md)

## Where to Find the Architecture & Contracts
The Master Specification has been decomposed into readable chunks. **Never duplicate these documents.**
- **Master Blueprint:** [Platform Specification v1.1](docs/architecture/platform_specification_v1.1.md)
- **Shared Contracts/Schemas:** [Shared Contracts Reference](docs/architecture/shared_contracts.md)
- **REST APIs:** [API Contracts](docs/architecture/api_contracts.md)
- **WebSockets:** [WebSocket Contracts](docs/architecture/websocket_contracts.md)
- **State Machines:** [System State Machines](docs/architecture/state_machines.md)

## Development & Git Workflow
1. **Branching:** Branch off `development`. Use `feature/module-name` (e.g., `feature/model-unet`).
2. **Commit Messages:** Follow Conventional Commits (`feat:`, `fix:`, `docs:`).
3. **Pull Requests:** Must be opened against `development`.
4. **Code Review:** Requires at least 1 approval from a different module owner.
5. **Definition of Done:** 
   - Code is complete and cleanly formatted (`black`).
   - Strict type hints applied and pass `mypy`.
   - Unit tests run successfully.
   - You have strictly adhered to the `common/contracts/` schemas.

Let's build the future of Healthcare AI!
