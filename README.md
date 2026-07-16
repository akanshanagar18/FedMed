# FedMed: Cross-Silo Federated Learning Engine

![FedMed Banner](https://via.placeholder.com/800x200.png?text=FedMed:+Cross-Silo+Federated+Learning)

Welcome to the **FedMed** project! This repository contains the source code for a privacy-preserving healthcare AI platform. 

FedMed enables multiple hospitals to collaboratively train deep learning models (3D U-Net) for Medical Image Segmentation (BraTS) without sharing sensitive patient data, leveraging Federated Learning and Homomorphic Encryption.

## 🚀 Getting Started
If you are a contributor, **STOP HERE**. 
Please read the [TEAM_START_HERE.md](TEAM_START_HERE.md) onboarding guide before writing any code.

## Documentation
The complete architecture and technical contracts are strictly documented in the `docs/` folder:
- [Platform Specification v1.1](docs/architecture/platform_specification_v1.1.md)
- [Module Guides](docs/modules/)

## Core Modules
1. **Federated Learning** (Flower orchestrator)
2. **Medical Model** (3D U-Net & BraTS)
3. **Privacy** (TenSEAL Homomorphic Encryption)
4. **Monitoring Platform** (FastAPI Backend & React Dashboard)

## Repository Structure

- `client/`: Federated learning client node implementation.
- `server/`: Centralized Flower server and aggregation logic.
- `model/`: Neural network architectures, loss functions, and evaluation metrics.
- `privacy/`: Encryption algorithms and differential privacy wrappers.
- `dashboard/`: Full-stack monitoring application (React frontend, FastAPI backend).
- `configs/`: Centralized configuration management and hyperparameters.
- `common/contracts/`: Shared schemas, types, and generic functions used across modules.
- `utils/`: Domain-specific processing utilities.
- `docs/`: Technical documentation and project guidelines.
- `tests/`: Unit and integration testing suites.
- `scripts/`: Helper bash/python scripts for environment setup.

## Current Progress
- ✅ Architecture Freeze (v1.1) completed.
- ⏳ Entering Implementation (Sprint 1)

## License
- [License Placeholder]
