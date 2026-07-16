# Folder Structure Guidelines

## Author
[Author Placeholder]

## Purpose
This document provides a strict map of the FedMed repository to ensure all code is placed correctly, maintaining architectural boundaries.

## Module Breakdown

### Core Engine
- `client/`: FL Client node implementation. Depends on `model/` and `privacy/`.
- `server/`: Central FL Server. Depends on `common/` for metrics schemas.

### Domain Logic
- `model/`: Neural architectures (3D U-Net), loss functions, evaluation metrics. **Must not import from `client/` or `server/`.**
- `privacy/`: TenSEAL homomorphic encryption wrappers. **Must not depend on `model/`.**
- `utils/`: Reusable scripts for image processing (NIfTI, OpenCV).

### Infrastructure
- `dashboard/`: Full-stack React/FastAPI monitoring platform.
- `configs/`: Centralized Pydantic settings.
- `common/`: Shared schemas and Python types.

### CI/CD & Tooling
- `tests/`: Unit and integration testing suites.
- `scripts/`: Shell scripts for environment setup and data downloading.
- `docs/`: Technical specifications.
