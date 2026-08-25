# Architecture Decision Records (ADR)

**Parent Document:** [Platform Specification v1.1](platform_specification_v1.1.md)

| Decision | Alternatives Considered | Pros | Cons |
|---|---|---|---|
| **Flower (flwr)** | PySyft, FATE | Lightweight, gRPC based. | Less native HE integration. |
| **FastAPI** | Flask, Django | Async, Pydantic, Auto-docs. | None for our scale. |
| **MONAI** | Pure PyTorch | Native 3D MRI transforms. | Steeper learning curve. |
| **TenSEAL** | SEAL (C++), PALISADE | Python native CKKS support. | Heavy ciphertext payload sizes. |
| **Monorepo** | Polyrepo | Single source of truth. | Complex CI/CD pipelines. |
