# Implementation Sequence

**Parent Document:** [Platform Specification v1.1](platform_specification_v1.1.md)

## Sprint 0: Architecture Freeze
- **Objective:** Finalize contracts and repository structure.
- **Deliverable:** `platform_specification_v1.1.md`, `TEAM_START_HERE.md`.

## Sprint 1: Backend Foundation & Common Contracts
- **Objective:** Establish the shared Pydantic models.
- **Deliverable:** `common/contracts/` schemas locked in.
- **Dependencies:** None.

## Sprint 2: Medical Model & Privacy
- **Objective:** Build local PyTorch pipelines and TenSEAL wrappers.
- **Deliverable:** Isolated training loops, encryption logic.
- **Dependencies:** None.

## Sprint 3: Flower Integration
- **Objective:** Wire the local model and encryption to Flower.
- **Deliverable:** Working gRPC client-server communication.
- **Dependencies:** Sprint 1, Sprint 2.

## Sprint 4: Monitoring Platform APIs
- **Objective:** Build FastAPI ingest endpoints.
- **Deliverable:** Working backend that persists FL Server telemetry.
- **Dependencies:** Sprint 1, Sprint 3.

## Sprint 5: React Dashboard
- **Objective:** Visualize the training process.
- **Deliverable:** Live charts updating via WebSocket.
- **Dependencies:** Sprint 4.

## Sprint 6 & 7: Integration and Performance Testing
- **Objective:** Run simulated hospitals locally and test TenSEAL overhead.

## Sprint 8: Final Demonstration
- **Objective:** End-to-end presentation of the privacy-preserving cross-silo engine.
