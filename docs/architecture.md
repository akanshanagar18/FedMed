# System Architecture

## Author
[Author Placeholder]

## Overview
This document outlines the high-level architecture of FedMed, detailing the interaction between the hospital clients, the central orchestration server, and the real-time monitoring dashboard.

## Component Architecture

1. **Hospital Nodes (Clients)**
   - Responsible for local training.
   - Hosts the 3D U-Net model and private BraTS MRI dataset.
   - Encrypts model weights locally before transmission.

2. **Central Server (Aggregator)**
   - Manages the federated learning rounds via `flwr`.
   - Aggregates encrypted weights using Federated Averaging (FedAvg).
   - Generates the updated global model and broadcasts it back to clients.

3. **Monitoring Platform (Dashboard)**
   - **Backend:** FastAPI server that ingests telemetry metrics from the Central Server.
   - **Frontend:** React application displaying live metrics (Dice score, loss, connected hospitals) via WebSockets.

## Sequence Diagram (Future Implementation)
- [ ] Diagram the exact RPC calls between `client` and `server`.
- [ ] Detail the homomorphic encryption key-exchange flow.
