# FEDMED OS — DISTRIBUTED ARCHITECTURE AUDIT & SPECIFICATION

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Flower gRPC Communication, Distributed Topologies, and Hardware Boundaries (Phase 8.5B.6)  

---

## 1. Execution Topologies & Definitions

FedMed OS distinguishes four distinct execution topologies to maintain absolute scientific integrity:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FEDMED TOPOLOGY SPECTRUM                        │
├──────────────────────────┬─────────────────────────┬───────────────────┤
│ Topology Level           │ Transport / Execution   │ Hardware Boundary │
├──────────────────────────┼─────────────────────────┼───────────────────┤
│ 1. LOCAL STRATEGY        │ In-Process Tensor /     │ Single Process,   │
│    (Pipeline Simulation) │ Direct Function Calls   │ Single Machine    │
│                          │                         │                   │
│ 2. MULTI-PROCESS NETWORK │ gRPC / Socket over      │ Multi-Process,    │
│    (Local Flower Net)    │ 127.0.0.1:8080          │ Single Machine    │
│                          │                         │                   │
│ 3. DISTRIBUTED           │ TLS-Secured gRPC / WAN  │ Multiple Physical │
│    MULTI-NODE            │ Network Sockets         │ Network Nodes     │
│                          │                         │                   │
│ 4. MULTI-GPU             │ PyTorch DDP / NCCL      │ Multiple Physical │
│    ACCELERATION          │ or Gloo inter-GPU       │ GPU Accelerators  │
└──────────────────────────┴─────────────────────────┴───────────────────┘
```

---

## 2. Current Environment Capabilities & Classification

Based on the automated environment audit ([`reports/environment/distributed_environment.json`](file:///Users/siddhant_patil/Projects/FedMed/reports/environment/distributed_environment.json)):

- **Host Machine:** Apple Silicon (macOS Darwin 25.6.0, arm64)
- **CPU:** 10 cores
- **RAM:** 16.0 GB total
- **Accelerators:** 1x Apple Silicon MPS (Metal Performance Shaders) GPU
- **Distributed Backends:** PyTorch Gloo (`True`), NCCL (`False`)
- **Topology Classification:** `SINGLE_NODE_SINGLE_GPU`

### Integrity Status:
- **Local In-Process Execution:** `AVAILABLE & OPERATIONAL`
- **Multi-Process Local Flower Network:** `AVAILABLE & OPERATIONAL`
- **Multi-Node Distributed Execution:** `BLOCKED_HARDWARE (Single Physical Host Available)`
- **Multi-GPU Parallelism:** `BLOCKED_HARDWARE (Single MPS Accelerator Available)`

---

## 3. Networked Flower Communication Path

For multi-process networked Flower execution:
1. **Server Process:** Launches Flower Server with strategy (`FedAvg`, `FedProx`) listening on `0.0.0.0:8080`.
2. **Client Processes:** `hospital_alpha` and `hospital_beta` launch in separate OS processes, connect via gRPC to `127.0.0.1:8080`, receive serialized weights, execute local training on private datasets, and stream model updates back to the coordinator.
