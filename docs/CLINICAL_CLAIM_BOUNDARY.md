# FEDMED OS — CLINICAL CLAIM BOUNDARY & REGULATORY SPECIFICATION

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Regulatory Status, Validation Boundaries, and Explicit Non-Claims  

---

## 1. Supported vs Blocked Claim Categories

| Claim Category | Status | Engineering Basis |
|---|---|---|
| **Software Architecture Validation** | `SUPPORTED` | Multi-hospital federated training, aggregation, and validation pipelines proven operational. |
| **Algorithmic Correctness** | `SUPPORTED` | FedAvg, FedProx, and Differential Privacy mathematically verified against exact analytical formulations. |
| **Cryptographic Integrity** | `SUPPORTED` | TenSEAL CKKS homomorphic aggregation verified on live parameter tensors with $< 10^{-7}$ approximation error. |
| **Real Medical AI Performance** | `BLOCKED` | Full BraTS 2021 cohort (1,251 patients, ~40 GB) is not present locally; 4-case mini-cohort cannot demonstrate clinical efficacy. |
| **Clinical Generalization** | `BLOCKED` | Insufficient patient cohort variance for multi-center generalization claims. |
| **Clinical Deployment as Medical Device** | `NOT CLAIMED` | FedMed is an experimental research platform, not an approved medical device. |
| **Regulatory Approval (FDA / CE-MDR / SaMD)**| `NOT CLAIMED` | No clinical trials, 510(k), or CE mark certifications are claimed or implied. |

---

## 2. Model Artifact Designation

All model artifacts produced prior to full-cohort real BraTS ingestion must strictly carry the artifact designation:

$$\text{ARTIFACT\_CLASSIFICATION} = \mathbf{DEVELOPMENT\_VALIDATION\_ARTIFACT}$$
$$\text{TRAINING\_DATASET\_MODE} = \mathbf{DEVELOPMENT\_SYNTHETIC}$$
