# FEDMED OS — REAL BRATS READINESS SPECIFICATION

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Dataset Transition, Zero Code Change Ingestion Pipeline  

---

## 1. Zero Code Change Ingestion Path

When the full real BraTS 2021 cohort (~40 GB, 1,251 cases) is placed in `data/BraTS2021/`:

1. **Discovery & Validation:**
   - `python3 -m data.real_brats_pipeline` executes automatically.
   - Automatically detects $240 \times 240 \times 155$ spatial dimensions and label `4` (Enhancing Tumor).
   - Upgrades `dataset_mode` from `DEVELOPMENT_SYNTHETIC` to `FULL_REAL_BRATS`.
2. **Deterministic Patient-Level Split:**
   - 70% Train (~875 subjects), 15% Validation (~188 subjects), 15% Held-out Test (~188 subjects).
   - Generates cryptographic split hash without subject-level leakage.
3. **Multi-Hospital Partitioning:**
   - Partitions ~875 training cases across `hospital_alpha`, `hospital_beta`, `hospital_gamma`, and `hospital_delta`.
   - Replaces `EMPTY_NO_TRAINING_DATA` with populated active queues (~218 cases per hospital).
4. **Execution Engines Unchanged:**
   - Canonical 3D UNet (`model/unet3d.py`), Adam optimizer, FedAvg, FedProx, DP, and HE engines execute identically without code refactoring.
