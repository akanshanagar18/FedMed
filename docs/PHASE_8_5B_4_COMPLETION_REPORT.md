# FEDMED OS — PHASE 8.5B.4 COMPLETION REPORT

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Centralized Baseline + FedAvg + FedProx Reproducible Experiment & Benchmark Integrity Gate (Phase 8.5B.4)  
**Execution Mode:** `DEVELOPMENT_SYNTHETIC`  
**Gate Status:** `PASS`  
**Real Data Status:** `BLOCKED — REAL BRATS DATASET NOT PRESENT`  
**Scientific Performance Status:** `BLOCKED — DEVELOPMENT COHORT INSUFFICIENT FOR CLINICAL CLAIMS`  

---

## 1. Executive Summary

Phase 8.5B.4 establishes the first scientifically rigorous, reproducible, multi-algorithm benchmarking framework in FedMed:
1. **Centralized Baseline, FedAvg, and FedProx** are evaluated under strictly identical conditions:
   - Identical dataset split (`4ce10ac52c093ec0971f306b6d4145e383ad92eeb231c551c6f841b64864384f`)
   - Identical canonical MONAI 3D UNet (4,810,074 parameters)
   - Identical initial parameter fingerprint ($H_0 = \text{5ba650f3408...}$)
   - Identical preprocessing and optimizer hyperparameters (`lr=1e-4, weight_decay=1e-5`)
   - Equal optimization budget (6 total cohort sample optimization passes)
2. **Zero Static Lookups / Zero Heuristics:**
   - Historical mock profiles (`algo_profiles`) in `evaluation/benchmark_runner.py` were replaced with live tensor forward/backward/Adam executions.
   - All Dice and IoU metrics are calculated with decoupled `evaluation.metrics` functions.
3. **True FedProx Proximal Regularization:**
   - Client-side proximal loss $\mathcal{L}_{\text{prox}}(w) = \mathcal{L}_{\text{local}}(w) + \frac{\mu}{2} \sum \|w_i - w_{\text{global}, i}\|^2$ implemented and verified across 8 unit test properties.
4. **Test Set Isolation & Single Final Evaluation:**
   - Zero test data access during training (`test_access_count = 0`).
   - Evaluated exactly once per algorithm upon training completion.
5. **Reproducibility & Persistence:**
   - Benchmark repeated twice with seed 42. Verified reproducibility within platform tolerance ($\Delta < 0.002$ on Apple Silicon MPS).
   - Results persisted to SQLite `fedmed.db` and JSON audit reports in `reports/experiments/benchmark/<run_id>/`.
6. **MLflow Dependency Resolution:**
   - Optional MLflow dependency handled cleanly with graceful fallback.

---

## 2. Actual Measured Benchmark Results (Run 1 vs Run 2)

| Algorithm | Val TC Dice | Val WT Dice | Val ET* Dice | Val Mean Dice | Val Mean IoU | Test TC Dice | Test WT Dice | Test ET* Dice | Test Mean Dice | Test Mean IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| **Centralized** | 0.4524 | 0.6320 | 0.0000 | **0.3615** | 0.2514 | 0.4522 | 0.6301 | 0.0000 | **0.3608** | 0.2507 |
| **FedAvg** | 0.4549 | 0.6269 | 0.0000 | **0.3606** | 0.2503 | 0.4526 | 0.6250 | 0.0000 | **0.3592** | 0.2490 |
| **FedProx ($\mu=0.01$)** | 0.4538 | 0.6283 | 0.0000 | **0.3607** | 0.2505 | 0.4517 | 0.6245 | 0.0000 | **0.3588** | 0.2486 |

*\*ET ground-truth positive voxels are 0 in development synthetic masks. ET metric evaluating to 0.0 reflects label limitations rather than algorithm failure.*

---

## 3. Scientific Claim Boundary

> [!WARNING]
> This experiment was conducted on a 4-case development cohort (`BraTS2021_00001` to `00004`). The resulting scores demonstrate **software execution correctness, algorithmic fairness, and measurement pipeline integrity**, but **MUST NEVER be used to make clinical, medical efficacy, or statistical generalization claims**. Full-cohort BraTS dataset ingestion is required for clinical benchmarking.
