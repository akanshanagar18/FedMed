# FEDMED OS — PHASE 10.1: REAL-DATA FEDAVG READINESS & AUDIT REPORT

**Date:** August 16, 2026  
**Auditor / Engineer:** Senior ML Systems Engineer  
**Scope:** Forensic Readiness Audit & Implementation Gate for Experiment B (Real-Data FedAvg across 4 Hospital Silos)  
**Dataset:** BraTS-GLI 2024 Adult Glioma Post-Treatment Training Cohort  
**Canonical Split Hash:** `d0358ca42d4bf510624bbe7e86c3e4bc1e25a6174521b1280c0001eb2c66005c`  
**Hospital Partitions Hash:** `4d5905c88558883cadf9463517d3c9f856a353e4f3a570c40515f16b763fc4f8`  
**Execution Environment:** Apple Silicon MPS (`mps` — 1 GPU, 16 GB Unified Memory)  
**Master JSON Audit Artifact:** [`reports/real_brats2024/fedavg_readiness_audit.json`](file:///Users/siddhant_patil/Projects/FedMed/reports/real_brats2024/fedavg_readiness_audit.json)  
**Final Audit Verdict:** **`READY_FOR_EXPERIMENT_B`**  

---

## 1. Executive Verification Matrix

| Audit Pillar | Focus of Investigation | Verification Finding | Verdict |
|---|---|---|---|
| **1. Data Integrity** | 944 Train (236/hospital), 202 Val, 204 Test | Zero overlap; disjoint silo partitions; all 1,350 cases on disk | **PASS** |
| **2. Model Architecture** | MONAI 3D U-Net (4 in, 3 out, 4,810,074 params) | Parameter count, tensor shapes, and $128^3$ resolution verified | **PASS** |
| **3. Preprocessing & Semantics** | Modalities `[t1, t1ce, t2, flair]`, labels TC, WT, ET, RC | Target canonicalizer verified; RC preserved in metadata | **PASS** |
| **4. Training Configuration** | `DiceCELoss(lambda_dice=1.0, lambda_ce=0.2)`, Adam | Exact match with frozen Experiment A configuration | **PASS** |
| **5. FedAvg Mathematics** | Sample-weighted aggregation vs NumPy reference | Element-wise equivalence verified: $\max \|W_{\text{server}} - W_{\text{numpy}}\| = 0.00$ | **PASS** |
| **6. MPS Execution Safety** | Sequential client simulation on Apple Silicon | Prevents multi-client MPS contention; resume checkpointing active | **PASS** |
| **7. Telemetry & Payload** | Per-client loss, global val loss, Dice/IoU, bandwidth | Payload: $18.35\text{ MB}$/client ($73.4\text{ MB}$/round total) | **PASS** |
| **8. Provenance Manifest** | Split hash, partition hash, config hash, seed | Fully recorded in machine-readable JSON manifests | **PASS** |
| **9. Test Firewall** | Locked 204 test cases; zero training/validation access | Static and runtime verification confirms 0 test accesses | **PASS** |
| **10. Real Smoke Test** | 1-round 4-client real-data execution at $128^3$ | Client divergence ($0.9688$) and global update ($0.4398$) verified | **PASS** |

---

## 2. Frozen Centralized Baseline Reference (Experiment A)

The centralized baseline established in Phase 10.0 provides the empirical benchmark against which Experiment B will be evaluated:

| Target Sub-Region | Metric | Frozen Centralized Baseline Score (Exp A) |
|---|---|---|
| **Whole Tumor (WT)** | **Dice Similarity Coefficient** | **`0.8056`** (80.56%) |
| **Tumor Core (TC)** | **Dice Similarity Coefficient** | **`0.5548`** (55.48%) |
| **Enhancing Tumor (ET)** | **Dice Similarity Coefficient** | **`0.5502`** (55.02%) |
| **Composite Multi-Region** | **Macro Average Dice** | **`0.6369`** (63.69%) |
| **Whole Tumor (WT)** | **Intersection over Union (IoU)** | **`0.6880`** (68.80%) |
| **Tumor Core (TC)** | **Intersection over Union (IoU)** | **`0.4639`** (46.39%) |
| **Enhancing Tumor (ET)** | **Intersection over Union (IoU)** | **`0.4581`** (45.81%) |
| **Best Model Checkpoint** | **Path / SHA-256** | `checkpoints/centralized_real/best.pt`<br>`a4e9199221cf5d622229f311dbf833748f175593d4b7784d1f390b877968f990` |

---

## 3. Detailed Audit Findings

### 1. Data Partitioning & Silo Boundaries
- **Cohort Verification:**
  - **Hospital Alpha:** 236 active training subjects
  - **Hospital Beta:** 236 active training subjects
  - **Hospital Gamma:** 236 active training subjects
  - **Hospital Delta:** 236 active training subjects
  - **Total Training Cases:** $236 \times 4 = 944$ cases
- **Held-Out Cohorts:**
  - **Validation Cohort:** 202 subjects (evaluated centrally after each round)
  - **Test Cohort:** 204 subjects (**LOCKED & FIREWALLED — 0 Accesses**)
- **Data Isolation:** Zero subject overlap across hospital silos, validation, and test cohorts.

---

### 2. Model & Preprocessing Invariance
- **Network Class:** `monai.networks.nets.UNet`
- **Tensors:** Input $(B, 4, 128, 128, 128)$, Target $(B, 3, 128, 128, 128)$
- **Parameter Count:** **4,810,074 float32 parameters** (exact match with Experiment A)
- **Modality Order:** Channel 0 $\leftarrow$ `t1n`, Channel 1 $\leftarrow$ `t1c`, Channel 2 $\leftarrow$ `t2w`, Channel 3 $\leftarrow$ `t2f`
- **Label Mapping:**
  - $\text{TC} = (\text{label} == 1) \lor (\text{label} == 3)$
  - $\text{WT} = (\text{label} == 1) \lor (\text{label} == 2) \lor (\text{label} == 3)$
  - $\text{ET} = (\text{label} == 3)$
  - $\text{RC} = (\text{label} == 4)$ preserved in provenance metadata.

---

### 3. Mathematical Verification of FedAvg
The server aggregation logic in `server/strategies/fedavg.py` was evaluated against an independent, vectorized NumPy implementation using 4 simulated client update vectors:
$$\mathbf{W}_{\text{server}} = \sum_{k=1}^4 \frac{n_k}{N} \mathbf{W}_k, \quad n_k = 236, \quad N = 944$$
- **Max Absolute Discrepancy:** **`0.00e+00`** (Exact floating point equivalence).

---

### 4. Apple Silicon MPS Safety & Execution Strategy
- **Sequential Client Execution:** In the single-node simulator, hospital clients are executed sequentially in loop (`hospital_alpha` $\to$ `hospital_beta` $\to$ `hospital_gamma` $\to$ `hospital_delta`), completely preventing GPU allocation contention on unified memory.
- **Checkpoint Resilience:** Automatically maintains `checkpoints/fedavg_real/best.pt` and `checkpoints/fedavg_real/latest.pt` with round resumption metadata.

---

### 5. Empirical Real-Data 1-Round Smoke Test Results

A controlled 1-round smoke test was executed on real BraTS-GLI 2024 subjects at full $128 \times 128 \times 128$ resolution on Apple Silicon MPS:
- **Client Alpha Mean Loss:** $0.994527$
- **Client Beta Mean Loss:** $0.997482$
- **Client Gamma Mean Loss:** $0.992684$
- **Client Delta Mean Loss:** $0.973616$
- **Client Divergence Norm ($\|W_\alpha - W_\beta\|_2$):** **`0.968795`** (confirmed genuine independent client learning)
- **Global Parameter Delta Norm ($\|W_{\text{new}} - W_{\text{old}}\|_2$):** **`0.439807`** (confirmed parameter update after FedAvg aggregation)
- **Round Time:** $100.26\text{ seconds}$ (including 202 validation evaluations)
- **Test Firewall Integrity:** `TRAINING_TEST_ACCESSES = 0` (zero test cases accessed).

---

## 4. Final Gate Verdict

### Verdict: **`READY_FOR_EXPERIMENT_B`**

The codebase, hospital silo partitions, data loaders, model architecture, loss function, optimizer, mathematical aggregation strategy, and test firewall are **100% verified and ready to execute Experiment B (Real-Data FedAvg across 4 Hospital Silos)**.

*Stopped execution as directed. Ready to launch Experiment B upon user confirmation.*
