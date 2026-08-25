# FEDMED OS — PHASE 9.1: REAL BraTS-GLI 2024 TRAINING READINESS AUDIT

**Date:** August 15, 2026  
**Auditor / Engineer:** Senior ML Systems Engineer  
**Dataset:** Official BraTS-GLI 2024 Adult Glioma Post-Treatment Training Cohort  
**Dataset Registry:** 1,350 Validated Subjects (70% Train = 944, 15% Val = 202, 15% Test = 204)  
**Hospital Silos:** 4 Hospital Silos (236 Train Subjects Each)  
**Execution Environment:** Apple Silicon MPS (1 GPU, 16 GB Unified Memory)  
**Regression Status:** 302 Passed, 1 Skipped, 0 Failed (100% Green)  
**Final Audit Verdict:** **`READY_WITH_CONFIGURATION_CHANGES`**  

---

## 1. Executive Summary & Gate Verdict

A comprehensive empirical audit was executed across all 12 operational pillars of the FedMed training engine using the official 1,350-subject BraTS-GLI 2024 cohort.

| Audit Item | Area Under Inspection | Current Implementation | Audit Finding | Verdict |
|---|---|---|---|---|
| **Audit 1** | Preprocessing Pipeline | Modalities: [t1n, t1c, t2w, t2f], RAS, 1.0mm, Non-zero Z-score | $(32, 32, 32)$ causes $220\times$ volume reduction; $(128, 128, 128)$ recommended | **ACCEPTABLE WITH CHANGE** |
| **Audit 2** | Target Construction | $\text{TC}=1\lor3$, $\text{WT}=1\lor2\lor3$, $\text{ET}=3$, $\text{RC}=4$ in provenance | Hierarchical binary multi-label tensor $(3, H, W, D)$ mathematically sound | **PASS** |
| **Audit 3** | Class Imbalance | 98.98% BG, 0.02% NETC, 0.68% SNFH, 0.12% ET, 0.19% RC | Unweighted BCE in `DiceCELoss` slows initial convergence; Dice emphasis needed | **PASS WITH TUNING** |
| **Audit 4** | Resolution Retention | Empirical 10-subject resolution test across $32^3$, $64^3$, $128^3$ | $128^3$ preserves 49-139 ET voxels vs 1-2 voxels at $32^3$ | **PASS ($128^3$)** |
| **Audit 5** | Training Profile | 4-subject single round computational profile on MPS | 0.0163s/step, 1.09s/round, 80 MB MPS memory at $32^3$, 261 MB at $128^3$ | **PASS** |
| **Audit 6** | Learning Sanity | 4-subject 5-epoch training and validation evaluation | Loss decreases monotonically ($1.0023 \to 0.9962$); WT Dice improves | **PASS** |
| **Audit 7** | Overfitting Sanity | Controlled 100-step single subject overfit (`BraTS-GLI-00009-100`) | Loss drops $0.9801 \to 0.7547$, WT Dice reaches $0.6041$ | **PASS** |
| **Audit 8** | Federated Profile | 4-hospital 1-round FedAvg profile on real subjects | Round time: 1.18s, Payload: 18.35 MB, $\Delta W = 0.373551$, Test firewalled | **PASS** |
| **Audit 9** | DP Profile | Opacus DP Engine + RDP Accountant on real gradient shapes | $C=1.0, \sigma=0.80$, $\epsilon=0.1906, \delta=10^{-5}$ after 1 step | **PASS** |
| **Audit 10** | HE Profile | TenSEAL CKKS 8192-poly encryption and public aggregation | Aggregation time: 3.59 ms, Max error $= 4.77 \times 10^{-7}$ | **PASS** |
| **Audit 11** | Resource Safety | Single-node Apple Silicon MPS capacity | Conservative config: $B=1, \text{shape}=(128,128,128)$, sequential client execution | **PASS** |
| **Audit 12** | Experiment Plan | Structured sequence: Centralized, FedAvg, FedProx, DP, HE, DP+HE | Fully defined specifications and artifact outputs | **PASS** |

---

## 2. Detailed Audit Reports

### AUDIT 1 — Preprocessing Pipeline Verification
- **Modality Ordering:** Channel 0 $\leftarrow$ `t1n` (T1), Channel 1 $\leftarrow$ `t1c` (T1CE), Channel 2 $\leftarrow$ `t2w` (T2), Channel 3 $\leftarrow$ `t2f` (FLAIR).
- **Spatial Alignment:** Standardized to RAS orientation (`Orientationd(axcodes="RAS")`).
- **Resolution & Spacing:** Native spacing is $1.0\times1.0\times1.0\text{ mm}^3$ isotropic; preserved with `Spacingd`.
- **Intensity Normalization:** Independent per-channel foreground Z-score normalization (`NormalizeIntensityd(nonzero=True, channel_wise=True)`).
- **Interpolation Integrity:** Images use `bilinear` interpolation; labels strictly use `nearest` neighbor to preserve integer discrete classes.
- **Evaluation of Spatial Shape $(32, 32, 32)$:**
  - **Verdict:** **B & C (Technically acceptable only for development/CI smoke tests; too aggressive for real scientific training).**
  - Native volume is $182 \times 218 \times 182 = 7,218,872$ voxels.
  - $(32, 32, 32)$ compresses volume down to $32,768$ voxels ($220\times$ reduction), which crushes small Enhancing Tumor structures.
  - **Recommendation:** Upgrade default experiment configuration to $(128, 128, 128)$.

---

### AUDIT 2 — Target Construction & Mathematical Formulation
- **Composite Regions:**
  $$\mathbf{TC} = (\text{label} == 1) \lor (\text{label} == 3)$$
  $$\mathbf{WT} = (\text{label} == 1) \lor (\text{label} == 2) \lor (\text{label} == 3)$$
  $$\mathbf{ET} = (\text{label} == 3)$$
- **Target Tensor Shape:** $(B, 3, H, W, D)$ float32 tensor with entries $\in \{0.0, 1.0\}$.
- **Mathematical Compatibility:**
  - MONAI 3D U-Net produces 3 raw logits without activation.
  - `DiceCELoss(sigmoid=True)` computes independent Sigmoids $\sigma(z_c) = \frac{1}{1 + e^{-z_c}}$ across each channel.
  - Compatible with multi-label overlapping segmentation where a single voxel belongs to both WT and TC.

---

### AUDIT 3 — Class Imbalance Analysis
From the full 1,350-subject ingestion audit ($9,748,393,200$ total voxels):
- **Background (0):** $9,649,363,112\text{ voxels}$ (**98.9841%**)
- **NETC (1):** $2,285,704\text{ voxels}$ (**0.0234%**)
- **SNFH (2):** $66,476,419\text{ voxels}$ (**0.6819%**)
- **ET (3):** $11,442,955\text{ voxels}$ (**0.1174%**)
- **RC (4):** $18,825,010\text{ voxels}$ (**0.1931%**)
- **Composite TC ($1\lor3$):** $13,728,659\text{ voxels}$ (**0.1408%**)
- **Composite WT ($1\lor2\lor3$):** $80,205,078\text{ voxels}$ (**0.8228%**)
- **Composite ET ($3$):** $11,442,955\text{ voxels}$ (**0.1174%**)

**Loss Formulation Assessment:**
- With 98.98% background, unweighted BCE strongly penalizes false positives and keeps sigmoid logits suppressed near $0.01$.
- **Recommendation:** Use pure `DiceLoss(sigmoid=True)` or weighted `DiceCELoss(lambda_dice=1.0, lambda_ce=0.2)` to ensure rapid gradient progression on foreground tumor masks.

---

### AUDIT 4 — Empirical Resolution Retention (10 Real Training Cases)

| Subject ID | Native ET Voxels | ET at $(32, 32, 32)$ | ET at $(64, 64, 64)$ | ET at $(128, 128, 128)$ |
|---|---|---|---|---|
| `BraTS-GLI-00005-100` | 0 | 0 | 0 | 0 |
| `BraTS-GLI-00006-100` | 0 | 0 | 0 | 0 |
| `BraTS-GLI-00008-101` | 66 | 1 | 3 | 49 |
| `BraTS-GLI-00008-103` | 178 | 2 | 20 | 139 |
| `BraTS-GLI-00009-100` | 6,311 | 76 | 563 | 4,491 |
| `BraTS-GLI-00020-100` | 732 | 11 | 80 | 545 |
| `BraTS-GLI-00020-101` | 2,752 | 34 | 266 | 2,052 |
| `BraTS-GLI-00027-100` | 24,082 | 291 | 2,339 | 17,914 |
| `BraTS-GLI-00033-101` | 8,976 | 99 | 803 | 6,554 |
| `BraTS-GLI-00063-101` | 12,410 | 154 | 1,189 | 9,142 |

**Key Finding:**
- Small ET structures (e.g. 66 voxels) survive with **49 voxels at $(128, 128, 128)$**, but collapse to **1 voxel at $(32, 32, 32)$**.
- Demonstrates why $(128, 128, 128)$ is the appropriate scientific resolution for real experimentation.

---

### AUDIT 5 — Local Training Computational Profile (Apple Silicon MPS)
- **Data Loading:** 0.0008 s
- **Forward Pass:** 0.0043 s
- **Backward Pass:** 0.0037 s
- **Optimizer Step:** 0.0076 s
- **Total Local Step Time:** **0.0163 s**
- **4-Subject Local Round Time:** **1.09 s**
- **Memory Footprint:**
  - At $(32, 32, 32)$: **80.11 MB** MPS allocated memory.
  - At $(128, 128, 128)$: **261.05 MB** MPS allocated memory (0.33s/step).

---

### AUDIT 6 — Learning Sanity Check (Multi-Epoch Training)
- **Epoch 1:** Train Loss $= 1.0023$, Train WT Dice $= 0.0117$, Val Loss $= 1.0010$
- **Epoch 2:** Train Loss $= 1.0004$, Train WT Dice $= 0.0144$, Val Loss $= 0.9986$
- **Epoch 3:** Train Loss $= 0.9988$, Train WT Dice $= 0.0175$, Val Loss $= 0.9963$
- **Epoch 4:** Train Loss $= 0.9971$, Train WT Dice $= 0.0194$, Val Loss $= 0.9955$
- **Epoch 5:** Train Loss $= 0.9962$, Train WT Dice $= 0.0208$, Val Loss $= 0.9955$
- **Verdict:** **PASS**. Loss strictly decreases; Dice improves above random initialization.

---

### AUDIT 7 — Controlled Single-Subject Overfitting Sanity Check
- **Subject:** `BraTS-GLI-00009-100`
- **Trajectory:**
  - Step 1: Loss $= 0.9801$, WT Dice $= 0.0456$
  - Step 20: Loss $= 0.9513$, WT Dice $= 0.3933$
  - Step 60: Loss $= 0.8422$, WT Dice $= 0.4664$
  - Step 80: Loss $= 0.7788$, WT Dice $= 0.5763$
  - Step 100: Loss $= 0.7547$, WT Dice $= \mathbf{0.6041}$
- **Verdict:** **PASS**. Model possesses the capacity to learn and segment 3D BraTS structures.

---

### AUDIT 8 — 4-Hospital Federated Profile (1 Round)
- **Hospitals:** `hospital_alpha`, `hospital_beta`, `hospital_gamma`, `hospital_delta`
- **Client Synchronization:** Verified across all 4 silos.
- **Round Time:** **1.18 s**
- **Client Payload:** **18.35 MB** (4,810,074 float32 parameters).
- **Parameter Delta:** $\Delta W = 0.373551$ (genuine multi-client parameter update).
- **Privacy Boundary:** 0 raw images egress client silos.
- **Test Firewall:** Verified.

---

### AUDIT 9 — Differential Privacy Profile
- **Clipping:** Max gradient norm $C = 1.0$ verified.
- **Noise Injection:** Gaussian $\sigma = 0.80$ verified.
- **Accounting:** RDP Accountant tracks cumulative budget ($\epsilon = 0.1906, \delta = 10^{-5}$).
- **Compatibility:** 100% compatible with real 3D U-Net gradient shapes.

---

### AUDIT 10 — Homomorphic Encryption Profile
- **Scheme:** TenSEAL CKKS ($N = 8192$, coeff bit sizes `[60, 40, 40, 60]`, scale $2^{40}$).
- **Public Context Aggregation:** 3.59 ms server aggregation latency.
- **Reconstruction Accuracy:** Max absolute error $= 4.77 \times 10^{-7}$, Mean absolute error $= 7.63 \times 10^{-8}$.
- **Compatibility:** 100% verified.

---

### AUDIT 11 — Resource Safety & Hardware Limits
- **System Profile:** 16 GB Unified Memory, Apple Silicon MPS (1 GPU).
- **Conservative Execution Parameters:**
  - `batch_size`: 1
  - `num_workers`: 0 or 2 (prevents socket leak)
  - `spatial_shape`: $(128, 128, 128)$
  - `client_execution`: Sequential client execution in single-node simulator
  - `storage_headroom`: <2 GB required for checkpoints & logs.

---

## 3. Concrete Recommended Experiment Sequence (Audit 12)

```
EXPERIMENT A: Real-Data Centralized Baseline
├── Cohort: 944 Train subjects, 202 Val, 204 Test
├── Training: 20 Epochs, Batch Size 1, Resolution (128, 128, 128)
├── Loss: DiceCELoss(lambda_dice=1.0, lambda_ce=0.2)
├── Est. Runtime: ~1.8 hours on Apple Silicon MPS
└── Output: checkpoints/centralized_real/best.pt, reports/real_brats2024/centralized.json

EXPERIMENT B: Real-Data FedAvg
├── Silos: 4 Hospitals (236 Train subjects each)
├── Rounds: 20 Rounds, 1 Local Epoch/Round
├── Aggregation: Sample-weighted FedAvg
├── Est. Runtime: ~2.1 hours
└── Output: checkpoints/fedavg_real/best.pt, reports/real_brats2024/fedavg.json

EXPERIMENT C: Real-Data FedProx
├── Silos: 4 Hospitals (236 Train subjects each)
├── Rounds: 20 Rounds, 1 Local Epoch/Round, mu = 0.01
├── Aggregation: Proximal FedAvg
├── Est. Runtime: ~2.2 hours
└── Output: checkpoints/fedprox_real/best.pt, reports/real_brats2024/fedprox.json

EXPERIMENT D: FedAvg + Differential Privacy
├── Privacy: C = 1.0, sigma = 0.80, target_eps = 5.0, target_delta = 1e-5
├── Rounds: 20 Rounds
├── Est. Runtime: ~2.4 hours
└── Output: reports/real_brats2024/dp_fedavg.json

EXPERIMENT E: FedAvg + Homomorphic Encryption (TenSEAL CKKS)
├── Cryptography: N = 8192, 128-bit security, server public context
├── Rounds: 20 Rounds
├── Est. Runtime: ~2.8 hours
└── Output: reports/real_brats2024/he_fedavg.json

EXPERIMENT F: FedAvg + DP + HE Composition
├── Full Privacy Composition: DP local clipping/noise + CKKS encrypted transport
├── Rounds: 20 Rounds
├── Est. Runtime: ~3.1 hours
└── Output: reports/real_brats2024/dp_he_fedavg.json
```

---

## 4. Final Gate Verdict

### Verdict: **`READY_WITH_CONFIGURATION_CHANGES`**

1. **What is Working:**
   - 1,350 BraTS-GLI 2024 subjects validated and indexed with zero missing files or corrupted NIfTI headers.
   - Deterministic 70/15/15 split ($944 / 202 / 204$) with strict test firewalling.
   - 4-hospital partitioning with 236 active subjects per silo.
   - Canonical modality and corrected label adapter ($\text{TC}=1\lor3, \text{WT}=1\lor2\lor3, \text{ET}=3, \text{RC}=4$).
   - Genuine backpropagation, gradient updates, learning dynamics, and overfitting capacity.
   - DP Engine and CKKS HE Aggregator verified on real tensor dimensions.
   - Full regression suite passing (**302 passed, 1 skipped, 0 failed**).

2. **What is Unsafe or Scientifically Questionable:**
   - Using `spatial_shape = (32, 32, 32)` for real training will compress small ET structures down to 1-2 voxels, leading to morphological degradation.
   - Unweighted BCE in `DiceCELoss` is overpowered by 98.98% background voxels.

3. **Exact Configuration Changes Required:**
   - Change training `spatial_shape` to `(128, 128, 128)`.
   - Configure loss weights to `lambda_dice=1.0, lambda_ce=0.2` (or pure `DiceLoss`).
   - Run clients sequentially during simulation on Apple Silicon MPS.

4. **Exact First Real Experiment to Run:**
   - **EXPERIMENT A:** Real-data centralized baseline on 944 training cases at $(128, 128, 128)$ for 20 epochs to establish the scientific empirical ceiling.

5. **Estimated Runtime:** ~1.8 hours for Centralized Baseline; ~2.1 hours for FedAvg.
6. **Estimated Storage:** <2 GB total (model checkpoints + reports).
7. **Whether Full Training Should Proceed:**
   - Full training should proceed once the two configuration updates (resolution $(128, 128, 128)$ and loss weighting) are applied in experiment configurations.
