# FEDMED OS — CANONICAL EXPERIMENT SPECIFICATION CONTRACT

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Status:** FROZEN MASTER CONTRACT (Phase 8.5B.8)  

---

## 1. Canonical Model Architecture

- **Class:** MONAI 3D UNet (`monai.networks.nets.UNet`)
- **Spatial Dimensions:** `3`
- **Input Channels:** `4` (`[t1, t1ce, t2, flair]`)
- **Output Channels:** `3` (`[TC, WT, ET]`)
- **Layer Channels:** `(16, 32, 64, 128, 256)`
- **Strides:** `(2, 2, 2, 2)`
- **Residual Units:** `2`
- **Total Trainable Parameters:** **4,810,074**

---

## 2. Canonical Optimization & Loss Hyperparameters

- **Optimizer:** Adam
- **Initial Learning Rate:** $1.0 \times 10^{-4}$
- **Weight Decay:** $1.0 \times 10^{-5}$
- **Loss Function:** MONAI `DiceCELoss(sigmoid=True)`
- **Random Seed:** `42`

---

## 3. Canonical Preprocessing Pipeline

- **Target Spatial Shape:** `(32, 32, 32)` isotropic
- **Modality Load Order:** `["t1", "t1ce", "t2", "flair"]`
- **Intensity Normalization:** Non-zero voxel Z-score standardization (`NormalizeIntensityd`)
- **Orientation:** Neurological RAS orientation (`Orientationd(keys=["image", "label"], axcodes="RAS")`)

---

## 4. Canonical Target Label Semantics

- **Tumor Core (TC):** $(\text{label} == 1) \lor (\text{label} == 4)$
- **Whole Tumor (WT):** $(\text{label} == 1) \lor (\text{label} == 2) \lor (\text{label} == 4)$
- **Enhancing Tumor (ET):** $(\text{label} == 4)$

---

## 5. Federated Training Budget

- **Centralized Baseline:** 3 epochs $\times$ 2 training subjects = 6 sample optimization passes
- **FedAvg / FedProx:** 3 rounds $\times$ 2 clients $\times$ 1 local epoch = 6 sample optimization passes
- **FedProx $\mu$ Regularization:** $\mu = 0.01$
- **Differential Privacy:** Sampled Gaussian mechanism, $C = 1.0$, $\sigma = 0.5$, $\delta = 10^{-5}$, RDP accounting
- **Homomorphic Encryption:** TenSEAL CKKS, $N = 8192$, coeff mod `[60, 40, 40, 60]`, scale $2^{40}$, chunk size $4096$

---

## 6. Test Firewall Contract

- **Training Phase:** Zero accesses to the TEST set (`TEST_SET_ACCESSED = False`).
- **Post-Training Evaluation:** Exactly 1 evaluation pass per model on `BraTS2021_00001` strictly after training completion.
