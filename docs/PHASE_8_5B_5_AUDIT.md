# FEDMED OS — PHASE 8.5B.5 AUDIT REPORT
# PRIVACY IMPLEMENTATION & TERMINOLOGY INTEGRITY

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Differential Privacy & TenSEAL Homomorphic Encryption Execution Audit  

---

## 1. Differential Privacy Audit Findings

### 1.1 Gradient Clipping Level
- **Observation:** `privacy.dp_engine.DifferentialPrivacyEngine` applies gradient clipping across model parameters on the accumulated batch gradient (`apply_gradient_clipping_and_noise(batch_size)`).
- **Classification:** **Batch-Level Gradient Clipping with Sample Calibration** (with batch size $B=1$ per subject in 3D MRI, batch-level is mathematically equivalent to per-sample clipping).
- **Provenance:** When `batch_size=1`, the batch gradient $g = \nabla \mathcal{L}(x_i)$ is strictly per-sample. For $B > 1$, micro-batching or Opacus per-sample gradient hooks are required for strict DP-SGD.

### 1.2 RDP Accounting and Sampling Rate ($q$)
- **Parameters:**
  - $T = 6$ total optimization steps across 3 rounds $\times$ 2 clients.
  - Subsampling ratio $q = \frac{1 \text{ sample trained per client}}{2 \text{ total train samples}} = 0.5$.
  - Noise multiplier $\sigma = 0.5$.
  - Target $\delta = 10^{-5}$.
- **Calculated $\epsilon$:** $\epsilon = 14.76$ derived from analytical RDP orders:
  $$\epsilon(\delta) = \min_{\alpha > 1} \left[ \frac{\alpha q^2 T}{2\sigma^2} + \frac{\ln(1/\delta)}{\alpha - 1} \right] = \min_{\alpha > 1} \left[ \frac{\alpha (0.25)(6)}{2(0.25)} + \frac{\ln(10^5)}{\alpha - 1} \right] = \min_{\alpha} \left[ 3\alpha + \frac{11.513}{\alpha - 1} \right]$$
  Minimizing at $\alpha \approx 2.958 \implies 3(2.958) + \frac{11.513}{1.958} \approx 8.874 + 5.880 = 14.754 \approx \mathbf{14.76}$.
- **Provenance Verdict:** Mathematically exact for the closed-form RDP upper bound with subsampled Gaussian mechanism.

---

## 2. Homomorphic Encryption Reconstruction Terminology Audit

### 2.1 CKKS Decryption Terminology
- **Correction:** CKKS (Cheon-Kim-Kim-Song) is a floating-point approximate homomorphic encryption scheme.
- **Audited Terminology:** CKKS decryption must strictly be defined as **approximate reconstruction with measured numerical error** (e.g. $|W_{\text{dec}} - W_{\text{expected}}| = 5.96 \times 10^{-8}$), and **MUST NOT** be referred to as "lossless" or "exact" reconstruction.
- **Cryptographic Security Parameter Provenance:**
  - $N = 8192$ (polynomial modulus degree)
  - Coefficient modulus bit sizes = `[60, 40, 40, 60]` ($\sum = 200$ bits)
  - Scale = $2^{40}$
  - Security Level: Conforms to 128-bit classical security under the Homomorphic Encryption Standard (HES) for $N=8192, \log_2 Q \le 218$.

---

## 3. Privacy Scientific Status
- **Differential Privacy Status:** `PASS FOR DEVELOPMENT COHORT (BATCH_SIZE=1 EQUIVALENCE)`
- **Homomorphic Encryption Status:** `PASS (APPROXIMATE RECONSTRUCTION WITH MEASURED ERROR < 1e-7)`
