# FEDMED OS — PHASE 8.5B.5 PRIVACY ARCHITECTURE SPECIFICATION

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Differential Privacy & Homomorphic Encryption Integrity Gate (Phase 8.5B.5)  
**Git Branch:** `release/stabilization-rc2`  
**Git Head:** `e266612 Phase 10: FedMed Omega production hardening and runtime validation`  

---

## 1. Existing Privacy Component Audit

1. **Differential Privacy (`privacy/dp_engine.py`):**
   - Implements `DifferentialPrivacyEngine` operating at the local client gradient level.
   - Enforces $L_2$ gradient clipping: $\|g\|_2 \le C$ (`max_grad_norm`).
   - Injects calibrated Gaussian noise: $g \leftarrow g + \mathcal{N}(0, \sigma^2 C^2 / B)$ (`noise_multiplier` $\sigma$).
   - Computes privacy loss accounting via Rényi Differential Privacy (RDP) conversion to $(\epsilon, \delta)$-DP:
     $$\epsilon(\delta) = \min_{\alpha > 1} \left[ \frac{\alpha q^2 T}{2 \sigma^2} + \frac{\ln(1/\delta)}{\alpha - 1} \right]$$
     where $q = \text{sample\_rate}$, $T = \text{steps}$, $\sigma = \text{noise\_multiplier}$.

2. **Homomorphic Encryption (`privacy/`):**
   - Implements TenSEAL CKKS vector encryption (`privacy/encrypt.py`), weighted ciphertext addition (`privacy/aggregation.py`), and decryption (`privacy/decrypt.py`).
   - Public evaluation context (`privacy/context.py`) drops the secret key so the central server can aggregate encrypted client updates homomorphically without learning plaintext parameters.
   - TenSEAL version: `0.3.16` (installed and functional).

3. **Existing Privacy Gaps (Pre-B5):**
   - DP and HE were previously tested in isolated mock scripts rather than bound into the canonical multi-hospital BraTS experiment pipeline.
   - The end-to-end multi-round execution matrix (Baseline, DP, HE, DP+HE) had not been executed against the canonical 3D UNet (4,810,074 parameters) with measured numerical errors and firewalled validation.

---

## 2. Proposed B5 Privacy Architecture

```
Hospital Alpha Node                       Server Aggregator                      Hospital Beta Node
===================                       =================                      ==================
Local 3D MRI Training                                                             Local 3D MRI Training
        │                                                                                 │
        ▼                                                                                 ▼
[DP Gradient Clipping]                                                            [DP Gradient Clipping]
||g|| <= C                                                                        ||g|| <= C
        │                                                                                 │
        ▼                                                                                 ▼
[DP Gaussian Noise]                                                               [DP Gaussian Noise]
g + N(0, (sigma*C)^2)                                                             g + N(0, (sigma*C)^2)
        │                                                                                 │
        ▼                                                                                 ▼
Optimizer Step & Update                                                           Optimizer Step & Update
        │                                                                                 │
        ▼                                                                                 ▼
[TenSEAL CKKS Encryption]                                                         [TenSEAL CKKS Encryption]
Chunk & Encrypt to [c_alpha]                                                      Chunk & Encrypt to [c_beta]
        │                                                                                 │
        └───────────────────────────► Transmit Ciphertexts ◄──────────────────────────────┘
                                              │
                                              ▼
                                 [Homomorphic Aggregation]
                                 (Server has Public Key ONLY)
                                 c_global = 0.5*c_alpha + 0.5*c_beta
                                              │
                                              ▼
                                 Transmit c_global to Clients
                                              │
        ┌─────────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                         ▼
[Authorized Decryption]                                                   [Authorized Decryption]
Decrypt c_global -> W_global                                              Decrypt c_global -> W_global
        │                                                                         │
        ▼                                                                         ▼
Local Model Synchronization                                               Local Model Synchronization
```

---

## 3. Cryptographic and Mathematical Parameters

- **DP Configuration:**
  - Mechanism: Sampled Gaussian Mechanism with Rényi Differential Privacy Accounting
  - Default $\text{max\_grad\_norm} (C) = 1.0$
  - Default $\text{noise\_multiplier} (\sigma) = 0.5$
  - Target $\delta = 10^{-5}$
- **CKKS HE Configuration:**
  - Scheme: CKKS (Cheon-Kim-Kim-Song)
  - Polynomial Modulus Degree ($N$): `8192`
  - Coefficient Modulus Bit Sizes: `[60, 40, 40, 60]` (128-bit classical security level)
  - Scale: $2^{40}$
  - Chunk size: `4096` slots per ciphertext vector
