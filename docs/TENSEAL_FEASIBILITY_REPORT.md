# FEDMED OS — TENSEAL / CKKS FEASIBILITY & BENCHMARK REPORT

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** TenSEAL CKKS Homomorphic Encryption Feasibility Study  
**Environment:** macOS (Apple Silicon), Python 3.9.6, TenSEAL 0.3.16  

---

## 1. Measured Micro-Benchmark Results (4,096 Slots per Vector)

- **TenSEAL Version:** `0.3.16`
- **CKKS Polynomial Modulus Degree ($N$):** `8192`
- **Coefficient Modulus Bit Sizes:** `[60, 40, 40, 60]`
- **Global Scale:** $2^{40}$
- **Security Level:** 128-bit classical security (homomorphic encryption standard)

### Performance Measurements (Single 4,096-element Vector):
- **Context Initialization Time:** `81.93 ms`
- **Encryption Time (2x 4,096-float vectors):** `4.15 ms`
- **Homomorphic Weighted Addition ($0.5 \cdot c_1 + 0.5 \cdot c_2$):** `0.97 ms`
- **Decryption Time:** `0.41 ms`
- **Ciphertext Size:** `326.37 KB` (334,200 bytes per 4,096 slots)
- **Plaintext Size:** `32.00 KB` (4,096 x 8 bytes float64)
- **Ciphertext Expansion Factor:** $\approx 10.2\times$
- **Approximation Absolute Error ($|W_{\text{dec}} - W_{\text{expected}}|$):** `3.5927e-07` ($< 10^{-6}$)
- **Approximation Relative Error:** `1.3397e-07`

---

## 2. Full-Model Feasibility Assessment (4,810,074 Parameters)

- **Total Parameter Count:** `4,810,074`
- **Vector Capacity per Chunk:** `4,096` slots
- **Number of Chunks Required:** $\lceil 4,810,074 / 4,096 \rceil = \mathbf{1,175}$ chunks
- **Measured Encryption Time per Client:** $\approx \mathbf{1.93\text{ seconds}}$
- **Measured Homomorphic Server Aggregation Time:** $\approx \mathbf{0.70\text{ seconds}}$
- **Measured Decryption Time:** $\approx \mathbf{0.39\text{ seconds}}$
- **Serialized Ciphertext Payload per Client:** $\approx \mathbf{383.05\text{ MB}}$

### Feasibility Conclusion:
Full-model CKKS chunked homomorphic encryption and aggregation is **100% FEASIBLE** on the development environment. End-to-end encryption, public aggregation, and decryption execute in approximately **3.0 seconds total round overhead**, maintaining high numerical fidelity ($< 10^{-6}$ error).
