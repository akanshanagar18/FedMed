# FEDMED OS — PRIVACY DATA FLOW & PHI BOUNDARY AUDIT

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Client Isolation, Server Visibility, Key Management, and PHI Safety  

---

## 1. Privacy Boundary & Data Flow

```
┌────────────────────────────────────────────────────────┐
│               LOCAL HOSPITAL SILO BOUNDARY             │
│                                                        │
│  [Private Storage]                                     │
│  Raw Patient MRI (T1, T1ce, T2, FLAIR) ◄── NEVER LEAVES │
│  Ground Truth Segmentations            ◄── LOCAL SILO  │
│        │                                               │
│        ▼                                               │
│  Local Model Forward/Backward Training                 │
│        │                                               │
│        ▼                                               │
│  [DP Engine] $L_2$ Gradient Clip & Gaussian Noise      │
│        │                                               │
│        ▼                                               │
│  [TenSEAL CKKS] Encrypt Weights with Client Secret Key │
│        │                                               │
│        ▼                                               │
│  Transmit CKKS Ciphertext Payload ONLY                 │
└────────┬───────────────────────────────────────────────┘
         │
         │ (Encrypted Ciphertext Over TLS Network)
         ▼
┌────────────────────────────────────────────────────────┐
│               CENTRAL COORDINATOR / SERVER             │
│                                                        │
│  - Receives opaque serialized ciphertexts ONLY         │
│  - Holds TenSEAL Public Context (Zero Secret Key)      │
│  - Executes Homomorphic Weighted Addition:             │
│      $c_{\text{global}} = \sum \frac{n_i}{N} c_i$      │
│  - Zero Raw Patient Image Visibility                   │
│  - Zero Plaintext Model Parameter Visibility           │
│  - Broadcasts $c_{\text{global}}$ to Hospitals         │
└────────────────────────────────────────────────────────┘
```

---

## 2. Key Ownership & Cryptographic Boundary

- **Client Private Keys:** Generated and held exclusively inside local hospital memory. Secret keys are never transmitted to the central server or stored in global checkpoints.
- **Server Public Context:** Contains evaluation and relinearization keys only; cannot decrypt any ciphertext vector.
- **Model Checkpoints:** Production model packages store public model weights only after authorized federated consensus and decryption.

---

## 3. PHI / Logging Safety Guarantees

- **No Patient Identifiers:** System logs use anonymized `subject_hash` or study identifiers (e.g. `BraTS2021_00001`). No patient names, MRNs, dates of birth, or clinical notes are ingested or logged.
- **Path Sanitization:** File logging uses relative or anonymized repository paths rather than local machine filesystem user directories.
