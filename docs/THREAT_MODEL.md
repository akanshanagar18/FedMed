# FEDMED OS — THREAT MODEL & SECURITY ASSUMPTIONS

**Date:** August 15, 2026  
**Document:** Security & Privacy Threat Model (FedMed v2.0)  

---

## 1. Adversary Model & Server Trust Profile

- **Server Model:** **Honest-but-Curious (Semi-Honest)**
  - The central coordinator / aggregation server correctly follows the defined protocol (FL aggregation, communication routing), but may attempt to passively infer private hospital patient information or reconstruct raw patient MRI images from observed messages.
- **Client Model:** **Isolated Participating Silos**
  - Participating hospital nodes (`hospital_alpha`, `hospital_beta`) hold private patient imaging cohorts that must never leave the hospital boundary.

---

## 2. Assets Protected vs Assets Exposed

### Protected Assets:
1. **Raw MRI Voxels & Metadata:** Never transmitted across the network or shared with any other silo or the server.
2. **Ground Truth Tumor Segmentation Masks:** Held strictly within local hospital storage.
3. **Local Intermediate Batch Activations & Gradients:** Kept strictly inside the client hardware memory during training.
4. **Individual Client Weight Updates (when HE is active):** Encrypted using TenSEAL CKKS with the client private key before network transmission. The server holds only the public evaluation key and cannot decrypt individual updates.
5. **Reconstruction Protection (when DP is active):** Injected Gaussian noise provides formal differential privacy $(\epsilon, \delta)$ against reconstruction and membership inference attacks on the aggregated global model.

### Exposed / Observed Assets:
1. **Aggregated Global Model Weights:** Publicly visible to all participating nodes after authorized decryption.
2. **Training & Validation Telemetry:** High-level loss and Dice scores intentionally reported for monitoring.
3. **Protocol Metadata:** Sample counts, client IDs, and cryptographic context headers required for protocol orchestration.

---

## 3. Explicit Boundaries and Non-Claims

> [!IMPORTANT]
> - This threat model does **NOT** protect against malicious servers that tamper with gradients or inject backdoors (Byzantine fault tolerance / robust aggregation is covered by separate strategies such as Krum/Bulyan).
> - This implementation does **NOT** constitute formal regulatory certification (e.g. HIPAA/GDPR legal compliance).
> - Differential privacy provides probabilistic $(\epsilon, \delta)$ bounds and inherently incurs a utility-privacy trade-off.
