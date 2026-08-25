# FEDMED OS — PHASE 10.2: EXPERIMENT B (REAL-DATA FEDAVG) REPORT

**Date:** August 16, 2026  
**Auditor / Engineer:** Senior ML Systems Engineer  
**Experiment Name:** `EXPERIMENT_B_FEDAVG_REAL_DATA`  
**Dataset:** BraTS-GLI 2024 Adult Glioma Post-Treatment Training Cohort (1,350 total cases)  
**Canonical Split Hash:** `d0358ca42d4bf510624bbe7e86c3e4bc1e25a6174521b1280c0001eb2c66005c`  
**Hospital Partitions Hash:** `4d5905c88558883cadf9463517d3c9f856a353e4f3a570c40515f16b763fc4f8`  
**Configuration Hash:** `afd7f5de63419df8a98fafe8ce0fc0f636d4c0048939cc5e8862b1b252886ccd`  
**Hardware Compute Device:** Apple Silicon MPS (`mps` — 1 GPU, 16 GB Unified Memory)  
**Status:** **`COMPLETED`**  

---

## 1. Executive Summary & Verification Matrix

Experiment B executed 20 complete federated communication rounds using sample-weighted Federated Averaging (FedAvg) across 4 decentralized hospital silos (236 subjects each = 944 total training subjects), evaluated across all 202 validation subjects after every round at full $(128, 128, 128)$ spatial resolution.

```
================================================================================
EXPERIMENT B STATUS:         COMPLETED (20/20 Rounds)
FEDAVG_EXPERIMENT_STATUS:    COMPLETE

FEDERATED CLIENTS:           4 Hospital Silos (hospital_alpha, beta, gamma, delta)
TRAINING PARTICIPATION:      100% (4/4 Clients Participating in All 20 Rounds)
TRAINING COHORT:             944 Real Subjects (236/hospital)
VALIDATION COHORT:           202 Real Subjects (Evaluated per Round)
TEST FIREWALL STATUS:        VERIFIED (204 Test Subjects Locked — 0 Accesses)

BEST FEDERATED ROUND:        Round 20
BEST COMPOSITE MEAN DICE:    0.3815 (38.15%)
BEST WHOLE TUMOR (WT) DICE:  0.7840 (78.40%)
BEST TUMOR CORE (TC) DICE:   0.1878 (18.78%)
BEST ENHANCING TUMOR (ET):   0.1729 (17.29%)

BEST WT IoU / JACCARD:       0.6559 (65.59%)
BEST TC IoU / JACCARD:       0.1242 (12.42%)
BEST ET IoU / JACCARD:       0.1106 (11.06%)

FINAL MEAN CLIENT LOSS:      0.679523 (Round 20)
FINAL GLOBAL VAL LOSS:       0.649858 (Round 20 Record Low)

TOTAL RUNTIME:               20,561.62 seconds (5.71 hours)
AVERAGE ROUND DURATION:      1,028.01 seconds (17.13 minutes)
PEAK MPS MEMORY USAGE:       484.80 MB
OOM STATUS:                  OOM_DETECTED = FALSE

COMMUNICATION PAYLOAD:       18.35 MB / client transfer (73.40 MB / round)
CUMULATIVE COMMUNICATION:    1,468.00 MB total wire transfer over 20 rounds

BEST CHECKPOINT PATH:        checkpoints/fedavg_real/best.pt
BEST CHECKPOINT SHA-256:     41e4f62ea2825097e7890ddc1b797807cfaac0b0ce31212430f2ed45e626aeb3
CANONICAL REPORT PATH:       reports/real_brats2024/fedavg_real.json
CANONICAL MANIFEST PATH:     reports/real_brats2024/fedavg_experiment_manifest.json
================================================================================
```

---

## 2. Head-to-Head Comparison: Centralized (Exp A) vs. FedAvg (Exp B)

| Segmentation Metric | Centralized Baseline (Exp A) | FedAvg 4-Silo (Exp B) | Absolute Gap (Exp A - Exp B) | Relative Performance Retention |
|---|:---:|:---:|:---:|:---:|
| **Whole Tumor (WT) Dice** | **`0.8056`** (80.56%) | **`0.7840`** (78.40%) | **`+0.0216`** (+2.16%) | **`97.32%`** 🌟 |
| **Tumor Core (TC) Dice** | **`0.5548`** (55.48%) | **`0.1878`** (18.78%) | **`+0.3670`** (+36.70%) | **`33.85%`** |
| **Enhancing Tumor (ET) Dice** | **`0.5502`** (55.02%) | **`0.1729`** (17.29%) | **`+0.3773`** (+37.73%) | **`31.43%`** |
| **Composite Macro Dice** | **`0.6369`** (63.69%) | **`0.3815`** (38.15%) | **`+0.2554`** (+25.54%) | **`59.90%`** |
| **Whole Tumor (WT) IoU** | `0.6880` (68.80%) | `0.6559` (65.59%) | `+0.0321` | **`95.33%`** |
| **Tumor Core (TC) IoU** | `0.4639` (46.39%) | `0.1242` (12.42%) | `+0.3397` | **`26.77%`** |
| **Enhancing Tumor (ET) IoU** | `0.4581` (45.81%) | `0.1106` (11.06%) | `+0.3475` | **`24.14%`** |
| **Best Round / Epoch** | Epoch 18 / 20 | Round 20 / 20 | — | — |
| **Validation Loss** | `0.425585` | `0.649858` | `+0.224273` | — |
| **Total Runtime** | `19,763.86s` (5.49h) | `20,561.62s` (5.71h) | `+797.76s` | — |

---

## 3. Complete 20-Round Metric Progression (Experiment B)

| Round | Mean Client Loss | Val Loss | Mean Dice | WT Dice | TC Dice | ET Dice | WT IoU | TC IoU | ET IoU | Client Divergence | Global Delta | Round Time | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **01** | 0.9788 | 0.9731 | 0.0617 | 0.1537 | 0.0086 | 0.0229 | 0.0895 | 0.0044 | 0.0121 | 11.4390 | 16.0582 | 15.5m | |
| **02** | 0.9713 | 0.9670 | 0.0883 | 0.2189 | 0.0099 | 0.0361 | 0.1357 | 0.0051 | 0.0197 | 9.5330 | 17.4917 | 16.8m | |
| **03** | 0.9661 | 0.9617 | 0.1081 | 0.2669 | 0.0097 | 0.0477 | 0.1724 | 0.0049 | 0.0266 | 8.0980 | 13.3947 | 17.2m | |
| **04** | 0.9610 | 0.9561 | 0.1275 | 0.3158 | 0.0097 | 0.0571 | 0.2106 | 0.0049 | 0.0323 | 7.4241 | 9.6604 | 17.4m | |
| **05** | 0.9551 | 0.9493 | 0.1424 | 0.3522 | 0.0101 | 0.0649 | 0.2396 | 0.0052 | 0.0369 | 9.5106 | 8.0201 | 17.9m | |
| **06** | 0.9483 | 0.9417 | 0.1525 | 0.3740 | 0.0135 | 0.0700 | 0.2582 | 0.0069 | 0.0397 | 9.7775 | 7.1228 | 17.5m | |
| **07** | 0.9392 | 0.9312 | 0.1718 | 0.4162 | 0.0163 | 0.0831 | 0.2974 | 0.0084 | 0.0477 | 9.9405 | 6.8858 | 16.7m | |
| **08** | 0.9267 | 0.9160 | 0.1988 | 0.4789 | 0.0194 | 0.0982 | 0.3559 | 0.0101 | 0.0572 | 10.4578 | 6.6437 | 16.8m | |
| **09** | 0.9085 | 0.8938 | 0.2307 | 0.5513 | 0.0245 | 0.1162 | 0.4283 | 0.0131 | 0.0691 | 10.4717 | 6.2230 | 17.0m | |
| **10** | 0.8837 | 0.8659 | 0.2641 | 0.6186 | 0.0372 | 0.1365 | 0.4950 | 0.0208 | 0.0827 | 10.8266 | 5.8679 | 17.0m | |
| **11** | 0.8529 | 0.8329 | 0.2972 | 0.6781 | 0.0560 | 0.1575 | 0.5542 | 0.0326 | 0.0978 | 10.6696 | 5.5684 | 16.8m | |
| **12** | 0.8193 | 0.7960 | 0.3236 | 0.7226 | 0.0772 | 0.1710 | 0.5987 | 0.0463 | 0.1082 | 10.4776 | 5.3789 | 17.0m | |
| **13** | 0.7876 | 0.7607 | 0.3444 | 0.7516 | 0.1093 | 0.1723 | 0.6277 | 0.0674 | 0.1087 | 10.3708 | 5.1769 | 17.2m | |
| **14** | 0.7621 | 0.7351 | 0.3547 | 0.7686 | 0.1378 | 0.1576 | 0.6437 | 0.0868 | 0.0984 | 10.3503 | 4.9602 | 17.2m | |
| **15** | 0.7426 | 0.7161 | 0.3614 | 0.7768 | 0.1606 | 0.1468 | 0.6516 | 0.1025 | 0.0906 | 10.3175 | 4.8898 | 16.8m | |
| **16** | 0.7257 | 0.6990 | 0.3654 | 0.7816 | 0.1643 | 0.1503 | 0.6561 | 0.1051 | 0.0930 | 11.1153 | 4.7445 | 17.0m | |
| **17** | 0.7105 | 0.6821 | 0.3725 | 0.7913 | 0.1704 | 0.1559 | 0.6677 | 0.1098 | 0.0971 | 11.3395 | 4.4586 | 17.0m | |
| **18** | 0.6980 | 0.6746 | 0.3742 | 0.7739 | 0.1820 | 0.1669 | 0.6446 | 0.1191 | 0.1056 | 11.6172 | 4.3654 | 17.4m | |
| **19** | 0.6874 | 0.6605 | 0.3751 | 0.7865 | 0.1768 | 0.1620 | 0.6618 | 0.1154 | 0.1021 | 10.7376 | 4.5137 | 17.1m | |
| **20** | **0.6795** | **0.6499** | **`0.3815`** | **`0.7840`** | **`0.1878`** | **`0.1729`** | **`0.6559`** | **`0.1242`** | **`0.1106`** | **10.3991** | **4.4878** | **17.0m** | 🏆 **BEST** |

---

## 4. Key Scientific Insights & Empirical Analysis

1. **Near-Centralized Retention on Whole Tumor (WT):**
   - Whole Tumor (WT) reached **`78.40%`** Dice similarity (compared to `80.56%` in centralized training), demonstrating an exceptional **`97.32%` relative performance retention**.
   - WT structures span broad contextual brain regions, allowing federated parameter averaging across the 4 silos to generalize seamlessly without severe client drift.

2. **Client Drift on Sub-Centimeter Structures (TC & ET):**
   - Tumor Core (TC) and Enhancing Tumor (ET) achieved **`18.78%`** and **`17.29%`** Dice respectively (vs `55.48%` and `55.02%` centralized).
   - In pure vanilla FedAvg without proximal regularization, localized gradients on extreme class-imbalanced voxels diverge across silos ($\|W_\alpha - W_\beta\|_2 \approx 10.4-11.6$), causing naive parameter averaging to dampen fine boundary sensitivity.
   - **Empirical Scientific Implication:** This result provides the exact scientific justification and foundation for **FedProx (Experiment C)**, where the proximal term $\frac{\mu}{2} \|w - w^t\|^2$ will constrain local client drift.

3. **Decentralized Privacy & Firewall Integrity:**
   - 0 raw images or labels egressed any hospital silo boundary.
   - All 204 test subjects remained completely locked and untouched (`TRAINING_TEST_ACCESSES = 0`).

---

## 5. Benchmark Reference Table across Completed Experiments

| Method | Execution Topology | Target Resolution | WT Dice | TC Dice | ET Dice | Composite Mean Dice | Status |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Experiment A (Centralized)** | Single Node (944 Cases) | $128 \times 128 \times 128$ | **`0.8056`** | **`0.5548`** | **`0.5502`** | **`0.6369`** | **`COMPLETED`** |
| **Experiment B (FedAvg)** | 4 Silos (236 Cases each) | $128 \times 128 \times 128$ | **`0.7840`** | **`0.1878`** | **`0.1729`** | **`0.3815`** | **`COMPLETED`** |
| *Experiment C (FedProx)* | 4 Silos (236 Cases each) | $128 \times 128 \times 128$ | *Pending* | *Pending* | *Pending* | *Pending* | *Upcoming* |
| *Experiment D (FedAvg + DP)* | 4 Silos (236 Cases each) | $128 \times 128 \times 128$ | *Pending* | *Pending* | *Pending* | *Pending* | *Upcoming* |
| *Experiment E (FedAvg + HE)* | 4 Silos (236 Cases each) | $128 \times 128 \times 128$ | *Pending* | *Pending* | *Pending* | *Pending* | *Upcoming* |
| *Experiment F (FedAvg + DP + HE)* | 4 Silos (236 Cases each) | $128 \times 128 \times 128$ | *Pending* | *Pending* | *Pending* | *Pending* | *Upcoming* |
