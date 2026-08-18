# FEDMED OS — PHASE 10.11: FINAL CANDIDATE FREEZE & LOCKED TEST EVALUATION REPORT
**Author:** Antigravity AI Engine (Autonomous Biomedical Systems Agent)  
**Date:** August 18, 2026  
**Status:** `FEDMED_FINAL_MODEL_EVALUATED`  
**Experiment ID:** `EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3`  
**Candidate Checkpoint:** `checkpoints/final/fedmed_dp_final_model.pt`  
**Checkpoint SHA-256:** `f6cd18dc5e05595ca88ad1675190aabb98595ee9f6d6dfed36c25b596f652f3a`  

---

## 1. Executive Summary

In **Phase 10.11**, the FedMed Differential Privacy candidate model (`EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3`) was formally frozen and subjected to an **exactly-once locked test evaluation** against the previously untouched 204-subject test cohort of the **BraTS-GLI 2024** benchmark.

### Strict Methodological Integrity & Protocol:
- **Zero Test Set Leakage**: Checkpoint selection was performed strictly on the 202-subject validation set (`best.pt` at Round 20 based on validation Macro Dice). Zero test set evaluations, threshold adjustments, or architecture tweaks occurred during development.
- **Inference Only**: Evaluated with `model.eval()`, `torch.no_grad()`, 0 optimizer steps, 0 gradient updates, and 0 training accesses.
- **Privacy Ledger Invariance**: Differential privacy parameters remain strictly frozen at $T = 4,720$ steps per silo, $\varepsilon = 2.8934$, $\delta = 10^{-5}$ (Poisson RDP $\alpha = 7$). Evaluation is pure post-processing.

---

## 2. Locked Test Evaluation Results (204 Subjects)

| Metric | Validation Mean (202 Subj) | Locked Test Mean (204 Subj) | 95% Bootstrap CI (Test) | Generalization Gap (Abs) | Generalization Gap (Rel %) | Descriptive Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Loss** | `0.9604` | `0.9628` | `[0.9589, 0.9667]` | `+0.0024` | `+0.25%` | **Stable** |
| **Macro Dice** | `0.0800` | `0.0741` | `[0.0656, 0.0831]` | `-0.0059` | `-7.38%` | **Stable** |
| **Tumor Core (TC) Dice** | `0.2177` | `0.2054` | `[0.1816, 0.2306]` | `-0.0123` | `-5.65%` | **Stable** |
| **Enhancing Tumor (ET) Dice** | `0.0147` | `0.0111` | `[0.0086, 0.0138]` | `-0.0036` | `-24.49%` | **Moderate Degradation** |
| **Whole Tumor (WT) Dice** | `0.0076` | `0.0059` | `[0.0046, 0.0073]` | `-0.0017` | `-22.37%` | **Moderate Degradation** |
| **Macro IoU** | `0.0497` | `0.0451` | `[0.0393, 0.0516]` | `-0.0046` | `-9.26%` | **Stable** |
| **TC IoU** | `0.1377` | `0.1268` | `[0.1102, 0.1452]` | `-0.0109` | `-7.92%` | **Stable** |
| **ET IoU** | `0.0076` | `0.0057` | `[0.0044, 0.0071]` | `-0.0019` | `-25.00%` | **Moderate Degradation** |
| **WT IoU** | `0.0038` | `0.0030` | `[0.0023, 0.0037]` | `-0.0008` | `-21.05%` | **Moderate Degradation** |

### Test Descriptive Summary Statistics:
- **Macro Dice**: Mean `0.0741`, Median `0.0522`, Std `0.0650`, Min `0.0006`, Max `0.3057`, P25 `0.0256`, P75 `0.1124`.
- **TC Dice**: Mean `0.2054`, Median `0.1449`, Std `0.1757`, Min `0.0017`, Max `0.7911`, P25 `0.0685`, P75 `0.3186`.
- **Inference Runtime**: 96.42 seconds across all 204 volumes ($0.47$ seconds per 3D subject on Apple Silicon MPS).

---

## 3. Comprehensive Multi-Experiment Benchmark Matrix

| Experiment / Milestone | Scope | Optimizer | $C$ | $\sigma$ | $\sigma_{\text{coord}}$ | $T$ | $\varepsilon (\delta=10^{-5})$ | Val Loss | Macro Dice | TC Dice | ET Dice | WT Dice | Macro IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Non-Private Baseline (FedAvg)** | 20 Rounds | Adam ($\eta=10^{-4}$) | $\infty$ | $0.0$ | $0.0$ | $0$ | $\infty$ | $0.5872$ | **$0.3815$** | $0.1878$ | $0.1729$ | **$0.7840$** | **$0.2641$** |
| **Experiment D (DP Baseline)** | 20 Rounds | Adam ($\eta=10^{-4}$) | $1.0$ | $0.87$ | $0.8700$ | $4720$ | $2.8934$ | $0.9888$ | $0.0190$ | $0.0437$ | $0.0080$ | $0.0052$ | $0.0102$ |
| **Experiment D1 ($C=0.06$)** | Stage B (3 Rnds)| Adam ($\eta=10^{-4}$) | $0.06$ | $0.87$ | $0.0522$ | $708$ | $1.9636$ | $0.9883$ | $0.0205$ | $0.0479$ | $0.0088$ | $0.0048$ | $0.0110$ |
| **Experiment D2 (SGD+Mom $10^{-4}$)**| Stage B (3 Rnds)| SGD+M ($\eta=10^{-4}$) | $0.06$ | $0.87$ | $0.0522$ | $708$ | $1.9636$ | $0.9902$ | $0.0210$ | $0.0498$ | $0.0090$ | $0.0044$ | $0.0113$ |
| **Experiment D2.1 (Validation)** | 20 Rounds | SGD+M ($\eta=10^{-3}$) | $0.06$ | $0.87$ | $0.0522$ | $4720$ | $2.8934$ | $0.9604$ | **$0.0800$** | **$0.2177$** | **$0.0147$** | **$0.0076$** | **$0.0497$** |
| **Experiment D2.1 (Locked Test)** | 204 Subjects | SGD+M ($\eta=10^{-3}$) | $0.06$ | $0.87$ | $0.0522$ | $4720$ | $2.8934$ | $0.9628$ | **$0.0741$** | **$0.2054$** | **$0.0111$** | **$0.0059$** | **$0.0451$** |

*Note: Screening / Stage B results for D1 and D2 reflect intermediate 3-round screening gates and are not equivalent to full 20-round convergence.*

---

## 4. Scientific Failure & Limitation Analysis

While Experiment D2.1 achieved a **$3.90\times$ Macro Dice improvement** on the locked test set over Baseline D ($0.0741$ vs $0.0190$) and a **$4.70\times$ improvement on Tumor Core** ($0.2054$ vs $0.0437$), two structural limitations were identified:

1. **Enhancing Tumor (ET) Sparsity & Background Gradient Dominance**:
   - ET represents the necrotic vascular border, comprising $<0.5\%$ of total brain voxels. Under sample-level DP noise, signal-to-noise ratio for sparse foreground structures is heavily suppressed by background gradients, leading to conservative sigmoid probabilities below the $0.5$ binarization threshold.
2. **Whole Tumor (WT) Infiltration & Soft Boundary Attenuation**:
   - WT includes vasogenic edema (SNFH label 2) which exhibits diffuse, low-gradient margins on T2/FLAIR. While non-private optimization learns these subtle transitions easily ($0.7840$), isotropic DP perturbation flattens subtle edema gradients, concentrating parameter updates exclusively on high-contrast Tumor Core boundaries.

---

## 5. Artifact & Immutability Verification

| Artifact Path | Description | SHA-256 Digest |
| :--- | :--- | :--- |
| `checkpoints/final/fedmed_dp_final_model.pt` | Frozen Final Candidate Model | `f6cd18dc5e05595ca88ad1675190aabb98595ee9f6d6dfed36c25b596f652f3a` |
| `reports/final/fedmed_final_candidate_manifest.json` | Final Candidate Manifest | `1686df047be88e998db669f91baae085600c3b88b22a2bb7f7f8582d92131ef7` |
| `reports/final/fedmed_pretest_audit.json` | Pre-Test Immutability Snapshot | `da6fa6bb8c499f57d6ea1ff6778f9f60bc9318b855523ce75d506d3389bc70b1` |
| `reports/final/fedmed_final_test_results.json` | Locked Test Results & Bootstrap CIs | `88c679b37c468e27c8ff7ea9eb4523da912f27599cb5103a4c0ae3c06e12ea99` |
| `reports/final/fedmed_test_subject_metrics.csv` | Per-Subject Metrics (204 Rows) | `9559c5d0b98bc83f0cae25e3630f5763b0bb3144ef9e776510f27bf237078e63` |
| `reports/final/fedmed_generalization_analysis.json` | Val-to-Test Generalization Analysis | `c0fc6694e9f783aa8ba3e6a2b851b272f69cebba2b757e74f8c9b3d9c73e0474` |
| `reports/final/fedmed_final_comparison.json` | Comparative Benchmark Matrix | `85a153282b0e6dfb4c0bf09228d48e895781a5d5e5bf5be045842880017f8b91` |

**Baseline Immutability Verified**: Experiments D, D1, D2, D2.1 Stage B archives remain 100% bit-for-bit identical to pre-test baseline hashes.
