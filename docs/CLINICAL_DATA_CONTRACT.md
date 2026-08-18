# FEDMED OS — CLINICAL DATA CONTRACT & LABEL SEMANTICS SPECIFICATION

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Clinical Data Contract, Modality Specifications, and BraTS Label Semantics  
**Phase:** Phase 8.5B.7 (Clinical Audit & Production Model Packaging Gate)  

---

## 1. Input Modality Specifications

FedMed requires four co-registered multi-parametric Magnetic Resonance Imaging (mpMRI) sequences per patient study:

| Modality Key | Clinical Description | NIfTI Suffix Pattern | Expected Tissue Contrast |
|---|---|---|---|
| **T1** | T1-weighted native | `*_t1.nii`, `*_t1.nii.gz` | Anatomical baseline, gray/white matter contrast |
| **T1ce** | T1-weighted contrast-enhanced (Gadolinium) | `*_t1ce.nii`, `*_t1ce.nii.gz` | Hyperintense active / enhancing tumor margins |
| **T2** | T2-weighted native | `*_t2.nii`, `*_t2.nii.gz` | Hyperintense fluid / peritumoral edema |
| **FLAIR** | Fluid-Attenuated Inversion Recovery | `*_flair.nii`, `*_flair.nii.gz` | Fluid suppression, whole tumor boundary distinction |

---

## 2. Structural & Numerical Image Integrity Requirements

- **File Format:** NIfTI-1.0 / NIfTI-2.0 standard (`.nii` or `.nii.gz`).
- **Spatial Dimensions:**
  - Real BraTS 2021: $240 \times 240 \times 155$ voxels.
  - Development Synthetic: $32 \times 32 \times 32$ voxels.
- **Voxel Spacing (Resolution):** $1.0 \times 1.0 \times 1.0\text{ mm}^3$ isotropic.
- **Numerical Integrity:**
  - All voxel values must be finite (strictly no `NaN`, `+Inf`, or `-Inf`).
  - Standard deviation across brain tissue must be $> 0$ (non-blank).
- **Affine & Orientation:**
  - All 4 modalities for a given subject must share identical spatial affine matrices ($\Delta_{\text{affine}} < 10^{-3}$).
  - Canonical orientation: Neurological RAS/LPS alignment.

---

## 3. BraTS Segmentation Label Semantics Audit

### 3.1 Raw Ground-Truth Voxel Labels:
- `0`: Background (non-tumor brain tissue or air).
- `1`: Necrotic core / non-enhancing tumor (NCR/NET).
- `2`: Peritumoral edema (ED).
- `4`: GD-enhancing tumor (ET).

### 3.2 Clinical Target Evaluation Channels:
1. **Tumor Core (TC):**
   $$\text{TC} = (\text{label} == 1) \lor (\text{label} == 4)$$
2. **Whole Tumor (WT):**
   $$\text{WT} = (\text{label} == 1) \lor (\text{label} == 2) \lor (\text{label} == 4)$$
3. **Enhancing Tumor (ET):**
   $$\text{ET} = (\text{label} == 4)$$

---

## 4. Development Cohort Label Limitation Audit

> [!IMPORTANT]
> - The local 4-case development mini-cohort contains ground-truth voxel labels `[0, 1, 2]`.
> - **ET Ground Truth Presence:** `ET_GROUND_TRUTH_PRESENT = FALSE`
> - **ET Metric Interpretation:** `ET_METRIC_INTERPRETATION = NOT_MEANINGFUL_ON_DEVELOPMENT_COHORT`
> - Raw ET Dice and IoU on development cases will evaluate to `0.0000` because the positive ground truth set for label 4 is empty.
> - TC and WT metrics remain active and mathematically valid on labels `1` and `1 + 2`.
