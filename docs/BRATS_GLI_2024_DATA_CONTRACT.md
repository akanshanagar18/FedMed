# FEDMED OS — BRATS-GLI 2024 CLINICAL DATA CONTRACT SPECIFICATION

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Dataset:** BraTS-GLI 2024 Official Cohort  
**Dataset Name:** `BraTS-GLI`  
**Dataset Version:** `2024`  
**Dataset Task:** `adult_glioma_post_treatment`  

---

## 1. Input Modality Specifications & Mapping

Every subject directory contains 4 multi-parametric MRI sequences and 1 segmentation mask:

| Raw File Suffix | Canonical Modality Key | Input Channel | Description |
|---|---|---|---|
| `*-t1n.nii.gz` | `t1` | Channel 0 | T1-weighted native non-contrast |
| `*-t1c.nii.gz` | `t1ce` | Channel 1 | T1-weighted post-contrast (Gadolinium) |
| `*-t2w.nii.gz` | `t2` | Channel 2 | T2-weighted native |
| `*-t2f.nii.gz` | `flair` | Channel 3 | T2 Fluid-Attenuated Inversion Recovery |
| `*-seg.nii.gz` | `seg` | Target Mask | Multi-class expert ground truth |

---

## 2. Geometric & Physical Requirements

- **Spatial Matrix Dimensions:** $182 \times 218 \times 182$ voxels.
- **Voxel Spacing (Resolution):** $1.0 \times 1.0 \times 1.0\text{ mm}^3$ isotropic.
- **Template Coordinate Space:** MNI152 standard space.
- **Affine Matrix Consistency:** All 4 MRI sequences for a given patient study must share identical spatial affine matrices ($\Delta_{\text{affine}} < 10^{-3}$).
- **Numerical Integrity:** Strictly finite voxel values (zero `NaN`, `+Inf`, `-Inf`).
