# FEDMED OS — BRATS-GLI 2024 PREPROCESSING AUDIT & VERIFICATION

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Transformation Pipeline Verification from Native $182 \times 218 \times 182$ to Canonical Target $(32, 32, 32)$ / $(128, 128, 128)$  

---

## 1. Preprocessing Pipeline Verification

```
┌───────────────────────────────────────────────────────────┐
│ RAW NIFTI VOLUMES (182 x 218 x 182, 1.0mm isotropic)      │
│ [t1n, t1c, t2w, t2f] + seg                                │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ 1. CANONICAL MODALITY & LABEL ADAPTER                      │
│    - Stack 4 sequences in order: [t1, t1ce, t2, flair]    │
│    - Map labels to composite (TC, WT, ET) targets         │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ 2. MONAI TRANSFORMS                                       │
│    - EnsureChannelFirstd: shape (4, 182, 218, 182)        │
│    - Orientationd: RAS orientation alignment              │
│    - Spacingd: 1.0mm isotropic (preserves native spacing)  │
│    - NormalizeIntensityd: Non-zero Z-score normalization  │
│    - Resized / SpatialCrop: Target spatial shape          │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ CANONICAL TENSOR INTERFACE                                │
│ Image: (4, 32, 32, 32) float32 tensor                     │
│ Target: (3, 32, 32, 32) float32 tensor                    │
└───────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Integrity Check

- **Non-zero Z-score Normalization:** Voxel values are normalized independently across each channel's foreground brain tissue ($v' = (v - \mu) / \sigma$).
- **Isotropic Grid Consistency:** Because native voxel spacing in BraTS-GLI 2024 is already $1.0\text{ mm}^3$ isotropic, spatial interpolation artifacts during orientation alignment are zero.
- **Dynamic Resampling:** The MONAI `Resized` transform operates smoothly across varying spatial aspect ratios ($182 \times 218 \times 182 \to 32 \times 32 \times 32$).
