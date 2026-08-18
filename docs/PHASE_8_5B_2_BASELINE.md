# FEDMED OS — PHASE 8.5B.2 BASELINE & LABEL SEMANTICS AUDIT

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Local Training Execution & Model Integrity Gate (Phase 8.5B.2)  
**Git Branch:** `release/stabilization-rc2`  
**Git Head:** `e266612 Phase 10: FedMed Omega production hardening and runtime validation`  

---

## 1. Step 0 Baseline State

- **Validated Dataset Manifest:** `reports/brats_dataset_manifest.json` (Mode: `DEVELOPMENT_SYNTHETIC`, 4 subjects).
- **Validated 3-Way Split:** `reports/dataset_split.json` (Seed: 42, Split Hash: `4ce10ac52c093ec0...`):
  - `TRAIN` (2 subjects): `['BraTS2021_00003', 'BraTS2021_00004']`
  - `VALIDATION` (1 subject): `['BraTS2021_00002']`
  - `TEST` (1 subject): `['BraTS2021_00001']`
- **Canonical Model Config:** `configs/experiments/real_brats_fedavg.yaml` specifying MONAI 3D UNet (4,810,074 parameters), `DiceCELoss(sigmoid=True)`, and Adam optimizer.

---

## 2. Step 1 Label Semantics & Sub-Region Mapping

### Real BraTS 2021 / MICCAI Standard:
- **Raw Labels in Real BraTS:**
  - `0`: Background (healthy tissue / CSF / air)
  - `1`: Necrotic core (NCR)
  - `2`: Peritumoral edema (ED)
  - `4`: GD-enhancing tumor (ET)

- **Target BraTS Sub-Regions (MONAI `ConvertToMultiChannelBasedOnBratsClasses`):**
  - **Channel 0 — Tumor Core (TC):** `(img == 1) | (img == 4)` (Necrotic core + Enhancing tumor)
  - **Channel 1 — Whole Tumor (WT):** `(img == 1) | (img == 4) | (img == 2)` (Necrotic core + Enhancing tumor + Edema)
  - **Channel 2 — Enhancing Tumor (ET):** `img == 4` (GD-enhancing tumor)

### Current Development Mini-Cohort (`data/BraTS2021/`):
- **Raw Labels Discovered:** `[0, 1, 2]`
- **Target Channel Transformations:**
  - **Channel 0 (TC):** Evaluates to `(img == 1)` (since label 4 is absent in this mini-cohort).
  - **Channel 1 (WT):** Evaluates to `(img == 1) | (img == 2)`.
  - **Channel 2 (ET):** Evaluates to `(img == 4)` (all-zeros mask in development cohort).
- **Semantics Policy:** No silent label remapping. The pipeline runs MONAI standard `ConvertToMultiChannelBasedOnBratsClasses` identically for development and real BraTS data.
