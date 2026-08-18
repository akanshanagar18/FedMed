# FEDMED OS — PHASE 8.5B.1 BASELINE AUDIT

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Data Foundation & Leakage Integrity Gate (Phase 8.5B.1)  
**Git Branch:** `release/stabilization-rc2`  
**Git Head:** `e266612 Phase 10: FedMed Omega production hardening and runtime validation`  

---

## 1. Current State Summary

- **Local NIfTI Storage:** `data/BraTS2021/` containing 4 development mini-cohort cases (`BraTS2021_00001` to `00004`, spatial shape `(32, 32, 32)`).
- **Real Full BraTS Cohort (~40 GB, 1,251 cases):** NOT present locally.
- **Dataset Classification Status:** Currently unclassified; must be explicitly classified as `DEVELOPMENT_SYNTHETIC`.
- **Subject-Level Split Status:** Currently 2-way random array slicing (`train_files`, `val_files`) in `BraTSDataset` without an independent held-out `TEST` split or persistent JSON manifest.
- **Hospital Partition Status:** `data/partitioner.py` partitioned all subjects across silos without isolating `TRAIN` subjects from `VALIDATION` / `TEST`.
- **Metric Decoupling Status:** Dice and IoU must be calculated independently from voxel overlaps without using analytical proxy formulas (`IoU = Dice / (2 - Dice)`).

---

## 2. Component Baseline Inspection

| Component | File Path | Baseline State | Required Phase 8.5B.1 Changes |
| :--- | :--- | :--- | :--- |
| **Dataset Validator** | `data/datasets/validators.py` | Basic file existence checker | Add comprehensive NIfTI integrity validator (`scripts/validate_brats_dataset.py`) checking affine, dimensions, spacing, finite values, and non-empty masks, generating `reports/brats_dataset_manifest.json`. |
| **Subject Splitter** | `data/splitter.py` | Missing | Create deterministic 3-way subject splitter (`TRAIN`, `VALIDATION`, `TEST`) generating `reports/dataset_split.json`. |
| **Data Leakage Tests** | `tests/integration/test_data_leakage.py` | Missing | Create tests verifying zero train/val, train/test, and val/test overlap. |
| **Hospital Partitioner** | `data/partitioner.py` | Partitions all subjects | Update to partition ONLY `TRAIN` subject IDs across 4 hospitals (`hospital_alpha`, `hospital_beta`, `hospital_gamma`, `hospital_delta`) and output `reports/hospital_partitions.json`. |
| **Canonical Model Config** | `configs/experiments/real_brats_fedavg.yaml` | Missing | Create canonical YAML config specifying MONAI 3D UNet (4,810,074 parameters) as single source of truth. |
| **Independent Metrics** | `evaluation/metrics.py` | Missing | Implement decoupled `compute_dice()` and `compute_iou()` functions. |
| **Metric Unit Tests** | `tests/unit/test_segmentation_metrics.py` | Missing | Implement unit tests for edge cases (perfect, zero, partial, empty pred, empty gt, multi-channel). |
| **Metric Provenance Audit** | `docs/PHASE_8_5B_1_METRIC_PROVENANCE.md` | Missing | Document all historical and active metric paths. |
