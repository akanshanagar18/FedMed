# FEDMED OS — PHASE 8.5B BASELINE AUDIT

**Audit Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Phase 8.5B Baseline State & Readiness Audit  
**Status:** AUDITED — READY FOR STEP 1 BASELINE REVIEW  

---

## 1. Executive Summary

Phase 8.5A established software execution integrity across the pipeline (data loading $\to$ preprocessing $\to$ 3D UNet $\to$ backpropagation $\to$ Flower aggregation $\to$ validation $\to$ telemetry). However, Phase 8.5A utilized a 4-case development mini-cohort.

Phase 8.5B establishes a rigorous, scientifically valid experimental protocol:
1. Explicit dataset classification (`DEVELOPMENT_SYNTHETIC`, `REAL_BRATS_SUBSET`, `FULL_REAL_BRATS`) to avoid confusing development mini-cohorts with genuine full-resolution cohorts.
2. Subject-level 3-way disjoint split (`TRAIN`, `VALIDATION`, `TEST`) with zero data leakage.
3. Hospital partitioning strictly applied only to the `TRAIN` split.
4. Independent, mathematically distinct `compute_dice()` and `compute_iou()` calculation functions (no formulaic coupling).
5. Canonical 3D UNet architecture with 4,810,074 trainable parameters used identically across centralized baseline, local training, and multi-hospital federated experiments.
6. Execution of real centralized baseline and 2-hospital / 4-hospital `FedAvg` experiments with JSON audit trails under `reports/experiments/`.

---

## 2. Baseline Audit Table

| Area | Current State | Evidence | Required Change | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Dataset** | 4 mini-NIfTI development cases (`(32, 32, 32)`) in `data/BraTS2021/`. Full 1,250 case cohort not present locally. | `data/BraTS2021/BraTS2021_00001` to `00004` on disk. | Implement explicit dataset classification (`DEVELOPMENT_SYNTHETIC`, `REAL_BRATS_SUBSET`, `FULL_REAL_BRATS`). Report required format for real BraTS ingestion. | **CODE READY — DATASET CLASSIFIED** |
| **Dataset Integrity** | `DatasetValidator` in `data/datasets/validators.py` checks file presence. | `DatasetValidator.validate_dataset_directory()` verified in Phase 8.5A. | Create `scripts/validate_brats_dataset.py` to inspect dimensions, affine, voxel spacing, finite values, non-empty labels, and output `reports/brats_dataset_manifest.json`. | **PLANNED** |
| **Patient Split** | Random index split in `BraTSDataset` into 2 splits (train/val). No explicit held-out test split. | `data/datasets/brats.py:110-119`. | Implement deterministic 3-way subject-level split (`TRAIN`, `VALIDATION`, `TEST`) with zero overlap; persist to `reports/dataset_split.json`. | **PLANNED** |
| **Preprocessing** | MONAI transform pipeline with RAS orientation, Spacing, CropForeground, SpatialPad, and BraTS sub-region mapping. | `data/datasets/transforms.py:30-95`. | Ensure preprocessing is applied identically across centralized baseline, local training, and hospital clients. | **WORKING** |
| **Model** | MONAI 3D UNet with 4,810,074 parameters. | `model/unet3d.py` (`channels=(16,32,64,128,256)`, `strides=(2,2,2,2)`, `res_units=2`). | Create canonical configuration in `configs/experiments/real_brats_fedavg.yaml` and resolve all parameter count documentation to 4,810,074. | **DOCUMENTED** |
| **Local Training** | `model/trainer.py` and `client/flower_client.py` perform forward pass, `DiceCELoss(sigmoid=True)`, backward pass, and `optimizer.step()`. | Verified parameter update $\Delta W > 0$ in `scripts/validate_real_brats_pipeline.py`. | Create standalone `scripts/run_real_local_training.py` with strict dataset mode reporting. | **PLANNED** |
| **FL Partitioning** | `data/partitioner.py` partitions full dataset among hospitals. | `data/partitioner.py:65-192`. | Partition *only* the `TRAIN` split among hospitals. Keep validation and test splits globally held out. Persist to `reports/hospital_partitions.json`. | **PLANNED** |
| **Validation** | Centralized script validates on validation split; global federated validation evaluated on held-out subject data. | `scripts/run_real_fedmed_smoke_test.py:186-218`. | Implement strict 2-stage validation: global validation during training rounds on `VALIDATION` split, final global model evaluation on `TEST` split. | **PLANNED** |
| **Metrics** | Exact voxel-wise Dice and IoU calculation in smoke tests; formulaic IoU removed from flower adapter in 8.5A. | `server/strategies/adapters/flower_adapter.py:136-143`. | Create separate, decoupled `compute_dice()` and `compute_iou()` functions with comprehensive unit tests for all edge cases (`tests/unit/test_segmentation_metrics.py`). | **PLANNED** |
| **Benchmarking** | Historical benchmark runner contained static lookup tables in `evaluation/benchmark_runner.py`. | `evaluation/benchmark_runner.py:32-39`. | Build new benchmark runner executing genuine Centralized, FedAvg, and FedProx experiment runs without lookup tables. | **PLANNED** |

---

## 3. Canonical Model Architecture Specification

- **Architecture:** MONAI 3D UNet (`UNet3D`)
- **Spatial Dimensions:** 3D
- **Input Channels:** 4 (`FLAIR`, `T1`, `T1ce`, `T2`)
- **Output Channels:** 3 (Multi-label BraTS sub-regions: `TC`, `WT`, `ET`)
- **Layer Feature Channels:** `(16, 32, 64, 128, 256)`
- **Strides:** `(2, 2, 2, 2)`
- **Residual Units:** 2 per block
- **Exact Parameter Count:** 4,810,074 trainable parameters
- **Loss Function:** MONAI `DiceCELoss(sigmoid=True)`
- **Optimizer:** `torch.optim.Adam(lr=1e-4, weight_decay=1e-5)`
- **Batch Size:** 2 (configurable to 1 for memory-constrained environments)
- **Local Epochs:** 1 per round
