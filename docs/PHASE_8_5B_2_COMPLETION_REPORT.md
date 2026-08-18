# FEDMED OS — PHASE 8.5B.2 COMPLETION REPORT

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Local Training Execution & Model Integrity Gate (Phase 8.5B.2)  
**Execution Mode:** `DEVELOPMENT_SYNTHETIC`  
**Gate Status:** `PASS`  
**Real Data Status:** `BLOCKED — REAL BRATS DATASET NOT PRESENT`  
**Scientific Performance Status:** `BLOCKED — DEVELOPMENT COHORT INSUFFICIENT FOR CLINICAL CLAIMS`  

---

## 1. What Was Executed & Verified

1. **Dataset Mode Safety & Authorization:**
   - Evaluated `reports/brats_dataset_manifest.json` and detected mode `DEVELOPMENT_SYNTHETIC`.
   - Verified that running without `--allow-development` halts with `DEVELOPMENT DATASET — EXPLICIT FLAG REQUIRED`.
   - Verified execution with `--allow-development` explicitly labels the run as `DEVELOPMENT_SYNTHETIC`.

2. **Label Semantics & BraTS Sub-Region Mapping:**
   - Raw development labels: `[0, 1, 2]`
   - MONAI `ConvertToMultiChannelBasedOnBratsClasses` transforms raw labels into:
     - Channel 0 (TC): `(img == 1) | (img == 4)` $\to$ Evaluates to `(img == 1)` (TC active)
     - Channel 1 (WT): `(img == 1) | (img == 4) | (img == 2)` $\to$ Evaluates to `(img == 1) | (img == 2)` (WT active)
     - Channel 2 (ET): `img == 4` $\to$ Evaluates to all-zeros mask in development mini-cohort.
   - Preserved zero silent remapping.

3. **Canonical Model Instantiation & Parameter Verification:**
   - Model loaded from `configs/experiments/real_brats_fedavg.yaml`: MONAI 3D UNet (`spatial_dims=3, in_channels=4, out_channels=3, channels=[16, 32, 64, 128, 256], strides=[2, 2, 2, 2], num_res_units=2`).
   - Programmatically measured trainable parameter count: **4,810,074 parameters**.

4. **Persisted Split Loading & Test Set Firewall:**
   - Loaded `reports/dataset_split.json` (Seed: 42, Split Hash: `4ce10ac52c093ec0...`).
   - `TRAIN` cohort: `['BraTS2021_00003', 'BraTS2021_00004']` (used exclusively for training updates).
   - `VALIDATION` cohort: `['BraTS2021_00002']` (used exclusively for validation metrics and checkpointing).
   - `TEST` cohort: `['BraTS2021_00001']` (strictly firewalled; NO DataLoader instantiated, zero access during training).
   - Recorded: `TEST_SET_ACCESSED_DURING_TRAINING = False`, `TEST_EVALUATION_STATUS = NOT_EVALUATED`.

5. **First Batch Reality Check:**
   - Verified 5D image tensor `(1, 4, 32, 32, 32)` and 5D target tensor `(1, 3, 32, 32, 32)`.
   - Verified float32 dtypes and finite values.
   - Verified initial forward pass: `logits.shape == target.shape`.

6. **Genuine Backpropagation & Parameter Updates:**
   - `DiceCELoss(sigmoid=True)` computed scalar loss: `1.876476`.
   - `loss.backward()` populated non-zero gradients: `gradient_norm = 1.386978` (`grad_norm > 0: True`).
   - `optimizer.step()` updated weights: `max_parameter_delta = 2.001400e-04`, `l2_parameter_delta = 0.334893` (`delta > 0: True`).

7. **Independent Validation Inference & Checkpointing:**
   - Evaluated validation inference with `torch.no_grad()` on held-out subject `BraTS2021_00002`:
     - TC Dice: `0.4540`, WT Dice: `0.6269`, ET Dice: `0.0000`, Mean Dice: `0.3603`
     - TC IoU: `0.2936`, WT IoU: `0.4566`, ET IoU: `0.0000`, Mean IoU: `0.2501`
   - Generated best checkpoint at `checkpoints/local_baseline_best.pt` (zero test metrics contained).
   - Generated experiment audit JSON at `reports/experiments/local_training/run_*.json`.

8. **Automated Integration Tests:**
   - Created `tests/integration/test_local_training.py` verifying model loading, split loading, firewall integrity, backpropagation, and checkpoint serialization.

---

## 2. Experimental Limitation

> [!WARNING]
> The current development cohort (2 train, 1 val, 1 test) is insufficient for meaningful medical or federated performance conclusions. This execution strictly verifies software pipeline integrity, loss backpropagation, weight updates, and independent validation without data leakage.

---

## 3. Exact Next Phase

**Phase 8.5B.3: Real 2-Hospital Federated Training Gate**
- Coordinate Hospital Alpha (`BraTS2021_00003`) and Hospital Beta (`BraTS2021_00004`) over 3 federated rounds.
- Verify initial parameter distribution, disjoint local training, non-identical local updates, Flower `FedAvg` sample-weighted aggregation, global weight delta, and validation monitoring on held-out subject `BraTS2021_00002`.
