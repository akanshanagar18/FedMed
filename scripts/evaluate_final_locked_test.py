"""
Script: scripts/evaluate_final_locked_test.py

Purpose:
FEDMED OS — PHASE 10.11: EXACTLY-ONCE LOCKED TEST EVALUATION & CHARACTERIZATION.
Evaluates the frozen final clinical candidate model (EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3)
against ALL 204 locked test subjects from BraTS-GLI 2024.

Strict Invariants:
- INFERENCE ONLY. model.eval(), torch.no_grad(), zero optimizer calls, zero gradient calculation.
- Evaluates exactly once against the locked 204 test subjects.
- No model selection, hyperparameter tuning, or threshold tuning.
- Decoupled Dice and IoU evaluation across all composite sub-regions.
- Computes comprehensive per-subject metrics, descriptive statistics, and 95% bootstrap confidence intervals.
- Generates all Phase 10.11 final reports.
"""

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.canonical_adapter import DatasetVersion, ModalityCanonicalizer
from data.datasets.transforms import get_brats_transforms
from data.real_brats_pipeline import RealBratsValidator
from evaluation.metrics import compute_dice, compute_iou

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fedmed_final_test_evaluation")


class LockedBratsTestDataset(Dataset):
    def __init__(self, subject_dirs: List[Path], spatial_shape: Tuple[int, int, int] = (128, 128, 128)):
        self.subject_dirs = subject_dirs
        self.spatial_shape = spatial_shape
        self.transforms = get_brats_transforms(
            mode="val",
            image_size=spatial_shape,
            dataset_version=DatasetVersion.BRATS_GLI_2024,
        )

    def __len__(self) -> int:
        return len(self.subject_dirs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s_dir = self.subject_dirs[idx]
        version, mod_paths, missing = ModalityCanonicalizer.discover_subject_modalities(s_dir)
        if missing:
            raise ValueError(f"Missing modalities in {s_dir.name}: {missing}")

        mod_files = [str(mod_paths[m]) for m in ("t1", "t1ce", "t2", "flair")]
        seg_file = str(mod_paths["seg"])

        sample = {"image": mod_files, "label": seg_file}
        processed = self.transforms(sample)

        return {
            "image": processed["image"],   # (4, 128, 128, 128)
            "target": processed["label"],  # (3, 128, 128, 128)
            "subject_id": s_dir.name,
        }


def compute_bootstrap_ci(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes bootstrap mean and percentile confidence intervals."""
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        resample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(resample)
    
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    mean_val = float(np.mean(data))
    return round(mean_val, 4), round(lower, 4), round(upper, 4)


def compute_descriptive_stats(data: List[float]) -> Dict[str, float]:
    """Computes standard descriptive summary statistics."""
    arr = np.array(data, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "p25": round(float(np.percentile(arr, 25.0)), 4),
        "p75": round(float(np.percentile(arr, 75.0)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="FedMed Final Locked Test Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final/fedmed_dp_final_model.pt")
    parser.add_argument("--split-file", type=str, default="reports/real_brats2024/dataset_split.json")
    parser.add_argument("--raw-data-dir", type=str, default="data/raw/BraTS2024")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device if torch.backends.mps.is_available() and args.device == "mps" else "cpu")
    logger.info("Evaluation Device: %s", device)

    # 1. Verify frozen checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Frozen checkpoint not found at %s", ckpt_path)
        sys.exit(1)

    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        hasher.update(f.read())
    ckpt_sha = hasher.hexdigest()
    logger.info("Evaluating Frozen Model Checkpoint: %s (SHA-256: %s)", ckpt_path, ckpt_sha)

    # 2. Instantiate Model and Load Weights
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.0,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    logger.info("Model loaded strictly (4,810,074 parameters). Zero parameters modifiable.")

    # 3. Instantiate Loss Function
    loss_fn = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.2)

    # 4. Discover All Subjects and Resolve the 204 Locked Test Subjects
    raw_dir = Path(args.raw_data_dir)
    validator = RealBratsValidator(raw_dir)
    all_subjs = {p.name: p for p in validator.discover_subject_directories()}
    logger.info("Discovered total %d subject directories across raw dataset", len(all_subjs))

    with open(args.split_file, "r") as f:
        split_data = json.load(f)

    test_subject_ids = split_data.get("test_subjects", [])
    if len(test_subject_ids) != 204:
        logger.error("Expected exactly 204 test subjects in split file, found %d", len(test_subject_ids))
        sys.exit(1)

    test_subject_dirs = [all_subjs[sid] for sid in test_subject_ids if sid in all_subjs]
    if len(test_subject_dirs) != 204:
        logger.error("Expected 204 resolved test subject directories, found %d", len(test_subject_dirs))
        sys.exit(1)

    logger.info("Discovered and resolved all 204 locked test subject directories.")

    # 5. Build Dataset and DataLoader
    test_dataset = LockedBratsTestDataset(test_subject_dirs, spatial_shape=(128, 128, 128))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 6. Execute Exactly-Once Evaluation
    logger.info("Starting EXACTLY-ONCE Inference on 204 Test Subjects...")
    start_time = time.time()

    per_subject_records: List[Dict[str, Any]] = []
    subject_losses: List[float] = []
    subject_wt_dices: List[float] = []
    subject_tc_dices: List[float] = []
    subject_et_dices: List[float] = []
    subject_macro_dices: List[float] = []

    subject_wt_ious: List[float] = []
    subject_tc_ious: List[float] = []
    subject_et_ious: List[float] = []
    subject_macro_ious: List[float] = []

    nan_count = 0
    inf_count = 0
    failed_count = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            s_id = batch["subject_id"][0]
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            try:
                logits = model(images)
                loss = loss_fn(logits, targets).item()

                if np.isnan(loss):
                    nan_count += 1
                if np.isinf(loss):
                    inf_count += 1

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Metric computation matching canonical validation pipeline
                d_res = compute_dice(preds_np, targets_np, channel_names=["WT", "TC", "ET"])
                i_res = compute_iou(preds_np, targets_np, channel_names=["WT", "TC", "ET"])

                wt_d = float(d_res.get("dice_WT", 0.0))
                tc_d = float(d_res.get("dice_TC", 0.0))
                et_d = float(d_res.get("dice_ET", 0.0))
                macro_d = float((wt_d + tc_d + et_d) / 3.0)

                wt_i = float(i_res.get("iou_WT", 0.0))
                tc_i = float(i_res.get("iou_TC", 0.0))
                et_i = float(i_res.get("iou_ET", 0.0))
                macro_i = float((wt_i + tc_i + et_i) / 3.0)

                subject_losses.append(loss)
                subject_wt_dices.append(wt_d)
                subject_tc_dices.append(tc_d)
                subject_et_dices.append(et_d)
                subject_macro_dices.append(macro_d)

                subject_wt_ious.append(wt_i)
                subject_tc_ious.append(tc_i)
                subject_et_ious.append(et_i)
                subject_macro_ious.append(macro_i)

                per_subject_records.append({
                    "subject_id": s_id,
                    "test_loss": round(loss, 4),
                    "macro_dice": round(macro_d, 4),
                    "tc_dice": round(tc_d, 4),
                    "et_dice": round(et_d, 4),
                    "wt_dice": round(wt_d, 4),
                    "macro_iou": round(macro_i, 4),
                    "tc_iou": round(tc_i, 4),
                    "et_iou": round(et_i, 4),
                    "wt_iou": round(wt_i, 4),
                })

                if (i + 1) % 25 == 0 or (i + 1) == len(test_loader):
                    logger.info(
                        "Evaluated [%d/%d] subjects | Running Mean Macro Dice: %.4f | TC: %.4f | ET: %.4f | WT: %.4f",
                        i + 1,
                        len(test_loader),
                        np.mean(subject_macro_dices),
                        np.mean(subject_tc_dices),
                        np.mean(subject_et_dices),
                        np.mean(subject_wt_dices),
                    )

            except Exception as e:
                logger.error("Error evaluating subject %s: %s", s_id, e)
                failed_count += 1

    total_eval_time = time.time() - start_time
    logger.info("Evaluation completed in %.2f seconds (%.2f s/subject).", total_eval_time, total_eval_time / len(test_loader))

    # 7. Compute Summary Statistics and Bootstrap Confidence Intervals
    mean_macro_dice, macro_ci_low, macro_ci_high = compute_bootstrap_ci(np.array(subject_macro_dices))
    mean_tc_dice, tc_ci_low, tc_ci_high = compute_bootstrap_ci(np.array(subject_tc_dices))
    mean_et_dice, et_ci_low, et_ci_high = compute_bootstrap_ci(np.array(subject_et_dices))
    mean_wt_dice, wt_ci_low, wt_ci_high = compute_bootstrap_ci(np.array(subject_wt_dices))

    mean_macro_iou, macro_iou_ci_low, macro_iou_ci_high = compute_bootstrap_ci(np.array(subject_macro_ious))
    mean_tc_iou, tc_iou_ci_low, tc_iou_ci_high = compute_bootstrap_ci(np.array(subject_tc_ious))
    mean_et_iou, et_iou_ci_low, et_iou_ci_high = compute_bootstrap_ci(np.array(subject_et_ious))
    mean_wt_iou, wt_iou_ci_low, wt_iou_ci_high = compute_bootstrap_ci(np.array(subject_wt_ious))

    test_results = {
        "status": "LOCKED_TEST_EVALUATION_COMPLETE",
        "experiment_id": "EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3",
        "timestamp_evaluation": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": ckpt_sha,
            "trainable_parameters": 4810074,
        },
        "evaluation_summary": {
            "total_test_subjects": len(test_subject_ids),
            "evaluated_subjects": len(per_subject_records),
            "failed_subjects": failed_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "total_runtime_seconds": round(total_eval_time, 2),
            "average_seconds_per_subject": round(total_eval_time / len(test_loader), 2),
            "device": str(device),
        },
        "primary_metrics": {
            "test_loss": round(float(np.mean(subject_losses)), 4),
            "macro_dice": mean_macro_dice,
            "tc_dice": mean_tc_dice,
            "et_dice": mean_et_dice,
            "wt_dice": mean_wt_dice,
            "macro_iou": mean_macro_iou,
            "tc_iou": mean_tc_iou,
            "et_iou": mean_et_iou,
            "wt_iou": mean_wt_iou,
        },
        "confidence_intervals_95_percent_bootstrap": {
            "macro_dice": {"mean": mean_macro_dice, "ci_lower": macro_ci_low, "ci_upper": macro_ci_high},
            "tc_dice": {"mean": mean_tc_dice, "ci_lower": tc_ci_low, "ci_upper": tc_ci_high},
            "et_dice": {"mean": mean_et_dice, "ci_lower": et_ci_low, "ci_upper": et_ci_high},
            "wt_dice": {"mean": mean_wt_dice, "ci_lower": wt_ci_low, "ci_upper": wt_ci_high},
            "macro_iou": {"mean": mean_macro_iou, "ci_lower": macro_iou_ci_low, "ci_upper": macro_iou_ci_high},
            "tc_iou": {"mean": mean_tc_iou, "ci_lower": tc_iou_ci_low, "ci_upper": tc_iou_ci_high},
            "et_iou": {"mean": mean_et_iou, "ci_lower": et_iou_ci_low, "ci_upper": et_iou_ci_high},
            "wt_iou": {"mean": mean_wt_iou, "ci_lower": wt_iou_ci_low, "ci_upper": wt_iou_ci_high},
        },
        "descriptive_statistics": {
            "test_loss": compute_descriptive_stats(subject_losses),
            "macro_dice": compute_descriptive_stats(subject_macro_dices),
            "tc_dice": compute_descriptive_stats(subject_tc_dices),
            "et_dice": compute_descriptive_stats(subject_et_dices),
            "wt_dice": compute_descriptive_stats(subject_wt_dices),
            "macro_iou": compute_descriptive_stats(subject_macro_ious),
            "tc_iou": compute_descriptive_stats(subject_tc_ious),
            "et_iou": compute_descriptive_stats(subject_et_ious),
            "wt_iou": compute_descriptive_stats(subject_wt_ious),
        }
    }

    # 8. Save Artifacts
    results_path = Path("reports/final/fedmed_final_test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info("Saved %s", results_path)

    csv_path = Path("reports/final/fedmed_test_subject_metrics.csv")
    df = pd.DataFrame(per_subject_records)
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s (%d rows)", csv_path, len(df))

    # 9. Perform Generalization Analysis (Step 5)
    with open("reports/real_brats2024/fedavg_dp_d2_1.json", "r") as f:
        val_rep = json.load(f)

    val_metrics = val_rep["performance_summary"]["best_validation_metrics"]

    def analyze_gap(v_val: float, t_val: float, metric_name: str) -> Dict[str, Any]:
        abs_gap = round(t_val - v_val, 4)
        rel_gap_pct = round(((t_val - v_val) / v_val) * 100.0, 2) if v_val != 0.0 else 0.0
        # Descriptive classification
        if abs(rel_gap_pct) <= 10.0:
            classification = "stable"
        elif abs(rel_gap_pct) <= 30.0:
            classification = "moderate degradation" if rel_gap_pct < 0 else "moderate gain"
        else:
            classification = "substantial degradation" if rel_gap_pct < 0 else "substantial gain"

        return {
            "validation_value": v_val,
            "test_value": t_val,
            "absolute_gap": abs_gap,
            "relative_gap_percent": rel_gap_pct,
            "classification": classification,
        }

    generalization_analysis = {
        "status": "GENERALIZATION_ANALYSIS_COMPLETE",
        "experiment_id": "EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cohort_sizes": {
            "validation_cohort": 202,
            "test_cohort": 204
        },
        "metric_comparisons": {
            "val_to_test_loss": analyze_gap(val_metrics.get("val_loss", 0.9604), test_results["primary_metrics"]["test_loss"], "loss"),
            "macro_dice": analyze_gap(val_metrics.get("macro_dice", 0.0800), test_results["primary_metrics"]["macro_dice"], "macro_dice"),
            "tc_dice": analyze_gap(val_metrics.get("tc_dice", 0.2177), test_results["primary_metrics"]["tc_dice"], "tc_dice"),
            "et_dice": analyze_gap(val_metrics.get("et_dice", 0.0147), test_results["primary_metrics"]["et_dice"], "et_dice"),
            "wt_dice": analyze_gap(val_metrics.get("wt_dice", 0.0076), test_results["primary_metrics"]["wt_dice"], "wt_dice"),
            "macro_iou": analyze_gap(val_metrics.get("macro_iou", 0.0497), test_results["primary_metrics"]["macro_iou"], "macro_iou"),
            "tc_iou": analyze_gap(val_metrics.get("tc_iou", 0.1377), test_results["primary_metrics"]["tc_iou"], "tc_iou"),
            "et_iou": analyze_gap(val_metrics.get("et_iou", 0.0076), test_results["primary_metrics"]["et_iou"], "et_iou"),
            "wt_iou": analyze_gap(val_metrics.get("wt_iou", 0.0038), test_results["primary_metrics"]["wt_iou"], "wt_iou"),
        }
    }

    gen_path = Path("reports/final/fedmed_generalization_analysis.json")
    with open(gen_path, "w") as f:
        json.dump(generalization_analysis, f, indent=2)
    logger.info("Saved %s", gen_path)

    # 10. Perform Baseline Comparative Analysis (Step 7)
    comparison_table = {
        "status": "COMPARATIVE_BENCHMARK_MATRIX_COMPLETE",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experiments": [
            {
                "experiment_id": "EXPERIMENT_CENTRALIZED_NON_PRIVATE",
                "label": "Non-Private FedAvg Baseline",
                "scope": "FULL_20_ROUNDS",
                "optimizer": "Adam (lr=1e-4)",
                "dp_mechanism": "NONE",
                "C": None,
                "sigma": 0.0,
                "epsilon": None,
                "delta": None,
                "T": 0,
                "macro_dice": 0.3815,
                "tc_dice": 0.1878,
                "et_dice": 0.1729,
                "wt_dice": 0.7840,
                "macro_iou": 0.2641,
                "val_loss": 0.5872,
            },
            {
                "experiment_id": "EXPERIMENT_D_FEDAVG_DP_REAL_DATA",
                "label": "Experiment D (DP-SGD Baseline)",
                "scope": "FULL_20_ROUNDS",
                "optimizer": "Adam (lr=1e-4)",
                "dp_mechanism": "Poisson DP-SGD",
                "C": 1.0,
                "sigma": 0.87,
                "epsilon": 2.8934,
                "delta": 1e-05,
                "T": 4720,
                "macro_dice": 0.0190,
                "tc_dice": 0.0437,
                "et_dice": 0.0080,
                "wt_dice": 0.0052,
                "macro_iou": 0.0102,
                "val_loss": 0.9888,
            },
            {
                "experiment_id": "EXPERIMENT_D1_CALIBRATED_CLIPPING",
                "label": "Experiment D1 (C=0.06 Calibration)",
                "scope": "SCREENING_STAGE_B_3_ROUNDS",
                "optimizer": "Adam (lr=1e-4)",
                "dp_mechanism": "Poisson DP-SGD",
                "C": 0.06,
                "sigma": 0.87,
                "epsilon": 1.9636,
                "delta": 1e-05,
                "T": 708,
                "macro_dice": 0.0205,
                "tc_dice": 0.0479,
                "et_dice": 0.0088,
                "wt_dice": 0.0048,
                "macro_iou": 0.0110,
                "val_loss": 0.9883,
            },
            {
                "experiment_id": "EXPERIMENT_D2_DP_SGD_MOMENTUM",
                "label": "Experiment D2 (SGD+Mom lr=1e-4)",
                "scope": "SCREENING_STAGE_B_3_ROUNDS",
                "optimizer": "SGD+Momentum (lr=1e-4, mom=0.9)",
                "dp_mechanism": "Poisson DP-SGD",
                "C": 0.06,
                "sigma": 0.87,
                "epsilon": 1.9636,
                "delta": 1e-05,
                "T": 708,
                "macro_dice": 0.0210,
                "tc_dice": 0.0498,
                "et_dice": 0.0090,
                "wt_dice": 0.0044,
                "macro_iou": 0.0113,
                "val_loss": 0.9902,
            },
            {
                "experiment_id": "EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3_VAL",
                "label": "Experiment D2.1 (Validation Cohort)",
                "scope": "FULL_20_ROUNDS",
                "optimizer": "SGD+Momentum (lr=1e-3, mom=0.9)",
                "dp_mechanism": "Poisson DP-SGD",
                "C": 0.06,
                "sigma": 0.87,
                "epsilon": 2.8934,
                "delta": 1e-05,
                "T": 4720,
                "macro_dice": 0.0800,
                "tc_dice": 0.2177,
                "et_dice": 0.0147,
                "wt_dice": 0.0076,
                "macro_iou": 0.0497,
                "val_loss": 0.9604,
            },
            {
                "experiment_id": "EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3_LOCKED_TEST",
                "label": "Experiment D2.1 (Locked Test Cohort)",
                "scope": "LOCKED_TEST_EVALUATION_204_SUBJECTS",
                "optimizer": "SGD+Momentum (lr=1e-3, mom=0.9)",
                "dp_mechanism": "Poisson DP-SGD (Post-Processing)",
                "C": 0.06,
                "sigma": 0.87,
                "epsilon": 2.8934,
                "delta": 1e-05,
                "T": 4720,
                "macro_dice": mean_macro_dice,
                "tc_dice": mean_tc_dice,
                "et_dice": mean_et_dice,
                "wt_dice": mean_wt_dice,
                "macro_iou": mean_macro_iou,
                "val_loss": round(float(np.mean(subject_losses)), 4),
            }
        ]
    }

    comp_path = Path("reports/final/fedmed_final_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison_table, f, indent=2)
    logger.info("Saved %s", comp_path)

    logger.info("============================================================")
    logger.info("FEDMED FINAL LOCKED TEST EVALUATION RESULTS:")
    logger.info("Test Loss:        %.4f", test_results["primary_metrics"]["test_loss"])
    logger.info("Macro Dice:       %.4f [95%% CI: %.4f - %.4f]", mean_macro_dice, macro_ci_low, macro_ci_high)
    logger.info("TC Dice:          %.4f [95%% CI: %.4f - %.4f]", mean_tc_dice, tc_ci_low, tc_ci_high)
    logger.info("ET Dice:          %.4f [95%% CI: %.4f - %.4f]", mean_et_dice, et_ci_low, et_ci_high)
    logger.info("WT Dice:          %.4f [95%% CI: %.4f - %.4f]", mean_wt_dice, wt_ci_low, wt_ci_high)
    logger.info("Macro IoU:        %.4f [95%% CI: %.4f - %.4f]", mean_macro_iou, macro_iou_ci_low, macro_iou_ci_high)
    logger.info("============================================================")


if __name__ == "__main__":
    main()
