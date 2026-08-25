"""
Script: scripts/run_real_centralized_baseline.py

Purpose:
FEDMED OS — PHASE 10.0 / EXPERIMENT A: REAL-DATA CENTRALIZED BASELINE.
Executes the centralized real-data training baseline across all 944 training subjects from the
BraTS-GLI 2024 Adult Glioma Post-Treatment Cohort at full (128, 128, 128) spatial resolution.
Strictly isolates validation (202 subjects) and maintains the test firewall (204 subjects locked, zero access).
"""

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.canonical_adapter import DatasetVersion, ModalityCanonicalizer
from data.datasets.transforms import get_brats_transforms
from evaluation.metrics import compute_dice, compute_iou

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("centralized_real_baseline")


def compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class RealCentralizedDataset(Dataset):
    def __init__(self, subject_dirs: List[Path], mode: str = "train", spatial_shape: Tuple[int, int, int] = (128, 128, 128)):
        self.subject_dirs = subject_dirs
        self.mode = mode
        self.spatial_shape = spatial_shape
        self.transforms = get_brats_transforms(
            mode=mode,
            image_size=spatial_shape,
            dataset_version=DatasetVersion.BRATS_GLI_2024,
        )

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
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


def evaluate_validation_cohort(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates model strictly over the 202-subject validation cohort.
    Computes validation loss, TC Dice, WT Dice, ET Dice, TC IoU, WT IoU, ET IoU.
    """
    model.eval()
    val_losses = []
    tc_dices, wt_dices, et_dices = [], [], []
    tc_ious, wt_ious, et_ious = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            logits = model(images)
            loss = loss_fn(logits, targets)
            val_losses.append(loss.item())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()

            d_res = compute_dice(preds_np, targets_np, channel_names=["TC", "WT", "ET"])
            i_res = compute_iou(preds_np, targets_np, channel_names=["TC", "WT", "ET"])

            tc_dices.append(d_res["dice_TC"])
            wt_dices.append(d_res["dice_WT"])
            et_dices.append(d_res["dice_ET"])
            tc_ious.append(i_res["iou_TC"])
            wt_ious.append(i_res["iou_WT"])
            et_ious.append(i_res["iou_ET"])

    mean_val_loss = float(np.mean(val_losses))
    mean_tc_dice = float(np.mean(tc_dices))
    mean_wt_dice = float(np.mean(wt_dices))
    mean_et_dice = float(np.mean(et_dices))
    mean_tc_iou = float(np.mean(tc_ious))
    mean_wt_iou = float(np.mean(wt_ious))
    mean_et_iou = float(np.mean(et_ious))
    composite_mean_dice = float((mean_tc_dice + mean_wt_dice + mean_et_dice) / 3.0)

    return {
        "val_loss": round(mean_val_loss, 6),
        "mean_dice": round(composite_mean_dice, 4),
        "tc_dice": round(mean_tc_dice, 4),
        "wt_dice": round(mean_wt_dice, 4),
        "et_dice": round(mean_et_dice, 4),
        "tc_iou": round(mean_tc_iou, 4),
        "wt_iou": round(mean_wt_iou, 4),
        "et_iou": round(mean_et_iou, 4),
    }


def run_centralized_experiment(
    config_path: Path = PROJECT_ROOT / "configs" / "experiments" / "canonical_real_brats_gli_2024.yaml",
    data_dir: Path = PROJECT_ROOT / "data" / "raw" / "BraTS2024",
    split_file: Path = PROJECT_ROOT / "reports" / "real_brats2024" / "dataset_split.json",
    epochs: int = 20,
    resume: bool = True,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("🧠 FEDMED OS — EXPERIMENT A: REAL-DATA CENTRALIZED BASELINE")
    logger.info("=" * 80)

    start_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    overall_start_time = time.perf_counter()

    # 1. Load Canonical Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config_str = yaml.dump(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

    seed = config.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Select Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Execution Device: {device}")

    # 3. Load & Verify Cohorts
    with open(split_file, "r") as f:
        split_data = json.load(f)
    split_hash = split_data["split_hash"]

    from data.real_brats_pipeline import RealBratsValidator
    validator = RealBratsValidator(data_dir)
    all_subjs = {p.name: p for p in validator.discover_subject_directories()}

    train_ids = split_data["train_subjects"]
    val_ids = split_data["validation_subjects"]
    test_ids = split_data["test_subjects"]

    train_dirs = [all_subjs[s] for s in train_ids if s in all_subjs]
    val_dirs = [all_subjs[s] for s in val_ids if s in all_subjs]
    test_dirs = [all_subjs[s] for s in test_ids if s in all_subjs]

    assert len(train_dirs) == 944, f"Expected 944 train subjects, found {len(train_dirs)}"
    assert len(val_dirs) == 202, f"Expected 202 validation subjects, found {len(val_dirs)}"
    assert len(test_dirs) == 204, f"Expected 204 test subjects, found {len(test_dirs)}"

    # Strictly verify zero test leakage
    train_val_set = set(train_ids).union(set(val_ids))
    test_set = set(test_ids)
    overlap = train_val_set.intersection(test_set)
    assert len(overlap) == 0, f"FATAL: Test split overlap detected: {overlap}"

    TRAINING_TEST_ACCESSES = 0
    logger.info(f"Cohort Discovery: 944 Train | 202 Val | 204 Test (Firewalled, 0 Accesses)")

    # 4. Instantiate Datasets and Loaders
    spatial_shape = tuple(config["training"].get("spatial_shape", [128, 128, 128]))
    train_ds = RealCentralizedDataset(train_dirs, mode="train", spatial_shape=spatial_shape)
    val_ds = RealCentralizedDataset(val_dirs, mode="val", spatial_shape=spatial_shape)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 5. Instantiate Canonical Model, Loss & Optimizer
    m_cfg = config["model"]
    model = UNet(
        spatial_dims=m_cfg.get("spatial_dims", 3),
        in_channels=m_cfg.get("in_channels", 4),
        out_channels=m_cfg.get("out_channels", 3),
        channels=tuple(m_cfg.get("channels", [16, 32, 64, 128, 256])),
        strides=tuple(m_cfg.get("strides", [2, 2, 2, 2])),
        num_res_units=m_cfg.get("num_res_units", 2),
        dropout=m_cfg.get("dropout", 0.0),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 4810074, f"Parameter count mismatch: {total_params} != 4810074"

    loss_params = config["training"].get("loss_params", {})
    loss_fn = DiceCELoss(
        sigmoid=loss_params.get("sigmoid", True),
        lambda_dice=loss_params.get("lambda_dice", 1.0),
        lambda_ce=loss_params.get("lambda_ce", 0.2),
    )

    t_cfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t_cfg.get("learning_rate", 1e-4),
        weight_decay=t_cfg.get("weight_decay", 1e-5),
    )

    # 6. Checkpoint Directory Setup
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "centralized_real"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best.pt"
    latest_ckpt_path = ckpt_dir / "latest.pt"

    start_epoch = 1
    best_mean_dice = -1.0
    best_epoch = -1
    best_val_metrics = {}
    epoch_history = []

    # Resume capability
    if resume and latest_ckpt_path.exists():
        logger.info(f"Resuming from latest checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_mean_dice = checkpoint.get("best_mean_dice", -1.0)
        best_epoch = checkpoint.get("best_epoch", -1)
        best_val_metrics = checkpoint.get("best_val_metrics", {})
        epoch_history = checkpoint.get("epoch_history", [])
        logger.info(f"Resumed at epoch {start_epoch}, prior best mean dice: {best_mean_dice:.4f}")

    # 7. Training Loop
    logger.info(f"Beginning Centralized Baseline Training for {epochs} epochs at resolution {spatial_shape}...")
    oom_detected = False
    numerical_instability = False
    step_times = []
    epoch_durations = []

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        train_losses = []

        logger.info(f"--- Starting Epoch {epoch}/{epochs} (944 training subjects) ---")
        for step_idx, batch in enumerate(train_loader, start=1):
            t_step_start = time.perf_counter()
            try:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    numerical_instability = True
                    logger.error(f"FATAL: Numerical instability (NaN/Inf) at Epoch {epoch}, Step {step_idx}")
                    break

                loss.backward()

                # Gradient norm check
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                if torch.isnan(total_grad_norm) or torch.isinf(total_grad_norm):
                    numerical_instability = True
                    logger.error(f"FATAL: Exploding gradients at Epoch {epoch}, Step {step_idx}")
                    break

                optimizer.step()
                t_step = time.perf_counter() - t_step_start
                step_times.append(t_step)
                train_losses.append(loss.item())

                if step_idx % 100 == 0 or step_idx == 1 or step_idx == len(train_loader):
                    mps_mem = round(torch.mps.current_allocated_memory() / (1024 * 1024), 1) if device.type == "mps" else 0
                    logger.info(
                        f"Epoch {epoch:02d} [{step_idx:03d}/{len(train_loader):03d}] | "
                        f"Loss: {loss.item():.6f} | Step Time: {t_step:.3f}s | MPS Mem: {mps_mem} MB"
                    )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_detected = True
                    logger.error(f"FATAL: OOM encountered at Epoch {epoch}, Step {step_idx}: {e}")
                    break
                raise e

        if oom_detected or numerical_instability:
            # Emergency checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "failure_reason": "OOM" if oom_detected else "Numerical Instability",
            }, ckpt_dir / "emergency_failure_checkpoint.pt")
            break

        mean_train_loss = float(np.mean(train_losses))
        train_duration = time.perf_counter() - epoch_start_time

        # Validation Phase
        t_val_start = time.perf_counter()
        logger.info(f"Evaluating validation cohort (202 subjects)...")
        val_metrics = evaluate_validation_cohort(model, val_loader, loss_fn, device)
        val_duration = time.perf_counter() - t_val_start

        epoch_total_duration = time.perf_counter() - epoch_start_time
        epoch_durations.append(epoch_total_duration)

        logger.info(
            f"Epoch {epoch:02d} Summary: Train Loss={mean_train_loss:.6f} | "
            f"Val Loss={val_metrics['val_loss']:.6f} | WT Dice={val_metrics['wt_dice']:.4f} | "
            f"TC Dice={val_metrics['tc_dice']:.4f} | ET Dice={val_metrics['et_dice']:.4f} | "
            f"Epoch Time={epoch_total_duration:.1f}s"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 6),
            "val_loss": val_metrics["val_loss"],
            "mean_dice": val_metrics["mean_dice"],
            "tc_dice": val_metrics["tc_dice"],
            "wt_dice": val_metrics["wt_dice"],
            "et_dice": val_metrics["et_dice"],
            "tc_iou": val_metrics["tc_iou"],
            "wt_iou": val_metrics["wt_iou"],
            "et_iou": val_metrics["et_iou"],
            "train_time_sec": round(train_duration, 2),
            "val_time_sec": round(val_duration, 2),
            "epoch_total_sec": round(epoch_total_duration, 2),
        }
        epoch_history.append(epoch_record)

        # Check if best model
        is_best = val_metrics["mean_dice"] > best_mean_dice
        if is_best:
            best_mean_dice = val_metrics["mean_dice"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mean_dice": best_mean_dice,
                "best_epoch": best_epoch,
                "val_metrics": val_metrics,
                "config_hash": config_hash,
                "split_hash": split_hash,
                "seed": seed,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, best_ckpt_path)
            logger.info(f"🏆 New best validation model saved at Epoch {epoch} (Mean Dice: {best_mean_dice:.4f})")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_mean_dice": best_mean_dice,
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "epoch_history": epoch_history,
            "config_hash": config_hash,
            "split_hash": split_hash,
            "seed": seed,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, latest_ckpt_path)

    total_experiment_time = time.perf_counter() - overall_start_time
    avg_step_time = float(np.mean(step_times)) if step_times else 0.0
    avg_epoch_time = float(np.mean(epoch_durations)) if epoch_durations else 0.0

    peak_mps_mb = None
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        peak_mps_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)

    best_ckpt_sha256 = compute_file_sha256(best_ckpt_path) if best_ckpt_path.exists() else None

    # 8. Compile Comprehensive Experiment Report
    final_report = {
        "experiment_name": "EXPERIMENT_A_CENTRALIZED_REAL_DATA_BASELINE",
        "timestamp_start": start_timestamp,
        "timestamp_end": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "COMPLETED" if not (oom_detected or numerical_instability) else "FAILED",
        "dataset": {
            "name": "BraTS-GLI",
            "version": "2024",
            "task": "adult_glioma_post_treatment",
            "split_hash": split_hash,
            "total_subjects": 1350,
            "train_subjects": len(train_dirs),
            "validation_subjects": len(val_dirs),
            "test_subjects": len(test_dirs),
            "training_test_accesses": TRAINING_TEST_ACCESSES,
            "test_firewall_verified": True,
        },
        "model": {
            "architecture": "MONAI 3D U-Net",
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 3,
            "channels": [16, 32, 64, 128, 256],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "total_parameters": total_params,
        },
        "preprocessing": {
            "spatial_resolution": list(spatial_shape),
            "modalities": ["t1n", "t1c", "t2w", "t2f"],
            "orientation": "RAS",
            "spacing": [1.0, 1.0, 1.0],
            "normalization": "Non-zero channel-wise Z-score",
            "image_interpolation": "bilinear",
            "label_interpolation": "nearest",
        },
        "label_semantics": {
            "0": "Background",
            "1": "NETC",
            "2": "SNFH",
            "3": "ET",
            "4": "RC",
            "composite_targets": {
                "TC": "label == 1 | label == 3",
                "WT": "label == 1 | label == 2 | label == 3",
                "ET": "label == 3",
            },
        },
        "training_configuration": {
            "loss_function": "DiceCELoss",
            "lambda_dice": 1.0,
            "lambda_ce": 0.2,
            "sigmoid": True,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "batch_size": 1,
            "epochs": epochs,
            "seed": seed,
            "device": str(device),
            "config_hash": config_hash,
        },
        "performance_summary": {
            "best_epoch": best_epoch,
            "best_validation_metrics": best_val_metrics,
            "final_train_loss": epoch_history[-1]["train_loss"] if epoch_history else None,
            "final_val_loss": epoch_history[-1]["val_loss"] if epoch_history else None,
            "total_runtime_seconds": round(total_experiment_time, 2),
            "average_epoch_seconds": round(avg_epoch_time, 2),
            "average_step_seconds": round(avg_step_time, 4),
            "samples_per_second": round(1.0 / avg_step_time, 4) if avg_step_time > 0 else 0,
            "peak_mps_memory_mb": peak_mps_mb,
            "oom_detected": oom_detected,
        },
        "checkpoint": {
            "best_checkpoint_path": str(best_ckpt_path),
            "best_checkpoint_sha256": best_ckpt_sha256,
            "latest_checkpoint_path": str(latest_ckpt_path),
        },
        "epoch_history": epoch_history,
    }

    report_path = PROJECT_ROOT / "reports" / "real_brats2024" / "centralized_real.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    history_path = PROJECT_ROOT / "reports" / "real_brats2024" / "centralized_real_history.json"
    with open(history_path, "w") as f:
        json.dump(epoch_history, f, indent=2)

    logger.info("=" * 80)
    logger.info("🎉 EXPERIMENT A COMPLETE: REAL-DATA CENTRALIZED BASELINE")
    logger.info(f"Best Epoch:               {best_epoch}")
    logger.info(f"Best WT Dice:             {best_val_metrics.get('wt_dice', 0.0):.4f}")
    logger.info(f"Best TC Dice:             {best_val_metrics.get('tc_dice', 0.0):.4f}")
    logger.info(f"Best ET Dice:             {best_val_metrics.get('et_dice', 0.0):.4f}")
    logger.info(f"Total Runtime:            {total_experiment_time:.2f}s ({total_experiment_time/3600:.2f}h)")
    logger.info(f"Average Epoch Time:       {avg_epoch_time:.2f}s")
    logger.info(f"Best Checkpoint:          {best_ckpt_path}")
    logger.info(f"Best Checkpoint SHA-256:  {best_ckpt_sha256}")
    logger.info(f"Report Path:              {report_path}")
    logger.info("=" * 80)

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Real-Data Centralized Baseline (Experiment A)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint")
    args = parser.parse_args()

    run_centralized_experiment(epochs=args.epochs, resume=not args.no_resume)
