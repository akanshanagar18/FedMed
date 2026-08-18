"""
Script: scripts/run_centralized_baseline.py

Purpose:
Phase 8.5B.4 Real Centralized Baseline Training Engine.
Executes genuine multi-subject centralized training across all training cohort subjects
(BraTS2021_00003 and BraTS2021_00004) under identical architecture, loss, optimizer, and preprocessing.
Strictly isolates validation (BraTS2021_00002) and firewalls test (BraTS2021_00001).

Usage:
  python3 scripts/run_centralized_baseline.py \
      --data-dir data/BraTS2021 \
      --config configs/experiments/real_brats_fedavg.yaml \
      --epochs 3 \
      --seed 42 \
      --allow-development
"""

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset

# Setup Path & Imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets.transforms import get_brats_transforms
from evaluation.metrics import compute_dice, compute_iou

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("centralized_baseline")

REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "centralized"


def compute_param_hash(model_or_weights: Any) -> str:
    """Computes deterministic SHA-256 hash over flattened parameter arrays in canonical layer order."""
    hasher = hashlib.sha256()
    if isinstance(model_or_weights, torch.nn.Module):
        arrays = [val.detach().cpu().numpy() for val in model_or_weights.state_dict().values()]
    elif isinstance(model_or_weights, dict):
        arrays = [val.detach().cpu().numpy() for val in model_or_weights.values()]
    elif isinstance(model_or_weights, list):
        arrays = model_or_weights
    else:
        raise TypeError(f"Unsupported parameter structure for hashing: {type(model_or_weights)}")

    for arr in arrays:
        hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()


class MultiPatientDataset(Dataset):
    """Loads MRI dictionary strictly for a designated list of subject IDs."""

    def __init__(
        self,
        data_dir: Path,
        subject_ids: List[str],
        transforms,
        modalities: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
        mask_modality: str = "seg",
    ):
        self.data_dir = data_dir
        self.subject_ids = sorted(subject_ids)
        self.transforms = transforms
        self.modalities = modalities

        self.samples = []
        for s_id in self.subject_ids:
            subj_dir = self.data_dir / s_id
            if not subj_dir.exists():
                raise FileNotFoundError(f"Subject directory '{subj_dir}' not found!")

            mod_files = []
            for m in self.modalities:
                matches = sorted(list(subj_dir.glob(f"*{m}.nii*")))
                if not matches:
                    raise FileNotFoundError(f"Missing modality '{m}' for subject '{s_id}'")
                mod_files.append(str(matches[0]))

            seg_matches = sorted(list(subj_dir.glob(f"*{mask_modality}.nii*")))
            if not seg_matches:
                raise FileNotFoundError(f"Missing segmentation mask for subject '{s_id}'")

            self.samples.append({
                "image": mod_files,
                "label": str(seg_matches[0]),
                "patient_id": s_id,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.transforms:
            return self.transforms(self.samples[idx])
        return self.samples[idx]


def run_centralized_training(
    data_dir: str = "data/BraTS2021",
    config_path: str = "configs/experiments/real_brats_fedavg.yaml",
    epochs: int = 3,
    seed: int = 42,
    allow_development: bool = False,
    initial_weights: Optional[List[np.ndarray]] = None,
    evaluate_test_at_end: bool = False,
) -> Dict[str, Any]:
    print("=" * 70)
    print("🏥 FEDMED OS — REAL CENTRALIZED BASELINE TRAINING ENGINE")
    print("=" * 70)

    # 1. Deterministic Seeds & Device
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon MPS (GPU)"
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"Hardware Compute Device: {device_name}")

    # 2. Load Configuration
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = PROJECT_ROOT / cfg_file
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)

    # 3. Verify Manifest & Dataset Mode
    manifest_file = REPORTS_DIR / "brats_dataset_manifest.json"
    with open(manifest_file, "r") as f:
        manifest_data = json.load(f)
    dataset_mode = manifest_data.get("dataset_mode", "DEVELOPMENT_SYNTHETIC")
    print(f"Discovered Dataset Mode: {dataset_mode}")

    if dataset_mode == "DEVELOPMENT_SYNTHETIC":
        if not allow_development:
            print("\n❌ DEVELOPMENT DATASET — EXPLICIT FLAG REQUIRED")
            sys.exit(1)
        execution_mode = "DEVELOPMENT_SYNTHETIC"
        print("⚠️ EXECUTION MODE: DEVELOPMENT_SYNTHETIC (Authorized for centralized baseline verification)")
    else:
        execution_mode = dataset_mode

    # 4. Load Validated Split & Assert Hash
    split_file = REPORTS_DIR / "dataset_split.json"
    with open(split_file, "r") as f:
        split_data = json.load(f)

    split_hash = split_data.get("split_hash", "")
    train_subjects = split_data.get("train_subjects", [])
    val_subjects = split_data.get("validation_subjects", [])
    test_subjects = split_data.get("test_subjects", [])

    print(f"SPLIT_HASH={split_hash}")
    print(f"TRAIN Cohort ({len(train_subjects)}):      {train_subjects}")
    print(f"VALIDATION Cohort ({len(val_subjects)}): {val_subjects}")
    print(f"TEST Cohort ({len(test_subjects)}):       {test_subjects} [FIREWALLED DURING TRAINING]")

    d_path = Path(data_dir)
    if not d_path.is_absolute():
        d_path = PROJECT_ROOT / d_path

    first_subj_meta = manifest_data["subjects"][0]
    spatial_shape = tuple(first_subj_meta.get("spatial_shape", (32, 32, 32)))

    # 5. Data Loaders
    train_transforms = get_brats_transforms(mode="train", image_size=spatial_shape)
    train_dataset = MultiPatientDataset(data_dir=d_path, subject_ids=train_subjects, transforms=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    val_transforms = get_brats_transforms(mode="val", image_size=spatial_shape)
    val_dataset = MultiPatientDataset(data_dir=d_path, subject_ids=val_subjects, transforms=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 6. Instantiate Canonical Model
    m_cfg = config.get("model", {})
    model = UNet(
        spatial_dims=m_cfg.get("spatial_dims", 3),
        in_channels=m_cfg.get("in_channels", 4),
        out_channels=m_cfg.get("out_channels", 3),
        channels=tuple(m_cfg.get("channels", [16, 32, 64, 128, 256])),
        strides=tuple(m_cfg.get("strides", [2, 2, 2, 2])),
        num_res_units=m_cfg.get("num_res_units", 2),
    ).to(device)

    # Load initial weights if provided (for identical initialization)
    if initial_weights is not None:
        keys = list(model.state_dict().keys())
        sd = {k: torch.tensor(arr).to(device) for k, arr in zip(keys, initial_weights)}
        model.load_state_dict(sd)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MODEL_PARAMETER_COUNT={param_count}")
    assert param_count == 4810074, f"Parameter count mismatch: {param_count} != 4810074"

    initial_hash = compute_param_hash(model)
    print(f"Initial Parameter Hash: {initial_hash}")

    t_cfg = config.get("training", {})
    lr = float(t_cfg.get("learning_rate", 1e-4))
    weight_decay = float(t_cfg.get("weight_decay", 1e-5))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = DiceCELoss(sigmoid=True)

    weights_at_start = [val.clone().detach() for val in model.state_dict().values()]

    epoch_records = []
    best_val_dice = -1.0
    best_checkpoint_path = ""
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_grad_norms = []

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()

            grad_sq = sum(float(p.grad.norm(2).item() ** 2) for p in model.parameters() if p.grad is not None)
            grad_norm = float(np.sqrt(grad_sq))
            epoch_grad_norms.append(grad_norm)

            optimizer.step()
            epoch_losses.append(float(loss.item()))

        avg_loss = float(np.mean(epoch_losses))
        avg_grad_norm = float(np.mean(epoch_grad_norms))

        # Model validation
        model.eval()
        with torch.no_grad():
            v_batch = next(iter(val_loader))
            v_images = v_batch["image"].to(device)
            v_targets = v_batch["label"].to(device)
            v_logits = model(v_images)

            d_res = compute_dice(v_logits, v_targets, threshold=0.5, channel_names=["TC", "WT", "ET"])
            i_res = compute_iou(v_logits, v_targets, threshold=0.5, channel_names=["TC", "WT", "ET"])

        val_metrics_combined = {
            "dice_TC": d_res["dice_TC"],
            "dice_WT": d_res["dice_WT"],
            "dice_ET": d_res["dice_ET"],
            "mean_dice": d_res["mean_dice"],
            "iou_TC": i_res["iou_TC"],
            "iou_WT": i_res["iou_WT"],
            "iou_ET": i_res["iou_ET"],
            "mean_iou": i_res["mean_iou"],
        }

        print(f"  Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Grad Norm: {avg_grad_norm:.4f} | Val Mean Dice: {d_res['mean_dice']:.4f} | IoU: {i_res['mean_iou']:.4f}")

        # Checkpointing
        if d_res["mean_dice"] >= best_val_dice:
            best_val_dice = d_res["mean_dice"]
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            best_checkpoint_path = str(CHECKPOINTS_DIR / "centralized_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "config": config,
                "split_hash": split_hash,
                "dataset_mode": dataset_mode,
                "validation_metrics": val_metrics_combined,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, best_checkpoint_path)

        epoch_records.append({
            "epoch": epoch,
            "training_loss": avg_loss,
            "gradient_norm": avg_grad_norm,
            "validation_metrics": val_metrics_combined,
        })

    weights_at_end = [val.clone().detach() for val in model.state_dict().values()]
    final_hash = compute_param_hash(model)

    max_delta = max((w_a - w_b).abs().max().item() for w_a, w_b in zip(weights_at_end, weights_at_start))
    mean_delta = float(np.mean([(w_a - w_b).abs().mean().item() for w_a, w_b in zip(weights_at_end, weights_at_start)]))
    l2_delta = float(np.sqrt(sum((w_a - w_b).norm(2).item() ** 2 for w_a, w_b in zip(weights_at_end, weights_at_start))))

    # Optional Final TEST Set Evaluation
    test_metrics = None
    if evaluate_test_at_end:
        print(f"\n  🔬 Evaluating Firewalled TEST Cohort ({test_subjects})...")
        test_transforms = get_brats_transforms(mode="val", image_size=spatial_shape)
        test_dataset = MultiPatientDataset(data_dir=d_path, subject_ids=test_subjects, transforms=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model.eval()
        with torch.no_grad():
            t_batch = next(iter(test_loader))
            t_images = t_batch["image"].to(device)
            t_targets = t_batch["label"].to(device)
            t_logits = model(t_images)

            t_d_res = compute_dice(t_logits, t_targets, threshold=0.5, channel_names=["TC", "WT", "ET"])
            t_i_res = compute_iou(t_logits, t_targets, threshold=0.5, channel_names=["TC", "WT", "ET"])

            test_metrics = {
                "dice_TC": t_d_res["dice_TC"],
                "dice_WT": t_d_res["dice_WT"],
                "dice_ET": t_d_res["dice_ET"],
                "mean_dice": t_d_res["mean_dice"],
                "iou_TC": t_i_res["iou_TC"],
                "iou_WT": t_i_res["iou_WT"],
                "iou_ET": t_i_res["iou_ET"],
                "mean_iou": t_i_res["mean_iou"],
            }
        print(f"    TEST Mean Dice: {test_metrics['mean_dice']:.4f} | Mean IoU: {test_metrics['mean_iou']:.4f}")

    # Generate Report
    exp_dir = REPORTS_DIR / "experiments" / "centralized"
    exp_dir.mkdir(parents=True, exist_ok=True)
    report_path = exp_dir / f"run_{timestamp_str}.json"

    report = {
        "experiment_id": f"centralized_{timestamp_str}",
        "algorithm": "Centralized",
        "execution_mode": execution_mode,
        "dataset_mode": dataset_mode,
        "split_hash": split_hash,
        "seed": seed,
        "epochs": epochs,
        "training_budget": {
            "epochs": epochs,
            "total_sample_passes": epochs * len(train_subjects),
        },
        "model_configuration": {
            "name": config["model"]["name"],
            "parameter_count": param_count,
        },
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": final_hash,
        "max_parameter_delta": max_delta,
        "mean_parameter_delta": mean_delta,
        "l2_parameter_delta": l2_delta,
        "epoch_records": epoch_records,
        "final_validation_metrics": epoch_records[-1]["validation_metrics"],
        "final_test_metrics": test_metrics,
        "test_firewall": {
            "TEST_ACCESSED_DURING_TRAINING": False,
            "FINAL_TEST_EVALUATED": evaluate_test_at_end,
        },
        "checkpoint_path": best_checkpoint_path,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized Baseline Runner")
    parser.add_argument("--data-dir", type=str, default="data/BraTS2021", help="Dataset path")
    parser.add_argument("--config", type=str, default="configs/experiments/real_brats_fedavg.yaml", help="Config YAML")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--allow-development", action="store_true", help="Authorize development cohort execution")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate final TEST set after training")
    args = parser.parse_args()

    run_centralized_training(
        data_dir=args.data_dir,
        config_path=args.config,
        epochs=args.epochs,
        seed=args.seed,
        allow_development=args.allow_development,
        evaluate_test_at_end=args.evaluate_test,
    )
