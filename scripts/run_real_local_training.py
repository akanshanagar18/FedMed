"""
Script: scripts/run_real_local_training.py

Purpose:
Phase 8.5B.2 Real Local Training Execution and Model Integrity Engine.
Executes genuine local training on the validated patient-level TRAIN cohort
and evaluates held-out VALIDATION cohort with independent Dice/IoU metrics.
Strictly isolates and firewalls the TEST split (TEST_SET_ACCESSED_DURING_TRAINING = False).

Usage:
  python3 scripts/run_real_local_training.py \
      --data-dir data/BraTS2021 \
      --config configs/experiments/real_brats_fedavg.yaml \
      --epochs 1 \
      --batch-size 1 \
      --seed 42 \
      --allow-development
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import sys
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

from data.datasets.transforms import get_brats_transforms
from evaluation.metrics import compute_dice, compute_iou

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("local_training")

REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


class ExplicitPatientDataset(Dataset):
    """
    Loads MONAI MRI dictionary entries strictly for a designated list of patient IDs.
    """

    def __init__(
        self,
        data_dir: Path,
        subject_ids: List[str],
        transforms,
        modalities: List[str] = ("t1", "t1ce", "t2", "flair"),
        mask_modality: str = "seg",
    ):
        self.data_dir = data_dir
        self.subject_ids = sorted(subject_ids)
        self.transforms = transforms
        self.modalities = modalities
        self.mask_modality = mask_modality

        self.samples: List[Dict[str, Any]] = []
        for s_id in self.subject_ids:
            subj_dir = self.data_dir / s_id
            if not subj_dir.exists():
                raise FileNotFoundError(f"Subject folder '{subj_dir}' does not exist!")

            mod_files: List[str] = []
            for m in self.modalities:
                matches = sorted(list(subj_dir.glob(f"*{m}.nii*")))
                if not matches:
                    raise FileNotFoundError(f"Subject '{s_id}' missing modality '{m}'")
                mod_files.append(str(matches[0]))

            seg_matches = sorted(list(subj_dir.glob(f"*{self.mask_modality}.nii*")))
            if not seg_matches:
                raise FileNotFoundError(f"Subject '{s_id}' missing segmentation mask")

            self.samples.append({
                "image": mod_files,
                "label": str(seg_matches[0]),
                "patient_id": s_id,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        if self.transforms:
            return self.transforms(item)
        return item


def load_canonical_model(config: Dict[str, Any], device: torch.device) -> Tuple[torch.nn.Module, int]:
    """
    Instantiates MONAI 3D UNet strictly from YAML configuration parameters.
    Calculates parameter count programmatically.
    """
    m_cfg = config.get("model", {})
    spatial_dims = m_cfg.get("spatial_dims", 3)
    in_channels = m_cfg.get("in_channels", 4)
    out_channels = m_cfg.get("out_channels", 3)
    channels = tuple(m_cfg.get("channels", [16, 32, 64, 128, 256]))
    strides = tuple(m_cfg.get("strides", [2, 2, 2, 2]))
    num_res_units = m_cfg.get("num_res_units", 2)
    dropout = m_cfg.get("dropout", 0.0)

    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        dropout=dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, param_count


def run_local_training_experiment(
    data_dir: str = "data/BraTS2021",
    config_path: str = "configs/experiments/real_brats_fedavg.yaml",
    epochs: int = 1,
    batch_size: int = 1,
    seed: int = 42,
    allow_development: bool = False,
) -> Dict[str, Any]:
    print("=" * 70)
    print("🧠 FEDMED OS — REAL LOCAL TRAINING & MODEL INTEGRITY GATE (PHASE 8.5B.2)")
    print("=" * 70)

    # 1. Deterministic Seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Select Device
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

    # 3. Load & Verify YAML Config
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = PROJECT_ROOT / cfg_file
    if not cfg_file.exists():
        print(f"❌ Configuration file not found at '{cfg_file}'")
        sys.exit(1)

    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)

    # 4. Instantiate Model & Measure Parameters
    model, param_count = load_canonical_model(config, device)
    print(f"MODEL_PARAMETER_COUNT={param_count}")
    print(f"Model Architecture:      MONAI 3D UNet (channels={config['model']['channels']}, strides={config['model']['strides']})")

    # 5. Dataset Mode Safety & Manifest Check
    manifest_file = REPORTS_DIR / "brats_dataset_manifest.json"
    if not manifest_file.exists():
        print("❌ Dataset manifest missing! Please run scripts/validate_brats_dataset.py first.")
        sys.exit(1)

    with open(manifest_file, "r") as f:
        manifest_data = json.load(f)

    dataset_mode = manifest_data.get("dataset_mode", "DEVELOPMENT_SYNTHETIC")
    print(f"Discovered Dataset Mode: {dataset_mode}")

    if dataset_mode == "DEVELOPMENT_SYNTHETIC":
        if not allow_development:
            print("\n❌ DEVELOPMENT DATASET — EXPLICIT FLAG REQUIRED")
            print("  The dataset consists of mini development cases. To authorize pipeline verification, pass '--allow-development'.")
            sys.exit(1)
        execution_mode = "DEVELOPMENT_SYNTHETIC"
        print("⚠️ EXECUTION MODE: DEVELOPMENT_SYNTHETIC (Authorized for software pipeline verification)")
    elif dataset_mode in ("REAL_BRATS_SUBSET", "FULL_REAL_BRATS"):
        execution_mode = dataset_mode
        print(f"✅ EXECUTION MODE: {dataset_mode}")
    else:
        execution_mode = "UNKNOWN"

    # 6. Load Validated Split & Verify Firewall
    split_file = REPORTS_DIR / "dataset_split.json"
    if not split_file.exists():
        print("❌ Dataset split missing! Please run data/splitter.py first.")
        sys.exit(1)

    with open(split_file, "r") as f:
        split_data = json.load(f)

    train_subjects = split_data.get("train_subjects", [])
    val_subjects = split_data.get("validation_subjects", [])
    test_subjects = split_data.get("test_subjects", [])
    split_hash = split_data.get("split_hash", "UNKNOWN")

    print(f"SPLIT_HASH={split_hash}")
    print(f"TRAIN Cohort ({len(train_subjects)}):       {train_subjects}")
    print(f"VALIDATION Cohort ({len(val_subjects)}):  {val_subjects}")
    print(f"TEST Cohort ({len(test_subjects)}):        {test_subjects} [FIREWALLED]")

    # Assert Disjointness
    set_tr = set(train_subjects)
    set_va = set(val_subjects)
    set_te = set(test_subjects)
    assert len(set_tr.intersection(set_va)) == 0, "TRAIN / VALIDATION overlap!"
    assert len(set_tr.intersection(set_te)) == 0, "TRAIN / TEST overlap!"
    assert len(set_va.intersection(set_te)) == 0, "VALIDATION / TEST overlap!"

    # 7. Build TRAIN and VALIDATION DataLoaders ONLY (Never instantiate TEST DataLoader)
    d_path = Path(data_dir)
    if not d_path.is_absolute():
        d_path = PROJECT_ROOT / d_path

    # Determine spatial size: if development mini cohort, adapt ROI to volume shape
    first_subj_manifest = manifest_data["subjects"][0]
    raw_shape = tuple(first_subj_manifest.get("spatial_shape", (128, 128, 128)))
    target_spatial_size = raw_shape if dataset_mode == "DEVELOPMENT_SYNTHETIC" else (128, 128, 128)

    train_transforms = get_brats_transforms(mode="train", image_size=target_spatial_size)
    val_transforms = get_brats_transforms(mode="val", image_size=target_spatial_size)

    train_ds = ExplicitPatientDataset(data_dir=d_path, subject_ids=train_subjects, transforms=train_transforms)
    val_ds = ExplicitPatientDataset(data_dir=d_path, subject_ids=val_subjects, transforms=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    print("\n" + "-" * 70)
    print("🔬 STEP 7: FIRST BATCH REALITY CHECK")
    print("-" * 70)
    first_batch = next(iter(train_loader))
    b_images = first_batch["image"].to(device)
    b_targets = first_batch["label"].to(device)

    print(f"IMAGE_SHAPE:   {list(b_images.shape)}")
    print(f"TARGET_SHAPE:  {list(b_targets.shape)}")
    print(f"IMAGE_DTYPE:   {b_images.dtype}")
    print(f"TARGET_DTYPE:  {b_targets.dtype}")
    print(f"IMAGE_MIN:     {float(b_images.min()):.4f}")
    print(f"IMAGE_MAX:     {float(b_images.max()):.4f}")
    print(f"TARGET_MIN:    {float(b_targets.min()):.4f}")
    print(f"TARGET_MAX:    {float(b_targets.max()):.4f}")
    print(f"FINITE_IMAGE:  {bool(torch.all(torch.isfinite(b_images)))}")
    print(f"FINITE_TARGET: {bool(torch.all(torch.isfinite(b_targets)))}")

    # Verify tensor shapes
    assert b_images.ndim == 5, f"Expected 5D image tensor (B, 4, D, H, W), got {b_images.ndim}D"
    assert b_targets.ndim == 5, f"Expected 5D target tensor (B, 3, D, H, W), got {b_targets.ndim}D"
    assert b_images.shape[1] == 4, f"Expected 4 input channels (T1, T1ce, T2, FLAIR), got {b_images.shape[1]}"
    assert b_targets.shape[1] == 3, f"Expected 3 target sub-region channels (TC, WT, ET), got {b_targets.shape[1]}"

    # Initial Forward Pass Check
    model.eval()
    with torch.no_grad():
        initial_logits = model(b_images)
    print(f"INITIAL_LOGITS_SHAPE: {list(initial_logits.shape)}")
    assert initial_logits.shape == b_targets.shape, f"Logits shape {initial_logits.shape} != target shape {b_targets.shape}"
    print("✅ First batch reality check PASSED!")

    # 8. Setup Optimizer & Loss
    t_cfg = config.get("training", {})
    lr = float(t_cfg.get("learning_rate", 1e-4))
    weight_decay = float(t_cfg.get("weight_decay", 1e-5))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = DiceCELoss(sigmoid=True)

    # 9. Genuine Training Loop
    print("\n" + "-" * 70)
    print("🏋️ STEP 8: GENUINE LOCAL TRAINING & BACKPROPAGATION")
    print("-" * 70)

    epoch_logs = []
    best_val_dice = -1.0
    best_epoch = 0
    best_checkpoint_path = ""

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_grad_norms = []
        batch_count = 0

        # Snapshot weights before update
        weights_before = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

        for batch_idx, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, targets)
            loss_val = float(loss.item())

            loss.backward()

            # Calculate genuine gradient norm
            grad_sq_sum = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_sq_sum += float(p.grad.norm(2).item() ** 2)
            grad_norm = float(np.sqrt(grad_sq_sum))
            epoch_grad_norms.append(grad_norm)

            optimizer.step()

            epoch_loss += loss_val
            batch_count += 1
            logger.info(f"Epoch {epoch}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss_val:.4f} | Grad Norm: {grad_norm:.4f}")

        avg_train_loss = epoch_loss / max(1, batch_count)
        avg_grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        # Snapshot weights after update & measure genuine delta
        weights_after = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}
        max_delta = max((weights_after[k] - weights_before[k]).abs().max().item() for k in weights_before)
        l2_delta = float(np.sqrt(sum((weights_after[k] - weights_before[k]).norm(2).item() ** 2 for k in weights_before)))

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Training Loss:       {avg_train_loss:.6f}")
        print(f"  Avg Gradient Norm:   {avg_grad_norm:.6f} (grad_norm > 0: {avg_grad_norm > 0})")
        print(f"  Max Parameter Delta: {max_delta:.6e} (max_delta > 0: {max_delta > 0})")
        print(f"  L2 Parameter Delta:  {l2_delta:.6e} (l2_delta > 0: {l2_delta > 0})")

        assert avg_grad_norm > 0.0, "Gradient norm is zero! Backpropagation failed."
        assert max_delta > 0.0, "Model parameters did not change! Optimizer step failed."

        # 10. Validation Inference (Strictly on VALIDATION Cohort)
        print("\n" + "-" * 70)
        print(f"📊 STEP 9: VALIDATION INFERENCE (Epoch {epoch})")
        print("-" * 70)
        model.eval()
        val_dices: List[Dict[str, float]] = []
        val_ious: List[Dict[str, float]] = []

        with torch.no_grad():
            for val_batch in val_loader:
                v_imgs = val_batch["image"].to(device)
                v_targets = val_batch["label"].to(device)

                v_logits = model(v_imgs)
                # Compute independent metrics
                d_res = compute_dice(v_logits, v_targets, threshold=0.5, channel_names=["TC", "WT", "ET"])
                i_res = compute_iou(v_logits, v_targets, threshold=0.5, channel_names=["TC", "WT", "ET"])

                val_dices.append(d_res)
                val_ious.append(i_res)

        # Average validation scores
        mean_tc_dice = float(np.mean([d["dice_TC"] for d in val_dices]))
        mean_wt_dice = float(np.mean([d["dice_WT"] for d in val_dices]))
        mean_et_dice = float(np.mean([d["dice_ET"] for d in val_dices]))
        overall_mean_dice = float(np.mean([d["mean_dice"] for d in val_dices]))

        mean_tc_iou = float(np.mean([i["iou_TC"] for i in val_ious]))
        mean_wt_iou = float(np.mean([i["iou_WT"] for i in val_ious]))
        mean_et_iou = float(np.mean([i["iou_ET"] for i in val_ious]))
        overall_mean_iou = float(np.mean([i["mean_iou"] for i in val_ious]))

        print(f"  Validation TC Dice:   {mean_tc_dice:.4f}")
        print(f"  Validation WT Dice:   {mean_wt_dice:.4f}")
        print(f"  Validation ET Dice:   {mean_et_dice:.4f}")
        print(f"  Validation Mean Dice: {overall_mean_dice:.4f}")
        print(f"  Validation TC IoU:    {mean_tc_iou:.4f}")
        print(f"  Validation WT IoU:    {mean_wt_iou:.4f}")
        print(f"  Validation ET IoU:    {mean_et_iou:.4f}")
        print(f"  Validation Mean IoU:  {overall_mean_iou:.4f}")

        val_metrics_payload = {
            "dice_TC": round(mean_tc_dice, 6),
            "dice_WT": round(mean_wt_dice, 6),
            "dice_ET": round(mean_et_dice, 6),
            "mean_dice": round(overall_mean_dice, 6),
            "iou_TC": round(mean_tc_iou, 6),
            "iou_WT": round(mean_wt_iou, 6),
            "iou_ET": round(mean_et_iou, 6),
            "mean_iou": round(overall_mean_iou, 6),
        }

        epoch_record = {
            "epoch": epoch,
            "training_loss": round(avg_train_loss, 6),
            "gradient_norm": round(avg_grad_norm, 6),
            "max_parameter_delta": round(max_delta, 8),
            "l2_parameter_delta": round(l2_delta, 8),
            "validation_metrics": val_metrics_payload,
        }
        epoch_logs.append(epoch_record)

        # 11. Checkpointing Best Model
        if overall_mean_dice >= best_val_dice:
            best_val_dice = overall_mean_dice
            best_epoch = epoch
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            best_checkpoint_path = str(CHECKPOINTS_DIR / "local_baseline_best.pt")

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_metrics": val_metrics_payload,
                "config": config,
                "dataset_mode": dataset_mode,
                "split_hash": split_hash,
                "seed": seed,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, best_checkpoint_path)
            print(f"💾 Saved best model checkpoint -> {best_checkpoint_path}")

    # 12. Generate Experiment Report
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_dir = REPORTS_DIR / "experiments" / "local_training"
    exp_dir.mkdir(parents=True, exist_ok=True)
    report_path = exp_dir / f"run_{timestamp_str}.json"

    exp_report = {
        "experiment_id": f"local_train_{timestamp_str}",
        "execution_mode": execution_mode,
        "dataset_mode": dataset_mode,
        "subject_count": len(manifest_data["subjects"]),
        "train_subjects": train_subjects,
        "validation_subjects": val_subjects,
        "test_subjects": test_subjects,
        "test_firewall": {
            "TEST_SET_ACCESSED_DURING_TRAINING": False,
            "TEST_METRICS_GENERATED": False,
            "TEST_EVALUATION_STATUS": "NOT_EVALUATED",
        },
        "split_hash": split_hash,
        "seed": seed,
        "hardware_device": device_name,
        "software_versions": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "monai": "1.3+",
        },
        "model_configuration": {
            "name": config["model"]["name"],
            "channels": config["model"]["channels"],
            "strides": config["model"]["strides"],
            "num_res_units": config["model"]["num_res_units"],
            "parameter_count": param_count,
        },
        "training_configuration": {
            "loss": "DiceCELoss(sigmoid=True)",
            "optimizer": "Adam",
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "epoch_logs": epoch_logs,
        "best_epoch": best_epoch,
        "best_validation_metrics": epoch_logs[best_epoch - 1]["validation_metrics"] if epoch_logs else {},
        "metric_source": "MODEL_INFERENCE_VS_VALIDATION_GROUND_TRUTH",
        "checkpoint_path": best_checkpoint_path,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experimental_limitation": (
            "The current development cohort is insufficient for meaningful medical or federated performance conclusions. "
            "This execution verifies software pipeline integrity, loss backpropagation, weight updates, and independent validation."
        ),
    }

    with open(report_path, "w") as f:
        json.dump(exp_report, f, indent=2)

    print("\n" + "=" * 70)
    print("🏆 LOCAL TRAINING EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Execution Mode:             {execution_mode}")
    print(f"Model Parameter Count:      {param_count}")
    print(f"TRAIN Subjects Used:        {train_subjects}")
    print(f"VALIDATION Subjects Used:   {val_subjects}")
    print(f"TEST Firewall Enforced:     TEST_SET_ACCESSED_DURING_TRAINING = FALSE")
    print(f"Initial / Final Loss:       {epoch_logs[0]['training_loss']:.6f}")
    print(f"Gradient Norm (> 0):        {epoch_logs[0]['gradient_norm']:.6f}")
    print(f"Parameter Delta (> 0):      {epoch_logs[0]['max_parameter_delta']:.6e}")
    print(f"Validation Mean Dice:       {epoch_logs[0]['validation_metrics']['mean_dice']:.4f}")
    print(f"Validation Mean IoU:        {epoch_logs[0]['validation_metrics']['mean_iou']:.4f}")
    print(f"Best Checkpoint:            {best_checkpoint_path}")
    print(f"Experiment Report:          {report_path}")
    print("=" * 70)

    return exp_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Local Training Execution Engine")
    parser.add_argument("--data-dir", type=str, default="data/BraTS2021", help="Dataset directory")
    parser.add_argument("--config", type=str, default="configs/experiments/real_brats_fedavg.yaml", help="Config YAML")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--allow-development", action="store_true", help="Authorize execution on development mini-dataset")
    args = parser.parse_args()

    run_local_training_experiment(
        data_dir=args.data_dir,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        allow_development=args.allow_development,
    )
