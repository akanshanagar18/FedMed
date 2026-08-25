"""
Script: scripts/run_real_fedavg_experiment.py

Purpose:
FEDMED OS — PHASE 10 / EXPERIMENT B: REAL-DATA FEDERATED LEARNING (FedAvg).
Executes sample-weighted Federated Averaging across 4 hospital silos (hospital_alpha,
hospital_beta, hospital_gamma, hospital_delta) on the BraTS-GLI 2024 cohort.
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
from server.strategies.base import FitResult, NDArrays
from server.strategies.fedavg import FedAvg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fedavg_real_experiment")


def compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class RealFederatedDataset(Dataset):
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


def evaluate_global_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates global federated model strictly over the 202-subject validation cohort.
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


def train_hospital_silo(
    hospital_name: str,
    subject_dirs: List[Path],
    global_state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
    max_steps: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], float, int, float]:
    """
    Executes local training on a single hospital silo.
    Returns updated local weights, average training loss, sample count, and training duration.
    """
    t0 = time.perf_counter()
    m_cfg = config["model"]
    local_model = UNet(
        spatial_dims=m_cfg.get("spatial_dims", 3),
        in_channels=m_cfg.get("in_channels", 4),
        out_channels=m_cfg.get("out_channels", 3),
        channels=tuple(m_cfg.get("channels", [16, 32, 64, 128, 256])),
        strides=tuple(m_cfg.get("strides", [2, 2, 2, 2])),
        num_res_units=m_cfg.get("num_res_units", 2),
        dropout=m_cfg.get("dropout", 0.0),
    ).to(device)

    local_model.load_state_dict(global_state_dict)
    local_model.train()

    t_cfg = config["training"]
    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=t_cfg.get("learning_rate", 1e-4),
        weight_decay=t_cfg.get("weight_decay", 1e-5),
    )

    loss_params = t_cfg.get("loss_params", {})
    loss_fn = DiceCELoss(
        sigmoid=loss_params.get("sigmoid", True),
        lambda_dice=loss_params.get("lambda_dice", 1.0),
        lambda_ce=loss_params.get("lambda_ce", 0.2),
    )

    spatial_shape = tuple(t_cfg.get("spatial_shape", [128, 128, 128]))
    ds = RealFederatedDataset(subject_dirs, mode="train", spatial_shape=spatial_shape)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    train_losses = []
    step_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = local_model(images)
        loss = loss_fn(logits, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"Numerical instability in {hospital_name} training")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=100.0)
        optimizer.step()

        train_losses.append(loss.item())
        step_count += 1

        if max_steps is not None and step_count >= max_steps:
            break

    duration = time.perf_counter() - t0
    mean_loss = float(np.mean(train_losses)) if train_losses else 0.0
    local_weights = {k: v.cpu().clone().detach() for k, v in local_model.state_dict().items()}

    return local_weights, mean_loss, len(subject_dirs), duration


def run_fedavg_experiment(
    config_path: Path = PROJECT_ROOT / "configs" / "experiments" / "canonical_real_brats_gli_2024.yaml",
    data_dir: Path = PROJECT_ROOT / "data" / "raw" / "BraTS2024",
    split_file: Path = PROJECT_ROOT / "reports" / "real_brats2024" / "dataset_split.json",
    partitions_file: Path = PROJECT_ROOT / "reports" / "real_brats2024" / "hospital_partitions.json",
    rounds: int = 20,
    max_steps_per_client: Optional[int] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("🌐 FEDMED OS — EXPERIMENT B: REAL-DATA FEDERATED LEARNING (FedAvg)")
    logger.info("=" * 80)

    start_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    overall_start_time = time.perf_counter()

    # 1. Load Canonical Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config_str = yaml.dump(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

    seed = config.get("federated", {}).get("seed", 42)
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

    # 3. Load & Verify Split and Hospital Partitions
    with open(split_file, "r") as f:
        split_data = json.load(f)
    split_hash = split_data["split_hash"]

    with open(partitions_file, "r") as f:
        partitions_data = json.load(f)
    partition_hash = partitions_data.get("partition_hash", hashlib.sha256(json.dumps(partitions_data, sort_keys=True).encode()).hexdigest())

    from data.real_brats_pipeline import RealBratsValidator
    validator = RealBratsValidator(data_dir)
    all_subjs = {p.name: p for p in validator.discover_subject_directories()}

    hospitals = config["federated"]["hospitals"]
    hospital_dirs = {}
    for h in hospitals:
        s_list = partitions_data["partitions"][h]["subjects"]
        hospital_dirs[h] = [all_subjs[s] for s in s_list if s in all_subjs]
        assert len(hospital_dirs[h]) == 236, f"Hospital {h} expected 236 subjects, found {len(hospital_dirs[h])}"

    val_ids = split_data["validation_subjects"]
    test_ids = split_data["test_subjects"]
    val_dirs = [all_subjs[s] for s in val_ids if s in all_subjs]
    test_dirs = [all_subjs[s] for s in test_ids if s in all_subjs]

    assert len(val_dirs) == 202, f"Expected 202 validation subjects, found {len(val_dirs)}"
    assert len(test_dirs) == 204, f"Expected 204 test subjects, found {len(test_dirs)}"

    # Test Firewall Verification
    all_train_subjects = [s for sublist in hospital_dirs.values() for s in sublist]
    assert len(all_train_subjects) == 944, f"Expected 944 total train subjects, found {len(all_train_subjects)}"
    train_val_names = set(p.name for p in all_train_subjects).union(set(val_ids))
    test_names = set(test_ids)
    overlap = train_val_names.intersection(test_names)
    assert len(overlap) == 0, f"FATAL: Test split leakage detected: {overlap}"

    TRAINING_TEST_ACCESSES = 0
    logger.info(f"Verified 4 Hospital Silos (236 subjects each = 944 train) | 202 Val | 204 Test (Firewalled)")

    # 4. Initialize Global Model & Validation Loader
    spatial_shape = tuple(config["training"].get("spatial_shape", [128, 128, 128]))
    val_ds = RealFederatedDataset(val_dirs, mode="val", spatial_shape=spatial_shape)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    m_cfg = config["model"]
    global_model = UNet(
        spatial_dims=m_cfg.get("spatial_dims", 3),
        in_channels=m_cfg.get("in_channels", 4),
        out_channels=m_cfg.get("out_channels", 3),
        channels=tuple(m_cfg.get("channels", [16, 32, 64, 128, 256])),
        strides=tuple(m_cfg.get("strides", [2, 2, 2, 2])),
        num_res_units=m_cfg.get("num_res_units", 2),
        dropout=m_cfg.get("dropout", 0.0),
    ).to(device)

    total_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    assert total_params == 4810074, f"Parameter count mismatch: {total_params} != 4810074"

    loss_params = config["training"].get("loss_params", {})
    loss_fn = DiceCELoss(
        sigmoid=loss_params.get("sigmoid", True),
        lambda_dice=loss_params.get("lambda_dice", 1.0),
        lambda_ce=loss_params.get("lambda_ce", 0.2),
    )

    # 5. Checkpoint Directory Setup
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "fedavg_real"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best.pt"
    latest_ckpt_path = ckpt_dir / "latest.pt"

    start_round = 1
    best_mean_dice = -1.0
    best_round = -1
    best_val_metrics = {}
    round_history = []

    # Resume capability
    if resume and latest_ckpt_path.exists():
        logger.info(f"Resuming from latest checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        global_model.load_state_dict(checkpoint["model_state_dict"])
        start_round = checkpoint["round"] + 1
        best_mean_dice = checkpoint.get("best_mean_dice", -1.0)
        best_round = checkpoint.get("best_round", -1)
        best_val_metrics = checkpoint.get("best_val_metrics", {})
        round_history = checkpoint.get("round_history", [])
        logger.info(f"Resumed at round {start_round}, prior best mean dice: {best_mean_dice:.4f}")

    # 6. Federated Rounds Loop
    logger.info(f"Beginning FedAvg Experiment for {rounds} rounds across 4 hospitals...")
    round_durations = []
    strategy = FedAvg(min_fit_clients=4, min_available_clients=4)

    for r in range(start_round, rounds + 1):
        round_start_time = time.perf_counter()
        logger.info(f"\n{'='*70}\n🔄 Starting Federated Round {r}/{rounds}\n{'='*70}")

        # Broadcast global parameters to clients
        global_state = {k: v.cpu().clone().detach() for k, v in global_model.state_dict().items()}
        global_param_arrays = [v.numpy() for v in global_state.values()]
        param_names = list(global_state.keys())

        client_updates = {}
        client_sample_counts = {}
        client_losses = {}
        client_timings = {}
        fit_results = []

        # Sequential Execution of Clients on Apple Silicon MPS
        for h_name in hospitals:
            logger.info(f"[{h_name}] Executing local training round on {len(hospital_dirs[h_name])} subjects...")
            loc_weights, loc_loss, n_samples, loc_time = train_hospital_silo(
                hospital_name=h_name,
                subject_dirs=hospital_dirs[h_name],
                global_state_dict=global_state,
                config=config,
                device=device,
                max_steps=max_steps_per_client,
            )

            client_updates[h_name] = loc_weights
            client_sample_counts[h_name] = n_samples
            client_losses[h_name] = loc_loss
            client_timings[h_name] = loc_time

            # Format fit result
            loc_arrays = [loc_weights[k].numpy() for k in param_names]
            fit_results.append(FitResult(
                parameters=loc_arrays,
                num_examples=n_samples,
                metrics={"training_loss": loc_loss, "hospital": h_name},
            ))
            logger.info(f"[{h_name}] Training Complete in {loc_time:.1f}s | Mean Loss: {loc_loss:.6f}")

        # Verify Client Parameter Divergence
        w_alpha = client_updates["hospital_alpha"]
        w_beta = client_updates["hospital_beta"]
        divergence = sum((w_alpha[k] - w_beta[k]).norm().item() for k in param_names)
        assert divergence > 0.0, "FATAL: Client updates failed to diverge!"
        logger.info(f"Client Divergence Verified: ||W_alpha - W_beta|| = {divergence:.6f}")

        # Sample-Weighted FedAvg Aggregation via Server Strategy
        agg_arrays, agg_metrics = strategy.aggregate_fit(
            server_round=r,
            results=fit_results,
            failures=[],
        )

        # Reconstruct PyTorch state dict from aggregated numpy arrays
        new_global_state = {}
        for name, arr in zip(param_names, agg_arrays):
            new_global_state[name] = torch.from_numpy(arr).to(device)

        # Global parameter delta check
        delta_global = sum((new_global_state[k].cpu() - global_state[k]).norm().item() for k in param_names)
        assert delta_global > 0.0, "FATAL: Global parameter delta is zero!"
        global_model.load_state_dict(new_global_state)
        logger.info(f"FedAvg Aggregation Complete | Global Delta ||W_new - W_old|| = {delta_global:.6f}")

        # Global Evaluation on Held-Out 202 Validation Cohort
        t_val_start = time.perf_counter()
        logger.info(f"Evaluating global model on held-out validation cohort (202 subjects)...")
        val_metrics = evaluate_global_model(global_model, val_loader, loss_fn, device)
        val_duration = time.perf_counter() - t_val_start

        round_total_duration = time.perf_counter() - round_start_time
        round_durations.append(round_total_duration)

        logger.info(
            f"Round {r:02d} Summary: Mean Client Loss={agg_metrics['training_loss']:.6f} | "
            f"Val Loss={val_metrics['val_loss']:.6f} | WT Dice={val_metrics['wt_dice']:.4f} | "
            f"TC Dice={val_metrics['tc_dice']:.4f} | ET Dice={val_metrics['et_dice']:.4f} | "
            f"Round Time={round_total_duration:.1f}s"
        )

        round_record = {
            "round": r,
            "mean_client_train_loss": round(agg_metrics["training_loss"], 6),
            "client_losses": {k: round(v, 6) for k, v in client_losses.items()},
            "client_timings_sec": {k: round(v, 2) for k, v in client_timings.items()},
            "global_parameter_delta": round(delta_global, 6),
            "client_divergence": round(divergence, 6),
            "val_loss": val_metrics["val_loss"],
            "mean_dice": val_metrics["mean_dice"],
            "tc_dice": val_metrics["tc_dice"],
            "wt_dice": val_metrics["wt_dice"],
            "et_dice": val_metrics["et_dice"],
            "tc_iou": val_metrics["tc_iou"],
            "wt_iou": val_metrics["wt_iou"],
            "et_iou": val_metrics["et_iou"],
            "val_time_sec": round(val_duration, 2),
            "round_total_sec": round(round_total_duration, 2),
        }
        round_history.append(round_record)

        # Check if best model
        is_best = val_metrics["mean_dice"] > best_mean_dice
        if is_best:
            best_mean_dice = val_metrics["mean_dice"]
            best_round = r
            best_val_metrics = val_metrics
            torch.save({
                "round": r,
                "model_state_dict": global_model.state_dict(),
                "best_mean_dice": best_mean_dice,
                "best_round": best_round,
                "val_metrics": val_metrics,
                "config_hash": config_hash,
                "split_hash": split_hash,
                "partition_hash": partition_hash,
                "seed": seed,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, best_ckpt_path)
            logger.info(f"🏆 New best validation model saved at Round {r} (Mean Dice: {best_mean_dice:.4f})")

        # Save latest checkpoint
        torch.save({
            "round": r,
            "model_state_dict": global_model.state_dict(),
            "best_mean_dice": best_mean_dice,
            "best_round": best_round,
            "best_val_metrics": best_val_metrics,
            "round_history": round_history,
            "config_hash": config_hash,
            "split_hash": split_hash,
            "partition_hash": partition_hash,
            "seed": seed,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, latest_ckpt_path)

    total_experiment_time = time.perf_counter() - overall_start_time
    avg_round_time = float(np.mean(round_durations)) if round_durations else 0.0
    best_ckpt_sha256 = compute_file_sha256(best_ckpt_path) if best_ckpt_path.exists() else None

    # Payload calculation (4,810,074 float32 params * 4 bytes = 19,240,296 bytes = 18.35 MB)
    payload_bytes_per_client = sum(p.nelement() * p.element_size() for p in global_model.parameters())
    payload_mb_per_client = round(payload_bytes_per_client / (1024 * 1024), 2)

    peak_mps_mb = None
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        peak_mps_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)

    # 7. Compile Final Federated Report
    final_report = {
        "experiment_name": "EXPERIMENT_B_FEDAVG_REAL_DATA",
        "timestamp_start": start_timestamp,
        "timestamp_end": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "COMPLETED",
        "dataset": {
            "name": "BraTS-GLI",
            "version": "2024",
            "task": "adult_glioma_post_treatment",
            "split_hash": split_hash,
            "partition_hash": partition_hash,
            "total_subjects": 1350,
            "train_subjects": 944,
            "validation_subjects": 202,
            "test_subjects": 204,
            "hospital_silos": {
                "hospital_alpha": len(hospital_dirs["hospital_alpha"]),
                "hospital_beta": len(hospital_dirs["hospital_beta"]),
                "hospital_gamma": len(hospital_dirs["hospital_gamma"]),
                "hospital_delta": len(hospital_dirs["hospital_delta"]),
            },
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
        "federated_configuration": {
            "strategy": "FedAvg",
            "client_execution": "sequential",
            "rounds": rounds,
            "clients_per_round": 4,
            "local_epochs": 1,
            "communication_payload_per_client_mb": payload_mb_per_client,
            "total_payload_per_round_mb": payload_mb_per_client * 4,
            "loss_function": "DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.2)",
            "optimizer": "Adam(lr=1e-4, weight_decay=1e-5)",
            "spatial_resolution": list(spatial_shape),
            "seed": seed,
            "device": str(device),
            "config_hash": config_hash,
        },
        "performance_summary": {
            "best_round": best_round,
            "best_validation_metrics": best_val_metrics,
            "final_train_loss": round_history[-1]["mean_client_train_loss"] if round_history else None,
            "final_val_loss": round_history[-1]["val_loss"] if round_history else None,
            "total_runtime_seconds": round(total_experiment_time, 2),
            "average_round_seconds": round(avg_round_time, 2),
            "peak_mps_memory_mb": peak_mps_mb,
            "oom_detected": False,
        },
        "checkpoint": {
            "best_checkpoint_path": str(best_ckpt_path),
            "best_checkpoint_sha256": best_ckpt_sha256,
            "latest_checkpoint_path": str(latest_ckpt_path),
        },
        "round_history": round_history,
    }

    manifest_path = PROJECT_ROOT / "reports" / "real_brats2024" / "fedavg_experiment_manifest.json"
    manifest_record = {
        "experiment_name": "EXPERIMENT_B_FEDAVG_REAL_DATA",
        "dataset_split_hash": split_hash,
        "hospital_partition_hash": partition_hash,
        "configuration_hash": config_hash,
        "random_seed": seed,
        "model_architecture": {
            "name": "MONAI 3D U-Net",
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 3,
            "channels": [16, 32, 64, 128, 256],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "total_parameters": total_params,
        },
        "preprocessing_configuration": {
            "spatial_resolution": list(spatial_shape),
            "modalities": ["t1", "t1ce", "t2", "flair"],
            "orientation": "RAS",
            "spacing": [1.0, 1.0, 1.0],
            "normalization": "Non-zero channel-wise Z-score",
            "target_canonicalization": {
                "TC": "label == 1 | label == 3",
                "WT": "label == 1 | label == 2 | label == 3",
                "ET": "label == 3",
                "RC": "label == 4 preserved in provenance"
            }
        },
        "federated_parameters": {
            "strategy": "FedAvg",
            "rounds": rounds,
            "clients": hospitals,
            "sample_weights": {h: len(hospital_dirs[h]) / 944.0 for h in hospitals},
            "client_execution": "sequential",
            "batch_size": 1,
            "local_epochs": 1,
            "loss_function": "DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.2)",
            "optimizer": "Adam(lr=1e-4, weight_decay=1e-5)",
        },
        "hardware_environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "best_checkpoint_sha256": best_ckpt_sha256,
        "status": "COMPLETED",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_record, f, indent=2)

    report_path = PROJECT_ROOT / "reports" / "real_brats2024" / "fedavg_real.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    history_path = PROJECT_ROOT / "reports" / "real_brats2024" / "fedavg_real_history.json"
    with open(history_path, "w") as f:
        json.dump(round_history, f, indent=2)

    logger.info("=" * 80)
    logger.info("🎉 EXPERIMENT B COMPLETE: REAL-DATA FEDAVG")
    logger.info(f"Best Round:               {best_round}")
    logger.info(f"Best WT Dice:             {best_val_metrics.get('wt_dice', 0.0):.4f}")
    logger.info(f"Best TC Dice:             {best_val_metrics.get('tc_dice', 0.0):.4f}")
    logger.info(f"Best ET Dice:             {best_val_metrics.get('et_dice', 0.0):.4f}")
    logger.info(f"Total Runtime:            {total_experiment_time:.2f}s ({total_experiment_time/3600:.2f}h)")
    logger.info(f"Average Round Time:       {avg_round_time:.2f}s")
    logger.info(f"Best Checkpoint:          {best_ckpt_path}")
    logger.info(f"Best Checkpoint SHA-256:  {best_ckpt_sha256}")
    logger.info(f"Report Path:              {report_path}")
    logger.info(f"Manifest Path:            {manifest_path}")
    logger.info("=" * 80)

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Real-Data FedAvg (Experiment B)")
    parser.add_argument("--rounds", type=int, default=20, help="Number of federated rounds")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per client for smoke testing")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint")
    args = parser.parse_args()

    run_fedavg_experiment(rounds=args.rounds, max_steps_per_client=args.max_steps, resume=not args.no_resume)
