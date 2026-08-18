"""
Script: scripts/run_real_fedavg_dp_d1_experiment.py

Purpose:
FEDMED OS — PHASE 10.9 / EXPERIMENT D1: REAL-DATA FEDAVG + CALIBRATED CLIPPING DP-SGD (C = 0.06).
Executes Differential Privacy-preserving Federated Averaging across 4 hospital silos
(hospital_alpha, hospital_beta, hospital_gamma, hospital_delta) on the BraTS-GLI 2024 cohort.
Applies per-sample gradient norm clipping with calibrated threshold (C = 0.06) and Gaussian noise (sigma = 0.87)
with formal Renyi Differential Privacy (RDP) accounting.
Strictly isolates validation (202 subjects) and maintains the test firewall (204 subjects locked, zero access).
All artifacts are strictly isolated under the 'fedavg_dp_d1' namespace.
"""

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
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
from privacy.dp_engine import (
    PoissonBatchSampler,
    compute_rdp_epsilon,
    compute_rdp_budget_detailed,
)
from server.strategies.base import FitResult, NDArrays
from server.strategies.fedavg import FedAvg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fedavg_dp_d1_experiment")


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

    def __len__(self) -> int:
        return len(self.subject_dirs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
    model: nn.Module,
    val_data: Union[DataLoader, List[Path]],
    loss_fn: nn.Module,
    device: torch.device,
    spatial_shape: Tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Evaluates the global model on the 202 validation subjects.
    Computes validation loss and multi-class Dice (WT, TC, ET) and IoU scores.
    Strictly touches only validation subjects.
    """
    model.eval()
    if isinstance(val_data, DataLoader):
        val_loader = val_data
    else:
        val_dataset = RealFederatedDataset(val_data, mode="val", spatial_shape=spatial_shape)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    val_losses = []
    wt_dices = []
    tc_dices = []
    et_dices = []
    wt_ious = []
    tc_ious = []
    et_ious = []

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

            d_res = compute_dice(preds_np, targets_np, channel_names=["WT", "TC", "ET"])
            i_res = compute_iou(preds_np, targets_np, channel_names=["WT", "TC", "ET"])

            wt_dices.append(d_res.get("dice_WT", 0.0))
            tc_dices.append(d_res.get("dice_TC", 0.0))
            et_dices.append(d_res.get("dice_ET", 0.0))
            wt_ious.append(i_res.get("iou_WT", 0.0))
            tc_ious.append(i_res.get("iou_TC", 0.0))
            et_ious.append(i_res.get("iou_ET", 0.0))

    mean_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
    mean_wt_dice = float(np.mean(wt_dices)) if wt_dices else 0.0
    mean_tc_dice = float(np.mean(tc_dices)) if tc_dices else 0.0
    mean_et_dice = float(np.mean(et_dices)) if et_dices else 0.0
    mean_macro_dice = float((mean_wt_dice + mean_tc_dice + mean_et_dice) / 3.0)

    mean_wt_iou = float(np.mean(wt_ious)) if wt_ious else 0.0
    mean_tc_iou = float(np.mean(tc_ious)) if tc_ious else 0.0
    mean_et_iou = float(np.mean(et_ious)) if et_ious else 0.0
    mean_macro_iou = float((mean_wt_iou + mean_tc_iou + mean_et_iou) / 3.0)

    return {
        "val_loss": round(mean_val_loss, 4),
        "mean_dice": round(mean_macro_dice, 4),
        "macro_dice": round(mean_macro_dice, 4),
        "wt_dice": round(mean_wt_dice, 4),
        "tc_dice": round(mean_tc_dice, 4),
        "et_dice": round(mean_et_dice, 4),
        "wt_iou": round(mean_wt_iou, 4),
        "tc_iou": round(mean_tc_iou, 4),
        "et_iou": round(mean_et_iou, 4),
        "macro_iou": round(mean_macro_iou, 4),
    }


def train_hospital_silo_dp(
    hospital_name: str,
    subject_dirs: List[Path],
    global_state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
    max_steps: Optional[int] = None,
    current_round: int = 1,
    seed: int = 42,
) -> Tuple[Dict[str, torch.Tensor], float, int, float, Dict[str, Any]]:
    """
    Executes local DP-SGD training on a single hospital silo with calibrated clipping C = 0.06.
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

    p_cfg = config.get("privacy", {})
    max_grad_norm = p_cfg.get("max_grad_norm", 0.06)
    noise_multiplier = p_cfg.get("noise_multiplier", 0.87)
    target_delta = p_cfg.get("target_delta", 1e-5)
    sample_rate = p_cfg.get("sample_rate", 1.0 / len(subject_dirs))

    spatial_shape = tuple(t_cfg.get("spatial_shape", [128, 128, 128]))
    ds = RealFederatedDataset(subject_dirs, mode="train", spatial_shape=spatial_shape)

    num_steps = max_steps if max_steps is not None else len(subject_dirs)
    seed_offset = current_round * 1000 + abs(hash(hospital_name)) % 1000
    sampler = PoissonBatchSampler(
        dataset_size=len(subject_dirs),
        sample_rate=sample_rate,
        num_steps=num_steps,
        seed=seed + seed_offset,
    )

    train_losses = []
    pre_clip_norms = []
    batch_sizes = []
    clipped_count = 0
    total_samples_evaluated = 0
    step_count = 0

    for step_idx, indices in enumerate(sampler):
        optimizer.zero_grad()
        actual_batch_size = len(indices)
        batch_sizes.append(actual_batch_size)

        accumulated_grads = [torch.zeros_like(p.data) for p in local_model.parameters() if p.requires_grad]
        step_loss_sum = 0.0

        if actual_batch_size > 0:
            for idx in indices:
                sample = ds[idx]
                image = sample["image"].unsqueeze(0).to(device)
                target = sample["target"].unsqueeze(0).to(device)

                local_model.zero_grad()
                logits = local_model(image)
                loss = loss_fn(logits, target)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Numerical instability in {hospital_name} training at step {step_idx}")

                loss.backward()
                step_loss_sum += loss.item()
                total_samples_evaluated += 1

                # Per-sample gradient norm calculation and clipping to max_grad_norm
                active_params = [p for p in local_model.parameters() if p.grad is not None]
                sample_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in active_params]), 2
                ).item()
                pre_clip_norms.append(sample_norm)

                if sample_norm > max_grad_norm:
                    clipped_count += 1
                    clip_factor = max_grad_norm / sample_norm
                else:
                    clip_factor = 1.0

                for acc, p in zip(accumulated_grads, active_params):
                    acc.add_(p.grad * clip_factor)

            step_loss_avg = step_loss_sum / actual_batch_size
            train_losses.append(step_loss_avg)
        else:
            # Empty Poisson batch (|S_t| = 0): Zero gradient, loss recorded as 0.0
            train_losses.append(0.0)

        # Calibrated Gaussian noise addition:
        # Under Poisson DP-SGD, sensitivity of sum of clipped gradients is C = max_grad_norm
        noise_std = noise_multiplier * max_grad_norm
        active_params = [p for p in local_model.parameters() if p.requires_grad]

        with torch.no_grad():
            for acc, p in zip(accumulated_grads, active_params):
                noise = torch.randn_like(acc) * noise_std
                p.grad = acc + noise

        optimizer.step()
        step_count += 1

    duration = time.perf_counter() - t0
    mean_loss = float(np.mean(train_losses)) if train_losses else 0.0
    mean_pre_clip_norm = float(np.mean(pre_clip_norms)) if pre_clip_norms else 0.0
    post_clip_norms = [min(norm, max_grad_norm) for norm in pre_clip_norms]
    mean_clipped_norm = float(np.mean(post_clip_norms)) if post_clip_norms else 0.0
    clipping_fraction = float(clipped_count / total_samples_evaluated) if total_samples_evaluated > 0 else 0.0

    mean_batch_size = float(np.mean(batch_sizes)) if batch_sizes else 0.0
    median_batch_size = float(np.median(batch_sizes)) if batch_sizes else 0.0
    std_batch_size = float(np.std(batch_sizes)) if batch_sizes else 0.0
    min_batch_size = int(np.min(batch_sizes)) if batch_sizes else 0
    max_batch_size = int(np.max(batch_sizes)) if batch_sizes else 0
    empty_batch_count = sum(1 for b in batch_sizes if b == 0)
    singleton_batch_count = sum(1 for b in batch_sizes if b == 1)
    multi_sample_batch_count = sum(1 for b in batch_sizes if b >= 2)

    # Calculate cumulative epsilon across all rounds up to now using exact RDP
    total_cumulative_steps = current_round * step_count
    budget = compute_rdp_budget_detailed(
        steps=total_cumulative_steps,
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        sample_rate=sample_rate,
    )

    dp_telemetry = {
        "max_grad_norm": max_grad_norm,
        "noise_multiplier": noise_multiplier,
        "noise_std": round(noise_multiplier * max_grad_norm, 4),
        "mean_pre_clip_norm": round(mean_pre_clip_norm, 4),
        "mean_unclipped_gradient_norm": round(mean_pre_clip_norm, 4),
        "mean_clipped_gradient_norm": round(mean_clipped_norm, 4),
        "clipping_fraction": round(clipping_fraction, 4),
        "mean_batch_size": round(mean_batch_size, 4),
        "median_batch_size": round(median_batch_size, 4),
        "std_batch_size": round(std_batch_size, 4),
        "min_batch_size": min_batch_size,
        "max_batch_size": max_batch_size,
        "empty_batch_count": empty_batch_count,
        "singleton_batch_count": singleton_batch_count,
        "multi_sample_batch_count": multi_sample_batch_count,
        "total_samples_evaluated": total_samples_evaluated,
        "steps_in_round": step_count,
        "cumulative_steps": total_cumulative_steps,
        "current_epsilon": round(budget["epsilon"], 4),
        "optimal_alpha": budget["optimal_alpha"],
        "target_delta": target_delta,
        "sampling_mechanism": "Poisson / Bernoulli Subsampling (q=1/236)",
        "accountant": "Exact Analytical Rényi Differential Privacy (Poisson RDP)",
    }

    local_weights = {k: v.cpu().clone().detach() for k, v in local_model.state_dict().items()}
    return local_weights, mean_loss, len(subject_dirs), duration, dp_telemetry


def run_fedavg_dp_d1_experiment(
    config_path: Path = PROJECT_ROOT / "configs" / "experiments" / "real_brats_dp_d1.yaml",
    data_dir: Path = PROJECT_ROOT / "data" / "raw" / "BraTS2024",
    split_file: Path = PROJECT_ROOT / "reports" / "real_brats2024" / "dataset_split.json",
    partitions_file: Path = PROJECT_ROOT / "reports" / "real_brats2024" / "hospital_partitions.json",
    rounds: int = 20,
    max_steps_per_client: Optional[int] = None,
    resume: bool = True,
    ckpt_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
    progress_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("🔒 FEDMED OS — EXPERIMENT D1: CALIBRATED CLIPPING DP-SGD (C = 0.06)")
    logger.info("=" * 80)

    start_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    overall_start_time = time.perf_counter()

    # 1. Load Canonical Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config_str = yaml.dump(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

    fed_cfg = config.get("federated", {})
    seed = fed_cfg.get("seed", 42)
    p_cfg = config.get("privacy", {})

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Select Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Execution Device: {device} | DP Mechanism: {p_cfg.get('mechanism')} (σ={p_cfg.get('noise_multiplier')}, C={p_cfg.get('max_grad_norm')})")

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

    hospitals = fed_cfg.get("hospitals", ["hospital_alpha", "hospital_beta", "hospital_gamma", "hospital_delta"])
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

    # 5. Checkpoint Directory Setup (Isolated D1 paths)
    if ckpt_dir is None:
        ckpt_dir = PROJECT_ROOT / "checkpoints" / "fedavg_dp_d1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best.pt"
    latest_ckpt_path = ckpt_dir / "latest.pt"

    if report_path is None:
        report_path = PROJECT_ROOT / "reports" / "real_brats2024" / "fedavg_dp_d1.json"
    if history_path is None:
        history_path = PROJECT_ROOT / "reports" / "real_brats2024" / "fedavg_dp_d1_history.json"
    if progress_path is None:
        progress_path = PROJECT_ROOT / "reports" / "real_brats2024" / "fedavg_dp_d1_progress.json"
    if manifest_path is None:
        manifest_path = PROJECT_ROOT / "reports" / "real_brats2024" / "dp_d1_experiment_manifest.json"

    start_round = 1
    best_mean_dice = -1.0
    best_round = -1
    best_val_metrics = {}
    round_history = []

    # Resume capability strictly within D1 checkpoint
    if resume and latest_ckpt_path.exists():
        logger.info(f"Resuming from latest D1 checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        global_model.load_state_dict(checkpoint["model_state_dict"])
        start_round = checkpoint["round"] + 1
        best_mean_dice = checkpoint.get("best_mean_dice", -1.0)
        best_round = checkpoint.get("best_round", -1)
        best_val_metrics = checkpoint.get("best_val_metrics", {})
        round_history = checkpoint.get("round_history", [])
        logger.info(f"Resumed at round {start_round}, prior best macro dice: {best_mean_dice:.4f}")

    # 6. Federated Rounds Loop
    logger.info(f"Beginning FedAvg+DP D1 Experiment for {rounds} rounds across 4 hospitals (C={p_cfg.get('max_grad_norm')})...")
    round_durations = []
    strategy = FedAvg(min_fit_clients=4, min_available_clients=4)

    for r in range(start_round, rounds + 1):
        round_start_time = time.perf_counter()
        logger.info(f"\n{'='*70}\n🔄 Starting Federated Round {r}/{rounds} (FedAvg + DP-SGD D1, C={p_cfg.get('max_grad_norm')})\n{'='*70}")

        global_state = {k: v.cpu().clone().detach() for k, v in global_model.state_dict().items()}
        param_names = list(global_state.keys())

        client_updates = {}
        client_sample_counts = {}
        client_losses = {}
        client_timings = {}
        client_dp_telemetry = {}
        fit_results = []

        # Sequential Execution of Clients on Apple Silicon MPS
        for h_name in hospitals:
            logger.info(f"[{h_name}] Executing local DP-SGD training on {len(hospital_dirs[h_name])} subjects (C={p_cfg.get('max_grad_norm')})...")
            loc_weights, loc_loss, n_samples, loc_time, dp_tel = train_hospital_silo_dp(
                hospital_name=h_name,
                subject_dirs=hospital_dirs[h_name],
                global_state_dict=global_state,
                config=config,
                device=device,
                max_steps=max_steps_per_client,
                current_round=r,
                seed=seed,
            )

            client_updates[h_name] = loc_weights
            client_sample_counts[h_name] = n_samples
            client_losses[h_name] = loc_loss
            client_timings[h_name] = loc_time
            client_dp_telemetry[h_name] = dp_tel

            loc_arrays = [loc_weights[k].numpy() for k in param_names]
            fit_results.append(FitResult(
                parameters=loc_arrays,
                num_examples=n_samples,
                metrics={
                    "training_loss": loc_loss,
                    "hospital": h_name,
                    "clipping_fraction": dp_tel["clipping_fraction"],
                    "current_epsilon": dp_tel["current_epsilon"],
                },
            ))
            logger.info(
                f"[{h_name}] DP Training Complete in {loc_time:.1f}s | "
                f"Loss: {loc_loss:.6f} | Clip Fraction: {dp_tel['clipping_fraction']*100:.1f}% | ε={dp_tel['current_epsilon']:.4f}"
            )

        # Verify Client Parameter Divergence
        w_alpha = client_updates["hospital_alpha"]
        w_beta = client_updates["hospital_beta"]
        divergence = sum((w_alpha[k] - w_beta[k]).norm().item() for k in param_names)
        assert divergence > 0.0, "FATAL: Client updates failed to diverge!"

        # Sample-Weighted FedAvg Aggregation
        agg_arrays, agg_metrics = strategy.aggregate_fit(
            server_round=r,
            results=fit_results,
            failures=[],
        )

        new_global_state = {}
        for name, arr in zip(param_names, agg_arrays):
            new_global_state[name] = torch.from_numpy(arr).to(device)

        delta_global = sum((new_global_state[k].cpu() - global_state[k]).norm().item() for k in param_names)
        assert delta_global > 0.0, "FATAL: Global parameter delta is zero!"
        global_model.load_state_dict(new_global_state)

        # Global Evaluation on Held-Out 202 Validation Cohort
        t_val_start = time.perf_counter()
        logger.info(f"Evaluating global model on held-out validation cohort (202 subjects)...")
        val_metrics = evaluate_global_model(global_model, val_loader, loss_fn, device)
        val_duration = time.perf_counter() - t_val_start

        round_total_duration = time.perf_counter() - round_start_time
        round_durations.append(round_total_duration)

        mean_round_eps = float(np.mean([t["current_epsilon"] for t in client_dp_telemetry.values()]))
        mean_clip_frac = float(np.mean([t["clipping_fraction"] for t in client_dp_telemetry.values()]))
        mean_batch_sz = float(np.mean([t["mean_batch_size"] for t in client_dp_telemetry.values()]))
        median_batch_sz = float(np.mean([t["median_batch_size"] for t in client_dp_telemetry.values()]))
        std_batch_sz = float(np.mean([t["std_batch_size"] for t in client_dp_telemetry.values()]))
        min_batch_sz = int(min(t["min_batch_size"] for t in client_dp_telemetry.values()))
        max_batch_sz = int(max(t["max_batch_size"] for t in client_dp_telemetry.values()))
        empty_batches_total = int(sum(t["empty_batch_count"] for t in client_dp_telemetry.values()))
        singleton_batches_total = int(sum(t["singleton_batch_count"] for t in client_dp_telemetry.values()))
        multi_sample_batches_total = int(sum(t["multi_sample_batch_count"] for t in client_dp_telemetry.values()))
        mean_unclipped_norm = float(np.mean([t["mean_unclipped_gradient_norm"] for t in client_dp_telemetry.values()]))
        mean_clipped_norm = float(np.mean([t["mean_clipped_gradient_norm"] for t in client_dp_telemetry.values()]))
        noise_std_val = float(np.mean([t["noise_std"] for t in client_dp_telemetry.values()]))
        optimal_alpha_val = int(list(client_dp_telemetry.values())[0]["optimal_alpha"])
        cum_steps_val = int(list(client_dp_telemetry.values())[0]["cumulative_steps"])

        logger.info(
            f"Round {r:02d} Summary: Mean Client Loss={agg_metrics['training_loss']:.6f} | "
            f"Val Loss={val_metrics['val_loss']:.6f} | WT Dice={val_metrics['wt_dice']:.4f} | "
            f"TC Dice={val_metrics['tc_dice']:.4f} | ET Dice={val_metrics['et_dice']:.4f} | "
            f"Macro Dice={val_metrics['mean_dice']:.4f} | ε={mean_round_eps:.4f} | Clip%={mean_clip_frac*100:.1f}% | Time={round_total_duration:.1f}s"
        )

        round_record = {
            "round": r,
            "mean_client_loss": round(agg_metrics["training_loss"], 6),
            "client_losses": {k: round(v, 6) for k, v in client_losses.items()},
            "client_timings_sec": {k: round(v, 2) for k, v in client_timings.items()},
            "global_parameter_delta": round(delta_global, 6),
            "client_divergence": round(divergence, 6),
            "privacy_telemetry": {
                "max_grad_norm_C": p_cfg.get("max_grad_norm", 0.06),
                "noise_multiplier_sigma": p_cfg.get("noise_multiplier", 0.87),
                "noise_standard_deviation": round(noise_std_val, 4),
                "mean_batch_size": round(mean_batch_sz, 4),
                "median_batch_size": round(median_batch_sz, 4),
                "std_batch_size": round(std_batch_sz, 4),
                "min_batch_size": min_batch_sz,
                "max_batch_size": max_batch_sz,
                "empty_batch_count": empty_batches_total,
                "singleton_batch_count": singleton_batches_total,
                "multi_sample_batch_count": multi_sample_batches_total,
                "clipping_fraction": round(mean_clip_frac, 4),
                "mean_clipping_fraction": round(mean_clip_frac, 4),
                "mean_unclipped_gradient_norm": round(mean_unclipped_norm, 4),
                "mean_clipped_gradient_norm": round(mean_clipped_norm, 4),
                "cumulative_epsilon": round(mean_round_eps, 4),
                "accountant_epsilon": round(mean_round_eps, 4),
                "accountant_optimal_alpha": optimal_alpha_val,
                "accountant_step_count": cum_steps_val,
                "target_delta": p_cfg.get("target_delta", 1e-5),
                "sampling_mechanism": "Poisson / Bernoulli Subsampling (q=1/236)",
                "accountant": "Exact Analytical Rényi Differential Privacy (Poisson RDP)",
            },
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

        # Model selection: Macro Dice
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
                "privacy_telemetry": round_record["privacy_telemetry"],
                "config_hash": config_hash,
                "split_hash": split_hash,
                "partition_hash": partition_hash,
                "seed": seed,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, best_ckpt_path)
            logger.info(f"🏆 New best validation model saved at Round {r} (Macro Dice: {best_mean_dice:.4f})")

        torch.save({
            "round": r,
            "model_state_dict": global_model.state_dict(),
            "best_mean_dice": best_mean_dice,
            "best_round": best_round,
            "best_val_metrics": best_val_metrics,
            "round_history": round_history,
            "privacy_telemetry": round_record["privacy_telemetry"],
            "config_hash": config_hash,
            "split_hash": split_hash,
            "partition_hash": partition_hash,
            "seed": seed,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, latest_ckpt_path)

        # Continuously persist history after every completed round
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(round_history, f, indent=2)

        # Continuously update live progress file
        elapsed_sec = time.perf_counter() - overall_start_time
        avg_r_time = elapsed_sec / (r - start_round + 1)
        remaining_r = rounds - r
        est_rem_sec = avg_r_time * remaining_r
        est_rem_hrs = est_rem_sec / 3600.0

        mps_mem_mb = None
        if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            mps_mem_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)

        progress_record = {
            "status": "RUNNING" if r < rounds else "COMPLETED",
            "current_round": r,
            "total_rounds": rounds,
            "round_progress": f"{r}/{rounds}",
            "completed_clients": 4,
            "total_clients": 4,
            "current_client": None,
            "elapsed_seconds": round(elapsed_sec, 2),
            "estimated_remaining_seconds": round(est_rem_sec, 2),
            "estimated_remaining_hours": round(est_rem_hrs, 2),
            "latest_validation_loss": val_metrics["val_loss"],
            "latest_macro_dice": val_metrics["mean_dice"],
            "best_macro_dice": best_mean_dice,
            "best_round": best_round,
            "current_epsilon": round_record["privacy_telemetry"]["cumulative_epsilon"],
            "target_epsilon": 2.8934,
            "accountant_steps": round_record["privacy_telemetry"]["accountant_step_count"],
            "target_accountant_steps": 4720,
            "MPS_memory": mps_mem_mb,
            "last_checkpoint": str(latest_ckpt_path),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(progress_path, "w") as f:
            json.dump(progress_record, f, indent=2)

    total_experiment_time = time.perf_counter() - overall_start_time
    avg_round_time = float(np.mean(round_durations)) if round_durations else 0.0
    best_ckpt_sha256 = compute_file_sha256(best_ckpt_path) if best_ckpt_path.exists() else None

    peak_mps_mb = None
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        peak_mps_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)

    manifest_record = {
        "experiment_name": "EXPERIMENT_D1_CALIBRATED_CLIPPING",
        "dataset_split_hash": split_hash,
        "hospital_partition_hash": partition_hash,
        "configuration_hash": config_hash,
        "random_seed": seed,
        "invariant_parameters": {
            "dataset": "BraTS-GLI 2024 Adult Glioma Post-Treatment",
            "total_subjects": 1350,
            "train_subjects": 944,
            "validation_subjects": 202,
            "test_subjects": 204,
            "hospital_silos": ["hospital_alpha", "hospital_beta", "hospital_gamma", "hospital_delta"],
            "subjects_per_silo": 236,
            "model_architecture": "MONAI 3D U-Net",
            "parameter_count": total_params,
            "spatial_resolution": list(spatial_shape),
            "loss_function": "DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.2)",
            "optimizer": "Adam(lr=1e-4, weight_decay=1e-5)",
            "batch_size": 1,
            "local_epochs": 1,
            "rounds": rounds,
            "client_execution": "sequential",
            "seed": seed,
        },
        "privacy_parameters": {
            "mechanism": "DP-SGD",
            "level": "sample_level",
            "max_grad_norm_C": p_cfg.get("max_grad_norm", 0.06),
            "noise_multiplier_sigma": p_cfg.get("noise_multiplier", 0.87),
            "target_delta": p_cfg.get("target_delta", 1e-5),
            "calculated_epsilon_20_rounds": round(
                compute_rdp_epsilon(
                    steps=rounds * 236,
                    noise_multiplier=p_cfg.get("noise_multiplier", 0.87),
                    target_delta=p_cfg.get("target_delta", 1e-5),
                    sample_rate=p_cfg.get("sample_rate", 1.0 / 236.0),
                ),
                4,
            ),
            "sampling_mechanism": "Poisson / Bernoulli Subsampling (q=1/236)",
            "accountant": "Exact Analytical Rényi Differential Privacy (Poisson RDP)",
        },
        "best_checkpoint_sha256": best_ckpt_sha256,
        "status": "COMPLETED" if rounds >= 20 else f"PARTIAL_{rounds}_ROUNDS",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_record, f, indent=2)

    total_exp_steps = rounds * 236
    final_budget = compute_rdp_budget_detailed(
        steps=total_exp_steps,
        noise_multiplier=p_cfg.get("noise_multiplier", 0.87),
        target_delta=p_cfg.get("target_delta", 1e-5),
        sample_rate=p_cfg.get("sample_rate", 1.0 / 236.0),
    )

    final_report = {
        "experiment_name": "EXPERIMENT_D1_CALIBRATED_CLIPPING",
        "timestamp_start": start_timestamp,
        "timestamp_end": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "COMPLETED" if rounds >= 20 else f"PARTIAL_{rounds}_ROUNDS",
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
            "hospital_silos": {h: len(hospital_dirs[h]) for h in hospitals},
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
        "privacy": {
            "mechanism": "DP-SGD",
            "level": "sample_level",
            "max_grad_norm_C": p_cfg.get("max_grad_norm", 0.06),
            "noise_multiplier_sigma": p_cfg.get("noise_multiplier", 0.87),
            "target_delta": p_cfg.get("target_delta", 1e-5),
            "calculated_epsilon_20_rounds": round(final_budget["epsilon"], 4),
            "optimal_alpha": final_budget["optimal_alpha"],
            "total_steps": total_exp_steps,
            "sampling_mechanism": "Poisson / Bernoulli Subsampling (q=1/236)",
            "accountant": "Exact Analytical Rényi Differential Privacy (Poisson RDP)",
        },
        "performance_summary": {
            "primary_model_selection_metric": "macro_dice",
            "best_round": best_round,
            "best_validation_metrics": best_val_metrics,
            "final_train_loss": round_history[-1]["mean_client_loss"] if round_history else None,
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

    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info("=" * 80)
    logger.info("🎉 EXPERIMENT D1 EXECUTION FINISHED")
    logger.info(f"Rounds Executed:          {rounds}")
    logger.info(f"Best Round (Macro Dice):  {best_round}")
    logger.info(f"Best Macro Dice:          {best_val_metrics.get('mean_dice', 0.0):.4f}")
    logger.info(f"Best WT Dice:             {best_val_metrics.get('wt_dice', 0.0):.4f}")
    logger.info(f"Best TC Dice:             {best_val_metrics.get('tc_dice', 0.0):.4f}")
    logger.info(f"Best ET Dice:             {best_val_metrics.get('et_dice', 0.0):.4f}")
    logger.info(f"Final Privacy Budget:     ε = {round_history[-1]['privacy_telemetry']['cumulative_epsilon']:.4f} (δ = 1e-5)")
    logger.info(f"Total Runtime:            {total_experiment_time:.2f}s ({total_experiment_time/3600:.2f}h)")
    logger.info(f"Best Checkpoint:          {best_ckpt_path}")
    logger.info("=" * 80)

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Real-Data FedAvg + Calibrated Clipping DP-SGD (Experiment D1)")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "experiments" / "real_brats_dp_d1.yaml"))
    parser.add_argument("--rounds", type=int, default=20, help="Number of federated rounds")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per client for smoke testing")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint")
    args = parser.parse_args()

    run_fedavg_dp_d1_experiment(
        config_path=Path(args.config),
        rounds=args.rounds,
        max_steps_per_client=args.max_steps,
        resume=not args.no_resume,
    )
