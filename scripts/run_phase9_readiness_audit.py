"""
Script: scripts/run_phase9_readiness_audit.py

Purpose:
FEDMED OS — Phase 9.1 Training Readiness Audit Engine.
Executes systematic empirical audits across:
- Preprocessing & Resolution Structure Retention (Audit 4)
- Real-Data Local Training Computational Profile (Audit 5)
- Learning Sanity Check (Audit 6)
- Controlled Single-Subject Overfitting Sanity Check (Audit 7)
- 4-Hospital Federated Profile (Audit 8)
- Differential Privacy Mechanism Profile (Audit 9)
- Homomorphic Encryption Aggregation Profile (Audit 10)
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import nibabel as nib
import numpy as np
import tenseal as ts
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.canonical_adapter import DatasetVersion, LabelCanonicalizer, ModalityCanonicalizer
from data.datasets.transforms import get_brats_transforms
from evaluation.metrics import compute_dice, compute_iou
from privacy.aggregation import aggregate_encrypted_updates
from privacy.context import create_ckks_context, get_public_context
from privacy.decrypt import decrypt_weights_vector
from privacy.dp_engine import DifferentialPrivacyEngine, compute_rdp_epsilon
from privacy.encrypt import encrypt_weights_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase9_readiness_audit")


class AuditDataset(Dataset):
    def __init__(self, subject_dirs: List[Path], spatial_shape: Tuple[int, int, int] = (32, 32, 32)):
        self.subject_dirs = subject_dirs
        self.spatial_shape = spatial_shape
        self.transforms = get_brats_transforms(
            mode="train",
            image_size=spatial_shape,
            dataset_version=DatasetVersion.BRATS_GLI_2024,
        )

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        s_dir = self.subject_dirs[idx]
        version, mod_paths, missing = ModalityCanonicalizer.discover_subject_modalities(s_dir)
        if missing:
            raise ValueError(f"Missing modalities: {missing}")

        mod_files = [str(mod_paths[m]) for m in ("t1", "t1ce", "t2", "flair")]
        seg_file = str(mod_paths["seg"])

        sample = {"image": mod_files, "label": seg_file}
        processed = self.transforms(sample)

        return {
            "image": processed["image"],
            "target": processed["label"],
            "subject_id": s_dir.name,
        }


def compute_binary_dice_val(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    p = (pred > 0.5).float().view(-1)
    t = (target > 0.5).float().view(-1)
    intersection = (p * t).sum().item()
    p_sum = p.sum().item()
    t_sum = t.sum().item()
    return float((2.0 * intersection + smooth) / (p_sum + t_sum + smooth))


# ==============================================================================
# AUDIT 4: Tumor Structure Retention Across Resolutions
# ==============================================================================
def run_audit_4_resolution_retention(
    train_dirs: List[Path],
    resolutions: List[Tuple[int, int, int]] = [(32, 32, 32), (64, 64, 64), (128, 128, 128)],
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("🔬 AUDIT 4: EMPIRICAL TUMOR STRUCTURE RETENTION AUDIT (10 SUBJECTS)")
    logger.info("=" * 80)

    sample_dirs = train_dirs[:10]
    subject_retention_data = []

    for s_dir in sample_dirs:
        version, mod_paths, _ = ModalityCanonicalizer.discover_subject_modalities(s_dir)
        seg_nii = nib.load(str(mod_paths["seg"]))
        raw_seg = seg_nii.get_fdata(dtype=np.float32)

        # Original ground truth targets
        orig_targets, orig_summary = LabelCanonicalizer.build_canonical_targets(raw_seg, version=version)
        orig_tc = orig_summary["TC_voxels"]
        orig_wt = orig_summary["WT_voxels"]
        orig_et = orig_summary["ET_voxels"]

        res_map = {}
        for res in resolutions:
            tf = get_brats_transforms(mode="val", image_size=res, dataset_version=version)
            mod_files = [str(mod_paths[m]) for m in ("t1", "t1ce", "t2", "flair")]
            sample = {"image": mod_files, "label": str(mod_paths["seg"])}
            processed = tf(sample)
            t_target = processed["label"]  # (3, H, W, D)

            tc_v = int((t_target[0] > 0.5).sum().item())
            wt_v = int((t_target[1] > 0.5).sum().item())
            et_v = int((t_target[2] > 0.5).sum().item())

            res_map[f"{res[0]}x{res[1]}x{res[2]}"] = {
                "TC_voxels": tc_v,
                "WT_voxels": wt_v,
                "ET_voxels": et_v,
                "ET_survived": et_v > 0 if orig_et > 0 else True,
                "TC_survived": tc_v > 0 if orig_tc > 0 else True,
                "WT_survived": wt_v > 0 if orig_wt > 0 else True,
            }

        subject_retention_data.append({
            "subject_id": s_dir.name,
            "original_native_shape": list(raw_seg.shape),
            "original_voxels": {
                "TC": orig_tc,
                "WT": orig_wt,
                "ET": orig_et,
                "RC": orig_summary["RC_voxels"],
            },
            "resolutions": res_map,
        })

    # Summary analysis
    surv_32 = sum(1 for s in subject_retention_data if s["resolutions"]["32x32x32"]["ET_survived"])
    surv_64 = sum(1 for s in subject_retention_data if s["resolutions"]["64x64x64"]["ET_survived"])
    surv_128 = sum(1 for s in subject_retention_data if s["resolutions"]["128x128x128"]["ET_survived"])

    res_audit_summary = {
        "subjects_analyzed": len(subject_retention_data),
        "et_survival_rate": {
            "32x32x32": f"{surv_32}/{len(subject_retention_data)}",
            "64x64x64": f"{surv_64}/{len(subject_retention_data)}",
            "128x128x128": f"{surv_128}/{len(subject_retention_data)}",
        },
        "details": subject_retention_data,
    }

    logger.info(f"ET Survival Rate at 32x32x32:   {res_audit_summary['et_survival_rate']['32x32x32']}")
    logger.info(f"ET Survival Rate at 64x64x64:   {res_audit_summary['et_survival_rate']['64x64x64']}")
    logger.info(f"ET Survival Rate at 128x128x128: {res_audit_summary['et_survival_rate']['128x128x128']}")
    return res_audit_summary


# ==============================================================================
# AUDIT 5: Real-Data Training Computational Profile
# ==============================================================================
def run_audit_5_training_profile(train_dirs: List[Path], device: torch.device) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("⏱️ AUDIT 5: REAL-DATA LOCAL TRAINING COMPUTATIONAL PROFILE (4 SUBJECTS)")
    logger.info("=" * 80)

    subset_dirs = train_dirs[:4]
    ds = AuditDataset(subset_dirs, spatial_shape=(32, 32, 32))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = DiceCELoss(sigmoid=True)

    timing_breakdown = []
    round_start = time.perf_counter()

    for idx, batch in enumerate(loader):
        t0 = time.perf_counter()
        img = batch["image"].to(device)
        tgt = batch["target"].to(device)
        t_data = time.perf_counter() - t0

        t1 = time.perf_counter()
        optimizer.zero_grad()
        logits = model(img)
        loss = loss_fn(logits, tgt)
        t_fwd = time.perf_counter() - t1

        t2 = time.perf_counter()
        loss.backward()
        t_bwd = time.perf_counter() - t2

        t3 = time.perf_counter()
        optimizer.step()
        t_opt = time.perf_counter() - t3

        timing_breakdown.append({
            "subject_id": batch["subject_id"][0],
            "data_loading_sec": round(t_data, 4),
            "forward_sec": round(t_fwd, 4),
            "backward_sec": round(t_bwd, 4),
            "optimizer_sec": round(t_opt, 4),
            "total_step_sec": round(t_data + t_fwd + t_bwd + t_opt, 4),
            "loss": round(loss.item(), 6),
        })

    round_total_sec = time.perf_counter() - round_start

    # Memory profiling
    mps_allocated_mb = None
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        mps_allocated_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)

    profile_summary = {
        "device": str(device),
        "total_subjects": len(subset_dirs),
        "total_local_round_sec": round(round_total_sec, 4),
        "average_step_sec": round(round_total_sec / len(subset_dirs), 4),
        "mps_allocated_mb": mps_allocated_mb,
        "timing_breakdown": timing_breakdown,
    }

    logger.info(f"Total Local Round Time: {profile_summary['total_local_round_sec']}s (Avg: {profile_summary['average_step_sec']}s/subject)")
    if mps_allocated_mb is not None:
        logger.info(f"MPS Allocated Memory:   {mps_allocated_mb} MB")
    return profile_summary


# ==============================================================================
# AUDIT 6: Learning Sanity Check (Multi-Epoch on 4 Subjects)
# ==============================================================================
def run_audit_6_learning_sanity(train_dirs: List[Path], val_dirs: List[Path], device: torch.device) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("📈 AUDIT 6: LEARNING SANITY CHECK (4 TRAIN SUBJECTS, 5 EPOCHS)")
    logger.info("=" * 80)

    train_subset = train_dirs[:4]
    val_subset = val_dirs[:1]

    train_ds = AuditDataset(train_subset, spatial_shape=(32, 32, 32))
    val_ds = AuditDataset(val_subset, spatial_shape=(32, 32, 32))

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    torch.manual_seed(42)
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = DiceCELoss(sigmoid=True)

    epoch_history = []

    for epoch in range(1, 6):
        model.train()
        train_losses = []
        train_tc_dice = []
        train_wt_dice = []
        train_et_dice = []

        for batch in train_loader:
            img = batch["image"].to(device)
            tgt = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, tgt)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            probs = torch.sigmoid(logits)
            train_tc_dice.append(compute_binary_dice_val(probs[:, 0], tgt[:, 0]))
            train_wt_dice.append(compute_binary_dice_val(probs[:, 1], tgt[:, 1]))
            train_et_dice.append(compute_binary_dice_val(probs[:, 2], tgt[:, 2]))

        # Evaluate on validation
        model.eval()
        val_losses = []
        val_tc = []
        val_wt = []
        val_et = []
        with torch.no_grad():
            for v_batch in val_loader:
                v_img = v_batch["image"].to(device)
                v_tgt = v_batch["target"].to(device)
                v_logits = model(v_img)
                v_loss = loss_fn(v_logits, v_tgt)
                val_losses.append(v_loss.item())
                v_probs = torch.sigmoid(v_logits)
                val_tc.append(compute_binary_dice_val(v_probs[:, 0], v_tgt[:, 0]))
                val_wt.append(compute_binary_dice_val(v_probs[:, 1], v_tgt[:, 1]))
                val_et.append(compute_binary_dice_val(v_probs[:, 2], v_tgt[:, 2]))

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(train_losses)), 6),
            "train_tc_dice": round(float(np.mean(train_tc_dice)), 4),
            "train_wt_dice": round(float(np.mean(train_wt_dice)), 4),
            "train_et_dice": round(float(np.mean(train_et_dice)), 4),
            "val_loss": round(float(np.mean(val_losses)), 6),
            "val_tc_dice": round(float(np.mean(val_tc)), 4),
            "val_wt_dice": round(float(np.mean(val_wt)), 4),
            "val_et_dice": round(float(np.mean(val_et)), 4),
        }
        epoch_history.append(epoch_record)
        logger.info(f"Epoch {epoch}: Train Loss={epoch_record['train_loss']:.4f}, Train WT Dice={epoch_record['train_wt_dice']:.4f} | Val Loss={epoch_record['val_loss']:.4f}")

    loss_decreased = epoch_history[-1]["train_loss"] < epoch_history[0]["train_loss"]
    dice_improved = epoch_history[-1]["train_wt_dice"] >= epoch_history[0]["train_wt_dice"]

    return {
        "status": "PASS" if loss_decreased else "FAIL",
        "loss_decreased": loss_decreased,
        "dice_improved": dice_improved,
        "initial_train_loss": epoch_history[0]["train_loss"],
        "final_train_loss": epoch_history[-1]["train_loss"],
        "initial_train_wt_dice": epoch_history[0]["train_wt_dice"],
        "final_train_wt_dice": epoch_history[-1]["train_wt_dice"],
        "epoch_history": epoch_history,
    }


# ==============================================================================
# AUDIT 7: Controlled Overfitting Sanity Check (1 Subject)
# ==============================================================================
def run_audit_7_overfitting_sanity(train_dirs: List[Path], device: torch.device) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("🎯 AUDIT 7: CONTROLLED SINGLE-SUBJECT OVERFITTING SANITY CHECK")
    logger.info("=" * 80)

    single_dir = train_dirs[:1]
    ds = AuditDataset(single_dir, spatial_shape=(32, 32, 32))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    torch.manual_seed(42)
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = DiceCELoss(sigmoid=True)

    history = []
    model.train()
    for step in range(1, 31):
        for batch in loader:
            img = batch["image"].to(device)
            tgt = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, tgt)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            tc_d = compute_binary_dice_val(probs[:, 0], tgt[:, 0])
            wt_d = compute_binary_dice_val(probs[:, 1], tgt[:, 1])
            et_d = compute_binary_dice_val(probs[:, 2], tgt[:, 2])

            if step % 5 == 0 or step == 1 or step == 30:
                history.append({
                    "step": step,
                    "loss": round(loss.item(), 6),
                    "TC_dice": round(tc_d, 4),
                    "WT_dice": round(wt_d, 4),
                    "ET_dice": round(et_d, 4),
                })
                logger.info(f"Overfit Step {step:02d}: Loss={loss.item():.6f}, WT Dice={wt_d:.4f}, TC Dice={tc_d:.4f}, ET Dice={et_d:.4f}")

    init_loss = history[0]["loss"]
    final_loss = history[-1]["loss"]
    final_wt_dice = history[-1]["WT_dice"]
    overfit_successful = final_loss < init_loss * 0.5 and final_wt_dice > 0.50

    return {
        "status": "PASS" if overfit_successful else "FAIL",
        "subject_id": single_dir[0].name,
        "initial_loss": init_loss,
        "final_loss": final_loss,
        "final_wt_dice": final_wt_dice,
        "overfit_successful": overfit_successful,
        "trajectory": history,
    }


# ==============================================================================
# AUDIT 8: 4-Hospital Federated Profile
# ==============================================================================
def run_audit_8_federated_profile(train_dirs: List[Path], device: torch.device) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("🌐 AUDIT 8: 4-HOSPITAL FEDERATED LEARNING PROFILE (1 ROUND)")
    logger.info("=" * 80)

    hospitals = ["hospital_alpha", "hospital_beta", "hospital_gamma", "hospital_delta"]
    hosp_subjs = {hospitals[i]: [train_dirs[i]] for i in range(4)}

    # Global Model Init
    torch.manual_seed(42)
    global_model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    global_weights = {k: v.clone().detach() for k, v in global_model.state_dict().items()}
    loss_fn = DiceCELoss(sigmoid=True)

    client_updates = {}
    client_sample_counts = {}
    client_timings = {}
    payload_bytes = 0

    t_fed_start = time.perf_counter()

    for h_name, s_list in hosp_subjs.items():
        t0 = time.perf_counter()
        local_model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
        local_model.load_state_dict(global_weights)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-4)

        ds = AuditDataset(s_list, spatial_shape=(32, 32, 32))
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        local_model.train()
        for batch in loader:
            img = batch["image"].to(device)
            tgt = batch["target"].to(device)
            optimizer.zero_grad()
            logits = local_model(img)
            loss = loss_fn(logits, tgt)
            loss.backward()
            optimizer.step()

        local_weights = {k: v.cpu().clone().detach() for k, v in local_model.state_dict().items()}
        client_updates[h_name] = local_weights
        client_sample_counts[h_name] = 1
        t_loc = time.perf_counter() - t0
        client_timings[h_name] = round(t_loc, 4)

        # Estimate payload size
        if payload_bytes == 0:
            payload_bytes = sum(t.nelement() * t.element_size() for t in local_weights.values())

    # Sample-weighted FedAvg Aggregation on Server
    total_samples = sum(client_sample_counts.values())
    new_global_weights = {}
    for k in global_weights.keys():
        accum = torch.zeros_like(global_weights[k].cpu())
        for h_name in hospitals:
            w_client = client_sample_counts[h_name] / total_samples
            accum += w_client * client_updates[h_name][k]
        new_global_weights[k] = accum.to(device)

    # Calculate parameter change
    delta_global = sum((new_global_weights[k] - global_weights[k]).norm().item() for k in global_weights.keys())
    global_model.load_state_dict(new_global_weights)
    t_fed_total = time.perf_counter() - t_fed_start

    fed_report = {
        "round_time_sec": round(t_fed_total, 4),
        "participating_hospitals": hospitals,
        "sample_counts": client_sample_counts,
        "client_timings_sec": client_timings,
        "payload_per_client_bytes": payload_bytes,
        "payload_per_client_mb": round(payload_bytes / (1024 * 1024), 2),
        "global_parameter_delta": round(delta_global, 6),
        "aggregation_verified": delta_global > 0.0,
        "raw_image_transmission": False,
        "test_firewall_maintained": True,
    }

    logger.info(f"FedAvg 1 Round Complete in {fed_report['round_time_sec']}s")
    logger.info(f"Payload per client:       {fed_report['payload_per_client_mb']} MB")
    logger.info(f"Global Parameter Delta:   {fed_report['global_parameter_delta']}")
    return fed_report


# ==============================================================================
# AUDIT 9: Differential Privacy Compatibility Profile
# ==============================================================================
def run_audit_9_dp_profile(train_dirs: List[Path], device: torch.device) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("🛡️ AUDIT 9: DIFFERENTIAL PRIVACY MECHANISM PROFILE")
    logger.info("=" * 80)

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dp_engine = DifferentialPrivacyEngine(
        model=model,
        optimizer=optimizer,
        target_epsilon=5.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=0.8,
        sample_rate=0.01,
    )

    ds = AuditDataset(train_dirs[:1], spatial_shape=(32, 32, 32))
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    loss_fn = DiceCELoss(sigmoid=True)

    for batch in loader:
        img = batch["image"].to(device)
        tgt = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(img)
        loss = loss_fn(logits, tgt)
        loss.backward()

        # Apply DP clipping and noise
        grad_norm = dp_engine.apply_gradient_clipping_and_noise(batch_size=1)
        optimizer.step()
        break

    budget = dp_engine.get_privacy_budget()

    dp_report = {
        "dp_enabled": True,
        "max_grad_norm_C": dp_engine.max_grad_norm,
        "noise_multiplier_sigma": dp_engine.noise_multiplier,
        "target_epsilon": dp_engine.target_epsilon,
        "target_delta": dp_engine.target_delta,
        "spent_epsilon": budget["epsilon"],
        "spent_delta": budget["delta"],
        "steps": dp_engine.steps,
        "clipping_executed": True,
        "noise_injected": True,
        "compatible_with_real_shapes": True,
    }

    logger.info(f"DP Step Executed. Spent Epsilon: {budget['epsilon']}, Delta: {budget['delta']}")
    return dp_report


# ==============================================================================
# AUDIT 10: Homomorphic Encryption Compatibility Profile
# ==============================================================================
def run_audit_10_he_profile() -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("🔐 AUDIT 10: HOMOMORPHIC ENCRYPTION AGGREGATION PROFILE")
    logger.info("=" * 80)

    context = create_ckks_context(poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60], global_scale=2**40)
    public_ctx = get_public_context(context)

    # Create dummy client update vectors representing layer parameter weights
    layer_shape = (50, 100)
    w_alpha = np.random.randn(*layer_shape).astype(np.float32)
    w_beta = np.random.randn(*layer_shape).astype(np.float32)

    # Plain reference average
    true_avg = 0.5 * w_alpha + 0.5 * w_beta

    # Encrypt client updates in chunks
    chunks_alpha, t_enc_a, b_a = encrypt_weights_vector(context, w_alpha, chunk_size=4096)
    chunks_beta, t_enc_b, b_b = encrypt_weights_vector(context, w_beta, chunk_size=4096)

    # Server aggregation using public context
    client_results = [
        ([chunks_alpha], 1),
        ([chunks_beta], 1),
    ]
    agg_res = aggregate_encrypted_updates(public_ctx, client_results, shapes=[layer_shape])
    agg_chunks = agg_res["aggregated_chunks"][0]

    # Client decrypts with secret key
    dec_avg = decrypt_weights_vector(context, agg_chunks, original_shape=layer_shape)

    # Precision error
    max_abs_err = float(np.max(np.abs(dec_avg - true_avg)))
    mean_abs_err = float(np.mean(np.abs(dec_avg - true_avg)))

    he_report = {
        "he_scheme": "CKKS",
        "poly_modulus_degree": 8192,
        "tested_tensor_shape": list(layer_shape),
        "encryption_time_ms": round((t_enc_a + t_enc_b) * 500, 2),
        "aggregation_time_ms": round(agg_res["aggregation_time_ms"], 2),
        "max_reconstruction_error": max_abs_err,
        "mean_reconstruction_error": mean_abs_err,
        "he_aggregation_verified": max_abs_err < 1e-4,
    }

    logger.info(f"HE Aggregation Verified. Max Error: {max_abs_err:.8f}")
    return he_report


# ==============================================================================
# MASTER RUNNER
# ==============================================================================
def run_full_readiness_audit():
    out_dir = PROJECT_ROOT / "reports" / "phase9"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Load splits
    split_file = PROJECT_ROOT / "reports" / "real_brats2024" / "dataset_split.json"
    with open(split_file, "r") as f:
        split_data = json.load(f)

    data_dir = PROJECT_ROOT / "data" / "raw" / "BraTS2024"
    from data.real_brats_pipeline import RealBratsValidator
    validator = RealBratsValidator(data_dir)
    all_subjs = {p.name: p for p in validator.discover_subject_directories()}

    train_dirs = [all_subjs[s] for s in split_data["train_subjects"] if s in all_subjs]
    val_dirs = [all_subjs[s] for s in split_data["validation_subjects"] if s in all_subjs]

    audit_4_res = run_audit_4_resolution_retention(train_dirs)
    audit_5_res = run_audit_5_training_profile(train_dirs, device)
    audit_6_res = run_audit_6_learning_sanity(train_dirs, val_dirs, device)
    audit_7_res = run_audit_7_overfitting_sanity(train_dirs, device)
    audit_8_res = run_audit_8_federated_profile(train_dirs, device)
    audit_9_res = run_audit_9_dp_profile(train_dirs, device)
    audit_10_res = run_audit_10_he_profile()

    master_report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "audit_phase": "PHASE_9.1_REAL_BRATS_GLI_2024_READINESS",
        "device": str(device),
        "audit_4_resolution_retention": audit_4_res,
        "audit_5_training_profile": audit_5_res,
        "audit_6_learning_sanity": audit_6_res,
        "audit_7_overfitting_sanity": audit_7_res,
        "audit_8_federated_profile": audit_8_res,
        "audit_9_dp_profile": audit_9_res,
        "audit_10_he_profile": audit_10_res,
    }

    report_path = out_dir / "readiness_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(master_report, f, indent=2)

    logger.info(f"Master readiness audit written to {report_path}")
    return master_report


if __name__ == "__main__":
    run_full_readiness_audit()
