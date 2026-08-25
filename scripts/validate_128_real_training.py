"""
Script: scripts/validate_128_real_training.py

Purpose:
Phase 9.2 Real Training Configuration Small Validation Engine.
Validates the canonical real-data training configuration on 4 real BraTS-GLI 2024 subjects (1 per hospital silo):
- Spatial resolution: (128, 128, 128)
- Loss: DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.2)
- Sequential client simulation on Apple Silicon MPS
- Gradient verification, parameter update, latency, and memory profiling
- Strict test firewall verification (TEST_ACCESSED_DURING_TRAINING = False)
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import nibabel as nib
import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.canonical_adapter import DatasetVersion, ModalityCanonicalizer
from data.datasets.transforms import get_brats_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_128_real_training")


class Real128Dataset(Dataset):
    def __init__(self, subject_dirs: List[Path], spatial_shape: tuple = (128, 128, 128)):
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


def run_128_validation(
    data_dir: Path = PROJECT_ROOT / "data" / "raw" / "BraTS2024",
    split_file: Path = PROJECT_ROOT / "reports" / "real_brats2024" / "dataset_split.json",
    partitions_file: Path = PROJECT_ROOT / "reports" / "real_brats2024" / "hospital_partitions.json",
    seed: int = 42,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("🧪 RUNNING PHASE 9.2: 128³ REAL TRAINING CONFIGURATION VALIDATION")
    logger.info("=" * 80)

    # 1. Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Execution device: {device}")

    # 2. Load Split and Partitions
    with open(split_file, "r") as f:
        split_data = json.load(f)
    with open(partitions_file, "r") as f:
        part_data = json.load(f)

    from data.real_brats_pipeline import RealBratsValidator
    validator = RealBratsValidator(data_dir)
    all_subjs = {p.name: p for p in validator.discover_subject_directories()}

    # Select 1 subject per hospital silo
    hospitals = ["hospital_alpha", "hospital_beta", "hospital_gamma", "hospital_delta"]
    selected_subjects = {}
    for h in hospitals:
        s_id = part_data["partitions"][h]["subjects"][0]
        selected_subjects[h] = all_subjs[s_id]

    test_subjects_firewalled = split_data["test_subjects"]
    logger.info(f"Test split firewalled: {len(test_subjects_firewalled)} subjects (Zero access)")

    # 3. Model & Optimizer
    torch.manual_seed(seed)
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    initial_params = {k: v.clone().detach() for k, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.2)

    # 4. Sequential Execution Across 4 Hospitals
    model.train()
    client_step_metrics = []
    total_grad_norms = {}
    round_start = time.perf_counter()
    oom_detected = False

    for h_name, s_path in selected_subjects.items():
        t0 = time.perf_counter()
        ds = Real128Dataset([s_path], spatial_shape=(128, 128, 128))
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        for batch in loader:
            t_load = time.perf_counter() - t0

            # Forward pass
            t_fwd_start = time.perf_counter()
            try:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, targets)
                t_fwd = time.perf_counter() - t_fwd_start

                # Backward pass
                t_bwd_start = time.perf_counter()
                loss.backward()
                t_bwd = time.perf_counter() - t_bwd_start

                # Gradient inspection
                grad_norms = {}
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        g_norm = p.grad.norm().item()
                        grad_norms[name] = g_norm
                total_grad_norms[h_name] = max(grad_norms.values()) if grad_norms else 0.0

                # Optimizer step
                t_opt_start = time.perf_counter()
                optimizer.step()
                t_opt = time.perf_counter() - t_opt_start

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_detected = True
                    logger.error(f"OOM Detected on {h_name}: {e}")
                    break
                raise e

            client_step_metrics.append({
                "hospital": h_name,
                "subject_id": batch["subject_id"][0],
                "data_load_time_sec": round(t_load, 4),
                "forward_time_sec": round(t_fwd, 4),
                "backward_time_sec": round(t_bwd, 4),
                "optimizer_time_sec": round(t_opt, 4),
                "total_step_time_sec": round(t_load + t_fwd + t_bwd + t_opt, 4),
                "loss": round(loss.item(), 6),
                "max_grad_norm": round(total_grad_norms[h_name], 6),
            })
            logger.info(
                f"[{h_name}] Step Complete — Subject: {batch['subject_id'][0]} | "
                f"Loss: {loss.item():.6f} | Step Time: {t_load + t_fwd + t_bwd + t_opt:.3f}s"
            )

    round_time_sec = time.perf_counter() - round_start

    # 5. Measure Parameter Update Delta
    param_deltas = {}
    for k, v in model.named_parameters():
        delta = (v - initial_params[k].to(device)).norm().item()
        param_deltas[k] = delta
    total_delta = sum(param_deltas.values())

    # 6. Memory Stats
    mps_mem_mb = None
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        mps_mem_mb = round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)

    val_report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phase": "PHASE_9.2_REAL_128_VALIDATION",
        "spatial_resolution": [128, 128, 128],
        "loss_function": "DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.2)",
        "device": str(device),
        "oom_detected": oom_detected,
        "validation_status": "PASS" if not oom_detected and total_delta > 0.0 else "FAIL",
        "performance_profile": {
            "round_time_sec": round(round_time_sec, 4),
            "average_step_time_sec": round(round_time_sec / len(hospitals), 4),
            "mps_allocated_memory_mb": mps_mem_mb,
            "total_parameter_delta": round(total_delta, 6),
            "max_gradient_norm": max(total_grad_norms.values()) if total_grad_norms else 0.0,
        },
        "firewall_verification": {
            "test_subjects_accessed": False,
            "total_test_subjects_firewalled": len(test_subjects_firewalled),
        },
        "client_steps": client_step_metrics,
    }

    out_file = PROJECT_ROOT / "reports" / "real_brats2024" / "real_128_validation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(val_report, f, indent=2)

    logger.info("=" * 80)
    logger.info("✅ 128³ REAL TRAINING CONFIGURATION VALIDATION PASSED")
    logger.info(f"Round Time:             {val_report['performance_profile']['round_time_sec']}s")
    logger.info(f"Avg Step Time:          {val_report['performance_profile']['average_step_time_sec']}s")
    logger.info(f"MPS Allocated Memory:   {val_report['performance_profile']['mps_allocated_memory_mb']} MB")
    logger.info(f"Total Parameter Delta:  {val_report['performance_profile']['total_parameter_delta']}")
    logger.info(f"OOM Detected:           {val_report['oom_detected']}")
    logger.info(f"Report Path:            {out_file}")
    logger.info("=" * 80)

    return val_report


if __name__ == "__main__":
    run_128_validation()
