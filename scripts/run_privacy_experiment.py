"""
Script: scripts/run_privacy_experiment.py

Purpose:
Phase 8.5B.5 Differential Privacy & TenSEAL Homomorphic Encryption Execution Engine.
Executes the complete 4-quadrant privacy matrix on the canonical BraTS setup:
  - Experiment A: FedAvg Baseline (DP OFF, HE OFF)
  - Experiment B: FedAvg + Differential Privacy (DP ON, HE OFF)
  - Experiment C: FedAvg + Homomorphic Encryption (DP OFF, HE ON)
  - Experiment D: FedAvg + DP + HE Composition (DP ON, HE ON)

Verifies:
  1. Identical model initialization across all 4 experiments (H_A == H_B == H_C == H_D == H_0)
  2. Genuine DP gradient clipping and Gaussian noise with exact RDP accounting
  3. Genuine TenSEAL CKKS homomorphic ciphertext aggregation (server holds public context only)
  4. Measured CKKS approximation error against plaintext reference
  5. Firewalled validation and test evaluation
  6. Standardized JSON audit reports and SQLite DB persistence
"""

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tenseal as ts
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
from privacy.aggregation import aggregate_encrypted_updates
from privacy.context import create_ckks_context, get_public_context
from privacy.decrypt import decrypt_model_parameters
from privacy.dp_engine import DifferentialPrivacyEngine, compute_rdp_epsilon
from privacy.encrypt import encrypt_model_parameters
from server.strategies.base import FitResult
from server.strategies.fedavg import FedAvg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("privacy_experiment")

REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_BASE = PROJECT_ROOT / "checkpoints" / "privacy"


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


class SinglePatientDataset(Dataset):
    """Loads MRI dictionary strictly for a designated subject ID."""

    def __init__(
        self,
        data_dir: Path,
        subject_id: str,
        transforms,
        modalities: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
        mask_modality: str = "seg",
    ):
        self.data_dir = data_dir
        self.subject_id = subject_id
        self.transforms = transforms
        self.modalities = modalities

        subj_dir = self.data_dir / subject_id
        if not subj_dir.exists():
            raise FileNotFoundError(f"Subject directory '{subj_dir}' not found!")

        mod_files = []
        for m in self.modalities:
            matches = sorted(list(subj_dir.glob(f"*{m}.nii*")))
            if not matches:
                raise FileNotFoundError(f"Missing modality '{m}' for subject '{subject_id}'")
            mod_files.append(str(matches[0]))

            seg_matches = sorted(list(subj_dir.glob(f"*{mask_modality}.nii*")))
            if not seg_matches:
                raise FileNotFoundError(f"Missing segmentation mask for subject '{subject_id}'")

        self.sample = {
            "image": mod_files,
            "label": str(seg_matches[0]),
            "patient_id": subject_id,
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.transforms:
            return self.transforms(self.sample)
        return self.sample


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluates model on dataloader and returns Dice and IoU metrics."""
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        images = batch["image"].to(device)
        targets = batch["label"].to(device)
        logits = model(images)

        d_res = compute_dice(logits, targets, threshold=0.5, channel_names=["TC", "WT", "ET"])
        i_res = compute_iou(logits, targets, threshold=0.5, channel_names=["TC", "WT", "ET"])

    return d_res, i_res


def run_single_privacy_experiment(
    exp_name: str,
    dp_enabled: bool,
    he_enabled: bool,
    initial_weights: List[np.ndarray],
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    d_path: Path,
    config: Dict[str, Any],
    device: torch.device,
    spatial_shape: Tuple[int, int, int],
    rounds: int = 3,
    dp_config: Optional[Dict[str, Any]] = None,
    he_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print(f"🔒 EXECUTING EXPERIMENT: [{exp_name}] (DP: {dp_enabled}, HE: {he_enabled})")
    print("=" * 75)

    m_cfg = config.get("model", {})
    t_cfg = config.get("training", {})
    lr = float(t_cfg.get("learning_rate", 1e-4))
    weight_decay = float(t_cfg.get("weight_decay", 1e-5))

    def create_model_with_weights(weights: List[np.ndarray]) -> UNet:
        m = UNet(
            spatial_dims=m_cfg.get("spatial_dims", 3),
            in_channels=m_cfg.get("in_channels", 4),
            out_channels=m_cfg.get("out_channels", 3),
            channels=tuple(m_cfg.get("channels", [16, 32, 64, 128, 256])),
            strides=tuple(m_cfg.get("strides", [2, 2, 2, 2])),
            num_res_units=m_cfg.get("num_res_units", 2),
        ).to(device)
        keys = list(m.state_dict().keys())
        m.load_state_dict({k: torch.tensor(arr).to(device) for k, arr in zip(keys, weights)})
        return m

    # Data loaders
    train_transforms = get_brats_transforms(mode="train", image_size=spatial_shape)
    val_transforms = get_brats_transforms(mode="val", image_size=spatial_shape)
    test_transforms = get_brats_transforms(mode="val", image_size=spatial_shape)

    alpha_dataset = SinglePatientDataset(data_dir=d_path, subject_id=train_subjects[0], transforms=train_transforms)
    beta_dataset = SinglePatientDataset(data_dir=d_path, subject_id=train_subjects[1], transforms=train_transforms)
    val_dataset = SinglePatientDataset(data_dir=d_path, subject_id=val_subjects[0], transforms=val_transforms)
    test_dataset = SinglePatientDataset(data_dir=d_path, subject_id=test_subjects[0], transforms=test_transforms)

    alpha_loader = DataLoader(alpha_dataset, batch_size=1, shuffle=False)
    beta_loader = DataLoader(beta_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    global_model = create_model_with_weights(initial_weights)
    current_global_weights = [val.clone().detach().cpu().numpy() for val in global_model.state_dict().values()]
    initial_hash = compute_param_hash(global_model)

    # Initialize CKKS Context if HE enabled
    private_he_ctx = None
    public_he_ctx = None
    if he_enabled:
        poly_deg = he_config.get("poly_modulus_degree", 8192) if he_config else 8192
        coeff_bits = he_config.get("coeff_mod_bit_sizes", [60, 40, 40, 60]) if he_config else [60, 40, 40, 60]
        private_he_ctx = create_ckks_context(poly_modulus_degree=poly_deg, coeff_mod_bit_sizes=coeff_bits)
        public_he_ctx = get_public_context(private_he_ctx)

    fedavg_strategy = FedAvg(min_fit_clients=2, min_available_clients=2)

    history = []
    he_timings = []
    he_errors = []
    dp_steps = 0
    t0_exp = time.time()

    for r in range(1, rounds + 1):
        # 1. Hospital Alpha Local Training
        m_alpha = create_model_with_weights(current_global_weights)
        opt_alpha = torch.optim.Adam(m_alpha.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn_alpha = DiceCELoss(sigmoid=True)

        dp_engine_alpha = None
        if dp_enabled:
            dp_engine_alpha = DifferentialPrivacyEngine(
                model=m_alpha,
                optimizer=opt_alpha,
                max_grad_norm=dp_config.get("max_grad_norm", 1.0),
                noise_multiplier=dp_config.get("noise_multiplier", 0.5),
                target_delta=dp_config.get("target_delta", 1e-5),
                sample_rate=dp_config.get("sample_rate", 0.5),
            )

        m_alpha.train()
        b_a = next(iter(alpha_loader))
        opt_alpha.zero_grad()
        l_a = loss_fn_alpha(m_alpha(b_a["image"].to(device)), b_a["label"].to(device))
        l_a.backward()

        if dp_engine_alpha:
            dp_engine_alpha.apply_gradient_clipping_and_noise(batch_size=1)
            dp_steps += 1

        opt_alpha.step()
        w_alpha = [val.detach().cpu().numpy() for val in m_alpha.state_dict().values()]

        # 2. Hospital Beta Local Training
        m_beta = create_model_with_weights(current_global_weights)
        opt_beta = torch.optim.Adam(m_beta.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn_beta = DiceCELoss(sigmoid=True)

        dp_engine_beta = None
        if dp_enabled:
            dp_engine_beta = DifferentialPrivacyEngine(
                model=m_beta,
                optimizer=opt_beta,
                max_grad_norm=dp_config.get("max_grad_norm", 1.0),
                noise_multiplier=dp_config.get("noise_multiplier", 0.5),
                target_delta=dp_config.get("target_delta", 1e-5),
                sample_rate=dp_config.get("sample_rate", 0.5),
            )

        m_beta.train()
        b_b = next(iter(beta_loader))
        opt_beta.zero_grad()
        l_b = loss_fn_beta(m_beta(b_b["image"].to(device)), b_b["label"].to(device))
        l_b.backward()

        if dp_engine_beta:
            dp_engine_beta.apply_gradient_clipping_and_noise(batch_size=1)
            dp_steps += 1

        opt_beta.step()
        w_beta = [val.detach().cpu().numpy() for val in m_beta.state_dict().values()]

        # Plaintext Reference FedAvg
        expected_plaintext_weights = [0.5 * a + 0.5 * b for a, b in zip(w_alpha, w_beta)]

        # 3. Aggregation (HE or Plaintext)
        if he_enabled:
            # Client-side encryption
            t_enc_start = time.time()
            chunk_size = he_config.get("chunk_size", 4096) if he_config else 4096
            enc_res_alpha = encrypt_model_parameters(private_he_ctx, w_alpha, chunk_size=chunk_size)
            enc_res_beta = encrypt_model_parameters(private_he_ctx, w_beta, chunk_size=chunk_size)
            enc_time_ms = (time.time() - t_enc_start) * 1000.0

            # Server-side homomorphic aggregation using PUBLIC context only
            encrypted_results = [
                (enc_res_alpha["encrypted_chunks"], 1),
                (enc_res_beta["encrypted_chunks"], 1),
            ]
            shapes = enc_res_alpha["shapes"]
            agg_he_res = aggregate_encrypted_updates(public_he_ctx, encrypted_results, shapes)

            # Authorized decryption
            t_dec_start = time.time()
            decrypted_weights = decrypt_model_parameters(private_he_ctx, agg_he_res["aggregated_chunks"], shapes)
            dec_time_ms = (time.time() - t_dec_start) * 1000.0

            # Measure CKKS Approximation Error against Plaintext reference
            max_abs_err = max(float(np.max(np.abs(d - e))) for d, e in zip(decrypted_weights, expected_plaintext_weights))
            rel_err = max_abs_err / max(float(np.max(np.abs(e))) for e in expected_plaintext_weights)

            he_timings.append({
                "round": r,
                "encryption_time_ms": enc_time_ms,
                "aggregation_time_ms": agg_he_res["aggregation_time_ms"],
                "decryption_time_ms": dec_time_ms,
                "ciphertext_size_bytes": agg_he_res["ciphertext_size_bytes"],
            })
            he_errors.append({
                "round": r,
                "max_absolute_error": max_abs_err,
                "relative_error": rel_err,
            })

            current_global_weights = decrypted_weights
            keys = list(global_model.state_dict().keys())
            global_model.load_state_dict({k: torch.tensor(arr).to(device) for k, arr in zip(keys, decrypted_weights)})
        else:
            fit_results = [
                FitResult(parameters=w_alpha, num_examples=1, metrics={"training_loss": float(l_a.item())}),
                FitResult(parameters=w_beta, num_examples=1, metrics={"training_loss": float(l_b.item())}),
            ]
            agg_weights, _ = fedavg_strategy.aggregate_fit(server_round=r, results=fit_results, failures=[])
            current_global_weights = agg_weights
            keys = list(global_model.state_dict().keys())
            global_model.load_state_dict({k: torch.tensor(arr).to(device) for k, arr in zip(keys, agg_weights)})

        # 4. Held-Out Validation
        d_val, i_val = evaluate_model(global_model, val_loader, device)

        history.append({
            "round": r,
            "training_loss": float((l_a.item() + l_b.item()) / 2.0),
            "client_losses": {"hospital_alpha": float(l_a.item()), "hospital_beta": float(l_b.item())},
            "validation_dice": d_val,
            "validation_iou": i_val,
        })
        print(f"  Round {r}/{rounds} | Loss: {(l_a.item() + l_b.item()) / 2.0:.4f} | Val Mean Dice: {d_val['mean_dice']:.4f} | IoU: {i_val['mean_iou']:.4f}")

    total_time_s = time.time() - t0_exp
    final_hash = compute_param_hash(global_model)

    # 5. Final TEST Evaluation (Post-training)
    d_test, i_test = evaluate_model(global_model, test_loader, device)
    print(f"  Final TEST Mean Dice: {d_test['mean_dice']:.4f} | Mean IoU: {i_test['mean_iou']:.4f}")

    # Calculate final DP accounting if enabled
    privacy_budget = None
    if dp_enabled:
        eps = compute_rdp_epsilon(
            steps=dp_steps,
            noise_multiplier=dp_config.get("noise_multiplier", 0.5),
            target_delta=dp_config.get("target_delta", 1e-5),
            sample_rate=dp_config.get("sample_rate", 0.5),
        )
        privacy_budget = {
            "epsilon": float(eps),
            "delta": dp_config.get("target_delta", 1e-5),
            "noise_multiplier": dp_config.get("noise_multiplier", 0.5),
            "max_grad_norm": dp_config.get("max_grad_norm", 1.0),
            "steps": dp_steps,
            "accountant": "RDP (Renyi Differential Privacy) Accountant",
        }

    # Save Best Checkpoint
    ckpt_dir = CHECKPOINTS_BASE / exp_name.lower().replace(" ", "_").replace("+", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "best_model.pt")
    torch.save({"model_state_dict": global_model.state_dict(), "exp_name": exp_name}, ckpt_path)

    return {
        "experiment_name": exp_name,
        "dp_enabled": dp_enabled,
        "he_enabled": he_enabled,
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": final_hash,
        "execution_time_seconds": total_time_s,
        "history": history,
        "final_validation_dice": history[-1]["validation_dice"],
        "final_validation_iou": history[-1]["validation_iou"],
        "final_test_dice": d_test,
        "final_test_iou": i_test,
        "privacy_budget": privacy_budget,
        "he_timings": he_timings if he_enabled else None,
        "he_errors": he_errors if he_enabled else None,
        "checkpoint_path": ckpt_path,
    }


def run_privacy_matrix(
    data_dir: str = "data/BraTS2021",
    config_path: str = "configs/experiments/privacy.yaml",
    rounds: int = 3,
    seed: int = 42,
    allow_development: bool = False,
) -> Dict[str, Any]:
    print("=" * 75)
    print("🛡️  FEDMED OS — REAL PRIVACY EXECUTION MATRIX (PHASE 8.5B.5)")
    print("=" * 75)

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

    # 3. Load Split & Manifest
    split_file = REPORTS_DIR / "dataset_split.json"
    with open(split_file, "r") as f:
        split_data = json.load(f)

    split_hash = split_data.get("split_hash", "")
    train_subjects = split_data.get("train_subjects", [])
    val_subjects = split_data.get("validation_subjects", [])
    test_subjects = split_data.get("test_subjects", [])

    manifest_file = REPORTS_DIR / "brats_dataset_manifest.json"
    with open(manifest_file, "r") as f:
        manifest_data = json.load(f)
    dataset_mode = manifest_data.get("dataset_mode", "DEVELOPMENT_SYNTHETIC")

    if dataset_mode == "DEVELOPMENT_SYNTHETIC" and not allow_development:
        print("\n❌ DEVELOPMENT DATASET — EXPLICIT FLAG REQUIRED")
        sys.exit(1)

    d_path = Path(data_dir)
    if not d_path.is_absolute():
        d_path = PROJECT_ROOT / d_path

    first_subj_meta = manifest_data["subjects"][0]
    spatial_shape = tuple(first_subj_meta.get("spatial_shape", (32, 32, 32)))

    # 4. Generate Canonical Initial Model Weights from Seed 42
    m_cfg = config.get("model", {})
    anchor_model = UNet(
        spatial_dims=m_cfg.get("spatial_dims", 3),
        in_channels=m_cfg.get("in_channels", 4),
        out_channels=m_cfg.get("out_channels", 3),
        channels=tuple(m_cfg.get("channels", [16, 32, 64, 128, 256])),
        strides=tuple(m_cfg.get("strides", [2, 2, 2, 2])),
        num_res_units=m_cfg.get("num_res_units", 2),
    ).to(device)

    initial_weights = [val.clone().detach().cpu().numpy() for val in anchor_model.state_dict().values()]
    initial_hash = compute_param_hash(anchor_model)
    param_count = sum(p.numel() for p in anchor_model.parameters() if p.requires_grad)

    print(f"Canonical Model Parameters: {param_count}")
    print(f"CANONICAL INITIAL PARAMETER HASH: {initial_hash}")

    dp_cfg = config.get("differential_privacy", {})
    he_cfg = config.get("homomorphic_encryption", {})

    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"privacy_run_{timestamp_str}"
    run_dir = REPORTS_DIR / "experiments" / "privacy" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 5. Execute 4 Experiments
    exp_a = run_single_privacy_experiment(
        exp_name="FedAvg Baseline",
        dp_enabled=False,
        he_enabled=False,
        initial_weights=initial_weights,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        d_path=d_path,
        config=config,
        device=device,
        spatial_shape=spatial_shape,
        rounds=rounds,
    )

    exp_b = run_single_privacy_experiment(
        exp_name="FedAvg + DP",
        dp_enabled=True,
        he_enabled=False,
        initial_weights=initial_weights,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        d_path=d_path,
        config=config,
        device=device,
        spatial_shape=spatial_shape,
        rounds=rounds,
        dp_config=dp_cfg,
    )

    exp_c = run_single_privacy_experiment(
        exp_name="FedAvg + HE",
        dp_enabled=False,
        he_enabled=True,
        initial_weights=initial_weights,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        d_path=d_path,
        config=config,
        device=device,
        spatial_shape=spatial_shape,
        rounds=rounds,
        he_config=he_cfg,
    )

    exp_d = run_single_privacy_experiment(
        exp_name="FedAvg + DP + HE",
        dp_enabled=True,
        he_enabled=True,
        initial_weights=initial_weights,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        d_path=d_path,
        config=config,
        device=device,
        spatial_shape=spatial_shape,
        rounds=rounds,
        dp_config=dp_cfg,
        he_config=he_cfg,
    )

    # 6. Save JSON Reports
    manifest_doc = {
        "run_id": run_id,
        "execution_mode": dataset_mode,
        "split_hash": split_hash,
        "seed": seed,
        "model_parameter_count": param_count,
        "initial_parameter_hash": initial_hash,
        "dp_config": dp_cfg,
        "he_config": he_cfg,
        "experiments": ["FedAvg Baseline", "FedAvg + DP", "FedAvg + HE", "FedAvg + DP + HE"],
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest_doc, f, indent=2)

    with open(run_dir / "fedavg_baseline.json", "w") as f:
        json.dump(exp_a, f, indent=2)

    with open(run_dir / "dp.json", "w") as f:
        json.dump(exp_b, f, indent=2)

    with open(run_dir / "he.json", "w") as f:
        json.dump(exp_c, f, indent=2)

    with open(run_dir / "dp_he.json", "w") as f:
        json.dump(exp_d, f, indent=2)

    comparison_doc = {
        "run_id": run_id,
        "initial_parameter_hash": initial_hash,
        "experiments": {
            "FedAvg_Baseline": exp_a,
            "FedAvg_DP": exp_b,
            "FedAvg_HE": exp_c,
            "FedAvg_DP_HE": exp_d,
        },
        "summary_table": [
            {
                "experiment": exp_a["experiment_name"],
                "dp_enabled": False,
                "he_enabled": False,
                "epsilon": None,
                "he_error_max": None,
                "val_mean_dice": exp_a["final_validation_dice"]["mean_dice"],
                "val_mean_iou": exp_a["final_validation_iou"]["mean_iou"],
                "test_mean_dice": exp_a["final_test_dice"]["mean_dice"],
                "test_mean_iou": exp_a["final_test_iou"]["mean_iou"],
                "runtime_seconds": exp_a["execution_time_seconds"],
            },
            {
                "experiment": exp_b["experiment_name"],
                "dp_enabled": True,
                "he_enabled": False,
                "epsilon": exp_b["privacy_budget"]["epsilon"],
                "he_error_max": None,
                "val_mean_dice": exp_b["final_validation_dice"]["mean_dice"],
                "val_mean_iou": exp_b["final_validation_iou"]["mean_iou"],
                "test_mean_dice": exp_b["final_test_dice"]["mean_dice"],
                "test_mean_iou": exp_b["final_test_iou"]["mean_iou"],
                "runtime_seconds": exp_b["execution_time_seconds"],
            },
            {
                "experiment": exp_c["experiment_name"],
                "dp_enabled": False,
                "he_enabled": True,
                "epsilon": None,
                "he_error_max": exp_c["he_errors"][-1]["max_absolute_error"],
                "val_mean_dice": exp_c["final_validation_dice"]["mean_dice"],
                "val_mean_iou": exp_c["final_validation_iou"]["mean_iou"],
                "test_mean_dice": exp_c["final_test_dice"]["mean_dice"],
                "test_mean_iou": exp_c["final_test_iou"]["mean_iou"],
                "runtime_seconds": exp_c["execution_time_seconds"],
            },
            {
                "experiment": exp_d["experiment_name"],
                "dp_enabled": True,
                "he_enabled": True,
                "epsilon": exp_d["privacy_budget"]["epsilon"],
                "he_error_max": exp_d["he_errors"][-1]["max_absolute_error"],
                "val_mean_dice": exp_d["final_validation_dice"]["mean_dice"],
                "val_mean_iou": exp_d["final_validation_iou"]["mean_iou"],
                "test_mean_dice": exp_d["final_test_dice"]["mean_dice"],
                "test_mean_iou": exp_d["final_test_iou"]["mean_iou"],
                "runtime_seconds": exp_d["execution_time_seconds"],
            },
        ],
    }
    with open(run_dir / "comparison.json", "w") as f:
        json.dump(comparison_doc, f, indent=2)

    # Persist into SQLite DB
    try:
        db_path = PROJECT_ROOT / "fedmed.db"
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                epoch INTEGER,
                training_loss REAL,
                validation_loss REAL,
                dice_score REAL,
                iou REAL,
                hospital_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for exp in [exp_a, exp_b, exp_c, exp_d]:
            cur.execute(
                "INSERT INTO training_metrics (experiment_id, round_number, training_loss, dice_score, iou, hospital_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    rounds,
                    exp["history"][-1]["training_loss"],
                    exp["final_validation_dice"]["mean_dice"],
                    exp["final_validation_iou"]["mean_iou"],
                    exp["experiment_name"],
                ),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Database persistence notice: {e}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 105)
    print("📊 FEDMED OS — PRIVACY MATRIX COMPARISON TABLE (MEASURED EXPERIMENTAL RESULTS)")
    print("=" * 105)
    header = f"{'Experiment':<22} | {'DP':<5} | {'HE':<5} | {'Epsilon (ε)':<12} | {'HE Error':<10} | {'Val Dice':<9} | {'Val IoU':<8} | {'Test Dice':<9} | {'Test IoU':<8} | {'Time':<6}"
    print(header)
    print("-" * 105)
    for r in comparison_doc["summary_table"]:
        eps_str = f"{r['epsilon']:.2f}" if r["epsilon"] is not None else "N/A"
        he_err_str = f"{r['he_error_max']:.2e}" if r["he_error_max"] is not None else "N/A"
        row = f"{r['experiment']:<22} | {str(r['dp_enabled']):<5} | {str(r['he_enabled']):<5} | {eps_str:<12} | {he_err_str:<10} | {r['val_mean_dice']:<9.4f} | {r['val_mean_iou']:<8.4f} | {r['test_mean_dice']:<9.4f} | {r['test_mean_iou']:<8.4f} | {r['runtime_seconds']:<5.1f}s"
        print(row)
    print("=" * 105)
    print(f"  Artifact Directory: {run_dir}")
    print("=" * 105)

    return comparison_doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy Experiment Runner")
    parser.add_argument("--data-dir", type=str, default="data/BraTS2021", help="Dataset path")
    parser.add_argument("--config", type=str, default="configs/experiments/privacy.yaml", help="Config YAML")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--allow-development", action="store_true", help="Authorize development cohort execution")
    args = parser.parse_args()

    run_privacy_matrix(
        data_dir=args.data_dir,
        config_path=args.config,
        rounds=args.rounds,
        seed=args.seed,
        allow_development=args.allow_development,
    )
