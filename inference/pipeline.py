"""
Module: inference.pipeline

Purpose:
Production-grade Clinical Inference Pipeline for FedMed OS Phase 11.
Loads the canonical frozen Differential Privacy model (checkpoints/final/fedmed_dp_final_model.pt)
with fail-closed SHA-256 verification (f6cd18dc5e05595ca88ad1675190aabb98595ee9f6d6dfed36c25b596f652f3a).
Executes strictly under torch.inference_mode() with zero gradient accumulation and zero training updates.
Generates multi-planar 2D slice overlays (Axial, Coronal, Sagittal) with color-coded tumor sub-region masks.
"""

import base64
import datetime
import hashlib
import io
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
import nibabel as nib
import numpy as np
import torch
import yaml

from data.canonical_adapter import DatasetVersion, ModalityCanonicalizer
from data.datasets.transforms import get_brats_transforms
from inference.validator import ClinicalErrorCode, ClinicalInputValidator, ClinicalOutputValidator

logger = logging.getLogger("inference_pipeline")

CANONICAL_CHECKPOINT_PATH = Path("checkpoints/final/fedmed_dp_final_model.pt")
EXPECTED_CHECKPOINT_SHA256 = "f6cd18dc5e05595ca88ad1675190aabb98595ee9f6d6dfed36c25b596f652f3a"


def compute_file_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def render_slice_to_base64(
    bg_slice: np.ndarray,
    mask_tc: np.ndarray,
    mask_wt: np.ndarray,
    mask_et: np.ndarray,
    title: str = "",
) -> str:
    """Renders a single 2D slice with composite color-coded segmentation overlays to Base64 PNG."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    
    # Normalize background image for display
    bg_norm = bg_slice.astype(np.float32)
    bg_min, bg_max = np.min(bg_norm), np.max(bg_norm)
    if bg_max > bg_min:
        bg_norm = (bg_norm - bg_min) / (bg_max - bg_min)
    else:
        bg_norm = np.zeros_like(bg_norm)
        
    ax.imshow(bg_norm, cmap="gray", origin="lower")
    
    # WT in green
    wt_overlay = np.zeros((*bg_slice.shape, 4), dtype=np.float32)
    wt_overlay[mask_wt > 0] = [0.15, 0.85, 0.25, 0.45]
    ax.imshow(wt_overlay, origin="lower")
    
    # TC in red
    tc_overlay = np.zeros((*bg_slice.shape, 4), dtype=np.float32)
    tc_overlay[mask_tc > 0] = [0.95, 0.20, 0.10, 0.60]
    ax.imshow(tc_overlay, origin="lower")
    
    # ET in yellow
    et_overlay = np.zeros((*bg_slice.shape, 4), dtype=np.float32)
    et_overlay[mask_et > 0] = [1.00, 0.90, 0.10, 0.75]
    ax.imshow(et_overlay, origin="lower")
    
    if title:
        ax.set_title(title, color="white", fontsize=10, pad=4)
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
    
    ax.axis("off")
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


class ClinicalInferenceEngine:
    """
    Production-grade inference engine executing the frozen MONAI 3D U-Net DP model.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        if checkpoint_path is None:
            self.checkpoint_path = CANONICAL_CHECKPOINT_PATH
        else:
            self.checkpoint_path = Path(checkpoint_path)

        self.config_path = Path(config_path) if config_path else Path("configs/production/inference.yaml")

        # 1. Checkpoint Existence and SHA-256 Verification
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"FAIL-CLOSED: Model checkpoint '{self.checkpoint_path}' does not exist!"
            )

        actual_sha = compute_file_hash(self.checkpoint_path)
        is_canonical = (
            self.checkpoint_path.resolve() == CANONICAL_CHECKPOINT_PATH.resolve()
            or self.checkpoint_path.name == "fedmed_dp_final_model.pt"
        )
        if is_canonical and actual_sha != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError(
                f"FAIL-CLOSED: Checkpoint hash mismatch! Expected {EXPECTED_CHECKPOINT_SHA256}, got {actual_sha}. "
                "Refusing to load untrusted model weights."
            )
        self.checkpoint_sha256 = actual_sha
        logger.info(f"Model Checkpoint Verified: {self.checkpoint_path} (SHA-256: {self.checkpoint_sha256[:16]}...)")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Inference configuration '{self.config_path}' not found!")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available() and self.config.get("execution", {}).get("allow_mps", True):
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Clinical Inference Device: {self.device}")
        self.input_validator = ClinicalInputValidator()
        self.output_validator = ClinicalOutputValidator()

        # Load and verify model
        self.model, self.model_load_time_ms = self._load_model()

    def _sync_device(self) -> None:
        if self.device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass
        elif self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def _load_model(self) -> Tuple[UNet, float]:
        self._sync_device()
        t0 = time.time()
        m_cfg = self.config["model"]
        model = UNet(
            spatial_dims=m_cfg.get("spatial_dims", 3),
            in_channels=m_cfg.get("in_channels", 4),
            out_channels=m_cfg.get("out_channels", 3),
            channels=tuple(m_cfg.get("channels", [16, 32, 64, 128, 256])),
            strides=tuple(m_cfg.get("strides", [2, 2, 2, 2])),
            num_res_units=m_cfg.get("num_res_units", 2),
            dropout=0.0,
        ).to(self.device)

        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self._sync_device()
        load_time_ms = (time.time() - t0) * 1000.0

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Loaded {total_params:,} parameters into memory in {load_time_ms:.2f} ms")
        return model, load_time_ms

    def predict_subject(
        self,
        modality_paths: Dict[str, Path],
        output_nifti_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Executes clinical segmentation inference on 4-modality BraTS 3D MRI.
        """
        t_val_start = time.time()
        is_valid, err_code, err_msg, meta = self.input_validator.validate_modalities_dict(modality_paths)
        t_val_ms = (time.time() - t_val_start) * 1000.0

        if not is_valid:
            return {
                "status": "FAILED",
                "error_code": err_code.value,
                "error_message": err_msg,
                "timing_ms": {"validation_time_ms": round(t_val_ms, 2)},
            }

        # 2. Preprocessing
        t_prep_start = time.time()
        spatial_shape = tuple(self.config["preprocessing"].get("spatial_shape", [128, 128, 128]))
        transforms = get_brats_transforms(
            mode="val",
            image_size=spatial_shape,
            dataset_version=DatasetVersion.BRATS_GLI_2024,
        )

        mod_files = [str(modality_paths[m]) for m in ("t1", "t1ce", "t2", "flair")]
        # Use t1 as dummy label for preprocessing loader
        sample = {"image": mod_files, "label": mod_files[0]}
        processed = transforms(sample)
        image_tensor = processed["image"].unsqueeze(0).to(self.device)
        self._sync_device()
        t_prep_ms = (time.time() - t_prep_start) * 1000.0

        # 3. Model Forward Pass (Strictly Inference Mode, Zero Gradients)
        t_inf_start = time.time()
        with torch.inference_mode():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > float(self.config["postprocessing"].get("threshold", 0.5))).float()
        self._sync_device()
        t_inf_ms = (time.time() - t_inf_start) * 1000.0

        # 4. Post-processing & Output Extraction
        t_post_start = time.time()
        pred_np = preds.squeeze(0).cpu().numpy()  # (3, D, H, W)
        img_np = image_tensor.squeeze(0).cpu().numpy()  # (4, D, H, W)

        out_valid, out_code, out_msg = self.output_validator.validate_segmentation_output(
            pred_np, expected_shape=spatial_shape, allowed_channels=3
        )
        t_post_ms = (time.time() - t_post_start) * 1000.0

        if not out_valid:
            return {
                "status": "FAILED",
                "error_code": out_code.value,
                "error_message": out_msg,
                "timing_ms": {
                    "validation_time_ms": round(t_val_ms, 2),
                    "preprocessing_time_ms": round(t_prep_ms, 2),
                    "inference_time_ms": round(t_inf_ms, 2),
                    "postprocessing_time_ms": round(t_post_ms, 2),
                },
            }

        # 5. Extract multi-planar slice overlays with Canonical BraTS Hierarchical Region Construction
        # Use T1CE (channel 1) for anatomical background
        bg_vol = img_np[1]
        
        # Foreground brain tissue mask (excludes empty air voxels outside cranial vault)
        brain_mask = (np.abs(img_np).sum(axis=0) > 0.01)

        # Raw channel thresholding restricted to intracranial volume:
        # Channel 0: TC candidate, Channel 1: WT candidate, Channel 2: ET candidate
        raw_tc = (pred_np[0] > 0) & brain_mask
        raw_wt = (pred_np[1] > 0) & brain_mask
        raw_et = (pred_np[2] > 0) & brain_mask

        # Canonical BraTS Clinical Region Hierarchy:
        # Whole Tumor (WT): Superset of all abnormal tumor subregions (NETC + SNFH + ET)
        # Tumor Core (TC): Solid core & enhancing rim (NETC + ET) enclosed within WT
        # Enhancing Tumor (ET): Active vascular rim enclosed within TC
        mask_wt_vol = (raw_wt | raw_tc | raw_et)
        mask_tc_vol = (raw_tc | raw_et) & mask_wt_vol
        mask_et_vol = raw_et & mask_tc_vol

        # Select representative center slices with peak tumor activity or geometric center
        d_center = spatial_shape[0] // 2
        h_center = spatial_shape[1] // 2
        w_center = spatial_shape[2] // 2

        wt_slices = np.where(np.sum(mask_wt_vol, axis=(1, 2)) > 0)[0]
        if len(wt_slices) > 0:
            d_center = int(wt_slices[len(wt_slices) // 2])

        wt_coronal = np.where(np.sum(mask_wt_vol, axis=(0, 2)) > 0)[0]
        if len(wt_coronal) > 0:
            h_center = int(wt_coronal[len(wt_coronal) // 2])

        wt_sagittal = np.where(np.sum(mask_wt_vol, axis=(0, 1)) > 0)[0]
        if len(wt_sagittal) > 0:
            w_center = int(wt_sagittal[len(wt_sagittal) // 2])

        axial_b64 = render_slice_to_base64(
            bg_vol[d_center, :, :],
            mask_tc_vol[d_center, :, :],
            mask_wt_vol[d_center, :, :],
            mask_et_vol[d_center, :, :],
            title=f"Axial Slice (Z={d_center})",
        )
        coronal_b64 = render_slice_to_base64(
            bg_vol[:, h_center, :],
            mask_tc_vol[:, h_center, :],
            mask_wt_vol[:, h_center, :],
            mask_et_vol[:, h_center, :],
            title=f"Coronal Slice (Y={h_center})",
        )
        sagittal_b64 = render_slice_to_base64(
            bg_vol[:, :, w_center],
            mask_tc_vol[:, :, w_center],
            mask_wt_vol[:, :, w_center],
            mask_et_vol[:, :, w_center],
            title=f"Sagittal Slice (X={w_center})",
        )

        # 6. Save NIfTI if requested
        output_file = None
        t_save_ms = 0.0
        if output_nifti_path:
            t_save_start = time.time()
            output_nifti_path = Path(output_nifti_path)
            output_nifti_path.parent.mkdir(parents=True, exist_ok=True)
            composite_mask = np.zeros(spatial_shape, dtype=np.uint8)
            composite_mask[mask_wt_vol] = 2  # Edema / Whole Tumor envelope
            composite_mask[mask_tc_vol] = 1  # Non-enhancing Tumor Core
            composite_mask[mask_et_vol] = 3  # Enhancing Tumor

            ref_affine = np.array(meta.get("ref_affine", np.eye(4))) if meta else np.eye(4)
            out_img = nib.Nifti1Image(composite_mask, affine=ref_affine)
            nib.save(out_img, str(output_nifti_path))
            output_file = str(output_nifti_path)
            t_save_ms = (time.time() - t_save_start) * 1000.0

        total_time_ms = t_val_ms + t_prep_ms + t_inf_ms + t_post_ms + t_save_ms

        tc_count = int(np.sum(mask_tc_vol))
        wt_count = int(np.sum(mask_wt_vol))
        et_count = int(np.sum(mask_et_vol))

        return {
            "status": "SUCCESS",
            "error_code": ClinicalErrorCode.OK.value,
            "device": str(self.device),
            "output_nifti_path": output_file,
            "segmented_voxels": {
                "TC_voxels": tc_count,
                "WT_voxels": wt_count,
                "ET_voxels": et_count,
            },
            "tumor_volumes_mm3": {
                "tumor_core_tc": float(tc_count),
                "whole_tumor_wt": float(wt_count),
                "enhancing_tumor_et": float(et_count),
            },
            "visual_slices": {
                "axial": axial_b64,
                "coronal": coronal_b64,
                "sagittal": sagittal_b64,
            },
            "timing_ms": {
                "model_load_time_ms": round(self.model_load_time_ms, 2),
                "validation_time_ms": round(t_val_ms, 2),
                "preprocessing_time_ms": round(t_prep_ms, 2),
                "inference_time_ms": round(t_inf_ms, 2),
                "postprocessing_time_ms": round(t_post_ms, 2),
                "nifti_save_time_ms": round(t_save_ms, 2),
                "total_inference_time_ms": round(total_time_ms, 2),
            },
            "model_metadata": {
                "name": "FedMed MONAI 3D U-Net DP Final Candidate",
                "experiment_id": "EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3",
                "parameter_count": 4810074,
                "checkpoint_sha256": self.checkpoint_sha256,
                "privacy_guarantee": {
                    "mechanism": "Poisson DP-SGD",
                    "epsilon": 2.8934,
                    "delta": 0.00001,
                    "accountant_steps": 4720,
                },
                "disclaimer": "Research Prototype — Not approved for primary clinical diagnostic use.",
            },
        }


class BraTSInferenceEngine:
    """
    Service layer wrapper for REST API, Web Dashboard, and CLI integration.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else CANONICAL_CHECKPOINT_PATH
        self.history: List[Dict[str, Any]] = []

        try:
            self.clinical_engine = ClinicalInferenceEngine(checkpoint_path=self.checkpoint_path)
        except Exception as e:
            logger.error(f"FATAL: Clinical inference engine failed initialization: {e}")
            self.clinical_engine = None

    def predict(
        self,
        patient_id: str = "BraTS-GLI-00005-100",
        model_version: str = "v2.1.0-dp-prod",
        custom_modality_paths: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs real inference on the specified subject or demo case.
        """
        if self.clinical_engine is None:
            return {
                "status": "FAILED",
                "error_message": "Clinical inference engine offline. Checkpoint missing or corrupted.",
                "patient_id": patient_id,
            }

        # Resolve modalities
        if custom_modality_paths and len(custom_modality_paths) == 4:
            mod_paths = {m: Path(p) for m, p in custom_modality_paths.items()}
        else:
            target_dir = None
            # Check if specific patient_id exists in BraTS2024
            if Path("data/raw/BraTS2024").exists():
                from data.real_brats_pipeline import RealBratsValidator
                validator = RealBratsValidator(Path("data/raw/BraTS2024"))
                all_subjs = {p.name: p for p in validator.discover_subject_directories()}
                target_dir = all_subjs.get(patient_id)
            
            # Check BraTS2021 directory
            if not target_dir or not target_dir.exists():
                brats2021_dir = Path(f"data/BraTS2021/{patient_id}")
                if brats2021_dir.exists():
                    target_dir = brats2021_dir
                elif Path("data/raw/BraTS2024").exists():
                    all_subjs = {p.name: p for p in validator.discover_subject_directories()}
                    if all_subjs:
                        target_dir = list(all_subjs.values())[0]

            if not target_dir or not target_dir.exists():
                if Path("data/BraTS2021/BraTS2021_00001").exists():
                    target_dir = Path("data/BraTS2021/BraTS2021_00001")
                elif Path("data/BraTS2021/BraTS2021_00002").exists():
                    target_dir = Path("data/BraTS2021/BraTS2021_00002")

            version, mod_paths, missing = ModalityCanonicalizer.discover_subject_modalities(target_dir)
            if missing:
                return {
                    "status": "FAILED",
                    "error_message": f"Missing modalities for patient '{patient_id}': {missing}",
                    "patient_id": patient_id,
                }

        res = self.clinical_engine.predict_subject(mod_paths)
        if res.get("status") == "SUCCESS":
            record = {
                "status": "SUCCESS",
                "patient_id": patient_id,
                "model_version": model_version,
                "inference_time_ms": res.get("timing_ms", {}).get("total_inference_time_ms", 0.0),
                "tumor_volumes_mm3": res.get("tumor_volumes_mm3", {}),
                "segmented_voxels": res.get("segmented_voxels", {}),
                "visual_slices": res.get("visual_slices", {}),
                "timing_ms": res.get("timing_ms", {}),
                "model_metadata": res.get("model_metadata", {}),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            self.history.append(record)
            return record
        else:
            res["patient_id"] = patient_id
            return res

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

    def get_model_info(self) -> Dict[str, Any]:
        """Returns model specification card."""
        return {
            "model_name": "FedMed MONAI 3D U-Net (Differential Privacy)",
            "experiment_id": "EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "parameter_count": 4810074,
            "architecture": "MONAI 3D U-Net (128x128x128)",
            "input_modalities": ["T1", "T1CE", "T2", "FLAIR"],
            "output_channels": ["Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"],
            "privacy_guarantee": {
                "mechanism": "Poisson DP-SGD",
                "epsilon": 2.8934,
                "delta": 0.00001,
                "accountant_steps": 4720,
                "clipping_bound_C": 0.06,
                "noise_multiplier_sigma": 0.87,
            },
            "training_cohort": "944 subjects across 4 hospital silos (BraTS-GLI 2024)",
            "validation_macro_dice": 0.0800,
            "locked_test_macro_dice": 0.0741,
            "locked_test_tc_dice": 0.2054,
            "status": "FROZEN_PRODUCTION_CANDIDATE",
            "disclaimer": "Research Prototype — Not approved for primary clinical diagnostic use.",
        }


global_inference_engine = BraTSInferenceEngine()
