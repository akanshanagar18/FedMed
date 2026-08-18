"""
Script: run_production_inference.py
Purpose: Standalone production inference runner for FedMed Differential Privacy 3D Brain Tumor Segmentation.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import numpy as np
import nibabel as nib
import torch
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Resized,
    Spacingd,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fedmed_production_inference")

EXPECTED_SHA256 = "f6cd18dc5e05595ca88ad1675190aabb98595ee9f6d6dfed36c25b596f652f3a"


def compute_sha256(p: Path) -> str:
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="FedMed Standalone Production Inference")
    parser.add_argument("--t1", required=True, help="Path to T1-native NIfTI image")
    parser.add_argument("--t1ce", required=True, help="Path to T1-contrast NIfTI image")
    parser.add_argument("--t2", required=True, help="Path to T2-weighted NIfTI image")
    parser.add_argument("--flair", required=True, help="Path to T2-FLAIR NIfTI image")
    parser.add_argument("--model", default="model.pt", help="Path to model checkpoint")
    parser.add_argument("--output", default="segmentation_output.nii.gz", help="Output NIfTI path")
    parser.add_argument("--device", default="auto", help="Compute device ('mps', 'cuda', 'cpu', 'auto')")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("FAIL-CLOSED: Model checkpoint '%s' not found!", model_path)
        sys.exit(1)

    act_hash = compute_sha256(model_path)
    if act_hash != EXPECTED_SHA256:
        logger.error("FAIL-CLOSED: Model hash mismatch! Expected %s, got %s", EXPECTED_SHA256, act_hash)
        sys.exit(1)
    logger.info("✓ Verified Model Integrity: %s", act_hash)

    # Device selection
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Executing on device: %s", device)

    # Load Model
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(str(model_path), map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Preprocessing
    mod_files = [args.t1, args.t1ce, args.t2, args.flair]
    for mf in mod_files:
        if not Path(mf).exists():
            logger.error("Input modality file missing: %s", mf)
            sys.exit(1)

    transforms = Compose([
        LoadImaged(keys=["image"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
        Orientationd(keys=["image"], axcodes="RAS", allow_missing_keys=True),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear", allow_missing_keys=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image", allow_missing_keys=True),
        Resized(keys=["image"], spatial_size=(128, 128, 128), mode="bilinear", allow_missing_keys=True),
        EnsureTyped(keys=["image"], data_type="tensor", allow_missing_keys=True),
    ])

    sample = {"image": mod_files}
    t0 = time.perf_counter()
    processed = transforms(sample)
    image_tensor = processed["image"].unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    t_inf = (time.perf_counter() - t0) * 1000.0
    preds_np = preds.squeeze(0).cpu().numpy()

    tc_vox = int(np.sum(preds_np[0] > 0))
    wt_vox = int(np.sum(preds_np[1] > 0))
    et_vox = int(np.sum(preds_np[2] > 0))

    # Build composite output mask
    composite_mask = np.zeros((128, 128, 128), dtype=np.uint8)
    composite_mask[preds_np[0] > 0] = 1
    composite_mask[(preds_np[1] > 0) & ~(preds_np[0] > 0)] = 2
    composite_mask[preds_np[2] > 0] = 4

    out_img = nib.Nifti1Image(composite_mask, affine=np.eye(4))
    nib.save(out_img, args.output)

    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE (Latency: %.2f ms)", t_inf)
    logger.info("Tumor Core (TC):     %d voxels (%.1f mm³)", tc_vox, float(tc_vox))
    logger.info("Whole Tumor (WT):    %d voxels (%.1f mm³)", wt_vox, float(wt_vox))
    logger.info("Enhancing Tumor (ET): %d voxels (%.1f mm³)", et_vox, float(et_vox))
    logger.info("Saved Segmentation:  %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
