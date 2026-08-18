# FedMed Differential Privacy 3D Brain Tumor Segmentation — Production Release

## Overview
This standalone production bundle contains the finalized, locked clinical model trained via Differential Privacy Federated Averaging (DP-SGD) on the BraTS-GLI 2024 cohort (1,350 subjects, 4 hospital silos).

## Model Specifications
- **Architecture:** MONAI 3D U-Net (128x128x128 spatial resolution)
- **Trainable Parameters:** 4,810,074
- **Input Channels:** 4 (T1-native, T1-contrast, T2-weighted, T2-FLAIR)
- **Output Channels:** 3 composite channels:
  - Channel 0: Tumor Core (TC)
  - Channel 1: Whole Tumor (WT)
  - Channel 2: Enhancing Tumor (ET)
- **Privacy Guarantee:** $(\varepsilon = 2.8934, \delta = 10^{-5})$ under Poisson Subsampled Renyi Differential Privacy (RDP $\alpha = 7$, $T = 4,720$ steps per silo, $C = 0.06$, $\sigma = 0.87$).
- **Benchmark Performance (Locked Test Cohort, 204 Subjects):**
  - **Macro Dice:** `0.0741` (95% CI: `[0.0656, 0.0831]`)
  - **Tumor Core (TC) Dice:** `0.2054` (95% CI: `[0.1816, 0.2306]`)
  - **Enhancing Tumor (ET) Dice:** `0.0111` (95% CI: `[0.0086, 0.0138]`)
  - **Whole Tumor (WT) Dice:** `0.0059` (95% CI: `[0.0046, 0.0073]`)
  - **Macro IoU:** `0.0451` (95% CI: `[0.0393, 0.0516]`)
  - **Test Loss:** `0.9628`

## Standalone Inference Usage
```bash
pip install -r requirements.txt

python run_production_inference.py \
    --t1 path/to/t1n.nii.gz \
    --t1ce path/to/t1c.nii.gz \
    --t2 path/to/t2w.nii.gz \
    --flair path/to/t2f.nii.gz \
    --model model.pt \
    --output segmentation_output.nii.gz
```

## Integrity Verification
Verify that `model.pt` matches the official release hash:
```bash
sha256sum -c SHA256SUMS
```
Expected SHA-256:
`f6cd18dc5e05595ca88ad1675190aabb98595ee9f6d6dfed36c25b596f652f3a`

## Clinical Disclaimer
This model is a **Research Prototype** for clinical decision support and evaluation. It is not cleared as a standalone primary diagnostic device without board-certified neuroradiology review.
