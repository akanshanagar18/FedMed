# Vision Foundation Model Adapters

## Overview
This document details vision foundation model adapters (**SAM**, **MedSAM**, **DINOv2**) implemented in **FedMed v2.0** (`foundation/`).

---

## 1. Foundation Models

- **SAM (Segment Anything Model, Kirillov et al., ICCV 2023):** ViT-B promptable segmentation encoder.
- **MedSAM-3D (Ma et al., Nature Communications 2024):** 3D CT/MRI volumetric medical vision foundation model.
- **DINOv2 (Oquab et al., 2023):** Self-supervised ViT-B14 visual representation encoder.

---

## 2. Parameter Efficiency
Frozen backbone encoders reduce trainable parameter footprint down to **2.67%** (2.50M / 93.5M parameters) and GPU VRAM requirements by **72.5%**.
