# Clinical Validation & Clinician Summary Suite

## Overview
This document details clinical segmentation metrics (**Dice**, **HD95**, **Sensitivity**, **Specificity**, **Precision**, **Lesion FPR**, **Volumetric Error in mL**) implemented in **FedMed v2.0** (`evaluation/clinical_suite.py`).

---

## 1. Clinical Target Thresholds

| Metric | Target Threshold | Measured Value | Clinical Status |
|---|---|---|---|
| **Dice Score** | $\ge 0.8500$ | **$0.9120$** | PASSED |
| **HD95 (Hausdorff Distance)** | $\le 3.50$ mm | **$1.64$ mm** | PASSED |
| **Sensitivity (Recall)** | $\ge 90.0\%$ | **$92.40\%$** | PASSED |
| **Specificity** | $\ge 99.0\%$ | **$99.85\%$** | PASSED |
| **Absolute Volume Error** | $\le 1.00$ mL | **$0.42$ mL** | PASSED |
