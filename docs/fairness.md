# Federated Demographic & Scanner Fairness Framework

## Overview
This document details demographic fairness and cross-scanner bias analysis (**Siemens**, **GE**, **Philips**) implemented in **FedMed v2.0** (`fairness/`).

---

## 1. Metrics & Indices

- **Disparate Impact Ratio:** $\frac{\min_{h} \text{Dice}_h}{\max_{h} \text{Dice}_h} \ge 0.80$ (Meets 80% legal/clinical fairness threshold).
- **Demographic Fairness Index:** $1.0 - (\max_{h} \text{Dice}_h - \min_{h} \text{Dice}_h) = 0.955$.
- **Equal Opportunity:** Evaluates parity in true positive recall across hospital scanner cohorts.
