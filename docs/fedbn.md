# FedBN — Federated Learning with Local Batch Normalization

## Overview
This document details **FedBN** (*Federated Learning on Non-IID Data via Local Batch Normalization*, Li et al., ICLR 2021) implemented in **FedMed v2.0**.

---

## 1. Algorithmic Design
- Keeps Batch Normalization parameters (`running_mean`, `running_var`, `bn.weight`, `bn.bias`) local to each hospital node.
- Aggregates convolutional, linear, and attention weights globally while preserving local domain adaptation.

---

## 2. Configuration & Execution

```bash
python scripts/run_simulation.py --config configs/fedbn.yaml
```
