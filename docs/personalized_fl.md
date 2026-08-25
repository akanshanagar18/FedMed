# Personalized Federated Learning Framework

## Overview
This document details the personalized federated learning strategies (**FedPer**, **LG-FedAvg**, **FedRep**, **Per-FedAvg**) implemented in **FedMed v2.0**.

---

## 1. Algorithm Overview

### FedPer (Arivazhagan et al., 2019)
Aggregates shared feature representation backbone layers globally while maintaining personalized classification/segmentation heads locally on each hospital node.

### LG-FedAvg (Liang et al., 2020)
Keeps lower representation layers local to hospital clients while aggregating upper prediction head layers globally.

### FedRep (Collins et al., 2021)
Alternates optimization between local classification heads ($h_i$) and global feature representation backbone ($B$).

### Per-FedAvg (Fallah et al., 2020)
Applies Model-Agnostic Meta-Learning (MAML) gradient updates:
$$f_i(\theta - \alpha \nabla f_i(\theta))$$

---

## 2. Configuration & Execution

```bash
python scripts/run_simulation.py --strategy fedper
python scripts/run_simulation.py --strategy lgfedavg
python scripts/run_simulation.py --strategy fedrep
python scripts/run_simulation.py --strategy perfedavg
```
