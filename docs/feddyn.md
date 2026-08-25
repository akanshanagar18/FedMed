# FedDyn — Dynamic Regularization for Federated Learning

## Overview
This document details **FedDyn** (*Federated Learning Based on Dynamic Regularization*, Acar et al., ICLR 2021) implemented in **FedMed v2.0**.

---

## 1. Mathematical Formulation

### Server State Vector Update ($h_t$):
$$h_t = h_{t-1} - \alpha \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} (y_i^K - x^t)$$

### Server Parameter Update ($x^{t+1}$):
$$x^{t+1} = \frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} y_i^K - \frac{1}{\alpha} h_t$$

---

## 2. Configuration & Execution

```bash
python scripts/run_simulation.py --config configs/feddyn.yaml
```
