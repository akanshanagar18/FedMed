# FedNova — Federated Normalized Averaging

## Overview
This document details **FedNova** (*Tackling Heterogeneity in Federated Optimization via Normalized Averaging*, Wang et al., NeurIPS 2020) implemented in **FedMed v2.0**.

---

## 1. Mathematical Formulation
To eliminate objective inconsistency caused by non-uniform local steps $\tau_i$ across heterogeneous clients:

1. **Normalized Local Delta:**
   $$\Delta_i^t = \frac{y_i^K - x^t}{\tau_i}$$
2. **Effective Local Step Count:**
   $$\tau_{\text{eff}} = \sum_{i \in \mathcal{S}} p_i \tau_i$$
3. **Normalized Aggregated Update:**
   $$\Delta_{\text{Nova}} = \tau_{\text{eff}} \sum_{i \in \mathcal{S}} p_i \Delta_i^t$$
4. **Global Model Update:**
   $$x^{t+1} = x^t + \Delta_{\text{Nova}}$$

---

## 2. Configuration & Execution

```bash
python scripts/run_simulation.py --config configs/fednova.yaml
```
