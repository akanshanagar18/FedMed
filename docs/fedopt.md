# FedOpt Family — Adaptive Server Optimization for Federated Learning

## Overview
This document details the mathematical formulation, update equations, and implementation details for the **FedOpt** family of adaptive server optimization algorithms (**FedAdam**, **FedYogi**, **FedAdagrad**) implemented in **FedMed v2.0** (Reddi et al., ICLR 2021).

---

## 1. Mathematical Update Equations

At global round $t$, sampled clients compute local updates $y_i^K$. The server computes the aggregated pseudogradient delta:
$$\Delta_t = \sum_{i \in \mathcal{S}} \frac{n_i}{n} (y_i^K - x^t)$$

### FedAdam
1. 1st Moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) \Delta_t$
2. 2nd Moment: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) \Delta_t^2$
3. Global Update: $x^{t+1} = x^t + \eta \frac{m_t}{\sqrt{v_t} + \tau}$

### FedYogi
1. 1st Moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) \Delta_t$
2. 2nd Moment: $v_t = v_{t-1} - (1 - \beta_2) \Delta_t^2 \text{sign}(v_{t-1} - \Delta_t^2)$
3. Global Update: $x^{t+1} = x^t + \eta \frac{m_t}{\sqrt{v_t} + \tau}$

### FedAdagrad
1. 2nd Moment Accumulator: $v_t = v_{t-1} + \Delta_t^2$
2. Global Update: $x^{t+1} = x^t + \eta \frac{\Delta_t}{\sqrt{v_t} + \tau}$

---

## 2. Configuration & Execution

### YAML Configuration (`configs/fedadam.yaml`):
```yaml
federated:
  strategy: "FedAdam"
  learning_rate: 0.0001
  eta: 0.01
  beta_1: 0.9
  beta_2: 0.999
  tau: 0.001
```

### CLI Execution:
```bash
python scripts/run_simulation.py --config configs/fedadam.yaml
python scripts/run_simulation.py --config configs/fedyogi.yaml
python scripts/run_simulation.py --config configs/fedadagrad.yaml
```
