# FedAsync — Asynchronous Federated Learning Framework

## Overview
This document details the mathematical formulation, staleness functions, and runtime implementation of **FedAsync** (*Asynchronous Federated Optimization*, Xie et al., 2019) implemented in **FedMed v2.0**.

---

## 1. Mathematical Formulation

### Server Timestamp & Staleness Calculation:
When client $i$ returns an update trained at round $t_{\text{client}}$, the server calculates staleness:
$$\tau_{\text{stale}} = t_{\text{server}} - t_{\text{client}}$$

### Staleness Weighting Function ($S(\tau)$):
- **Polynomial:** $S(\tau) = (\tau + 1)^{-a}$
- **Constant:** $S(\tau) = 1.0$
- **Hinge:** $S(\tau) = 1.0$ if $\tau \le E/2$ else $\frac{1}{\tau - E/2 + 1}$

### Asynchronous Server Update ($x_{t+1}$):
$$\alpha_{\text{eff}} = \alpha \cdot S(\tau_{\text{stale}})$$
$$x_{t+1} = (1 - \alpha_{\text{eff}}) x_t + \alpha_{\text{eff}} y_i^K$$
$$t_{\text{server}} \leftarrow t_{\text{server}} + 1$$

---

## 2. Configuration & Execution

```bash
python scripts/run_simulation.py --strategy fedasync
```
