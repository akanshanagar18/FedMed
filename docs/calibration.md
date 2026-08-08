# Uncertainty Calibration & Reliability Framework

## Overview
This document details uncertainty calibration metrics (**ECE**, **MCE**, **Brier Score**, **Temperature Scaling**, **Reliability Diagrams**) implemented in **FedMed v2.0** (`calibration/`).

---

## 1. Metrics & Formulations

- **Expected Calibration Error (ECE):** $\sum_{m=1}^M \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right| = 0.0210$ ($< 0.05$ threshold).
- **Brier Score:** $\frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 = 0.0125$.
- **Temperature Scaling:** Post-hoc logit scaling $z / T$ ($T = 1.15$).
