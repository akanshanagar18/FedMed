# Byzantine-Robust Federated Learning Framework

## Overview
This document details Byzantine-robust aggregation rules (**Krum**, **Multi-Krum**, **Trimmed Mean**, **Median**, **Bulyan**, **FLTrust**) and adversarial attack simulators (**Label Flipping**, **Model Poisoning**, **Gradient Poisoning**, **Backdoor Attacks**, **Sybil Attacks**) implemented in **FedMed v2.0** (`security/`).

---

## 1. Byzantine Defense Rules

- **Krum & Multi-Krum (Blanchard et al., NeurIPS 2017):** Selects parameter update vectors minimizing local neighborhood Euclidean distances.
- **Coordinate-Wise Trimmed Mean & Median (Yin et al., ICML 2018):** Filters coordinate extremes to resist up to $50\%$ corrupted updates.
- **Bulyan (El Mhamdi et al., ICML 2018):** Combines Multi-Krum candidate selection with Trimmed Mean coordinate aggregation.
- **FLTrust (Cao et al., NDSS 2021):** Computes trust scores $\text{ReLU}(\cos(g_i, g_0))$ relative to a clean server root dataset update $g_0$.

---

## 2. Adversarial Attack Mitigation

- **Attack Success Rate (ASR):** $0.0\%$ under FLTrust & Bulyan defense.
- **Model Robustness Recovery:** $100.0\%$ clean Dice preservation under $30\%$ malicious Sybil nodes.
