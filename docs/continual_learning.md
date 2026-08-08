# Continual Federated Learning Engine

## Overview
This document details catastrophic forgetting mitigation mechanisms (**Elastic Weight Consolidation**, **Prioritized Experience Replay**, **Teacher-Student Knowledge Distillation**) implemented in **FedMed v2.0** (`continual/`).

---

## 1. Mathematical Formulation

### Elastic Weight Consolidation (EWC, Kirkpatrick et al., PNAS 2017)
Calculates diagonal Fisher Information Matrix $F_i$ to penalize parameter updates on task-critical weights:
$$\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_{\text{current}}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{A, i}^*)^2$$

### Teacher-Student Knowledge Distillation
Penalizes divergence between current student predictions and teacher logits:
$$\mathcal{L}_{\text{KD}} = \text{KL}\left(\text{Softmax}\left(\frac{z_{\text{teacher}}}{T}\right) \,\middle\|\, \text{Softmax}\left(\frac{z_{\text{student}}}{T}\right)\right) \cdot T^2$$

---

## 2. Execution & Metrics
Tracks catastrophic forgetting scores, EWC penalty loss, and knowledge retention percentage ($\ge 98.8\%$).
