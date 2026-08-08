# Medical AI Explainability & Uncertainty Suite

## Overview
This document details 3D visual explainability (**Grad-CAM**, **Attention Maps**) and epistemic uncertainty mapping implemented in **FedMed v2.0** (`explainability/`).

---

## 1. Explainability Methods

### 3D Grad-CAM (Selvaraju et al., ICCV 2017)
Generates 3D class activation saliency heatmaps:
$$\alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \sum_{k} \frac{\partial y^c}{\partial A^k_{i,j,k}}$$
$$L_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_k \alpha_k A^k \right)$$

### Epistemic Uncertainty & Entropy Mapping
Monte Carlo Dropout sampling ($N=10$) calculates prediction variance and information entropy:
$$\mathcal{H}(y) = - \sum_c p_c \log p_c$$
