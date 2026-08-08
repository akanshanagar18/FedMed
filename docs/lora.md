# Parameter-Efficient Fine-Tuning (PEFT) with LoRA

## Overview
This document details Low-Rank Adaptation (**LoRA**, Hu et al., ICLR 2022) implemented in **FedMed v2.0** (`peft/`).

---

## 1. Mathematical Formulation
For base frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA injects low-rank decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$:
$$W = W_0 + \frac{\alpha}{r} B A$$
where $r \ll \min(d, k)$ is the rank parameter (e.g. $r=8$).

---

## 2. Advantages for Federated AI
- Reduces communication payload from 350 MB to 10 MB per round.
- Enables resource-constrained hospital nodes to train large 100M+ parameter foundation models.
