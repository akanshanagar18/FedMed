# FedMed v2.0 — Advanced Strategy Engine Comparison & Research Matrix

## Overview
This document compares all 9 federated optimization strategies implemented in **FedMed v2.0**: **FedAvg**, **FedProx**, **SCAFFOLD**, **FedAdam**, **FedYogi**, **FedAdagrad**, **FedNova**, **FedDyn**, and **FedBN**.

---

## Algorithm Comparison Matrix

| Strategy | Authors & Year | Core Mechanism | Supported Privacy Modes | Primary Advantage |
|---|---|---|---|---|
| **FedAvg** | McMahan et al., AISTATS 2017 | Sample-Weighted Averaging | Plaintext, DP, HE | Baseline & Simple |
| **FedProx** | Li et al., MLSys 2020 | Proximal Loss Regularization | Plaintext, DP, HE | Prevents Client Drift |
| **SCAFFOLD** | Karimireddy et al., ICML 2020 | Control Variate Variance Reduction | Plaintext, DP, HE | Heterogeneity Independent |
| **FedAdam** | Reddi et al., ICLR 2021 | Server Adam Adaptive Momentum | Plaintext, DP, HE | Fast Convergence |
| **FedYogi** | Reddi et al., ICLR 2021 | Server Yogi Adaptive Variance | Plaintext, DP, HE | Robust Non-IID Variance |
| **FedAdagrad** | Reddi et al., ICLR 2021 | Server Adagrad Accumulator | Plaintext, DP, HE | Adaptive Scaling |
| **FedNova** | Wang et al., NeurIPS 2020 | Normalized Step Update Scaling | Plaintext, DP, HE | Objective Consistency |
| **FedDyn** | Acar et al., ICLR 2021 | Dynamic Server State Vector | Plaintext, DP, HE | Asymptotic Exactness |
| **FedBN** | Li et al., ICLR 2021 | Local Batch Normalization | Plaintext, DP, HE | Domain Shift Personalization |
