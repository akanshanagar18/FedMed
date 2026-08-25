# Masked Secure Aggregation Protocol

## Overview
This document details pairwise additive secret masking secure aggregation (Bonawitz et al., CCS 2017) implemented in **FedMed v2.0** (`security/secure_aggregation.py`).

---

## 1. Protocol Architecture

Pairwise secret masks $\delta_{i, j}$ are exchanged between hospital nodes:
$$\sum_{i=1}^N (w_i + \Delta_i) = \sum_{i=1}^N w_i + \sum_{i=1}^N \sum_{j \neq i} S_{i, j} = \sum_{i=1}^N w_i$$
Masks cancel out exactly to zero, preventing the central server from inspecting unmasked gradient updates.
