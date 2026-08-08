# Communication Compression Engine

## Overview
This document details the communication compression algorithms (**INT8/INT4 Quantization**, **Top-K/Random-K Sparsification**, **SignSGD with Error Feedback**) implemented in **FedMed v2.0** (`communication/`).

---

## 1. Supported Compression Methods

### Uniform Quantization (INT8 & INT4)
- Scales floating point arrays $[w_{\min}, w_{\max}]$ into unsigned integer ranges $[0, 2^b - 1]$.
- Achieves **4.0x (INT8)** and **8.0x (INT4)** payload compression.

### Gradient Sparsification (Top-K & Random-K)
- Transmits only top fraction $k$ (e.g. $k=10\%$) of non-zero gradient values alongside index coordinates.
- Achieves **5.0x - 10.0x** compression ratio.

### SignSGD with Error Feedback
- Quantizes values into 1-bit signs $\{-1, +1\}$ with scalar mean magnitude scale.
- Accumulates quantization residuals $e_{t+1} = w_{\text{target}} - (\text{scale} \cdot \text{sign}(w_{\text{target}}))$ in local error feedback buffers.
- Achieves **32.0x (96.9%)** compression.

---

## 2. Configuration & Execution

```bash
python scripts/run_simulation.py --compression quantization
python scripts/run_simulation.py --compression topk
python scripts/run_simulation.py --compression sign_sgd
```
