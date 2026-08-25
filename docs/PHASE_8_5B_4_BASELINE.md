# FEDMED OS — PHASE 8.5B.4 BASELINE & AUDIT DOCUMENT

**Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Scope:** Centralized Baseline + FedAvg + FedProx Reproducible Experiment & Benchmark Integrity Gate (Phase 8.5B.4)  
**Git Branch:** `release/stabilization-rc2`  
**Git Head:** `e266612 Phase 10: FedMed Omega production hardening and runtime validation`  

---

## 1. Baseline Audit of Systems

1. **Centralized Training Capability:**
   - Pre-existing script `scripts/run_real_local_training.py` trained a single hospital node on its assigned subject.
   - Centralized multi-subject training requires training on all subjects in `TRAIN` (`['BraTS2021_00003', 'BraTS2021_00004']`) while strictly isolating `VALIDATION` (`BraTS2021_00002`) and firewalled `TEST` (`BraTS2021_00001`).

2. **FedAvg Implementation:**
   - Canonical `server/strategies/fedavg.py` executes exact sample-weighted parameter aggregation:
     $$W_{\text{server}} = \sum \frac{n_k}{N} W_k$$
   - Tested and verified in Phase 8.5B.3 with $0.0$ error against independent calculation.

3. **FedProx Implementation Status:**
   - Server-side strategy `server/strategies/fedprox.py` exists in registry.
   - Client-side proximal loss regularization $\mathcal{L}_{\text{prox}}(w) = \mathcal{L}_{\text{local}}(w) + \frac{\mu}{2} \|w - w_{\text{global}}\|^2$ must be formally implemented in the client local training loop using the fixed global model received at the start of each round.

4. **Benchmark Runner Audit & Static/Mock Artifact Classification:**
   - `evaluation/benchmark_runner.py` previously contained a mock method `run_benchmark_matrix()` with static hardcoded profiles (`algo_profiles = {"FedAvg": {"dice": 0.852, ...}, "FedProx": {"dice": 0.864, ...}}`).
   - **Audit Action:** The active execution path (`evaluation/benchmark_runner.py` and `scripts/run_benchmark.py`) is refactored to execute real forward/backward passes and compute genuine measured metrics via `evaluation.metrics` without any static lookups.

5. **Configuration System:**
   - Single canonical configuration in `configs/experiments/real_brats_fedavg.yaml` defining MONAI 3D UNet (4,810,074 parameters), Adam optimizer (`lr=1e-4, weight_decay=1e-5`), and `DiceCELoss(sigmoid=True)`.

6. **Checkpoint System:**
   - Model checkpoints saved to `checkpoints/centralized/`, `checkpoints/fedavg/`, and `checkpoints/fedprox/`.

7. **Metric System:**
   - Standardized on `evaluation/metrics.py` (`compute_dice()`, `compute_iou()`).

8. **Database & Reporting System:**
   - SQLite table `training_metrics` in `fedmed.db`.
   - Experiment reports stored in `reports/experiments/benchmark/<run_id>/`.

9. **MLflow Dependency Behavior:**
   - Optional plugin design in `utils/mlflow_tracker.py`. Test suite updated to test fallback mode when `mlflow` is absent.

---

## 2. Step 1: B3 Flower Integrity Audit

- **Execution Path Analysis:**
  - `scripts/run_fedavg_experiment.py` orchestrates multi-round federated training by instantiating `SimulatedHospitalClient` (which implements local MONAI forward/backward/Adam updates) and directly passing `FitResult` to `FedAvg.aggregate_fit()`.
- **Classification:**
  - `FLOWER_EXECUTION_MODE = LOCAL_STRATEGY_EXECUTION`
- **Finding:**
  - The federated math, weight distribution, local backpropagation, parameter delta measurement, and sample-weighted aggregation execute genuine PyTorch/NumPy tensor operations. Full network-distributed gRPC Flower (`server/flower_server.py` and `client/flower_client.py`) is available for standalone daemon modes.

---

## 3. Step 2: Parameter Delta Provenance Audit

- In Phase 8.5B.3, `max_parameter_delta` was reported as $\approx 1.0000\text{e-}4$.
- **Audit of Tensor Formula:**
  - `max_delta = max((w_after - w_before).abs().max().item())`
- **Mathematical Explanation:**
  - The optimizer is Adam with learning rate $\eta = 1.0\text{e-}4$.
  - In step 1 of Adam optimization from initial momentum buffers ($m_0=0, v_0=0$):
    $$m_1 = (1-\beta_1) g = 0.1 g \implies \hat{m}_1 = g$$
    $$v_1 = (1-\beta_2) g^2 = 0.001 g^2 \implies \hat{v}_1 = g^2$$
    $$\Delta \theta = \eta \cdot \frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} \approx 10^{-4} \cdot \text{sign}(g) = \pm 10^{-4}$$
  - The maximum parameter update for any parameter with non-zero gradient is analytically bounded by $\eta = 10^{-4}$ in the initial steps.
- **Verification:**
  - The tensor delta is 100% genuine and derived directly from tensor subtraction.
  - In Phase 8.5B.4, we report `max_parameter_delta`, `mean_parameter_delta`, and `l2_parameter_delta` for full transparency.
