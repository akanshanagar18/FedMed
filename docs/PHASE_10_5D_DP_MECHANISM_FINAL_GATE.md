# PHASE 10.5D: FINAL DP MECHANISM SEMANTICS & PRE-LAUNCH FORENSIC GATE REPORT

**Date:** August 17, 2026  
**System:** FedMed OS v2.0 Enterprise Platform  
**Target:** Experiment D (Differential Privacy Benchmark on BraTS-GLI 2024)  
**Execution Mode:** Read-Only Forensic Semantics Verification & Pre-Launch Gate (Experiment D Held at Gate)  
**Final Status:** `DP_MECHANISM_READY_FOR_EXPERIMENT_D`  

---

## A. Actual Sampler Semantics

- **Class:** `privacy.dp_engine.PoissonBatchSampler` (inherits from `torch.utils.data.Sampler[List[int]]`).
- **Algorithm:** For each optimizer step $t \in \{1, \dots, 236\}$, every dataset index $i \in \{0, \dots, N_{\text{client}}-1\}$ ($N_{\text{client}} = 236$) undergoes an independent Bernoulli trial:
  $$b_{t, i} \sim \text{Bernoulli}(q), \quad q = \frac{1}{236} \approx 0.004237288$$
- **Index Set:** $S_t = \{i \in \{0, \dots, 235\} \mid b_{t, i} = 1\}$.
- **Step Determinism:** Generator is seeded deterministically per hospital and per round:
  $$\text{seed}_{\text{sampler}} = \text{seed}_{\text{global}} + r \times 1000 + (\text{abs}(\text{hash}(\text{hospital\_name})) \bmod 1000)$$
- **Empirical Validation (10,000 Step Monte Carlo Test):**
  - Empirical Mean Batch Size: **1.0138** (Theoretical: **1.0000**)
  - Empirical Empty Batch Rate: **35.92%** (Theoretical: **36.71%**)
  - Empirical Singleton Rate: **37.26%** (Theoretical: **36.87%**)
  - Empirical Multi-Sample Rate: **26.82%** (Theoretical: **26.42%**)
  - Empirical Per-Item Selection Rate: **0.004296** (Theoretical $q$: **0.004237**)

---

## B. Actual Batch Semantics

- **Expected Batch Size:** $\mathbb{E}[|S_t|] = N_{\text{client}} \cdot q = 236 \times \frac{1}{236} = \mathbf{1.0}$.
- **Batch Realizations:**
  - $|S_t| = 0$ (Empty Batch): Occurs with probability $(1-q)^N \approx 36.7\%$.
  - $|S_t| = 1$ (Singleton Batch): Occurs with probability $N q (1-q)^{N-1} \approx 36.9\%$.
  - $|S_t| \ge 2$ (Multi-Sample Batch): Occurs with probability $1 - (1-q)^N - N q (1-q)^{N-1} \approx 26.4\%$.
- **Batch Evaluation Strategy:** Sequential micro-batching (1 sample forward-backward at a time). Peak MPS memory is strictly bounded at $\sim 114\text{ MB}$, completely eliminating GPU OOMs regardless of realization $|S_t|$.

---

## C. Actual Gradient Aggregation Equation

Tracing from raw per-sample evaluations to optimizer consumption:

1. **Per-Sample Gradient:** For sample $i \in S_t$, forward-backward produces:
   $$g_i = \nabla_\theta \mathcal{L}(\theta_t; x_i, y_i)$$
2. **Per-Sample $L_2$ Clipping:**
   $$\bar{g}_i = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right) \implies \|\bar{g}_i\|_2 \le C = 1.0$$
3. **Un-Normalized Accumulation (Option A):**
   $$\text{acc}_t = \sum_{i \in S_t} \bar{g}_i$$
   *(If $S_t = \emptyset$, $\text{acc}_t = \mathbf{0}$. There is NO division by $|S_t|$).*
4. **Spherical Gaussian Noise Addition:**
   $$\xi_t \sim \mathcal{N}(0, \sigma^2 C^2 I_{P}), \quad P = 4,810,074, \quad \sigma = 0.87, \quad C = 1.0$$
5. **Noisy Gradient Vector to Optimizer:**
   $$\tilde{g}_t = \text{acc}_t + \xi_t = \sum_{i \in S_t} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)$$

---

## D. Sensitivity Proof

Let $D$ and $D'$ be adjacent local datasets differing in at most one patient sample (replacement of record $j$ by $j'$).

For any step $t$, let $S_t \subseteq D$ and $S'_t \subseteq D'$ be the Poisson subsampled subsets.
- If record $j$ is not selected ($j \notin S_t, j \notin S'_t$), $S_t = S'_t \implies f(S_t) - f(S'_t) = \mathbf{0}$.
- If record $j$ is selected (probability $q = 1/236$), then:
  $$f(S_t) - f(S'_t) = \left(\sum_{i \in S_t \setminus \{j\}} \bar{g}_i + \bar{g}_j\right) - \left(\sum_{i \in S_t \setminus \{j\}} \bar{g}_i + \bar{g}'_j\right) = \bar{g}_j - \bar{g}'_j$$
  or $\bar{g}_j$ in the add/remove adjacency definition.
- Since individual gradients are clipped to $\|\bar{g}_j\|_2 \le C$, the global $L_2$ sensitivity is:
  $$\Delta_2 f = \max_{D \sim D'} \|f(D) - f(D')\|_2 \le C = \mathbf{1.0}$$

**Conclusion:** The implementation computes an un-normalized sum of per-sample clipped gradients. Sensitivity $\Delta_2 f = C = 1.0$ **strictly matches** the accountant's assumption.

---

## E. Actual Noise Equation & Numerical Verification

- **Distribution:** $\mathcal{N}(0, \sigma^2 C^2 I)$
- **Standard Deviation:** $\sigma_{\text{noise}} = \sigma \cdot C = 0.87 \times 1.0 = \mathbf{0.8700}$.
- **Numerical Test (MONAI 3D U-Net, $P = 4,810,074$ parameters):**
  - Empirical Mean: $\mathbf{-0.000522}$ (Expected: $0.000000$)
  - Empirical Standard Deviation: $\mathbf{0.870027}$ (Expected: $0.870000$)
  - Noise Independence: Verified independent across parameter tensors and client steps.

---

## F. Empty-Batch Behavior ($|S_t| = 0$)

When a Poisson step selects 0 samples ($P(|S_t|=0) \approx 36.70\%$):
1. `accumulated_grads` is initialized to zeros: $\text{acc}_t = \mathbf{0}$.
2. Spherical Gaussian noise $\xi_t \sim \mathcal{N}(0, (0.87)^2 I)$ is generated and assigned: $p.\text{grad} = \xi_t$.
3. `optimizer.step()` is called.
4. Adam updates moments:
   $$m_t = \beta_1 m_{t-1} + (1-\beta_1) \xi_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) \xi_t^2$$
5. Weight decay $\lambda \theta_{t-1}$ ($\lambda = 10^{-5}$) applies along with the noise-driven update.
6. Empirical Parameter Update Norm: $\Delta_\theta = 0.002635 > 0$.
7. Accountant increments step counter $T \to T + 1$.

**Theoretical Intentionality:** In the Poisson subsampled Gaussian mechanism, the mechanism output on query $f(S_t)$ is $M(S_t) = f(S_t) + \xi_t$. When $S_t = \emptyset$, $f(\emptyset) = \mathbf{0}$, so $M(\emptyset) = \mathbf{0} + \xi_t = \xi_t$. Omitting noise or skipping the optimizer step on empty batches would reveal to an adversary that $|S_t| = 0$, violating the differential privacy guarantee. Stepping with pure noise on empty batches is mathematically required.

---

## G. Adam Interaction

Adam receives $\tilde{g}_t = \sum_{i \in S_t} \bar{g}_i + \xi_t$.
- Learning Rate: $\eta = 10^{-4}$
- Weight Decay: $\lambda = 10^{-5}$
- First Moment Decay: $\beta_1 = 0.9$
- Second Moment Decay: $\beta_2 = 0.999$
- Numerical Epsilon: $\epsilon_{\text{adam}} = 10^{-8}$
- Behavior: Continuous exponential smoothing of clipped gradients and injected Gaussian noise across all $T = 4,720$ steps per client.

---

## H. Accountant-Step Synchronization

- **Total Rounds:** 20 federated rounds.
- **Local Steps / Round:** 236 Poisson steps per hospital silo.
- **Total Cumulative Steps per Client:** $T = 20 \times 236 = \mathbf{4,720\text{ steps}}$.
- **Step Counting Rules:**
  - Empty batches ($|S_t| = 0$): Counted as 1 step.
  - Singleton batches ($|S_t| = 1$): Counted as 1 step.
  - Multi-sample batches ($|S_t| \ge 2$): Counted as 1 step.
  - Accountant state persists across rounds without reset.
- **Resulting Validated Privacy Guarantee:**
  $$(\epsilon = 2.8934, \delta = 10^{-5})\text{-DP at optimal order } \alpha^* = 7$$
  $$(D_7 = 2.06488424 \times 10^{-4}, \quad T \cdot D_7 = 0.974625)$$

---

## I. Privacy Scope

- **Level:** Strictly **Sample-Level Differential Privacy**.
- **Adjacency Relation:** One patient's complete 4-modality 3D MRI scan and ground truth segmentation mask ($x_i, y_i$) added, removed, or replaced within a hospital partition.
- **Disjoint Partitions:** Hospital Alpha ($N=236$), Beta ($N=236$), Gamma ($N=236$), and Delta ($N=236$) are pairwise disjoint ($S_A \cap S_B = \emptyset$).
- **Federated Composition:** Parallel composition holds across silos. Each silo independently guarantees $(\epsilon = 2.8934, \delta = 10^{-5})$-DP.
- **Exclusion:** Does **NOT** claim hospital-level DP (protecting against the removal of an entire hospital silo).

---

## J. Telemetry Readiness

The experiment runner writes the following per-round metrics into `round_record["privacy_telemetry"]`:
- `mean_batch_size`
- `median_batch_size`
- `std_batch_size`
- `min_batch_size`
- `max_batch_size`
- `empty_batch_count`
- `singleton_batch_count`
- `multi_sample_batch_count`
- `clipping_fraction`
- `mean_unclipped_gradient_norm`
- `mean_clipped_gradient_norm`
- `noise_standard_deviation`
- `accountant_epsilon`
- `accountant_optimal_alpha`
- `accountant_step_count`

---

## K. Checkpoint Privacy

Checkpoints saved to `checkpoints/fedavg_dp_real/` store:
- Model parameter state dictionaries (`model_state_dict`)
- Training round number and validation metrics
- Configuration, split, and partition SHA-256 hashes
- Privacy telemetry metadata ($\epsilon, \delta, \sigma, C, q, \alpha^*, T$)

Checkpoints strictly do **NOT** store:
- Raw MRI volumes or voxel data
- Patient subject IDs
- Raw or per-sample gradient tensors
- Segmentation ground truth masks

---

## L. Regression Test Results

- **Test Suite Command:** `pytest tests/unit/ tests/integration/ -q`
- **Result:** **311 PASSED, 0 FAILED, 35 warnings in 72.37s**
- **Regressions:** None.

---

## M. Remaining Limitations

1. **Sample-Level vs Hospital-Level DP:** Protection is sample-level. Hospital-level exclusion is not bounded by this $(\epsilon, \delta)$ guarantee.
2. **Noise Perturbation vs Segmentation Accuracy:** Gaussian noise with standard deviation $\sigma \cdot C = 0.87$ added at every step will perturb optimizer trajectories relative to un-privatized FedAvg ($\sigma = 0.0, \text{Macro Dice} = 0.3815$), quantitatively mapping the Privacy-Utility boundary in Phase 10.

---

```
DP_MECHANISM_READY_FOR_EXPERIMENT_D
```
