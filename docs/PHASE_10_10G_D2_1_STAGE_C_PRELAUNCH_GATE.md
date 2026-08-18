# FEDMED OS — PHASE 10.10G: EXPERIMENT D2.1 STAGE C PRE-LAUNCH FORENSIC GATE

**Experiment Target**: `EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3`  
**Parent Baseline**: `EXPERIMENT_D2_DP_SGD_MOMENTUM`  
**Execution Mode**: **STRICTLY READ-ONLY FORENSIC AUDIT (STAGE C HELD)**  
**Gate Status**: **`D2_1_STAGE_C_READY`**  
**Audit Timestamp**: August 18, 2026

---

## 1. Executive Summary & Verification Matrix

| Pre-Launch Requirement | Forensic Inspection Findings | Compliance Status |
| :--- | :--- | :---: |
| **Stage B Promotion Gate** | Round 3 Macro Dice = **`0.0291`** (Target: $> 0.0250$), Val Loss = **`0.9832`** | **PASS** |
| **Single Causal Intervention** | Exactly 1 parameter differs from D2: `training.learning_rate: 1e-4 -> 1e-3` | **PASS** |
| **Invariant Freezing** | SGD ($\mu = 0.9, \lambda = 10^{-5}$), $C = 0.06, \sigma = 0.87, q = 1/236, \delta = 10^{-5}$, Seed $42$ | **PASS** |
| **Privacy Accountant** | Analytical Poisson RDP: $T = 4,720 \implies \varepsilon = 2.8934$ ($\alpha = 7, \delta = 10^{-5}$) | **PASS** |
| **Model & Parameters** | MONAI 3D U-Net ($128^3$), strictly $4,810,074$ trainable parameters | **PASS** |
| **Cohort & Silo Structure** | 4 Hospital Silos ($236$ subjects/silo = $944$ train), $202$ validation, $204$ firewalled test | **PASS** |
| **Test Firewall** | `TRAINING_TEST_ACCESSES = 0`, zero test leakage | **PASS** |
| **Numerical Sanity** | $0\text{ NaN}, 0\text{ Inf}, 0\text{ OOM}$, peak MPS allocation $180.95\text{ MB}$ | **PASS** |
| **Baseline Immutability** | 100% SHA-256 integrity match across all D, D1, D2, and D2.1 Stage B artifacts | **PASS** |
| **Namespace Isolation** | Dedicated `fedavg_dp_d2_1` paths; zero collision with baseline namespaces | **PASS** |
| **Process State** | Zero conflicting background training processes running | **PASS** |

---

## 2. Initialization & Checkpoint Policy

- **Specification**: In FedMed OS, the 3-round Stage B run serves as a formal screening gate over the first 3 rounds ($T=708, \varepsilon=1.9636$) of the designated 20-round experiment ledger.
- **Audited Policy**:
  - `STAGE_C_INITIALIZATION_POLICY = STAGE_B_CHECKPOINT_RESUME` (Default)  
    The canonical runner (`scripts/run_real_fedavg_dp_d2_1_experiment.py --rounds 20`) detects `checkpoints/fedavg_dp_d2_1/latest.pt`, loads the audited Round 3 global weights and history, seamlessly advances from Round 4 to Round 20, and achieves the exact analytical budget ($T=4,720, \varepsilon=2.8934$).
  - If executed with `--no-resume`, it initiates clean random weights from `seed: 42` and runs Rounds 1–20 from $t=0$, producing the identical deterministic trajectory.

---

## 3. Configuration & Parameter Verification

```yaml
experiment_name: "EXPERIMENT_D2.1_DP_SGD_MOMENTUM_LR1E3"
parent_baseline: "EXPERIMENT_D2_DP_SGD_MOMENTUM"

# SINGLE CAUSAL INTERVENTION RELATIVE TO D2
learning_rate: 0.001       # 1e-3 (was 1e-4 in D2)

# STRICTLY FROZEN BENCHMARK INVARIANTS
optimizer: "SGD"
momentum: 0.9
weight_decay: 0.00001
model_architecture: "MONAI 3D U-Net (128^3)"
trainable_parameters: 4810074
loss: "DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=0.2)"
batch_size: 1
local_epochs: 1
rounds: 20
hospitals: ["hospital_alpha", "hospital_beta", "hospital_gamma", "hospital_delta"]
subjects_per_silo: 236
train_subjects: 944
validation_subjects: 202
locked_test_subjects: 204
seed: 42

# DIFFERENTIAL PRIVACY LEDGER (FROZEN)
max_grad_norm_C: 0.06
noise_multiplier_sigma: 0.87
physical_noise_std: 0.0522
sample_rate_q: 0.004237288
target_delta: 0.00001
total_steps_T: 4720
final_epsilon_20_rounds: 2.8934
optimal_renyi_alpha: 7
```

---

## 4. Protected Baseline Hashes Audit

| Protected File | Expected SHA-256 | Audited SHA-256 | Integrity Status |
| :--- | :---: | :---: | :---: |
| `configs/experiments/real_brats_dp.yaml` | `6b7699951687...` | `6b7699951687654965935db414338313260e65de2f61f2c1dfb44a74c2bb5ef7` | **100% UNTOUCHED** |
| `configs/experiments/real_brats_dp_d1.yaml` | `a7def27c80d1...` | `a7def27c80d1d10b738b9f7f4e8dc4e06a4cf0a39a00fb08c13f08f011763520` | **100% UNTOUCHED** |
| `configs/experiments/real_brats_dp_d2.yaml` | `80dc6726110e...` | `80dc6726110e1ceabff279ca60b78a39a0be4f4b708b294f7a3e2669199d368c` | **100% UNTOUCHED** |
| `configs/experiments/real_brats_dp_d2_1.yaml` | `c1229d6cb041...` | `c1229d6cb041d595b4b6d11c3c7ab40c928459a2ad5a0bd2bdd501751fbd292e` | **100% UNTOUCHED** |
| `checkpoints/fedavg_dp_real/best.pt` | `3684075cc207...` | `3684075cc20783c86354ab3ae38b3c9d91c543e79e8b9c6a2b3df0a247654086` | **100% UNTOUCHED** |
| `reports/real_brats2024/fedavg_dp_real.json` | `78e54472bb44...` | `78e54472bb44de3f330e1d78ee9829f17a993b471ac93987f576dc5334de3f5b` | **100% UNTOUCHED** |
| `reports/real_brats2024/dp_experiment_manifest.json` | `382103fabe65...` | `382103fabe657bc58a985abb14565e38b93f5edda192f5f4178042e7191f9104` | **100% UNTOUCHED** |
| `reports/real_brats2024/fedavg_dp_d1.json` | `f64407613799...` | `f64407613799cd111215e46371f66ca3aca281e6cccd169fc35af0ae8655e56a` | **100% UNTOUCHED** |
| `checkpoints/fedavg_dp_d2/best.pt` | `bbc1d1e76a9e...` | `bbc1d1e76a9e4bbc96125e94c17c1dd17e4eb92bf652b171547fff7ae71f4493` | **100% UNTOUCHED** |
| `reports/real_brats2024/fedavg_dp_d2.json` | `25608e9d69c1...` | `25608e9d69c17d19c2d9bcda42606b6a78ba473480132223792483be68b13cc2` | **100% UNTOUCHED** |
| `checkpoints/fedavg_dp_d2_1/best.pt` | `fb96cb7b4fb1...` | `fb96cb7b4fb1866280e57fe35d376506c79bda195ecb5489862b2af0561efd00` | **100% UNTOUCHED** |
| `reports/real_brats2024/fedavg_dp_d2_1.json` | `01908f5b7866...` | `01908f5b7866e31e5250d683c44ea865de250238dcdb18a1e3829b1833345a06` | **100% UNTOUCHED** |

---

## 5. Runtime Projections for Stage C (20 Rounds)

- **Observed Stage B Round Timings**: R1 = $976.33\text{ s}$, R2 = $1056.58\text{ s}$, R3 = $1097.34\text{ s}$
- **Observed Average Round Duration**: **`1043.42 s`** (~$17.39\text{ min}$)
- **Observed Validation Overhead per Round**: ~$103.0\text{ s}$ (~$1.72\text{ min}$)
- **Stage C Projections**:
  - **Optimistic Estimate** ($16.0\text{ min/round} \times 20$): **`5.33 hours`**
  - **Expected Estimate** ($17.39\text{ min/round} \times 20$): **`5.80 hours`** (or **`4.93 hours`** for resumed Rounds 4–20)
  - **Conservative Estimate** ($19.0\text{ min/round} \times 20$): **`6.33 hours`**

---

## 6. Stage C Monitoring Contract & Live Stability Safeguards

During Stage C execution, telemetry will track:
1. **Parameter Displacement Stability**: Confirming $\Delta_{\text{global}}$ remains bounded in the expected $\sim 29.5 \pm 0.5$ range without explosion.
2. **Client Update Divergence**: Confirming divergence remains bounded at $\sim 83.5 \pm 1.0$.
3. **Validation Trajectory**: Tracking monotonic descent of validation loss (below $0.9832$) and expansion of Macro Dice (from $0.0291$).
4. **Privacy Budget**: Advancing $T$ by $+236$ steps per round, matching exact RDP composition to $T=4,720 \implies \varepsilon = 2.8934$.
5. **Firewall Invariance**: Zero test set accesses (`TRAINING_TEST_ACCESSES = 0`).

---

## Final Pre-Launch Decision

```
D2_1_STAGE_C_READY
```
