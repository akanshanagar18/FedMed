# FedMed Research Publication Report: FedMed Strategy Benchmark
**Benchmark ID:** `bm_1786211336` | **Date:** 2026-08-08  
**Git Commit:** `49bd4ab4ec24123e61c80e5e2b2c864c4b0f190d` (Branch: `feature/integration-green`)  
**Environment:** Python 3.9.6 | PyTorch 2.8.0 | MONAI 1.5.2 | Flower 1.23.0  

## 1. Strategy Rankings & Statistical Metrics
| Rank | Strategy | Mean Dice (± Std) | 95% CI | Mean Loss | Mean Runtime (s) | Mean Conv Round | Total Runs |
|---|---|---|---|---|---|---|---|
| #1 | **FedAvg** | 0.4482 ± 0.0000 | [0.4482, 0.4482] | 0.8043 | 60.04s | Round 9.0 | 1 |
| #2 | **FedProx** | 0.4482 ± 0.0000 | [0.4482, 0.4482] | 0.8043 | 3.61s | Round 9.0 | 1 |

## 2. Hypothesis Testing & Significance Analysis
| Comparison | t-statistic | p-value (t-test) | Cohen's d | Statistically Significant |
|---|---|---|---|---|
| FedAvg vs FedProx | 0.0 | 1.0 | 0.0 | No |

## 3. Detailed Experiment Leaderboard
| Rank | Experiment ID | Strategy | Partition | Seed | Best Dice | Avg Loss | Runtime (s) |
|---|---|---|---|---|---|---|---|
| 1 | `bm_1786211336_exp_001_FedAvg_42` | FedAvg | IID | 42 | **0.4482** | 0.8043 | 60.04s |
| 2 | `bm_1786211336_exp_002_FedProx_42` | FedProx | IID | 42 | **0.4482** | 0.8043 | 3.61s |