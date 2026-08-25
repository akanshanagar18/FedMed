# FEDMED OS — METRIC PROVENANCE & REALITY AUDIT (PHASE 8.5B.1)

**Audit Date:** August 15, 2026  
**Auditor:** Senior ML Systems Engineer  
**Objective:** Audit all occurrences of metrics, lookup profiles, and formulas across the codebase to ensure zero fake or static metrics exist in the active execution path.

---

## 1. Classification Methodology

Every metric occurrence in the repository is classified under one of four explicit categories:
1. **`ACTIVE_EXECUTION_PATH`**: Code actively executed during real training, validation, federated aggregation, or API telemetry. **MUST ONLY USE GENUINE MEASURED TENSOR METRICS.**
2. **`HISTORICAL_ARTIFACT`**: Frozen historical demo outputs or archived runs retained for auditability.
3. **`DEVELOPMENT_TEST`**: Deterministic synthetic unit or mock tests verifying class signatures and fallback handlers.
4. **`DOCUMENTATION`**: Markdown files, audit reports, or schema documentation.

---

## 2. Metric Provenance Audit Table

| Source File | Line(s) | Metric Pattern / Value | Classification | Assessment & Status |
| :--- | :--- | :--- | :--- | :--- |
| `evaluation/metrics.py` | 38-160 | `compute_dice()`, `compute_iou()` | `ACTIVE_EXECUTION_PATH` | **GENUINE**: Evaluates binary intersection/union independently from raw model logits and target masks. No proxy formula. |
| `model/trainer.py` | 100-112 | `avg_dice = total_dice / num_batches` | `ACTIVE_EXECUTION_PATH` | **GENUINE**: Accumulates true batch-level Dice computed by MONAI `DiceCELoss` during local epoch. |
| `server/strategies/adapters/flower_adapter.py` | 131-143 | `metrics.get("training_loss")`, `metrics.get("dice_score")` | `ACTIVE_EXECUTION_PATH` | **GENUINE**: Extracts real client metrics and genuine validation metrics. Heuristic multiplier removed in Phase 8.5A. |
| `evaluation/benchmark_runner.py` | 31-45 | `algo_profiles = {"FedAvg": ...}` | `HISTORICAL_ARTIFACT` | **NON-ACTIVE**: Historical benchmark profile lookup; superseded in Phase 8.5B by genuine experiment orchestration. |
| `scripts/generate_evidence.py` | 24-80 | Static sample metrics | `HISTORICAL_ARTIFACT` | **NON-ACTIVE**: Retained only as Phase 10 demo artifact; excluded from 8.5B real validation. |
| `analytics/executive_reports.py` | 24 | `avg_dice_score: float = 0.865` | `DEVELOPMENT_TEST` | **DEFAULT FALLBACK**: Default parameter value for CLI signature when called without args in demo mode. |
| `tests/unit/test_executive_reports.py` | 16 | `avg_dice_score=0.87` | `DEVELOPMENT_TEST` | **TEST FIXTURE**: Test input scalar for PDF report formatting. |
| `dashboard/frontend/src/App.jsx` | 808 | `'0.0420'` | `ACTIVE_EXECUTION_PATH` | **SAFE FALLBACK**: Fallback string only if backend is disconnected; populated by real API response `/api/v1/governance/drift`. |

---

## 3. Active Execution Path Guarantee

In the Phase 8.5B execution path:
- **Zero hardcoded Dice scores**
- **Zero hardcoded IoU scores**
- **Zero formulaic coupling** between Dice and IoU ($\text{IoU} \neq \frac{\text{Dice}}{2 - \text{Dice}}$)
- **Zero fabricated patient data**
