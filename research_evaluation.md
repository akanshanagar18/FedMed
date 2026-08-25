# FedMed v2.0 — Scientific Research Evaluation & Statistical Analysis Platform

## Overview
FedMed v2.0 incorporates a rigorous statistical hypothesis testing, publication visualization, and multi-dimensional benchmark matrix evaluation platform.

---

## 1. Statistical Analysis Methodology

### A. Descriptive Statistics
For every strategy $S$, metrics $M = \{m_1, m_2, \dots, m_n\}$ across random seeds and partitions are summarized by:
- **Sample Mean ($\mu$):** $\mu = \frac{1}{n} \sum_{i=1}^n m_i$
- **Sample Standard Deviation ($\sigma$):** $\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (m_i - \mu)^2}$
- **95% Confidence Interval ($95\%\text{ CI}$):** $[\mu - t_{0.975, n-1} \cdot \frac{\sigma}{\sqrt{n}}, \, \mu + t_{0.975, n-1} \cdot \frac{\sigma}{\sqrt{n}}]$

### B. Inferential Hypothesis Testing
To validate whether Strategy $A$ statistically outperforms Strategy $B$:
- **Paired Student $t$-test:** Tests null hypothesis $H_0: \mu_A = \mu_B$.
- **Wilcoxon Signed-Rank Test:** Non-parametric test for non-normal metric distributions.
- **Significance Level ($\alpha$):** Results are declared statistically significant when $p < 0.05$.

### C. Effect Size Estimation
- **Cohen's $d$:** Parametric standardized mean difference:
  $$d = \frac{\mu_A - \mu_B}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_A - 1)s_A^2 + (n_B - 1)s_B^2}{n_A + n_B - 2}}$$
- **Cliff's $\delta$:** Non-parametric effect size measuring probability that a value from sample $A$ is greater than a value from sample $B$.

---

## 2. Publication Figure Visualizations

The `BenchmarkVisualizer` automatically outputs 10 high-resolution publication figures (in both high-DPI `.png` and vector `.pdf` formats):
1. **Figure 1:** Loss vs Round Convergence (`fig1_loss_vs_round.png/.pdf`)
2. **Figure 2:** Dice Similarity vs Round (`fig2_dice_vs_round.png/.pdf`)
3. **Figure 3:** Mean IoU Progression (`fig3_iou_vs_round.png/.pdf`)
4. **Figure 4:** Execution Runtime by Strategy (`fig4_runtime_vs_strategy.png/.pdf`)
5. **Figure 5:** Communication Payload Overhead (`fig5_communication_cost.png/.pdf`)
6. **Figure 6:** Privacy-Utility Trade-off Curve ($\epsilon$ vs Dice) (`fig6_privacy_budget_tradeoff.png/.pdf`)
7. **Figure 7:** Homomorphic Encryption (CKKS) Latency (`fig7_encryption_overhead.png/.pdf`)
8. **Figure 8:** IID vs Non-IID Dirichlet Heterogeneity (`fig8_iid_vs_non_iid.png/.pdf`)
9. **Figure 9:** FedAvg vs FedProx Convergence (`fig9_fedavg_vs_fedprox.png/.pdf`)
10. **Figure 10:** Client Scalability Sweep (`fig10_scalability.png/.pdf`)

---

## 3. Publication Export Artifacts

For each benchmark run `results/benchmark_<id>/`:
- `benchmark_report.pdf`: Publication PDF report.
- `benchmark_report.md`: Markdown summary.
- `benchmark_summary.json`: Aggregated JSON schema.
- `leaderboard.csv`: Complete experiment matrix rankings.
- `runtime.csv`: Per-experiment runtime breakdown.
- `privacy.csv`: Differential Privacy & Homomorphic Encryption parameters.
- `communication.csv`: Network communication bytes transferred per round.
- `statistical_tests.csv`: Paired t-tests, Wilcoxon $p$-values, and effect sizes.
- `tables.tex`: Ready-to-use LaTeX tables for IEEE / ACM / Springer / NeurIPS / MICCAI research paper submissions.
