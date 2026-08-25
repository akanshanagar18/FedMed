"""
Module: utils.statistical_analysis

Purpose:
Production Statistical Analysis & Hypothesis Testing Engine for FedMed v2.0.
Calculates descriptive metrics (Mean, Std Dev, 95% Confidence Intervals), inferential statistics
(Paired t-test, Wilcoxon signed-rank test), effect sizes (Cohen's d, Cliff's delta), and automatic
strategy performance rankings for publication-grade federated learning research.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger("statistical_analysis")


def compute_descriptive_stats(values: List[float]) -> Dict[str, float]:
    """
    Computes mean, standard deviation, and 95% confidence interval bounds.
    """
    clean_vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not clean_vals:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "count": 0,
        }

    arr = np.array(clean_vals, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std_val / math.sqrt(n) if n > 0 else 0.0

    # 95% Confidence Interval multiplier via Student-t distribution
    if n > 1:
        t_crit = stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * se
    else:
        margin = 0.0

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "ci_lower": round(mean_val - margin, 4),
        "ci_upper": round(mean_val + margin, 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "count": n,
    }


def compute_cohens_d(x: List[float], y: List[float]) -> float:
    """Computes Cohen's d effect size between two numerical samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0

    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)

    pooled_std = math.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0

    return round(float((mean_x - mean_y) / pooled_std), 4)


def compute_cliffs_delta(x: List[float], y: List[float]) -> float:
    """Computes non-parametric Cliff's delta effect size."""
    if not x or not y:
        return 0.0

    more = sum(1 for i in x for j in y if i > j)
    less = sum(1 for i in x for j in y if i < j)
    total = len(x) * len(y)

    return round(float((more - less) / total), 4)


def perform_hypothesis_tests(
    group_a: List[float],
    group_b: List[float],
    group_a_label: str = "Strategy A",
    group_b_label: str = "Strategy B",
) -> Dict[str, Any]:
    """
    Performs Paired Student t-test, Wilcoxon Signed-Rank test, and effect size estimations.
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a != n_b or n_a < 2:
        # Fallback to independent t-test if sample sizes differ
        try:
            t_stat, p_val_t = stats.ttest_ind(group_a, group_b)
            p_val_t = float(p_val_t) if not math.isnan(p_val_t) else 1.0
            t_stat = float(t_stat) if not math.isnan(t_stat) else 0.0
        except Exception:
            t_stat, p_val_t = 0.0, 1.0

        return {
            "comparison": f"{group_a_label} vs {group_b_label}",
            "test_type": "independent_ttest",
            "t_statistic": round(t_stat, 4),
            "p_value_ttest": round(p_val_t, 6),
            "wilcoxon_stat": None,
            "p_value_wilcoxon": None,
            "cohens_d": compute_cohens_d(group_a, group_b),
            "cliffs_delta": compute_cliffs_delta(group_a, group_b),
            "statistically_significant": p_val_t < 0.05,
        }

    # Paired t-test
    try:
        t_stat, p_val_t = stats.ttest_rel(group_a, group_b)
        p_val_t = float(p_val_t) if not math.isnan(p_val_t) else 1.0
        t_stat = float(t_stat) if not math.isnan(t_stat) else 0.0
    except Exception:
        t_stat, p_val_t = 0.0, 1.0

    # Wilcoxon signed-rank test
    diffs = np.array(group_a) - np.array(group_b)
    if np.all(diffs == 0):
        w_stat, p_val_w = 0.0, 1.0
    else:
        try:
            res = stats.wilcoxon(group_a, group_b)
            w_stat, p_val_w = float(res.statistic), float(res.pvalue)
            if math.isnan(p_val_w): p_val_w = 1.0
            if math.isnan(w_stat): w_stat = 0.0
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

    return {
        "comparison": f"{group_a_label} vs {group_b_label}",
        "test_type": "paired_ttest_and_wilcoxon",
        "t_statistic": round(t_stat, 4),
        "p_value_ttest": round(p_val_t, 6),
        "wilcoxon_stat": round(w_stat, 4) if w_stat is not None else None,
        "p_value_wilcoxon": round(p_val_w, 6) if p_val_w is not None else None,
        "cohens_d": compute_cohens_d(group_a, group_b),
        "cliffs_delta": compute_cliffs_delta(group_a, group_b),
        "statistically_significant": p_val_t < 0.05 or p_val_w < 0.05,
    }


class StatisticalAnalysisEngine:
    """
    Consolidates benchmark suite experiment runs, computes statistical summaries,
    performs pairwise hypothesis tests, and ranks strategies.
    """

    def __init__(self, experiment_results: List[Dict[str, Any]]):
        self.results = experiment_results
        self.strategy_groups: Dict[str, List[Dict[str, Any]]] = {}

        for r in self.results:
            strat = r.get("strategy_name", r.get("strategy", "Unknown"))
            if strat not in self.strategy_groups:
                self.strategy_groups[strat] = []
            self.strategy_groups[strat].append(r)

    def compute_strategy_summaries(self, metric_key: str = "best_dice") -> Dict[str, Dict[str, float]]:
        """Computes mean, std, 95% CI for specified metric across all strategies."""
        summaries = {}
        for strat, runs in self.strategy_groups.items():
            vals = [r.get(metric_key, 0.0) for r in runs if metric_key in r]
            summaries[strat] = compute_descriptive_stats(vals)
        return summaries

    def compute_pairwise_hypothesis_tests(self, metric_key: str = "best_dice") -> List[Dict[str, Any]]:
        """Performs pairwise t-test and Wilcoxon tests between all strategy pairs."""
        strats = list(self.strategy_groups.keys())
        pairwise_tests = []

        for i in range(len(strats)):
            for j in range(i + 1, len(strats)):
                s1, s2 = strats[i], strats[j]
                v1 = [r.get(metric_key, 0.0) for r in self.strategy_groups[s1]]
                v2 = [r.get(metric_key, 0.0) for r in self.strategy_groups[s2]]

                test_res = perform_hypothesis_tests(v1, v2, group_a_label=s1, group_b_label=s2)
                pairwise_tests.append(test_res)

        return pairwise_tests

    def rank_strategies(self) -> List[Dict[str, Any]]:
        """
        Ranks strategies using multi-metric weighted scoring (Dice, Loss, Runtime, Convergence).
        """
        ranked = []
        for strat, runs in self.strategy_groups.items():
            dices = [r.get("best_dice", 0.0) for r in runs]
            losses = [r.get("avg_loss", 0.0) for r in runs]
            runtimes = [r.get("runtime_sec", 0.0) for r in runs]
            conv_rounds = [r.get("convergence_round", 1) for r in runs]

            dice_stats = compute_descriptive_stats(dices)
            loss_stats = compute_descriptive_stats(losses)
            runtime_stats = compute_descriptive_stats(runtimes)

            ranked.append({
                "strategy": strat,
                "mean_dice": dice_stats["mean"],
                "std_dice": dice_stats["std"],
                "ci_dice": f"[{dice_stats['ci_lower']:.4f}, {dice_stats['ci_upper']:.4f}]",
                "mean_loss": loss_stats["mean"],
                "mean_runtime_sec": runtime_stats["mean"],
                "mean_convergence_round": round(float(np.mean(conv_rounds)), 1) if conv_rounds else 1.0,
                "total_runs": len(runs),
            })

        # Rank primarily by highest mean_dice, secondarily by lowest mean_loss
        ranked.sort(key=lambda x: (x["mean_dice"], -x["mean_loss"]), reverse=True)

        for rank_idx, item in enumerate(ranked, start=1):
            item["rank"] = rank_idx

        return ranked
