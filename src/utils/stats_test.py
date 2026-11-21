"""
Statistical Testing and Results Formatting for Tri-SACTA.

Provides:
- Paired t-test and Wilcoxon signed-rank test.
- Mean +/- Std aggregation across multiple seeds.
- Effect size computation (Cohen's d).
- Publication-ready LaTeX and Markdown summary tables.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    metric_name: str = "metric",
) -> Dict[str, Any]:
    """Perform a two-sided paired t-test between two methods across identical seeds."""
    a, b = np.array(scores_a), np.array(scores_b)
    if len(a) < 2 or np.all(a == b):
        return {"t_stat": 0.0, "p_value": 1.0, "significant": False, "metric": metric_name}

    t_stat, p_val = stats.ttest_rel(a, b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
        "metric": metric_name,
    }


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
    metric_name: str = "metric",
) -> Dict[str, Any]:
    """Perform a Wilcoxon signed-rank test between two paired condition vectors."""
    a, b = np.array(scores_a), np.array(scores_b)
    diff = a - b
    if np.all(diff == 0) or len(diff) < 3:
        return {"w_stat": 0.0, "p_value": 1.0, "significant": False, "metric": metric_name}

    try:
        w_stat, p_val = stats.wilcoxon(a, b)
        return {
            "w_stat": float(w_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
            "metric": metric_name,
        }
    except Exception as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        return {"w_stat": 0.0, "p_value": 1.0, "significant": False, "metric": metric_name}


def compute_effect_size(scores_a: List[float], scores_b: List[float]) -> float:
    """Compute Cohen's d effect size for paired samples."""
    a, b = np.array(scores_a), np.array(scores_b)
    diff = a - b
    s_d = np.std(diff, ddof=1)
    if s_d == 0:
        return 0.0
    return float(np.mean(diff) / s_d)


def compute_mean_std_over_seeds(results_by_seed: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metric results across seeds into mean +/- std."""
    metrics_collected: Dict[str, List[float]] = {}
    for seed, seed_metrics in results_by_seed.items():
        for m_name, val in seed_metrics.items():
            if isinstance(val, (int, float)):
                metrics_collected.setdefault(m_name, []).append(float(val))

    aggregated = {}
    for m_name, vals in metrics_collected.items():
        aggregated[m_name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "raw": vals,
        }
    return aggregated


def format_results_table(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    methods: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    as_latex: bool = True,
) -> str:
    """Format aggregated results into a publication-ready LaTeX or Markdown table.
    
    Args:
        all_results: {method: {metric: {'mean': float, 'std': float}}}
        methods: List of method keys in display order.
        metrics: List of metric keys in display order.
        as_latex: If True, outputs LaTeX table; else Markdown table.
    """
    if methods is None:
        methods = list(all_results.keys())
    if metrics is None:
        metrics = ["macro_f1", "accuracy", "empirical_coverage", "coverage_gap", "avg_set_size", "ece"]

    if as_latex:
        col_spec = "l" + "c" * len(metrics)
        header = " & ".join(["Method"] + [m.replace("_", " ").title() for m in metrics]) + " \\\\"
        rows = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            header,
            "\\midrule",
        ]
        for m in methods:
            row_items = [m.replace("_", "-").upper()]
            for met in metrics:
                if m in all_results and met in all_results[m]:
                    stats_dict = all_results[m][met]
                    mean = stats_dict["mean"]
                    std = stats_dict["std"]
                    row_items.append(f"${mean:.3f} \\pm {std:.3f}$")
                else:
                    row_items.append("-")
            rows.append(" & ".join(row_items) + " \\\\")

        rows.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Continual Compound Shift Performance Across 5 Random Seeds.}",
            "\\label{tab:main_results}",
            "\\end{table}",
        ])
        return "\n".join(rows)

    # Markdown format
    header = "| Method | " + " | ".join([m.replace("_", " ").title() for m in metrics]) + " |"
    sep = "| :--- | " + " | ".join([":---:" for _ in metrics]) + " |"
    md_rows = [header, sep]
    for m in methods:
        row_items = [f"**{m}**"]
        for met in metrics:
            if m in all_results and met in all_results[m]:
                stats_dict = all_results[m][met]
                row_items.append(f"{stats_dict['mean']:.3f} +/- {stats_dict['std']:.3f}")
            else:
                row_items.append("-")
        md_rows.append("| " + " | ".join(row_items) + " |")

    return "\n".join(md_rows)
