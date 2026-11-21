"""
05_aggregate_results.py: Multi-Seed Statistical Aggregation and Q1 Table Generator.

Aggregates experiment results across seeds to produce:
1. Mean ± Std tables (Macro-F1, Accuracy, Coverage, Coverage Gap, Set Size, Peak VRAM, Latency).
2. Paired t-test and Wilcoxon signed-rank test p-values against baselines.
3. Publication-ready Markdown and LaTeX table outputs.
"""

from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_runs(input_dir: str, target_data_mode: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load and group run results by configuration key, strictly filtering by data_mode (AUD-10)."""
    grouped = defaultdict(list)
    files = glob.glob(os.path.join(input_dir, "*.json"))

    for fpath in files:
        # Ignore legacy runs preserved in results/legacy/ or un-tagged legacy files
        if "legacy" in fpath or "_mosei_seed42.json" in fpath:
            continue
        with open(fpath, "r") as f:
            data = json.load(f)

        data_mode = data.get("data_mode", "synthetic")
        if target_data_mode and data_mode != target_data_mode:
            continue

        method = data.get("method", "unknown")
        conformal = data.get("conformal", "unknown")
        delay = data.get("delay", 0)
        key = f"{method}_{conformal}_d{delay}"
        grouped[key].append(data)

    return grouped


def format_mean_std(values: List[float], scale: float = 1.0, is_pct: bool = False) -> str:
    """Format list of floats as 'Mean ± Std'."""
    arr = np.array(values) * scale
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    if is_pct:
        return f"{mean:.2f} ± {std:.2f}%"
    return f"{mean:.4f} ± {std:.4f}"


def run_statistical_tests(ours_values: List[float], baseline_values: List[float]) -> Tuple[float, float]:
    """Compute Paired t-test and Wilcoxon signed-rank test p-values."""
    if len(ours_values) < 2 or len(ours_values) != len(baseline_values):
        return 1.0, 1.0

    diff = np.array(ours_values) - np.array(baseline_values)
    if np.all(diff == 0.0):
        return 1.0, 1.0

    # Paired t-test
    _, t_pval = stats.ttest_rel(ours_values, baseline_values)

    # Wilcoxon signed-rank test
    try:
        _, w_pval = stats.wilcoxon(ours_values, baseline_values)
    except Exception:
        w_pval = 1.0

    return float(t_pval), float(w_pval)


def generate_tables(grouped: Dict[str, List[Dict[str, Any]]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    ours_key = "tri_sacta_online_aci_d0"
    ours_runs = grouped.get(ours_key, [])
    ours_seeds = {r["seed"]: r for r in ours_runs}

    for key, runs in sorted(grouped.items()):
        seeds = [r["seed"] for r in runs]
        n_seeds = len(runs)
        f1s = [r["overall_metrics"]["macro_f1"] for r in runs]
        accs = [r["overall_metrics"]["accuracy"] for r in runs]
        covs = [r["overall_metrics"]["empirical_coverage"] for r in runs]
        gaps = [r["overall_metrics"]["coverage_gap"] for r in runs]
        sizes = [r["overall_metrics"]["avg_set_size"] for r in runs]
        vrams = [r["efficiency"]["peak_vram_mb"] for r in runs]
        lats = [r["efficiency"]["avg_latency_ms_per_sample"] for r in runs]

        # Calculate p-value against ours on matched seeds
        t_pval_str = "-"
        if key != ours_key and len(runs) > 1:
            matched_ours = []
            matched_base = []
            for r in runs:
                s = r["seed"]
                if s in ours_seeds:
                    matched_base.append(r["overall_metrics"]["macro_f1"])
                    matched_ours.append(ours_seeds[s]["overall_metrics"]["macro_f1"])
            if len(matched_ours) >= 2:
                tp, wp = run_statistical_tests(matched_ours, matched_base)
                t_pval_str = f"p={tp:.3f}"

        row = {
            "key": key,
            "seeds": n_seeds,
            "macro_f1": format_mean_std(f1s),
            "acc": format_mean_std(accs, scale=100.0, is_pct=True),
            "coverage": format_mean_std(covs, scale=100.0, is_pct=True),
            "gap": format_mean_std(gaps, scale=100.0, is_pct=True),
            "set_size": f"{np.mean(sizes):.2f} ± {np.std(sizes, ddof=1) if len(sizes)>1 else 0.0:.2f}",
            "vram_mb": f"{np.mean(vrams):.1f} MB",
            "latency_ms": f"{np.mean(lats):.2f} ms",
            "stat_pval": t_pval_str,
        }
        summary_rows.append(row)

    # 1. Output Markdown Table
    md_path = os.path.join(output_dir, "multi_seed_statistical_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Tri-SACTA Multi-Seed Benchmark Summary (Mean ± Std)\n\n")
        f.write("| Method / Model | Seeds | Macro-F1 | Accuracy | Coverage (90%) | Coverage Gap | Set Size | Peak VRAM | Latency | Paired t-test |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in summary_rows:
            f.write(f"| `{r['key']}` | {r['seeds']} | **{r['macro_f1']}** | {r['acc']} | **{r['coverage']}** | {r['gap']} | {r['set_size']} | {r['vram_mb']} | {r['latency_ms']} | {r['stat_pval']} |\n")

    # 2. Output LaTeX Table
    tex_path = os.path.join(output_dir, "table_q1_publication.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Tri-SACTA Q1 Journal Publication Table\n")
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Continual Test-Time Adaptation and Online Conformal Prediction performance across compound distribution shift regimes (Mean $\\pm$ Std over seeds).}\n")
        f.write("\\label{tab:main_benchmark}\n")
        f.write("\\begin{tabular}{lcccccccc}\n\\toprule\n")
        f.write("Method & Macro-F1 $\\uparrow$ & Acc (\\%) $\\uparrow$ & Coverage (\\%) & Cov. Gap (\\%) $\\downarrow$ & $|\\mathcal{C}_t| \\downarrow$ & Peak VRAM & Latency & $p$-val \\\\\n\\midrule\n")
        for r in summary_rows:
            escaped_key = r['key'].replace('_', r'\_')
            f.write(f"{escaped_key} & {r['macro_f1']} & {r['acc']} & {r['coverage']} & {r['gap']} & {r['set_size']} & {r['vram_mb']} & {r['latency_ms']} & {r['stat_pval']} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    logger.info(f"Generated Markdown table: {md_path}")
    logger.info(f"Generated LaTeX publication table: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Multi-Seed Benchmark Results")
    parser.add_argument("--input_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/summary")
    parser.add_argument("--data-mode", type=str, default="synthetic", choices=["synthetic", "real", "all"])
    args = parser.parse_args()

    mode_filter = None if args.data_mode == "all" else args.data_mode
    grouped = load_runs(args.input_dir, target_data_mode=mode_filter)
    if not grouped:
        logger.warning(f"No result runs found in {args.input_dir} for data-mode {args.data_mode}")
        return

    generate_tables(grouped, args.output_dir)
    logger.info("Statistical aggregation completed successfully.")


if __name__ == "__main__":
    main()
