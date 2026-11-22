"""
05_aggregate_results.py: Multi-Seed Statistical Aggregation and Q1 Table Generator — V3 (R3-11).

Key scientific improvements in V3:
1. Strict Schema Grouping:
   Groups exclusively by (dataset, data_mode, protocol, method, conformal, delay).
   Prevents cross-dataset or cross-mode pollution.
2. Unique Seed Deduplication:
   Rejects or de-duplicates repeated runs for the same seed; reports true unique seed count.
3. Rigorous Uncertainty Reporting:
   When n=1: reports 'mean (n=1)' with NA variance, eliminating misleading '± 0.0000'.
   When n>=2: reports sample standard deviation (ddof=1) and paired statistical tests.
4. Independent Tables for Real vs Synthetic:
   Strictly separates synthetic pilot results from real dataset benchmarks.
"""

from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_runs(input_dir: str, target_data_mode: Optional[str] = None) -> Dict[Tuple[str, str, str, str, str, int], Dict[int, Dict[str, Any]]]:
    """Load and group runs by (dataset, data_mode, protocol, method, conformal, delay) with unique seed mapping.
    
    Strict scientific provenance enforcement (V5-01):
    - Strictly rejects any path containing 'quarantine' or 'quarantined'
    - Rejects files containing legacy, summary, or corrupted formats
    - Strictly prevents quarantined pseudo-real runs from entering real-mode aggregation
    """
    grouped: Dict[Tuple[str, str, str, str, str, int], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    
    files = glob.glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(input_dir, "*.json"))

    for fpath in files:
        norm_path = os.path.normpath(fpath).replace("\\", "/").lower()
        # V5-01: Strict quarantine rejection at file path level
        if "quarantin" in norm_path or "legacy" in norm_path or "summary" in norm_path or "_mosei_seed42.json" in norm_path:
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict) or "overall_metrics" not in data:
            continue

        # V5-01: Metadata-level quarantine rejection
        if data.get("quarantined", False) or "quarantined" in str(data.get("provenance", "")).lower():
            continue
        if data.get("is_synthetic_fixture", False):
            if target_data_mode == "real":
                continue

        dataset = data.get("dataset", "mosei")
        data_mode = data.get("data_mode", "synthetic")
        protocol = data.get("protocol", "prequential")
        method = data.get("method", "unknown")
        conformal = data.get("conformal", "unknown")
        delay = int(data.get("delay", 0))
        seed = int(data.get("seed", 42))

        if target_data_mode and data_mode != target_data_mode:
            continue

        group_key = (dataset, data_mode, protocol, method, conformal, delay)
        # Store by seed to ensure uniqueness (R3-11)
        grouped[group_key][seed] = data

    return grouped


def format_stat(values: List[float], scale: float = 1.0, is_pct: bool = False) -> str:
    """Format list of floats as 'Mean ± Std' for n>=2, or 'Mean (n=1)' for n=1 (R3-11)."""
    if len(values) == 0:
        return "N/A"
    arr = np.array(values) * scale
    mean = np.mean(arr)
    if len(arr) == 1:
        if is_pct:
            return f"{mean:.2f}% (n=1)"
        return f"{mean:.4f} (n=1)"
    
    std = np.std(arr, ddof=1)
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

    _, t_pval = stats.ttest_rel(ours_values, baseline_values)
    try:
        _, w_pval = stats.wilcoxon(ours_values, baseline_values)
    except Exception:
        w_pval = 1.0

    return float(t_pval), float(w_pval)


def generate_tables(grouped: Dict[Tuple[str, str, str, str, str, int], Dict[int, Dict[str, Any]]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Separate tables by (dataset, data_mode, protocol)
    contexts = set((k[0], k[1], k[2]) for k in grouped.keys())

    for dataset, data_mode, protocol in sorted(contexts):
        summary_rows = []
        ours_key = (dataset, data_mode, protocol, "tri_sacta", "online_aci", 0)
        ours_runs_by_seed = grouped.get(ours_key, {})

        matching_keys = [k for k in grouped.keys() if k[0] == dataset and k[1] == data_mode and k[2] == protocol]
        for key in sorted(matching_keys):
            _, _, _, method, conformal, delay = key
            runs_by_seed = grouped[key]
            unique_seeds = sorted(list(runs_by_seed.keys()))
            n_seeds = len(unique_seeds)
            runs = [runs_by_seed[s] for s in unique_seeds]

            f1s = [r["overall_metrics"]["macro_f1"] for r in runs]
            accs = [r["overall_metrics"]["accuracy"] for r in runs]
            covs = [r["overall_metrics"]["empirical_coverage"] for r in runs]
            gaps = [r["overall_metrics"]["coverage_gap"] for r in runs]
            sizes = [r["overall_metrics"]["avg_set_size"] for r in runs]
            vrams = [r["efficiency"]["peak_vram_mb"] for r in runs]
            latencies = [r["efficiency"]["avg_latency_ms_per_sample"] for r in runs]

            # Paired significance vs Tri-SACTA
            paired_f1_ours = []
            paired_f1_base = []
            for s in unique_seeds:
                if s in ours_runs_by_seed:
                    paired_f1_ours.append(ours_runs_by_seed[s]["overall_metrics"]["macro_f1"])
                    paired_f1_base.append(runs_by_seed[s]["overall_metrics"]["macro_f1"])

            t_p, w_p = run_statistical_tests(paired_f1_ours, paired_f1_base)

            row_name = f"{method} + {conformal}"
            if delay > 0:
                row_name += f" (Delay={delay})"

            summary_rows.append({
                "Method": row_name,
                "Seeds": str(n_seeds),
                "Macro-F1": format_stat(f1s),
                "Accuracy": format_stat(accs, scale=100.0, is_pct=True),
                "Coverage": format_stat(covs, scale=100.0, is_pct=True),
                "Coverage Gap": format_stat(gaps, scale=100.0, is_pct=True),
                "Set Size": format_stat(sizes),
                "Peak VRAM": f"{np.mean(vrams):.1f} MB" if vrams else "N/A",
                "Latency": f"{np.mean(latencies):.2f} ms" if latencies else "N/A",
                "p-val (t)": f"{t_p:.4f}" if n_seeds >= 2 and method != "tri_sacta" else "—",
                "p-val (W)": f"{w_p:.4f}" if n_seeds >= 2 and method != "tri_sacta" else "—",
            })

        # Generate Markdown table
        md_path = os.path.join(output_dir, f"summary_{dataset}_{data_mode}_{protocol}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Multi-Seed Benchmark Summary: {dataset.upper()} ({data_mode.upper()}, {protocol})\n\n")
            f.write(f"Generated from verified runs across unique random seeds.\n\n")
            headers = ["Method", "Seeds", "Macro-F1", "Accuracy", "Coverage", "Coverage Gap", "Set Size", "Peak VRAM", "Latency", "p-val (t)", "p-val (W)"]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for r in summary_rows:
                f.write("| " + " | ".join([r[h] for h in headers]) + " |\n")

        logger.info(f"Saved summary table to {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Multi-Seed Results (V3)")
    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/summary")
    parser.add_argument("--data-mode", type=str, default=None, choices=["synthetic", "real"])
    args = parser.parse_args()

    grouped = load_runs(args.input_dir, target_data_mode=args.data_mode)
    if not grouped:
        logger.warning(f"No valid run results found in {args.input_dir}.")
        return

    logger.info(f"Loaded {len(grouped)} distinct experiment groups.")
    generate_tables(grouped, args.output_dir)
    logger.info("Summary tables generated successfully.")


if __name__ == "__main__":
    main()
