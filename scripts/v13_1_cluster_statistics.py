"""
scripts/v13_1_cluster_statistics.py: Paired Clustered Block Bootstrap for Tri-SACTA V13.1.

Executes B=2,000 paired block bootstrap resamples clustered by unique video_id.
Evaluates paired differences (Delta Macro-F1 and % Cost Reduction):
1. SequentialConfidence_0.55_0.45 vs AlwaysOnFusion
2. SequentialConfidence_0.55_0.45 vs Static_TA
3. SequentialConfidence_0.55_0.45 vs ConfidenceRouting_0.55
4. SequentialConfidence_0.55_0.45 vs UtilityRouter_V13
5. SequentialConfidence_0.55_0.45 vs FixedFallback
6. ConditionalUtility_0.05 vs SequentialConfidence_0.55_0.45

Outputs:
- results/v13_1/statistics/paired_bootstrap_summary.json
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    k = num_classes
    m = k * y_true + y_pred
    counts = np.bincount(m, minlength=k * k).reshape(k, k)
    f1_sum = 0.0
    for c in range(k):
        tp = counts[c, c]
        fp = counts[:, c].sum() - tp
        fn = counts[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1_sum += (2.0 * tp / denom) if denom > 0 else 0.0
    return f1_sum / k


def run_paired_cluster_bootstrap(B: int = 2000, seed: int = 1301):
    logger.info(f"=== Running Paired Video-Grouped Block Bootstrap (B={B}, Seed={seed}) ===")
    traces_path = os.path.join(REPO_ROOT, "results", "v13_1", "evaluation", "query_traces.json")
    out_dir = os.path.join(REPO_ROOT, "results", "v13_1", "statistics")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(traces_path):
        raise FileNotFoundError(f"Missing traces file: {traces_path}. Run evaluate_policies first.")

    with open(traces_path, "r", encoding="utf-8") as f:
        traces = json.load(f)

    df = pd.DataFrame(traces)
    logger.info(f"Loaded {len(df)} query traces. Unique videos: {df['video_id'].nunique()}, Policies: {df['policy'].nunique()}")

    target_policy = "SequentialConfidence_0.55_0.45"
    comparators = [
        "AlwaysOnFusion",
        "Static_TA",
        "ConfidenceRouting_0.55",
        "UtilityRouter_V13",
        "FixedFallback",
        "ConditionalUtility_0.05",
    ]

    # Pre-structure data by policy for fast aligned indexing
    # We group by (sample_idx, seed, workload)
    pivoted_preds = df.pivot(index=["sample_idx", "video_id", "seed", "workload"], columns="policy", values="y_pred")
    pivoted_costs = df.pivot(index=["sample_idx", "video_id", "seed", "workload"], columns="policy", values="cost")
    y_true_series = df.pivot(index=["sample_idx", "video_id", "seed", "workload"], columns="policy", values="y_true").iloc[:, 0]

    index_df = pivoted_preds.reset_index()
    unique_vids = index_df["video_id"].unique()
    K = len(unique_vids)
    logger.info(f"Unique video clusters for bootstrap: {K}")

    # Map video_id to row indices
    vid_to_rows: Dict[str, np.ndarray] = {}
    for vid in unique_vids:
        vid_to_rows[vid] = np.where(index_df["video_id"].values == vid)[0]

    rng = np.random.default_rng(seed)

    # Pre-extract arrays
    y_true_all = y_true_series.values
    target_preds_all = pivoted_preds[target_policy].values
    target_costs_all = pivoted_costs[target_policy].values

    comp_preds_all = {c: pivoted_preds[c].values for c in comparators}
    comp_costs_all = {c: pivoted_costs[c].values for c in comparators}

    # Observed point estimates on full dataset
    observed_results = {}
    f1_target_full = fast_macro_f1(y_true_all, target_preds_all)
    cost_target_full = float(np.mean(target_costs_all))

    for c in comparators:
        f1_c = fast_macro_f1(y_true_all, comp_preds_all[c])
        cost_c = float(np.mean(comp_costs_all[c]))
        delta_f1 = f1_target_full - f1_c
        cost_reduction_pct = ((cost_c - cost_target_full) / cost_c) * 100.0 if cost_c > 0 else 0.0

        observed_results[c] = {
            "target_macro_f1": float(f1_target_full),
            "comparator_macro_f1": float(f1_c),
            "delta_macro_f1": float(delta_f1),
            "target_mean_cost": float(cost_target_full),
            "comparator_mean_cost": float(cost_c),
            "cost_reduction_pct": float(cost_reduction_pct),
        }

    # Bootstrap loop
    boot_delta_f1 = {c: [] for c in comparators}
    boot_cost_red = {c: [] for c in comparators}

    for b in range(B):
        # Resample video clusters with replacement
        sampled_vids = rng.choice(unique_vids, size=K, replace=True)
        row_indices = np.concatenate([vid_to_rows[v] for v in sampled_vids])

        b_y_true = y_true_all[row_indices]
        b_target_preds = target_preds_all[row_indices]
        b_target_cost = float(np.mean(target_costs_all[row_indices]))
        b_target_f1 = fast_macro_f1(b_y_true, b_target_preds)

        for c in comparators:
            b_c_preds = comp_preds_all[c][row_indices]
            b_c_cost = float(np.mean(comp_costs_all[c][row_indices]))
            b_c_f1 = fast_macro_f1(b_y_true, b_c_preds)

            d_f1 = b_target_f1 - b_c_f1
            c_red = ((b_c_cost - b_target_cost) / b_c_cost) * 100.0 if b_c_cost > 0 else 0.0

            boot_delta_f1[c].append(d_f1)
            boot_cost_red[c].append(c_red)

    # Aggregate bootstrap percentiles
    summary_comparisons = {}
    for c in comparators:
        d_f1_arr = np.array(boot_delta_f1[c])
        c_red_arr = np.array(boot_cost_red[c])

        f1_ci = [float(np.percentile(d_f1_arr, 2.5)), float(np.percentile(d_f1_arr, 97.5))]
        cost_ci = [float(np.percentile(c_red_arr, 2.5)), float(np.percentile(c_red_arr, 97.5))]

        obs = observed_results[c]
        summary_comparisons[f"{target_policy}_vs_{c}"] = {
            "target_policy": target_policy,
            "comparator_policy": c,
            "observed_delta_macro_f1": obs["delta_macro_f1"],
            "delta_macro_f1_95_ci": f1_ci,
            "observed_cost_reduction_pct": obs["cost_reduction_pct"],
            "cost_reduction_pct_95_ci": cost_ci,
            "p_superiority": float(np.mean(d_f1_arr > 0.0)),
            "p_non_inferiority_neg001": float(np.mean(d_f1_arr >= -0.01)),
            "target_f1": obs["target_macro_f1"],
            "comparator_f1": obs["comparator_macro_f1"],
            "target_cost": obs["target_mean_cost"],
            "comparator_cost": obs["comparator_mean_cost"],
        }

        logger.info(
            f"Comparison vs {c:25s} | Delta F1: {obs['delta_macro_f1']:+.4f} [95% CI: {f1_ci[0]:+.4f}, {f1_ci[1]:+.4f}] | "
            f"Cost Saved: {obs['cost_reduction_pct']:+.1f}% [95% CI: {cost_ci[0]:+.1f}%, {cost_ci[1]:+.1f}%]"
        )

    final_artifact = {
        "bootstrap_parameters": {
            "resamples_B": B,
            "random_seed": seed,
            "clustering_variable": "video_id",
            "cluster_count": K,
            "total_evaluated_instances": len(index_df),
        },
        "target_primary_policy": target_policy,
        "comparisons": summary_comparisons,
    }

    out_json = os.path.join(out_dir, "paired_bootstrap_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_artifact, f, indent=2)

    logger.info(f"Saved paired bootstrap summary to: {out_json}")
    return final_artifact


if __name__ == "__main__":
    run_paired_cluster_bootstrap(B=2000, seed=1301)
