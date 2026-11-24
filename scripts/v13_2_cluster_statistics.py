"""
scripts/v13_2_cluster_statistics.py: Paired Video-Grouped Block Bootstrap for Tri-SACTA V13.2 (Phase I-C).

Mathematical Methodology:
1. Resamples whole unique video clusters with replacement (B = 2,000 resamples).
2. Uses exact linear sufficiency: cluster-level confusion vectors (TP, FP, FN) and costs
   are linearly additive under multinomial cluster count vectors w ~ Multinomial(K, 1/K).
3. Recomputes exact primary estimand (unweighted mean per workload/seed Macro-F1) in every resample.
4. Reports original observed point estimate, 95% paired bootstrap CI, and cost saving CI.
5. Exports results to results/v13_2/statistics/paired_bootstrap_summary.json.
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

LOG_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "logs")
STAT_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "statistics")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STAT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "cluster_statistics.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_paired_video_bootstrap(B: int = 2000, seed: int = 1320):
    logger.info(f"=== Running Paired Video-Grouped Block Bootstrap (B={B:,}, Seed={seed}) ===")
    traces_path = os.path.join(REPO_ROOT, "results", "v13_2", "development", "query_traces.json")
    if not os.path.exists(traces_path):
        raise FileNotFoundError(f"Missing traces file: {traces_path}")

    with open(traces_path, "r", encoding="utf-8") as f:
        traces = json.load(f)

    df = pd.DataFrame(traces)
    unique_videos = np.array(sorted(df["video_id"].unique()))
    num_clusters = len(unique_videos)
    logger.info(f"Loaded {len(df):,} query traces across {num_clusters} unique video clusters.")

    # Candidate policy vs comparators
    candidate = "SequentialConfidence_0.55_0.45"
    comparators = [
        "AlwaysOnFusion",
        "BudgetAdmittedMaxSubset",
        "Static_TA",
        "ConfidenceRouting_0.55",
        "MatchedBudgetRandom_1.45",
        "ConditionalUtility_0.05",
        "Static_T",
    ]
    all_policies_to_eval = [candidate] + comparators

    unique_ws = sorted(list(set(zip(df["workload"], df["seed"]))))
    num_ws = len(unique_ws)
    ws_to_idx = {ws: i for i, ws in enumerate(unique_ws)}
    vid_to_idx = {v: i for i, v in enumerate(unique_videos)}

    # Pre-aggregate cluster statistics for each policy and (workload, seed)
    # Shapes: [policy, ws, cluster, 3] for TP, FP, FN; [policy, ws, cluster] for cost, [cluster] for sample count
    tp_table = {p: np.zeros((num_ws, num_clusters, 3), dtype=np.float64) for p in all_policies_to_eval}
    fp_table = {p: np.zeros((num_ws, num_clusters, 3), dtype=np.float64) for p in all_policies_to_eval}
    fn_table = {p: np.zeros((num_ws, num_clusters, 3), dtype=np.float64) for p in all_policies_to_eval}
    cost_table = {p: np.zeros((num_ws, num_clusters), dtype=np.float64) for p in all_policies_to_eval}
    sample_count = np.zeros(num_clusters, dtype=np.float64)

    logger.info("Pre-aggregating linear cluster statistics for ultra-fast exact bootstrap...")
    for t in traces:
        pol = t["policy"]
        if pol not in tp_table:
            continue
        v_idx = vid_to_idx[t["video_id"]]
        ws_idx = ws_to_idx[(t["workload"], t["seed"])]
        yt = t["true_label"]
        yp = t["prediction"]
        c = t["spent_cost"]

        cost_table[pol][ws_idx, v_idx] += c

        if pol == candidate and ws_idx == 0:
            sample_count[v_idx] += 1

        for cls_i in [0, 1, 2]:
            if yt == cls_i and yp == cls_i:
                tp_table[pol][ws_idx, v_idx, cls_i] += 1.0
            elif yt != cls_i and yp == cls_i:
                fp_table[pol][ws_idx, v_idx, cls_i] += 1.0
            elif yt == cls_i and yp != cls_i:
                fn_table[pol][ws_idx, v_idx, cls_i] += 1.0

    def compute_metrics_from_weights(w: np.ndarray, pol: str) -> Tuple[float, float]:
        # w shape: [num_clusters]
        # tp shape: [num_ws, 3]
        tp = np.einsum("k,wkc->wc", w, tp_table[pol])
        fp = np.einsum("k,wkc->wc", w, fp_table[pol])
        fn = np.einsum("k,wkc->wc", w, fn_table[pol])

        denom = 2.0 * tp + fp + fn
        f1_wc = np.where(denom > 0, 2.0 * tp / denom, 0.0)
        macro_f1_w = np.mean(f1_wc, axis=1)  # mean over 3 classes
        primary_f1 = float(np.mean(macro_f1_w))  # mean over (workload, seed)

        total_cost = np.sum(np.einsum("k,wk->w", w, cost_table[pol]))
        total_queries = np.sum(w * sample_count) * num_ws
        mean_cost = float(total_cost / total_queries) if total_queries > 0 else 0.0
        return primary_f1, mean_cost

    # 1. Observed Point Estimates (w = ones)
    w_obs = np.ones(num_clusters, dtype=np.float64)
    cand_obs_f1, cand_obs_cost = compute_metrics_from_weights(w_obs, candidate)

    observed_comparisons = {}
    for comp in comparators:
        comp_f1, comp_cost = compute_metrics_from_weights(w_obs, comp)
        delta_f1 = cand_obs_f1 - comp_f1
        cost_saving = ((comp_cost - cand_obs_cost) / comp_cost * 100.0) if comp_cost > 0 else 0.0
        observed_comparisons[comp] = {
            "comp_macro_f1": comp_f1,
            "comp_cost": comp_cost,
            "observed_delta_f1": delta_f1,
            "observed_cost_saving_pct": cost_saving,
        }

    # 2. Resample B times via Multinomial cluster weights
    rng = np.random.default_rng(seed)
    # Multinomial draws of size num_clusters across num_clusters bins
    w_resamples = rng.multinomial(num_clusters, [1.0 / num_clusters] * num_clusters, size=B)

    bootstrap_deltas: Dict[str, List[float]] = {c: [] for c in comparators}
    bootstrap_savings: Dict[str, List[float]] = {c: [] for c in comparators}

    logger.info(f"Executing {B:,} exact paired cluster bootstrap resamples...")
    for b in range(B):
        w_b = w_resamples[b].astype(np.float64)
        b_cand_f1, b_cand_c = compute_metrics_from_weights(w_b, candidate)

        for comp in comparators:
            b_comp_f1, b_comp_c = compute_metrics_from_weights(w_b, comp)
            d_f1 = b_cand_f1 - b_comp_f1
            c_save = ((b_comp_c - b_cand_c) / b_comp_c * 100.0) if b_comp_c > 0 else 0.0
            bootstrap_deltas[comp].append(d_f1)
            bootstrap_savings[comp].append(c_save)

    # 3. Percentile 95% Confidence Intervals
    comparison_summary = {}
    for comp in comparators:
        d_arr = np.array(bootstrap_deltas[comp])
        s_arr = np.array(bootstrap_savings[comp])

        ci_f1_lower = float(np.percentile(d_arr, 2.5))
        ci_f1_upper = float(np.percentile(d_arr, 97.5))
        ci_cost_lower = float(np.percentile(s_arr, 2.5))
        ci_cost_upper = float(np.percentile(s_arr, 97.5))

        obs = observed_comparisons[comp]
        p_sup = float(np.mean(d_arr <= 0.0))

        comparison_summary[comp] = {
            "comparator": comp,
            "candidate_observed_f1": cand_obs_f1,
            "comparator_observed_f1": obs["comp_macro_f1"],
            "observed_delta_macro_f1": obs["observed_delta_f1"],
            "delta_macro_f1_95_ci": [ci_f1_lower, ci_f1_upper],
            "candidate_observed_cost": cand_obs_cost,
            "comparator_observed_cost": obs["comp_cost"],
            "observed_cost_saving_pct": obs["observed_cost_saving_pct"],
            "cost_saving_pct_95_ci": [ci_cost_lower, ci_cost_upper],
            "bootstrap_fraction_delta_le_zero": p_sup,
            "non_inferiority_margin_0_01_pass": bool(ci_f1_lower >= -0.010),
            "resource_reduction_15pct_pass": bool(ci_cost_lower >= 15.0),
        }
        logger.info(
            f"vs {comp:25s}: Delta F1 = {obs['observed_delta_f1']:+.4f} [{ci_f1_lower:+.4f}, {ci_f1_upper:+.4f}] | "
            f"Cost Saved = {obs['observed_cost_saving_pct']:+.1f}% [{ci_cost_lower:+.1f}%, {ci_cost_upper:+.1f}%] | "
            f"Non-Inf (-0.01): {'PASS' if ci_f1_lower >= -0.010 else 'FAIL'} | "
            f"Savings (>=15%): {'PASS' if ci_cost_lower >= 15.0 else 'FAIL'}"
        )

    out_artifact = {
        "analysis_version": "13.2",
        "methodology": "Paired Video-Grouped Block Bootstrap",
        "num_bootstrap_resamples": B,
        "rng_seed": seed,
        "num_video_clusters": num_clusters,
        "total_queries_analyzed": len(df),
        "primary_estimand": "unweighted_mean_per_workload_seed_macro_f1",
        "candidate_policy": candidate,
        "candidate_observed_point_estimates": {
            "macro_f1": cand_obs_f1,
            "mean_cost": cand_obs_cost,
        },
        "comparisons": comparison_summary,
    }

    out_path = os.path.join(STAT_DIR, "paired_bootstrap_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_artifact, f, indent=2)

    logger.info(f"Paired cluster bootstrap summary successfully saved to {out_path}")
    return out_artifact


if __name__ == "__main__":
    run_paired_video_bootstrap()
