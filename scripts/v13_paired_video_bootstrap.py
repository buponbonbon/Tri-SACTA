"""
scripts/v13_paired_video_bootstrap.py: Rigorous Paired Video-Grouped Block Bootstrap for Tri-SACTA V13.

Performs 1,000 clustered resamples by unique video_id to calculate genuine paired differences
between UtilitySequentialPolicy and all baselines across each workload and overall aggregated benchmark.

Tests:
1. Non-Inferiority: Lower 95% CI of Delta Macro-F1 >= -0.010 (1.0 F1 point margin)
2. Cost Efficiency: Lower 95% CI of % Cost Reduction >= 15.0%
3. Pareto Dominance: Both Non-Inferiority and Cost Efficiency satisfied simultaneously.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 100, 2026]
WORKLOADS = [
    "CLEAN_MULTIMODAL",
    "TEXT_EMBEDDING_NOISE",
    "AUDIO_FEATURE_MASKING",
    "VISUAL_ZERO_MASK",
    "COMPOUND_FAULT",
    "INTERMITTENT_MISSINGNESS",
    "HETEROGENEOUS_DEADLINE_STRESS",
]
BASELINES = [
    "ConfidenceRouting_0.55",
    "ConfidenceRouting_0.65",
    "FixedFallback",
    "MatchedBudgetRandom",
    "AlwaysOnFusion",
]
B_BOOT = 1000
MARGIN_F1 = -0.010
TARGET_COST_RED_PCT = 15.0


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    """Vectorized Macro-F1 computation (100x faster than sklearn for bootstrap)."""
    cm = np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    return float(np.mean(f1))


def video_grouped_bootstrap_indices(video_ids: List[str], rng: np.random.Generator) -> List[int]:
    """Resamples unique video clusters with replacement and returns pooled utterance indices."""
    unique_vids = np.unique(video_ids)
    K = len(unique_vids)
    sampled_vids = rng.choice(unique_vids, size=K, replace=True)

    vid_to_indices: Dict[str, List[int]] = {}
    for idx, v in enumerate(video_ids):
        vid_to_indices.setdefault(v, []).append(idx)

    resampled_indices = []
    for v in sampled_vids:
        resampled_indices.extend(vid_to_indices[v])
    return resampled_indices


def run_paired_video_bootstrap(split_name: str = "val"):
    logger.info(f"=== Running Paired Video-Grouped Block Bootstrap on split: '{split_name}' (B={B_BOOT}) ===")
    eval_dir = os.path.join(repo_root, "results", "v13_utility", "evaluation", split_name)
    out_dir = os.path.join(repo_root, "results", "v13_utility", "statistics", split_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load all per-query traces
    # traces[workload][policy][seed] -> list of query dicts
    traces: Dict[str, Dict[str, Dict[int, List[Dict[str, Any]]]]] = {}

    for w_name in WORKLOADS:
        traces[w_name] = {}
        for pol in ["UtilitySequentialPolicy"] + BASELINES:
            traces[w_name][pol] = {}
            for s in SEEDS:
                fpath = os.path.join(eval_dir, f"trace_{w_name}_{pol}_seed{s}.json")
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Missing trace file: {fpath}")
                with open(fpath, "r", encoding="utf-8") as f:
                    traces[w_name][pol][s] = json.load(f)

    # Reference labels and video_ids from Clean
    sample_trace = traces["CLEAN_MULTIMODAL"]["UtilitySequentialPolicy"][42]
    ref_labels = np.array([r["label"] for r in sample_trace])
    ref_vids = [r["video_id"] for r in sample_trace]
    N = len(ref_labels)
    unique_vids = np.unique(ref_vids)
    logger.info(f"Loaded traces. N={N} queries across {len(unique_vids)} unique video clusters.")

    rng = np.random.default_rng(2026)
    bootstrap_results: Dict[str, Any] = {
        "split": split_name,
        "sample_count": N,
        "unique_video_clusters": len(unique_vids),
        "bootstrap_resamples": B_BOOT,
        "non_inferiority_margin_f1": MARGIN_F1,
        "cost_reduction_target_pct": TARGET_COST_RED_PCT,
        "workloads": {},
        "overall_aggregate": {},
    }

    # Pre-extract arrays for fast bootstrap computation:
    # arrays[w_name][pol][s] = {"preds": np.ndarray, "costs": np.ndarray}
    data_arrays: Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]] = {}
    for w in WORKLOADS:
        data_arrays[w] = {}
        for pol in ["UtilitySequentialPolicy"] + BASELINES:
            data_arrays[w][pol] = {}
            for s in SEEDS:
                t_list = traces[w][pol][s]
                data_arrays[w][pol][s] = {
                    "preds": np.array([r["prediction"] for r in t_list], dtype=np.int64),
                    "costs": np.array([r["spent_cost"] for r in t_list], dtype=np.float32),
                }

    # Pre-sample 1000 video-grouped cluster resample indices
    boot_indices_list = [video_grouped_bootstrap_indices(ref_vids, rng) for _ in range(B_BOOT)]

    # Pre-compute UtilitySequentialPolicy metrics for all resamples:
    # util_metrics[w_name][b] = {"f1": float, "cost": float}
    logger.info("Pre-computing Utility policy bootstrap distributions...")
    util_metrics: Dict[str, List[Tuple[float, float]]] = {w: [] for w in WORKLOADS}
    for w_name in WORKLOADS:
        for b in range(B_BOOT):
            idx = boot_indices_list[b]
            y_boot = ref_labels[idx]
            u_f1_s, u_c_s = [], []
            for s in SEEDS:
                u_p = data_arrays[w_name]["UtilitySequentialPolicy"][s]["preds"][idx]
                u_c = data_arrays[w_name]["UtilitySequentialPolicy"][s]["costs"][idx]
                u_f1_s.append(fast_macro_f1(y_boot, u_p))
                u_c_s.append(np.mean(u_c))
            util_metrics[w_name].append((float(np.mean(u_f1_s)), float(np.mean(u_c_s))))

    # 2. Evaluate per workload
    for w_name in WORKLOADS:
        logger.info(f"\n--- Bootstrapping Workload: {w_name} ---")
        bootstrap_results["workloads"][w_name] = {}

        for base_name in BASELINES:
            delta_f1s = np.zeros(B_BOOT, dtype=np.float64)
            delta_costs = np.zeros(B_BOOT, dtype=np.float64)
            cost_red_pcts = np.zeros(B_BOOT, dtype=np.float64)

            for b in range(B_BOOT):
                idx = boot_indices_list[b]
                y_boot = ref_labels[idx]

                mean_u_f1, mean_u_c = util_metrics[w_name][b]
                base_f1_seeds, base_c_seeds = [], []

                for s in SEEDS:
                    b_p = data_arrays[w_name][base_name][s]["preds"][idx]
                    b_c = data_arrays[w_name][base_name][s]["costs"][idx]
                    base_f1_seeds.append(fast_macro_f1(y_boot, b_p))
                    base_c_seeds.append(np.mean(b_c))

                mean_b_f1 = np.mean(base_f1_seeds)
                mean_b_c = np.mean(base_c_seeds)

                delta_f1s[b] = mean_u_f1 - mean_b_f1
                delta_costs[b] = mean_u_c - mean_b_c
                cost_red_pcts[b] = 100.0 * (mean_b_c - mean_u_c) / (mean_b_c + 1e-8)

            f1_ci_low = float(np.percentile(delta_f1s, 2.5))
            f1_ci_high = float(np.percentile(delta_f1s, 97.5))
            c_ci_low = float(np.percentile(delta_costs, 2.5))
            c_ci_high = float(np.percentile(delta_costs, 97.5))
            red_ci_low = float(np.percentile(cost_red_pcts, 2.5))
            red_ci_high = float(np.percentile(cost_red_pcts, 97.5))

            is_non_inferior = bool(f1_ci_low >= MARGIN_F1)
            is_cost_dominant = bool(red_ci_low >= TARGET_COST_RED_PCT)
            is_pareto = bool(is_non_inferior and is_cost_dominant)

            res_entry = {
                "delta_macro_f1_mean": float(np.mean(delta_f1s)),
                "delta_macro_f1_95ci": [f1_ci_low, f1_ci_high],
                "delta_cost_mean": float(np.mean(delta_costs)),
                "delta_cost_95ci": [c_ci_low, c_ci_high],
                "cost_reduction_pct_mean": float(np.mean(cost_red_pcts)),
                "cost_reduction_pct_95ci": [red_ci_low, red_ci_high],
                "non_inferior": is_non_inferior,
                "cost_dominant": is_cost_dominant,
                "pareto_dominant": is_pareto,
            }
            bootstrap_results["workloads"][w_name][base_name] = res_entry

            verdict_str = "PARETO DOMINANT" if is_pareto else ("NON-INFERIOR" if is_non_inferior else "INFERIOR")
            logger.info(f"vs {base_name:24s} | Delta F1: {res_entry['delta_macro_f1_mean']:+.4f} [{f1_ci_low:+.4f}, {f1_ci_high:+.4f}] | "
                        f"Cost Red: {res_entry['cost_reduction_pct_mean']:5.1f}% [{red_ci_low:5.1f}%, {red_ci_high:5.1f}%] -> {verdict_str}")

    # 3. Overall Macro-Averaged Benchmark across all 7 Workloads
    logger.info(f"\n--- Bootstrapping Overall Macro-Averaged Benchmark across all 7 Workloads ---")
    for base_name in BASELINES:
        agg_delta_f1s = np.zeros(B_BOOT, dtype=np.float64)
        agg_delta_costs = np.zeros(B_BOOT, dtype=np.float64)
        agg_cost_red_pcts = np.zeros(B_BOOT, dtype=np.float64)

        for b in range(B_BOOT):
            idx = boot_indices_list[b]
            y_boot = ref_labels[idx]

            w_u_f1s = [util_metrics[w_name][b][0] for w_name in WORKLOADS]
            w_u_cs = [util_metrics[w_name][b][1] for w_name in WORKLOADS]
            w_b_f1s, w_b_cs = [], []

            for w_name in WORKLOADS:
                b_f1_s, b_c_s = [], []
                for s in SEEDS:
                    b_p = data_arrays[w_name][base_name][s]["preds"][idx]
                    b_c = data_arrays[w_name][base_name][s]["costs"][idx]
                    b_f1_s.append(fast_macro_f1(y_boot, b_p))
                    b_c_s.append(np.mean(b_c))

                w_b_f1s.append(np.mean(b_f1_s))
                w_b_cs.append(np.mean(b_c_s))

            mean_w_u_f1 = np.mean(w_u_f1s)
            mean_w_b_f1 = np.mean(w_b_f1s)
            mean_w_u_c = np.mean(w_u_cs)
            mean_w_b_c = np.mean(w_b_cs)

            agg_delta_f1s[b] = mean_w_u_f1 - mean_w_b_f1
            agg_delta_costs[b] = mean_w_u_c - mean_w_b_c
            agg_cost_red_pcts[b] = 100.0 * (mean_w_b_c - mean_w_u_c) / (mean_w_b_c + 1e-8)

        f1_ci_low = float(np.percentile(agg_delta_f1s, 2.5))
        f1_ci_high = float(np.percentile(agg_delta_f1s, 97.5))
        red_ci_low = float(np.percentile(agg_cost_red_pcts, 2.5))
        red_ci_high = float(np.percentile(agg_cost_red_pcts, 97.5))

        is_non_inferior = bool(f1_ci_low >= MARGIN_F1)
        is_cost_dominant = bool(red_ci_low >= TARGET_COST_RED_PCT)
        is_pareto = bool(is_non_inferior and is_cost_dominant)

        agg_entry = {
            "delta_macro_f1_mean": float(np.mean(agg_delta_f1s)),
            "delta_macro_f1_95ci": [f1_ci_low, f1_ci_high],
            "delta_cost_mean": float(np.mean(agg_delta_costs)),
            "cost_reduction_pct_mean": float(np.mean(agg_cost_red_pcts)),
            "cost_reduction_pct_95ci": [red_ci_low, red_ci_high],
            "non_inferior": is_non_inferior,
            "cost_dominant": is_cost_dominant,
            "pareto_dominant": is_pareto,
        }
        bootstrap_results["overall_aggregate"][base_name] = agg_entry

        verdict_str = "PARETO DOMINANT" if is_pareto else ("NON-INFERIOR" if is_non_inferior else "INFERIOR")
        logger.info(f"OVERALL vs {base_name:24s} | Delta F1: {agg_entry['delta_macro_f1_mean']:+.4f} [{f1_ci_low:+.4f}, {f1_ci_high:+.4f}] | "
                    f"Cost Red: {agg_entry['cost_reduction_pct_mean']:5.1f}% [{red_ci_low:5.1f}%, {red_ci_high:5.1f}%] -> {verdict_str}")

    summary_file = os.path.join(out_dir, "paired_bootstrap_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)

    logger.info(f"\nPaired bootstrap analysis complete. Saved to: {summary_file}")
    return bootstrap_results


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    run_paired_video_bootstrap(split)
