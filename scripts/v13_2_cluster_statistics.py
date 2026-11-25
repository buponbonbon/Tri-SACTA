"""
scripts/v13_2_cluster_statistics.py: Paired Video-Grouped Block Bootstrap for Tri-SACTA V13.2.

Computes B=2,000 paired video-grouped block bootstrap resamples (deterministic seed 1320)
for TWO strictly separated resource endpoints across the exact same 7-workload x 3-seed
validation population (250 utterances, 29 unique video clusters):
1. PRIMARY ENDPOINT: Measured Application-Protocol Bytes & Service-Quality Macro-F1
   computed directly from results/v13_2/systems/distributed_benchmark.csv.
   Jointly checks BOTH primary comparators (AlwaysOn_Direct_Serial, AlwaysOn_Direct_Parallel,
   Static_TA_Direct) plus secondary controls.
2. SECONDARY SIMULATION ENDPOINT: Nominal Acquisition Cost & Service-Quality Macro-F1
   computed directly from results/v13_2/development/query_traces.json.

Handles denominator-zero cleanly (reports absolute difference when comparator transmits 0 bytes)
and stores ordered video cluster list + RNG seed for exact deterministic reconstruction.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
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


def paired_cluster_bootstrap_on_df(
    df: pd.DataFrame,
    candidate: str,
    comparators: List[str],
    pred_col: str,
    resource_col: str,
    B: int = 2000,
    seed: int = 1320,
) -> Dict[str, Any]:
    """
    Executes exact paired video-grouped block bootstrap using linear confusion sufficiency.
    Preserves all samples, seeds, workloads, and policies per resampled video cluster.
    """
    unique_videos = sorted(df["video_id"].unique().tolist())
    num_clusters = len(unique_videos)
    unique_ws = sorted(list(set(zip(df["workload"].tolist(), df["seed"].tolist()))))
    num_ws = len(unique_ws)

    ws_to_idx = {ws: i for i, ws in enumerate(unique_ws)}
    vid_to_idx = {v: i for i, v in enumerate(unique_videos)}
    all_policies = [candidate] + comparators

    tp_table = {p: np.zeros((num_ws, num_clusters, 3), dtype=np.float64) for p in all_policies}
    fp_table = {p: np.zeros((num_ws, num_clusters, 3), dtype=np.float64) for p in all_policies}
    fn_table = {p: np.zeros((num_ws, num_clusters, 3), dtype=np.float64) for p in all_policies}
    res_table = {p: np.zeros((num_ws, num_clusters), dtype=np.float64) for p in all_policies}
    sample_count = np.zeros(num_clusters, dtype=np.float64)

    sub_df = df[df["policy"].isin(all_policies)]
    for row in sub_df.itertuples(index=False):
        pol = getattr(row, "policy")
        v_idx = vid_to_idx[getattr(row, "video_id")]
        ws_idx = ws_to_idx[(getattr(row, "workload"), getattr(row, "seed"))]
        yt = int(getattr(row, "true_label"))
        yp = int(getattr(row, pred_col))
        r_val = float(getattr(row, resource_col))

        res_table[pol][ws_idx, v_idx] += r_val
        if pol == candidate and ws_idx == 0:
            sample_count[v_idx] += 1.0

        for cls_i in (0, 1, 2):
            if yt == cls_i and yp == cls_i:
                tp_table[pol][ws_idx, v_idx, cls_i] += 1.0
            elif yt != cls_i and yp == cls_i:
                fp_table[pol][ws_idx, v_idx, cls_i] += 1.0
            elif yt == cls_i and yp != cls_i:
                fn_table[pol][ws_idx, v_idx, cls_i] += 1.0

    def compute_from_weights(w: np.ndarray, pol: str) -> Tuple[float, float]:
        tp = np.einsum("k,wkc->wc", w, tp_table[pol])
        fp = np.einsum("k,wkc->wc", w, fp_table[pol])
        fn = np.einsum("k,wkc->wc", w, fn_table[pol])
        denom = 2.0 * tp + fp + fn
        f1_wc = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        primary_f1 = float(np.mean(np.mean(f1_wc, axis=1)))
        total_res = float(np.sum(np.einsum("k,wk->w", w, res_table[pol])))
        total_q = float(np.sum(w * sample_count) * num_ws)
        mean_res = total_res / total_q if total_q > 0 else 0.0
        return primary_f1, mean_res

    w_obs = np.ones(num_clusters, dtype=np.float64)
    cand_obs_f1, cand_obs_res = compute_from_weights(w_obs, candidate)

    rng = np.random.default_rng(seed)
    w_resamples = rng.multinomial(num_clusters, [1.0 / num_clusters] * num_clusters, size=B)

    boot_deltas = {c: [] for c in comparators}
    boot_savings_pct = {c: [] for c in comparators}
    boot_abs_diff = {c: [] for c in comparators}
    obs_comp = {}

    for comp in comparators:
        comp_f1, comp_res = compute_from_weights(w_obs, comp)
        obs_comp[comp] = {
            "comp_f1": comp_f1,
            "comp_res": comp_res,
            "delta_f1": cand_obs_f1 - comp_f1,
            "abs_diff": cand_obs_res - comp_res,
            "saving_pct": ((comp_res - cand_obs_res) / comp_res * 100.0) if comp_res > 0.0 else None,
        }

    for b in range(B):
        w_b = w_resamples[b].astype(np.float64)
        b_cf1, b_cres = compute_from_weights(w_b, candidate)
        for comp in comparators:
            b_kf1, b_kres = compute_from_weights(w_b, comp)
            boot_deltas[comp].append(b_cf1 - b_kf1)
            boot_abs_diff[comp].append(b_cres - b_kres)
            if b_kres > 0.0:
                boot_savings_pct[comp].append((b_kres - b_cres) / b_kres * 100.0)
            else:
                boot_savings_pct[comp].append(np.nan)

    comparisons = {}
    for comp in comparators:
        d_arr = np.array(boot_deltas[comp], dtype=np.float64)
        abs_arr = np.array(boot_abs_diff[comp], dtype=np.float64)
        s_arr = np.array(boot_savings_pct[comp], dtype=np.float64)
        obs = obs_comp[comp]

        ci_f1 = [float(np.percentile(d_arr, 2.5)), float(np.percentile(d_arr, 97.5))]
        ci_abs = [float(np.percentile(abs_arr, 2.5)), float(np.percentile(abs_arr, 97.5))]

        if obs["saving_pct"] is not None and np.all(np.isfinite(s_arr)):
            ci_save = [float(np.percentile(s_arr, 2.5)), float(np.percentile(s_arr, 97.5))]
            saving_pass = bool(ci_save[0] >= 15.0)
            saving_defined = True
        else:
            ci_save = None
            saving_pass = False
            saving_defined = False

        comparisons[comp] = {
            "comparator": comp,
            "candidate_observed_f1": cand_obs_f1,
            "comparator_observed_f1": obs["comp_f1"],
            "observed_delta_macro_f1": obs["delta_f1"],
            "delta_macro_f1_95_ci": ci_f1,
            "candidate_observed_resource": cand_obs_res,
            "comparator_observed_resource": obs["comp_res"],
            "observed_resource_abs_diff": obs["abs_diff"],
            "resource_abs_diff_95_ci": ci_abs,
            "saving_pct_defined": saving_defined,
            "observed_resource_saving_pct": obs["saving_pct"],
            "resource_saving_pct_95_ci": ci_save,
            "bootstrap_fraction_delta_le_zero": float(np.mean(d_arr <= 0.0)),
            "non_inferiority_margin_0_01_pass": bool(ci_f1[0] >= -0.010),
            "resource_reduction_15pct_pass": saving_pass,
        }

    return {
        "resource_unit": resource_col,
        "prediction_column": pred_col,
        "num_bootstrap_resamples": B,
        "rng_seed": seed,
        "rng_algorithm": "numpy.random.default_rng(seed).multinomial(K, [1/K]*K, size=B)",
        "ci_method": "two_sided_percentile_2.5_97.5",
        "num_video_clusters": num_clusters,
        "ordered_video_clusters": unique_videos,
        "num_ws_cells": num_ws,
        "candidate_policy": candidate,
        "candidate_observed": {"macro_f1": cand_obs_f1, "mean_resource": cand_obs_res},
        "comparisons": comparisons,
    }


def run_paired_video_bootstrap(B: int = 2000, seed: int = 1320):
    utc_now = datetime.now(timezone.utc).isoformat()
    logger.info(f"=== Running Dual-Endpoint Paired Video-Grouped Block Bootstrap (B={B:,}, Seed={seed}) ===")

    # 1. PRIMARY MEASURED RESOURCE ENDPOINT (Application-Protocol Bytes + SQ Macro-F1 from TCP CSV)
    sys_csv = os.path.join(REPO_ROOT, "results", "v13_2", "systems", "distributed_benchmark.csv")
    if not os.path.exists(sys_csv):
        raise FileNotFoundError(f"Missing distributed benchmark CSV: {sys_csv}")
    df_sys = pd.read_csv(sys_csv)
    tcp_comparators = [
        "AlwaysOn_Direct_Serial",
        "AlwaysOn_Direct_Parallel",
        "Static_TA_Direct",
        "BudgetAdmittedMaxSubset",
        "OneShot_ConfidenceRouting_0.55",
        "MatchedBudgetRandom_1.45",
        "Unimodal_Text",
    ]
    primary_measured = paired_cluster_bootstrap_on_df(
        df_sys,
        candidate="SequentialConfidence_0.55_0.45",
        comparators=tcp_comparators,
        pred_col="service_quality_prediction",
        resource_col="application_protocol_bytes",
        B=B,
        seed=seed,
    )
    primary_measured["primary_comparators"] = ["AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel", "Static_TA_Direct"]

    # Joint intersection rule on PRIMARY endpoint
    primary_joint_pass = True
    for p_comp in primary_measured["primary_comparators"]:
        c_res = primary_measured["comparisons"][p_comp]
        if not (c_res["non_inferiority_margin_0_01_pass"] and c_res["resource_reduction_15pct_pass"]):
            primary_joint_pass = False
    primary_measured["joint_primary_criterion_pass"] = primary_joint_pass

    logger.info("--- PRIMARY ENDPOINT (Measured Application-Protocol Bytes & Service-Quality Macro-F1) ---")
    for comp, c_d in primary_measured["comparisons"].items():
        save_str = f"{c_d['observed_resource_saving_pct']:+.1f}% {c_d['resource_saving_pct_95_ci']}" if c_d["saving_pct_defined"] else f"N/A (abs diff {c_d['observed_resource_abs_diff']:+.1f}B)"
        logger.info(
            f"  vs {comp:30s}: Delta F1 = {c_d['observed_delta_macro_f1']:+.4f} {c_d['delta_macro_f1_95_ci']} | "
            f"AppBytes Saved = {save_str} | NI={'PASS' if c_d['non_inferiority_margin_0_01_pass'] else 'FAIL'} | "
            f"15%={'PASS' if c_d['resource_reduction_15pct_pass'] else 'FAIL'}"
        )

    # 2. SECONDARY SIMULATION ENDPOINT (Nominal Cost & SQ Macro-F1 from Dev Replay)
    traces_path = os.path.join(REPO_ROOT, "results", "v13_2", "development", "query_traces.json")
    if not os.path.exists(traces_path):
        raise FileNotFoundError(f"Missing dev traces: {traces_path}")
    with open(traces_path, "r", encoding="utf-8") as f:
        df_dev = pd.DataFrame(json.load(f))

    dev_comparators = [
        "AlwaysOnFusion",
        "Static_TA",
        "BudgetAdmittedMaxSubset",
        "ConfidenceRouting_0.55",
        "MatchedBudgetRandom_1.45",
        "ConditionalUtility_0.05",
        "Static_T",
    ]
    secondary_nominal = paired_cluster_bootstrap_on_df(
        df_dev,
        candidate="SequentialConfidence_0.55_0.45",
        comparators=dev_comparators,
        pred_col="service_quality_prediction",
        resource_col="spent_cost",
        B=B,
        seed=seed,
    )
    secondary_nominal["primary_comparators"] = ["AlwaysOnFusion", "Static_TA"]
    nominal_joint_pass = all(
        secondary_nominal["comparisons"][pc]["non_inferiority_margin_0_01_pass"]
        and secondary_nominal["comparisons"][pc]["resource_reduction_15pct_pass"]
        for pc in secondary_nominal["primary_comparators"]
    )
    secondary_nominal["joint_primary_criterion_pass"] = nominal_joint_pass

    logger.info("--- SECONDARY SIMULATION ENDPOINT (Nominal Acquisition Cost & SQ Macro-F1) ---")
    for comp, c_d in secondary_nominal["comparisons"].items():
        save_str = f"{c_d['observed_resource_saving_pct']:+.1f}% {c_d['resource_saving_pct_95_ci']}" if c_d["saving_pct_defined"] else "N/A"
        logger.info(
            f"  vs {comp:30s}: Delta F1 = {c_d['observed_delta_macro_f1']:+.4f} {c_d['delta_macro_f1_95_ci']} | "
            f"NominalCost Saved = {save_str} | NI={'PASS' if c_d['non_inferiority_margin_0_01_pass'] else 'FAIL'} | "
            f"15%={'PASS' if c_d['resource_reduction_15pct_pass'] else 'FAIL'}"
        )

    out_artifact = {
        "analysis_version": "13.2",
        "timestamp_utc": utc_now,
        "methodology": "Paired Video-Grouped Block Bootstrap (Linear Sufficiency)",
        "primary_estimand": "unweighted_mean_per_workload_seed_service_quality_macro_f1",
        "primary_measured_endpoint": primary_measured,
        "secondary_nominal_endpoint": secondary_nominal,
    }
    out_path = os.path.join(STAT_DIR, "paired_bootstrap_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_artifact, f, indent=2)

    logger.info(f"Saved dual-endpoint paired bootstrap summary to {out_path}")
    return out_artifact


if __name__ == "__main__":
    run_paired_video_bootstrap()
