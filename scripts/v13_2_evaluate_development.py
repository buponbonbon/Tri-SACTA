"""
scripts/v13_2_evaluate_development.py: Fair Development Policy Benchmarking for Tri-SACTA V13.2 (Phase I-C).

Resolves P0 Evaluator Deficiencies:
1. Enforces exact per-query cost_budget and deadline_ms propagation (no hardcoded 3.5).
2. Traps masked feature access and handles infeasible budget / blackout fail-closed.
3. Separates Always-On Unrestricted from Budget-Admitted Max-Subset.
4. Computes both the primary estimand (unweighted mean of per-workload/per-seed 3-class Macro-F1)
   and the secondary estimand (pooled confusion matrix Macro-F1).
5. Exports complete per-query traces to results/v13_2/development/query_traces.json.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import SUBSET_METADATA, load_split
from src.service.baseline_policies import (
    ConfidenceRoutingPolicy,
    FixedFallbackPolicy,
    AlwaysOnFusionPolicy,
    BudgetAdmittedMaxSubsetPolicy,
    MatchedBudgetRandomPolicy,
    DerivedThresholdRouter,
    SequentialConfidenceRouter,
    StaticSubsetPolicy,
)
from src.service.conditional_utility import ConditionalGainEstimator
from src.service.conditional_router import ConditionalUtilityRouter
from src.models.fusion_backbone import SimpleMLP

LOG_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "logs")
DEV_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "development")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DEV_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "development_evaluation.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

SEEDS = [42, 100, 2026]
WORKLOAD_NAMES = [
    "CLEAN_MULTIMODAL",
    "TEXT_EMBEDDING_NOISE",
    "AUDIO_FEATURE_MASKING",
    "VISUAL_ZERO_MASK",
    "COMPOUND_FAULT",
    "INTERMITTENT_MISSINGNESS",
    "HETEROGENEOUS_DEADLINE_STRESS",
]


def compute_3class_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact 3-class macro-F1 with rejections (-1) counted as false negatives."""
    f1s = []
    for c in [0, 1, 2]:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def load_predictor_banks(device: torch.device) -> Dict[int, Dict[str, nn.Module]]:
    """Loads all 21 subset MLPs grouped by seed."""
    ckpt_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "training", "checkpoints")
    banks: Dict[int, Dict[str, nn.Module]] = {s: {} for s in SEEDS}

    for seed in SEEDS:
        for s_key, meta in SUBSET_METADATA.items():
            in_dim = meta["in_dim"]
            h_dim = 64 if "text" in meta["mods"] else (32 if "audio" in meta["mods"] else 16)
            model = SimpleMLP(in_dim=in_dim, hidden_dim=h_dim, num_classes=3).to(device)
            fpath = os.path.join(ckpt_dir, f"mlp_{s_key}_seed{seed}.pt")
            model.load_state_dict(torch.load(fpath, weights_only=True, map_location=device))
            model.eval()
            banks[seed][s_key] = model
    return banks


def instantiate_policies(
    predictor_bank: Dict[str, nn.Module],
    device: torch.device,
    cond_estimator: ConditionalGainEstimator,
) -> Dict[str, Any]:
    """Instantiates all 16 comparator policies with matched predictor bank."""
    return {
        # Candidate Release Policies
        "SequentialConfidence_0.55_0.45": SequentialConfidenceRouter(
            predictor_bank, device, tau_stage1=0.55, tau_stage2=0.45, audio_first=True
        ),
        "SequentialConfidence_0.65_0.55": SequentialConfidenceRouter(
            predictor_bank, device, tau_stage1=0.65, tau_stage2=0.55, audio_first=True
        ),
        "SequentialConfidence_VisualFirst_0.55_0.45": SequentialConfidenceRouter(
            predictor_bank, device, tau_stage1=0.55, tau_stage2=0.45, audio_first=False
        ),
        # One-Shot Confidence Routing
        "ConfidenceRouting_0.55": ConfidenceRoutingPolicy(predictor_bank, device, tau_confidence=0.55),
        "ConfidenceRouting_0.65": ConfidenceRoutingPolicy(predictor_bank, device, tau_confidence=0.65),
        # Always-On and Budget-Admitted Max-Subset
        "AlwaysOnFusion": AlwaysOnFusionPolicy(predictor_bank, device),
        "BudgetAdmittedMaxSubset": BudgetAdmittedMaxSubsetPolicy(predictor_bank, device),
        # Static Subsets
        "Static_T": StaticSubsetPolicy(predictor_bank, device, target_subset="T"),
        "Static_TA": StaticSubsetPolicy(predictor_bank, device, target_subset="TA"),
        "Static_TV": StaticSubsetPolicy(predictor_bank, device, target_subset="TV"),
        "Static_AV": StaticSubsetPolicy(predictor_bank, device, target_subset="AV"),
        "Static_TAV": StaticSubsetPolicy(predictor_bank, device, target_subset="TAV"),
        "FixedFallback": FixedFallbackPolicy(predictor_bank, device),
        # Random Acquisition (calibrated to expected cost ~1.45 on validation)
        "MatchedBudgetRandom_1.45": MatchedBudgetRandomPolicy(predictor_bank, device, p_acquire=0.28, seed=42),
        # Historical Negative Controls
        "DerivedThreshold_0.05": DerivedThresholdRouter(predictor_bank, device, lambda_utility=0.05),
        "ConditionalUtility_0.05": ConditionalUtilityRouter(
            predictor_bank, device, lambda_utility=0.05, estimator=cond_estimator
        ),
    }


def evaluate_development():
    logger.info("=== Tri-SACTA V13.2: Phase I-C Fair Development Evaluation Initiated ===")
    device = torch.device("cpu")

    # Load validation split
    val_data = load_split("val", REPO_ROOT)
    y_true = val_data["labels"].numpy()
    video_ids = val_data["video_ids"]
    N = len(y_true)
    logger.info(f"Loaded validation split: {N} samples across {len(set(video_ids))} video clusters.")

    # Generate workloads if needed and load from disk
    from scripts.v13_generate_workloads import generate_workloads_for_split
    wl_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "workloads", "val")
    if not os.path.exists(wl_dir) or not os.path.exists(os.path.join(wl_dir, "workloads_manifest.json")):
        generate_workloads_for_split("val")

    workloads = {}
    for w_name in WORKLOAD_NAMES:
        fpath = os.path.join(wl_dir, f"{w_name}.pt")
        workloads[w_name] = torch.load(fpath, map_location="cpu", weights_only=False)

    # Load model banks & conditional utility estimator
    logger.info("Loading 21 subset MLPs across seeds [42, 100, 2026]...")
    predictor_banks = load_predictor_banks(device)
    cond_estimator = ConditionalGainEstimator()

    all_query_traces: List[Dict[str, Any]] = []
    # Dict: policy -> { (workload, seed) -> {f1, acc, cost, lat, ...} }
    run_records: Dict[str, List[Dict[str, Any]]] = {}

    for seed in SEEDS:
        p_bank = predictor_banks[seed]
        policies = instantiate_policies(p_bank, device, cond_estimator)

        for pol_name, policy in policies.items():
            if pol_name not in run_records:
                run_records[pol_name] = []

            for w_name in WORKLOAD_NAMES:
                wl = workloads[w_name]
                w_text = wl["text"]
                w_audio = wl["audio"]
                w_visual = wl["visual"]
                w_masks = wl["modality_mask"]
                w_deadlines = wl["deadline_ms"]
                w_budgets = wl["cost_budget"]

                preds = []
                costs = []
                latencies = []
                on_times = []
                b_violations = []

                for i in range(N):
                    sample_feat = {
                        "text": w_text[i : i + 1],
                        "audio": w_audio[i : i + 1],
                        "visual": w_visual[i : i + 1],
                    }
                    sample_mask = {
                        "text": bool(w_masks["text"][i]),
                        "audio": bool(w_masks["audio"][i]),
                        "visual": bool(w_masks["visual"][i]),
                    }
                    d_ms = float(w_deadlines[i])
                    b_cost = float(w_budgets[i])

                    # Exact per-query budget and deadline propagation!
                    res = policy.route_single_query(
                        sample_feat, sample_mask, deadline_ms=d_ms, cost_budget=b_cost
                    )

                    pred = int(res["prediction"])
                    c = float(res["spent_cost"])
                    lat = float(res["spent_latency_ms"])
                    on_time = int(lat <= d_ms)
                    b_viol = int(c > b_cost)

                    preds.append(pred)
                    costs.append(c)
                    latencies.append(lat)
                    on_times.append(on_time)
                    b_violations.append(b_viol)

                    all_query_traces.append({
                        "sample_idx": i,
                        "video_id": video_ids[i],
                        "seed": seed,
                        "workload": w_name,
                        "policy": pol_name,
                        "true_label": int(y_true[i]),
                        "prediction": pred,
                        "spent_cost": c,
                        "spent_latency_ms": lat,
                        "final_subset": res.get("final_subset", "NONE"),
                        "routing_reason": res.get("routing_reason", "unknown"),
                        "status": res.get("status", "SUCCESS"),
                        "budget_limit": b_cost,
                        "deadline_limit": d_ms,
                        "on_time": on_time,
                        "budget_violated": b_viol,
                    })

                y_pred_arr = np.array(preds)
                # Primary estimand: per-(workload, seed) 3-class Macro-F1
                m_f1 = compute_3class_macro_f1(y_true, y_pred_arr)
                acc = float(np.mean(y_true == y_pred_arr))
                m_cost = float(np.mean(costs))
                m_lat = float(np.mean(latencies))
                ot_rate = float(np.mean(on_times))
                bv_rate = float(np.mean(b_violations))
                accept_rate = float(np.mean(y_pred_arr != -1))

                run_records[pol_name].append({
                    "seed": seed,
                    "workload": w_name,
                    "macro_f1": m_f1,
                    "accuracy": acc,
                    "mean_cost": m_cost,
                    "mean_latency_ms": m_lat,
                    "on_time_rate": ot_rate,
                    "budget_violation_rate": bv_rate,
                    "acceptance_rate": accept_rate,
                    "num_samples": N,
                })

    # Summarize metrics across all runs
    summary_metrics = {}
    table_rows = []

    for pol_name, runs in run_records.items():
        df_runs = pd.DataFrame(runs)

        # Primary Estimand: unweighted mean of per-(workload, seed) Macro-F1
        primary_f1_mean = float(df_runs["macro_f1"].mean())
        primary_f1_std = float(df_runs["macro_f1"].std())

        mean_cost = float(df_runs["mean_cost"].mean())
        mean_lat = float(df_runs["mean_latency_ms"].mean())
        mean_ot = float(df_runs["on_time_rate"].mean())
        mean_bv = float(df_runs["budget_violation_rate"].mean())
        mean_acc = float(df_runs["accuracy"].mean())
        mean_acpt = float(df_runs["acceptance_rate"].mean())

        # Secondary Estimand: pooled confusion matrix Macro-F1
        pol_traces = [t for t in all_query_traces if t["policy"] == pol_name]
        pooled_y_true = np.array([t["true_label"] for t in pol_traces])
        pooled_y_pred = np.array([t["prediction"] for t in pol_traces])
        secondary_f1_pooled = compute_3class_macro_f1(pooled_y_true, pooled_y_pred)

        per_workload_breakdown = {}
        for w_name in WORKLOAD_NAMES:
            w_runs = df_runs[df_runs["workload"] == w_name]
            per_workload_breakdown[w_name] = {
                "macro_f1": float(w_runs["macro_f1"].mean()),
                "mean_cost": float(w_runs["mean_cost"].mean()),
                "mean_latency_ms": float(w_runs["mean_latency_ms"].mean()),
                "on_time_rate": float(w_runs["on_time_rate"].mean()),
            }

        summary_metrics[pol_name] = {
            "primary_estimand": {
                "metric": "unweighted_mean_per_workload_seed_macro_f1",
                "mean": primary_f1_mean,
                "std": primary_f1_std,
            },
            "secondary_estimand": {
                "metric": "pooled_confusion_macro_f1",
                "score": secondary_f1_pooled,
            },
            "mean_cost": mean_cost,
            "mean_latency_ms": mean_lat,
            "mean_accuracy": mean_acc,
            "on_time_rate": mean_ot,
            "budget_violation_rate": mean_bv,
            "acceptance_rate": mean_acpt,
            "num_runs": len(runs),
            "total_queries_evaluated": len(pol_traces),
            "per_workload_breakdown": per_workload_breakdown,
        }

        table_rows.append({
            "Policy": pol_name,
            "Primary_Macro_F1": primary_f1_mean,
            "F1_Std": primary_f1_std,
            "Secondary_Pooled_F1": secondary_f1_pooled,
            "Mean_Cost": mean_cost,
            "Mean_Latency_ms": mean_lat,
            "On_Time_Rate": mean_ot,
            "Budget_Violations": mean_bv,
            "Acceptance_Rate": mean_acpt,
        })

    # Save summary and query traces
    out_summary_json = os.path.join(DEV_DIR, "policy_metrics.json")
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    df_table = pd.DataFrame(table_rows).sort_values(by="Primary_Macro_F1", ascending=False)
    out_csv = os.path.join(DEV_DIR, "matched_budget_frontiers.csv")
    df_table.to_csv(out_csv, index=False)

    out_traces_json = os.path.join(DEV_DIR, "query_traces.json")
    with open(out_traces_json, "w", encoding="utf-8") as f:
        json.dump(all_query_traces, f)

    logger.info(f"=== Development Evaluation Complete: {len(all_query_traces):,} total queries evaluated ===")
    logger.info(f"\nTop Policies by Primary Macro-F1:\n{df_table.to_string(index=False)}")
    logger.info(f"Artifacts saved to {DEV_DIR}")
    return summary_metrics


if __name__ == "__main__":
    evaluate_development()
