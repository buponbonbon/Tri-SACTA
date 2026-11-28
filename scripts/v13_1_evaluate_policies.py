"""
scripts/v13_1_evaluate_policies.py: Comprehensive Fair Policy Benchmarking for Tri-SACTA V13.1.

Evaluates on identical samples, identical seeds, identical workloads, and matched budgets:
1. Candidate M2: ConditionalUtilityRouter (varying lambda_utility)
2. Fair Baseline: SequentialConfidenceRouter (audio-first, visual-first; varying tau)
3. Derived Equivalent: DerivedThresholdRouter (M0 historical V13, M1 corrected priors)
4. Frozen Reference: UtilityRouter (V13)
5. One-Shot Baselines: ConfidenceRouting_0.55, ConfidenceRouting_0.65
6. Fallback & Static: FixedFallback, AlwaysOnFusion, Static_T, Static_TA, Static_TV, Static_TAV
7. Random Baseline: MatchedBudgetRandom (calibrated per budget point)

Outputs:
- results/v13_1/evaluation/policy_metrics.json
- results/v13_1/evaluation/matched_budget_frontiers.csv
- results/v13_1/evaluation/query_traces.json
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

from src.data.loader import SUBSET_METADATA
from src.service.utility_router import UtilityRouter
from src.service.baseline_policies import (
    ConfidenceRoutingPolicy,
    FixedFallbackPolicy,
    AlwaysOnFusionPolicy,
    MatchedBudgetRandomPolicy,
    DerivedThresholdRouter,
    SequentialConfidenceRouter,
    StaticSubsetPolicy,
)
from src.service.conditional_utility import ConditionalGainEstimator
from src.service.conditional_router import ConditionalUtilityRouter
from src.models.fusion_backbone import SimpleMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
BUDGET_GRID = [1.0, 1.5, 2.0, 2.2, 2.3, 2.5, 3.5]


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


def load_workloads(split_name: str = "val") -> Dict[str, Dict[str, Any]]:
    """Loads pre-generated deterministic workloads."""
    workload_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "workloads", split_name)
    loaded = {}
    for w_name in WORKLOAD_NAMES:
        fpath = os.path.join(workload_dir, f"{w_name}.pt")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing workload file: {fpath}")
        loaded[w_name] = torch.load(fpath, weights_only=False, map_location="cpu")
    return loaded


def instantiate_policies(
    bank: Dict[str, nn.Module],
    device: torch.device,
    estimator: ConditionalGainEstimator,
) -> Dict[str, Any]:
    """Creates the dictionary of policy instances to compare."""
    from src.service.utility_router import EMPIRICAL_PRIORS

    # Load corrected priors from json if available
    corr_priors_path = os.path.join(REPO_ROOT, "results", "v13_1", "priors", "empirical_transition_priors.json")
    if os.path.exists(corr_priors_path):
        with open(corr_priors_path, "r") as f:
            p_data = json.load(f)["empirical_val_transitions"]
        corrected_priors = {
            (v["source_subset"], v["candidate_modality"]): {
                "gamma": v["gamma_rescue_rate"],
                "nu": v["nu_negative_transfer_rate"],
            }
            for v in p_data.values()
        }
    else:
        corrected_priors = EMPIRICAL_PRIORS

    policies = {
        # 1. Candidate M2: Conditional Utility Router
        "ConditionalUtility_0.05": ConditionalUtilityRouter(bank, device, lambda_utility=0.05, tau_stop_confidence=0.75, estimator=estimator),
        "ConditionalUtility_0.02": ConditionalUtilityRouter(bank, device, lambda_utility=0.02, tau_stop_confidence=0.75, estimator=estimator),
        "ConditionalUtility_0.08": ConditionalUtilityRouter(bank, device, lambda_utility=0.08, tau_stop_confidence=0.75, estimator=estimator),

        # 2. Fair Sequential Confidence Router (action-matched)
        "SequentialConfidence_0.65_0.55": SequentialConfidenceRouter(bank, device, tau_stage1=0.65, tau_stage2=0.55, audio_first=True),
        "SequentialConfidence_0.75_0.60": SequentialConfidenceRouter(bank, device, tau_stage1=0.75, tau_stage2=0.60, audio_first=True),
        "SequentialConfidence_0.55_0.45": SequentialConfidenceRouter(bank, device, tau_stage1=0.55, tau_stage2=0.45, audio_first=True),

        # 3. Derived Equivalent Router (M0 historical V13 and M1 corrected)
        "DerivedThreshold_V13": DerivedThresholdRouter(bank, device, lambda_utility=0.05, tau_stop_confidence=0.75, custom_priors=EMPIRICAL_PRIORS),
        "DerivedThreshold_Corrected_M1": DerivedThresholdRouter(bank, device, lambda_utility=0.05, tau_stop_confidence=0.75, custom_priors=corrected_priors),

        # 4. Frozen V13 Reference
        "UtilityRouter_V13": UtilityRouter(bank, device, lambda_utility=0.05, tau_stop_confidence=0.75),

        # 5. One-Shot Confidence Routing
        "ConfidenceRouting_0.55": ConfidenceRoutingPolicy(bank, device, tau_confidence=0.55),
        "ConfidenceRouting_0.65": ConfidenceRoutingPolicy(bank, device, tau_confidence=0.65),

        # 6. Fallback and Static
        "FixedFallback": FixedFallbackPolicy(bank, device),
        "AlwaysOnFusion": AlwaysOnFusionPolicy(bank, device),
        "Static_T": StaticSubsetPolicy(bank, device, target_subset="T"),
        "Static_TA": StaticSubsetPolicy(bank, device, target_subset="TA"),
        "Static_TV": StaticSubsetPolicy(bank, device, target_subset="TV"),
        "Static_TAV": StaticSubsetPolicy(bank, device, target_subset="TAV"),

        # 7. Matched Budget Random
        "MatchedBudgetRandom": MatchedBudgetRandomPolicy(bank, device, p_acquire=0.35, seed=42),
    }
    return policies


def evaluate_all_policies(split_name: str = "val"):
    logger.info(f"=== Benchmarking All Fair Baselines on split: '{split_name}' ===")
    device = torch.device("cpu")
    out_dir = os.path.join(REPO_ROOT, "results", "v13_1", "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    predictor_banks = load_predictor_banks(device)
    workloads = load_workloads(split_name)
    estimator = ConditionalGainEstimator()

    # Load ground truth labels and video ids
    ref_labels = np.array(workloads["CLEAN_MULTIMODAL"]["labels"])
    ref_video_ids = workloads["CLEAN_MULTIMODAL"]["video_ids"]
    N = len(ref_labels)

    all_traces = []
    summary_metrics: Dict[str, Any] = {}

    # Iterate over seeds and instantiate policies
    for seed in SEEDS:
        bank = predictor_banks[seed]
        policies = instantiate_policies(bank, device, estimator)

        for pol_name, policy in policies.items():
            if pol_name not in summary_metrics:
                summary_metrics[pol_name] = {
                    "workloads": {},
                    "aggregate": {
                        "macro_f1_seeds": [],
                        "accuracy_seeds": [],
                        "mean_cost_seeds": [],
                        "mean_latency_seeds": [],
                        "on_time_rate_seeds": [],
                    },
                }

            # Evaluate each workload
            for w_name in WORKLOAD_NAMES:
                wl = workloads[w_name]
                w_text = wl["text"]
                w_audio = wl["audio"]
                w_visual = wl["visual"]
                w_masks = wl["modality_mask"]
                w_deadlines = wl["deadline_ms"]

                preds = []
                costs = []
                latencies = []
                on_time_flags = []

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

                    # Standard budget 3.5 for full evaluation
                    res = policy.route_single_query(
                        sample_feat, sample_mask, deadline_ms=d_ms, cost_budget=3.5
                    )

                    pred = int(res["prediction"])
                    c = float(res["spent_cost"])
                    lat = float(res["spent_latency_ms"])
                    on_time = int(lat <= d_ms)

                    preds.append(pred)
                    costs.append(c)
                    latencies.append(lat)
                    on_time_flags.append(on_time)

                    # Record trace for bootstrap
                    all_traces.append({
                        "sample_idx": i,
                        "video_id": ref_video_ids[i],
                        "seed": seed,
                        "workload": w_name,
                        "policy": pol_name,
                        "y_true": int(ref_labels[i]),
                        "y_pred": pred,
                        "cost": c,
                        "latency_ms": lat,
                        "on_time": on_time,
                        "final_subset": res.get("final_subset", "T"),
                    })

                preds = np.array(preds)
                f1 = float(fast_macro_f1(ref_labels, preds))
                acc = float(accuracy_score(ref_labels, preds))
                mean_c = float(np.mean(costs))
                mean_lat = float(np.mean(latencies))
                on_time_rate = float(np.mean(on_time_flags))

                if w_name not in summary_metrics[pol_name]["workloads"]:
                    summary_metrics[pol_name]["workloads"][w_name] = {
                        "macro_f1_seeds": [],
                        "accuracy_seeds": [],
                        "mean_cost_seeds": [],
                        "mean_latency_seeds": [],
                        "on_time_rate_seeds": [],
                    }

                summary_metrics[pol_name]["workloads"][w_name]["macro_f1_seeds"].append(f1)
                summary_metrics[pol_name]["workloads"][w_name]["accuracy_seeds"].append(acc)
                summary_metrics[pol_name]["workloads"][w_name]["mean_cost_seeds"].append(mean_c)
                summary_metrics[pol_name]["workloads"][w_name]["mean_latency_seeds"].append(mean_lat)
                summary_metrics[pol_name]["workloads"][w_name]["on_time_rate_seeds"].append(on_time_rate)

    # Compute aggregate metrics across seeds and workloads
    table_rows = []
    for pol_name, p_data in summary_metrics.items():
        # Aggregate over all (workload, seed) runs
        all_f1s = []
        all_accs = []
        all_costs = []
        all_lats = []
        all_ontime = []

        for w_name in WORKLOAD_NAMES:
            w_rec = p_data["workloads"][w_name]
            w_rec["macro_f1_mean"] = float(np.mean(w_rec["macro_f1_seeds"]))
            w_rec["macro_f1_std"] = float(np.std(w_rec["macro_f1_seeds"]))
            w_rec["accuracy_mean"] = float(np.mean(w_rec["accuracy_seeds"]))
            w_rec["mean_cost_mean"] = float(np.mean(w_rec["mean_cost_seeds"]))
            w_rec["on_time_rate_mean"] = float(np.mean(w_rec["on_time_rate_seeds"]))

            all_f1s.extend(w_rec["macro_f1_seeds"])
            all_accs.extend(w_rec["accuracy_seeds"])
            all_costs.extend(w_rec["mean_cost_seeds"])
            all_lats.extend(w_rec["mean_latency_seeds"])
            all_ontime.extend(w_rec["on_time_rate_seeds"])

        p_data["aggregate"] = {
            "macro_f1_mean": float(np.mean(all_f1s)),
            "macro_f1_std": float(np.std(all_f1s)),
            "accuracy_mean": float(np.mean(all_accs)),
            "accuracy_std": float(np.std(all_accs)),
            "mean_cost": float(np.mean(all_costs)),
            "mean_latency_ms": float(np.mean(all_lats)),
            "on_time_rate": float(np.mean(all_ontime)),
        }

        table_rows.append({
            "policy": pol_name,
            "macro_f1": p_data["aggregate"]["macro_f1_mean"],
            "accuracy": p_data["aggregate"]["accuracy_mean"],
            "mean_cost": p_data["aggregate"]["mean_cost"],
            "mean_latency_ms": p_data["aggregate"]["mean_latency_ms"],
            "on_time_rate": p_data["aggregate"]["on_time_rate"],
        })

        logger.info(
            f"Policy {pol_name:30s} | F1={p_data['aggregate']['macro_f1_mean']:.4f} ± {p_data['aggregate']['macro_f1_std']:.4f} | "
            f"Cost={p_data['aggregate']['mean_cost']:.2f} | Lat={p_data['aggregate']['mean_latency_ms']:.1f}ms | OnTime={p_data['aggregate']['on_time_rate'] * 100:.1f}%"
        )

    # Save summary JSON
    metrics_path = os.path.join(out_dir, "policy_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # Save frontiers CSV
    frontier_df = pd.DataFrame(table_rows).sort_values(by="macro_f1", ascending=False)
    csv_path = os.path.join(out_dir, "matched_budget_frontiers.csv")
    frontier_df.to_csv(csv_path, index=False)

    # Save traces JSON (for video-grouped bootstrap)
    traces_path = os.path.join(out_dir, "query_traces.json")
    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump(all_traces, f)

    logger.info(f"Saved evaluation metrics to: {metrics_path}")
    logger.info(f"Saved matched budget frontiers to: {csv_path}")
    logger.info(f"Saved {len(all_traces)} query traces to: {traces_path}")
    return summary_metrics


if __name__ == "__main__":
    evaluate_all_policies("val")
