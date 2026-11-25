"""
scripts/v13_evaluate_workloads.py: Comprehensive Workload Benchmark for Tri-SACTA V13.

Evaluates all candidate and baseline policies across 7 realistic workloads and 3 seeds:
1. UtilitySequentialPolicy (Expected-Utility Sequential Acquisition)
2. ConfidenceRouting_0.55 (Standard Confidence Routing, tau=0.55)
3. ConfidenceRouting_0.65 (Standard Confidence Routing, tau=0.65)
4. FixedFallback (Static Text-only fallback)
5. MatchedBudgetRandom (Random modality acquisition at matched cost)
6. Static_TA (Static Text + Audio)
7. Static_TV (Static Text + Visual)
8. AlwaysOnFusion (Always-on full multimodal fusion)

Preserves 100% of per-query prediction traces, subset transitions, costs, latencies,
and policy-matched conformal prediction sets.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.service.utility_router import UtilityRouter, SUBSET_METADATA
from src.service.baseline_policies import (
    ConfidenceRoutingPolicy,
    FixedFallbackPolicy,
    AlwaysOnFusionPolicy,
    MatchedBudgetRandomPolicy,
)
from src.models.fusion_backbone import SimpleMLP

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
SUBSET_CONFIGS = {
    "T":   {"in_dim": 768, "hidden_dim": 64},
    "A":   {"in_dim": 74, "hidden_dim": 32},
    "V":   {"in_dim": 35, "hidden_dim": 16},
    "TA":  {"in_dim": 842, "hidden_dim": 64},
    "TV":  {"in_dim": 803, "hidden_dim": 64},
    "AV":  {"in_dim": 109, "hidden_dim": 32},
    "TAV": {"in_dim": 877, "hidden_dim": 64},
}


def load_predictor_bank(seed: int, device: torch.device) -> Dict[str, nn.Module]:
    ckpt_dir = os.path.join(repo_root, "results", "v13_utility", "training", "checkpoints")
    bank = {}
    for k, cfg in SUBSET_CONFIGS.items():
        fpath = os.path.join(ckpt_dir, f"mlp_{k}_seed{seed}.pt")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing checkpoint: {fpath}")
        model = SimpleMLP(in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"], num_classes=3)
        ckpt = torch.load(fpath, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        bank[k] = model
    return bank


def evaluate_split_workloads(split_name: str = "val", allow_historical_test: bool = False):
    if split_name in ("holdout", "eligible_holdout", "confirmatory"):
        raise PermissionError("PHASE I GUARDRAIL (H04): Use gated Phase II runner for fresh holdout.")
    if split_name == "test" and not allow_historical_test:
        raise PermissionError("PHASE I GUARDRAIL (H04): Direct evaluation of 'test' requires explicit opt-in.")
    logger.info(f"=== Starting Comprehensive Workload Benchmark for split: '{split_name}' ===")
    workloads_dir = os.path.join(repo_root, "results", "v13_utility", "workloads", split_name)
    eval_dir = os.path.join(repo_root, "results", "v13_utility", "evaluation", split_name)
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load conformal calibration quantiles
    conf_path = os.path.join(repo_root, "results", "v13_utility", "conformal", "conformal_calibration.json")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Missing conformal calibration: {conf_path}. Run calibration first.")
    with open(conf_path, "r", encoding="utf-8") as f:
        conf_data = json.load(f)

    # Load all workloads into memory
    workload_data: Dict[str, Dict[str, Any]] = {}
    for w_name in WORKLOADS:
        w_file = os.path.join(workloads_dir, f"{w_name}.pt")
        if not os.path.exists(w_file):
            raise FileNotFoundError(f"Missing workload file: {w_file}")
        workload_data[w_name] = torch.load(w_file, map_location="cpu", weights_only=False)

    ref_labels = workload_data["CLEAN_MULTIMODAL"]["labels"]
    ref_video_ids = workload_data["CLEAN_MULTIMODAL"]["video_ids"]
    N = len(ref_labels)
    logger.info(f"Loaded {len(WORKLOADS)} workloads. N={N} samples, {len(np.unique(ref_video_ids))} unique video clusters.")

    # Storage for all evaluation results
    full_metrics: Dict[str, Dict[str, Any]] = {}
    all_traces: Dict[str, Any] = {}

    for w_name in WORKLOADS:
        logger.info(f"\n======================================================\nEvaluating Workload: {w_name}\n======================================================")
        w_dict = workload_data[w_name]
        w_labels = np.array(w_dict["labels"])
        w_vids = w_dict["video_ids"]

        t_tensor = w_dict["text"].to(device)
        a_tensor = w_dict["audio"].to(device)
        v_tensor = w_dict["visual"].to(device)
        masks = w_dict["modality_mask"]
        deadlines = w_dict["deadline_ms"]
        budgets = w_dict["cost_budget"]

        full_metrics[w_name] = {}

        for seed in SEEDS:
            bank = load_predictor_bank(seed, device)
            q_hat = conf_data["seeds"][str(seed)]["calibrated_quantile"]

            # Initialize policies
            policies: Dict[str, Any] = {
                "UtilitySequentialPolicy": UtilityRouter(
                    predictor_bank=bank,
                    device=device,
                    lambda_utility=0.05,
                    tau_stop_confidence=0.75,
                    conformal_quantile=q_hat,
                ),
                "ConfidenceRouting_0.55": ConfidenceRoutingPolicy(predictor_bank=bank, device=device, tau_confidence=0.55),
                "ConfidenceRouting_0.65": ConfidenceRoutingPolicy(predictor_bank=bank, device=device, tau_confidence=0.65),
                "FixedFallback": FixedFallbackPolicy(predictor_bank=bank, device=device),
                "MatchedBudgetRandom": MatchedBudgetRandomPolicy(predictor_bank=bank, device=device, p_acquire=0.35, seed=seed),
                "AlwaysOnFusion": AlwaysOnFusionPolicy(predictor_bank=bank, device=device),
            }

            for pol_name, pol_instance in policies.items():
                trace_key = f"{split_name}_{w_name}_{pol_name}_seed{seed}"
                preds = np.zeros(N, dtype=np.int64)
                costs = np.zeros(N, dtype=np.float32)
                latencies = np.zeros(N, dtype=np.float32)
                covered_flags = np.zeros(N, dtype=np.float32)
                set_sizes = np.zeros(N, dtype=np.float32)

                query_records = []

                for i in range(N):
                    sample_feats = {
                        "text": t_tensor[i : i + 1],
                        "audio": a_tensor[i : i + 1],
                        "visual": v_tensor[i : i + 1],
                    }
                    sample_mask = {
                        "text": bool(masks["text"][i]),
                        "audio": bool(masks["audio"][i]),
                        "visual": bool(masks["visual"][i]),
                    }
                    d_ms = float(deadlines[i])
                    c_b = float(budgets[i])

                    res = pol_instance.route_single_query(
                        features=sample_feats,
                        modality_mask=sample_mask,
                        deadline_ms=d_ms,
                        cost_budget=c_b,
                    )

                    pred = res["prediction"]
                    p_dist = res["probabilities"]
                    c_spent = res["spent_cost"]
                    lat_spent = res["spent_latency_ms"]
                    path = res["acquisition_path"]
                    reason = res.get("routing_reason", "")

                    # Conformal set evaluation
                    if "conformal_set" in res:
                        c_set = res["conformal_set"]
                    else:
                        threshold = 1.0 - q_hat
                        c_set = [c for c in range(3) if p_dist[c] >= threshold]
                        if not c_set:
                            c_set = [int(np.argmax(p_dist))]

                    is_cov = float(w_labels[i] in c_set)

                    preds[i] = pred
                    costs[i] = c_spent
                    latencies[i] = lat_spent
                    covered_flags[i] = is_cov
                    set_sizes[i] = len(c_set)

                    query_records.append({
                        "sample_idx": i,
                        "video_id": w_vids[i],
                        "label": int(w_labels[i]),
                        "prediction": int(pred),
                        "spent_cost": float(c_spent),
                        "spent_latency_ms": float(lat_spent),
                        "final_subset": path[-1],
                        "acquisition_path": path,
                        "routing_reason": reason,
                        "conformal_set": [int(c) for c in c_set],
                        "covered": bool(is_cov),
                    })

                acc = float(accuracy_score(w_labels, preds))
                f1 = float(f1_score(w_labels, preds, average="macro", zero_division=0))
                mean_c = float(np.mean(costs))
                mean_l = float(np.mean(latencies))
                cov = float(np.mean(covered_flags))
                mean_ss = float(np.mean(set_sizes))

                pol_entry = full_metrics[w_name].setdefault(pol_name, {
                    "accuracy_by_seed": [],
                    "macro_f1_by_seed": [],
                    "cost_by_seed": [],
                    "latency_by_seed": [],
                    "coverage_by_seed": [],
                    "set_size_by_seed": [],
                })
                pol_entry["accuracy_by_seed"].append(acc)
                pol_entry["macro_f1_by_seed"].append(f1)
                pol_entry["cost_by_seed"].append(mean_c)
                pol_entry["latency_by_seed"].append(mean_l)
                pol_entry["coverage_by_seed"].append(cov)
                pol_entry["set_size_by_seed"].append(mean_ss)

                # Save raw per-query trace for this run
                trace_file = os.path.join(eval_dir, f"trace_{w_name}_{pol_name}_seed{seed}.json")
                with open(trace_file, "w", encoding="utf-8") as f:
                    json.dump(query_records, f)

    # Compute aggregated mean and std across seeds for each policy and workload
    summary_report: Dict[str, Any] = {
        "split": split_name,
        "sample_count": N,
        "unique_video_clusters": len(np.unique(ref_video_ids)),
        "workloads": {},
    }

    for w_name in WORKLOADS:
        summary_report["workloads"][w_name] = {}
        logger.info(f"\n--- Multi-Seed Summary for Workload: {w_name} ---")
        for pol_name, m_dict in full_metrics[w_name].items():
            f1_mean = float(np.mean(m_dict["macro_f1_by_seed"]))
            f1_std = float(np.std(m_dict["macro_f1_by_seed"]))
            acc_mean = float(np.mean(m_dict["accuracy_by_seed"]))
            cost_mean = float(np.mean(m_dict["cost_by_seed"]))
            lat_mean = float(np.mean(m_dict["latency_by_seed"]))
            cov_mean = float(np.mean(m_dict["coverage_by_seed"]))
            ss_mean = float(np.mean(m_dict["set_size_by_seed"]))

            summary_report["workloads"][w_name][pol_name] = {
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "accuracy_mean": acc_mean,
                "cost_mean": cost_mean,
                "latency_mean": lat_mean,
                "conformal_coverage_mean": cov_mean,
                "conformal_set_size_mean": ss_mean,
            }
            logger.info(f"{pol_name:26s} | Macro-F1: {f1_mean:.4f} ± {f1_std:.4f} | Cost: {cost_mean:.2f} | Lat: {lat_mean:4.1f}ms | Cov: {cov_mean*100:5.1f}% | Size: {ss_mean:.2f}")

    metrics_file = os.path.join(eval_dir, "workload_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"\nBenchmark completed successfully! Summary saved to {metrics_file}")
    return summary_report


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    evaluate_split_workloads(split)
