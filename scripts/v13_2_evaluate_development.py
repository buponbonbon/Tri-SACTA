"""
scripts/v13_2_evaluate_development.py: Fair Development Policy Benchmarking for Tri-SACTA V13.2.

Features:
- Enforces exact per-query cost_budget and deadline_ms propagation (no hardcoded 3.5).
- Traps masked feature access and handles infeasible budget / blackout fail-closed.
- Separates Unrestricted Always-On from Budget-Admitted Max-Subset.
- Computes BOTH Service-Quality Macro-F1 (primary estimand: late/failed/rejected -> -1 FN)
  AND Raw Macro-F1, plus secondary pooled confusion Macro-F1, useful-on-time, acceptance,
  and budget violation rates.
- Evaluates the predeclared development-only boundary/guardrail stress suite.
- Exports complete per-query traces to results/v13_2/development/query_traces.json.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
    MaskedFeatureAccessViolation,
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
    return {
        "SequentialConfidence_0.55_0.45": SequentialConfidenceRouter(
            predictor_bank, device, tau_stage1=0.55, tau_stage2=0.45, audio_first=True
        ),
        "SequentialConfidence_0.65_0.55": SequentialConfidenceRouter(
            predictor_bank, device, tau_stage1=0.65, tau_stage2=0.55, audio_first=True
        ),
        "SequentialConfidence_VisualFirst_0.55_0.45": SequentialConfidenceRouter(
            predictor_bank, device, tau_stage1=0.55, tau_stage2=0.45, audio_first=False
        ),
        "ConfidenceRouting_0.55": ConfidenceRoutingPolicy(predictor_bank, device, tau_confidence=0.55),
        "ConfidenceRouting_0.65": ConfidenceRoutingPolicy(predictor_bank, device, tau_confidence=0.65),
        "AlwaysOnFusion": AlwaysOnFusionPolicy(predictor_bank, device),
        "BudgetAdmittedMaxSubset": BudgetAdmittedMaxSubsetPolicy(predictor_bank, device),
        "Static_T": StaticSubsetPolicy(predictor_bank, device, target_subset="T"),
        "Static_TA": StaticSubsetPolicy(predictor_bank, device, target_subset="TA"),
        "Static_TV": StaticSubsetPolicy(predictor_bank, device, target_subset="TV"),
        "Static_AV": StaticSubsetPolicy(predictor_bank, device, target_subset="AV"),
        "Static_TAV": StaticSubsetPolicy(predictor_bank, device, target_subset="TAV"),
        "FixedFallback": FixedFallbackPolicy(predictor_bank, device),
        "MatchedBudgetRandom_1.45": MatchedBudgetRandomPolicy(predictor_bank, device, p_acquire=0.28, seed=42),
        "DerivedThreshold_0.05": DerivedThresholdRouter(predictor_bank, device, lambda_utility=0.05),
        "ConditionalUtility_0.05": ConditionalUtilityRouter(
            predictor_bank, device, lambda_utility=0.05, estimator=cond_estimator
        ),
    }


class PoisonedFeatureDict(dict):
    """Raises MaskedFeatureAccessViolation if an unavailable modality key is read."""
    def __init__(self, features: Dict[str, torch.Tensor], mask: Dict[str, bool]):
        super().__init__(features)
        self._mask = mask

    def __getitem__(self, key: str) -> torch.Tensor:
        if not self._mask.get(key, True):
            raise MaskedFeatureAccessViolation(f"Poisoned provider trap triggered on masked modality '{key}'.")
        return super().__getitem__(key)


def run_boundary_guardrail_suite(predictor_bank: Dict[str, nn.Module], device: torch.device) -> Dict[str, Any]:
    """Predeclared development-only boundary & guardrail suite (Section 3.2)."""
    router = SequentialConfidenceRouter(predictor_bank, device, tau_stage1=0.55, tau_stage2=0.45, audio_first=True)
    val_data = load_split("val", REPO_ROOT)
    feat = {"text": val_data["text"][0:1], "audio": val_data["audio"][0:1], "visual": val_data["visual"][0:1]}
    clean_mask = {"text": True, "audio": True, "visual": True}

    cases = []
    # 1. Below initial cost (0.99 < 1.0)
    r_low_c = router.route_single_query(feat, clean_mask, deadline_ms=50.0, cost_budget=0.99)
    cases.append({
        "case": "below_initial_cost_0.99",
        "pass": r_low_c["prediction"] == -1 and r_low_c["spent_cost"] == 0.0 and r_low_c["status"] == "REJECTED_INFEASIBLE",
        "prediction": r_low_c["prediction"], "spent_cost": r_low_c["spent_cost"], "status": r_low_c["status"],
    })
    # 2. Exact initial budget (1.0)
    r_exact_1 = router.route_single_query(feat, clean_mask, deadline_ms=50.0, cost_budget=1.0)
    cases.append({
        "case": "exact_budget_1.0",
        "pass": r_exact_1["spent_cost"] <= 1.0 and r_exact_1["final_subset"] == "T",
        "final_subset": r_exact_1["final_subset"], "spent_cost": r_exact_1["spent_cost"],
    })
    # 3. Just below vs above deadline (0.99ms vs 6.0ms vs 14.0ms)
    r_low_d = router.route_single_query(feat, clean_mask, deadline_ms=0.99, cost_budget=3.5)
    r_tight_d = router.route_single_query(feat, clean_mask, deadline_ms=5.9, cost_budget=3.5)
    r_exact_d = router.route_single_query(feat, clean_mask, deadline_ms=14.0, cost_budget=3.5)
    cases.append({
        "case": "deadline_boundaries_0.99_5.9_14.0",
        "pass": (
            r_low_d["prediction"] == -1
            and r_tight_d["spent_latency_ms"] <= 5.9
            and r_exact_d["spent_latency_ms"] <= 14.0
        ),
        "below_1ms_status": r_low_d["status"],
        "at_5.9ms_subset": r_tight_d["final_subset"],
        "at_14.0ms_subset": r_exact_d["final_subset"],
    })
    # 4. Complete blackout
    r_blk = router.route_single_query(feat, {"text": False, "audio": False, "visual": False}, deadline_ms=50.0, cost_budget=3.5)
    cases.append({
        "case": "complete_blackout",
        "pass": r_blk["prediction"] == -1 and r_blk["spent_cost"] == 0.0 and r_blk["status"] == "REJECTED_BLACKOUT",
        "prediction": r_blk["prediction"], "status": r_blk["status"],
    })
    # 5. Poisoned masked provider across partial masks
    partial_mask = {"text": True, "audio": False, "visual": False}
    poisoned = PoisonedFeatureDict(feat, partial_mask)
    r_poison = router.route_single_query(poisoned, partial_mask, deadline_ms=50.0, cost_budget=3.5)
    cases.append({
        "case": "poisoned_masked_provider_no_leak",
        "pass": r_poison["final_subset"] == "T" and r_poison["prediction"] in (0, 1, 2),
        "final_subset": r_poison["final_subset"],
    })

    all_pass = all(c["pass"] for c in cases)
    return {"all_passed": all_pass, "cases": cases}


def evaluate_development():
    utc_now = datetime.now(timezone.utc).isoformat()
    logger.info("=== Tri-SACTA V13.2: Phase I-C Fair Development Evaluation Initiated ===")
    device = torch.device("cpu")

    val_data = load_split("val", REPO_ROOT)
    y_true = val_data["labels"].numpy()
    video_ids = val_data["video_ids"]
    N = len(y_true)
    logger.info(f"Loaded validation split: {N} samples across {len(set(video_ids))} video clusters.")

    from scripts.v13_generate_workloads import generate_workloads_for_split
    wl_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "workloads", "val")
    if not os.path.exists(wl_dir) or not os.path.exists(os.path.join(wl_dir, "workloads_manifest.json")):
        generate_workloads_for_split("val")

    workloads = {}
    for w_name in WORKLOAD_NAMES:
        workloads[w_name] = torch.load(os.path.join(wl_dir, f"{w_name}.pt"), map_location="cpu", weights_only=False)

    predictor_banks = load_predictor_banks(device)
    cond_estimator = ConditionalGainEstimator()

    boundary_report = run_boundary_guardrail_suite(predictor_banks[42], device)
    with open(os.path.join(DEV_DIR, "boundary_guardrails.json"), "w", encoding="utf-8") as f:
        json.dump(boundary_report, f, indent=2)
    if not boundary_report["all_passed"]:
        raise RuntimeError(f"Development boundary guardrail suite FAILED: {boundary_report}")

    all_query_traces: List[Dict[str, Any]] = []
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

                raw_preds = []
                sq_preds = []
                costs = []
                latencies = []
                on_times = []
                b_violations = []
                accepts = []
                useful_ots = []

                for i in range(N):
                    sample_mask = {
                        "text": bool(w_masks["text"][i]),
                        "audio": bool(w_masks["audio"][i]),
                        "visual": bool(w_masks["visual"][i]),
                    }
                    sample_feat = PoisonedFeatureDict({
                        "text": w_text[i : i + 1],
                        "audio": w_audio[i : i + 1],
                        "visual": w_visual[i : i + 1],
                    }, sample_mask)

                    d_ms = float(w_deadlines[i])
                    b_cost = float(w_budgets[i])

                    res = policy.route_single_query(
                        sample_feat, sample_mask, deadline_ms=d_ms, cost_budget=b_cost
                    )

                    pred = int(res["prediction"])
                    c = float(res["spent_cost"])
                    lat = float(res["spent_latency_ms"])
                    on_time = int(lat <= d_ms)
                    b_viol = int(c > b_cost)
                    accepted = int(pred in (0, 1, 2))
                    sq_pred = pred if (accepted == 1 and on_time == 1) else -1
                    useful_ot = int(sq_pred in (0, 1, 2))

                    raw_preds.append(pred)
                    sq_preds.append(sq_pred)
                    costs.append(c)
                    latencies.append(lat)
                    on_times.append(on_time)
                    b_violations.append(b_viol)
                    accepts.append(accepted)
                    useful_ots.append(useful_ot)

                    all_query_traces.append({
                        "sample_idx": i,
                        "video_id": video_ids[i],
                        "seed": seed,
                        "workload": w_name,
                        "policy": pol_name,
                        "true_label": int(y_true[i]),
                        "prediction": pred,
                        "service_quality_prediction": sq_pred,
                        "spent_cost": c,
                        "spent_latency_ms": lat,
                        "final_subset": res.get("final_subset", "NONE"),
                        "acquisition_path": res.get("acquisition_path", []),
                        "routing_reason": res.get("routing_reason", "unknown"),
                        "status": res.get("status", "SUCCESS"),
                        "budget_limit": b_cost,
                        "deadline_limit": d_ms,
                        "modality_mask": sample_mask,
                        "on_time": on_time,
                        "budget_violated": b_viol,
                        "accepted": accepted,
                        "useful_on_time": useful_ot,
                    })

                raw_arr = np.array(raw_preds)
                sq_arr = np.array(sq_preds)
                sq_f1 = compute_3class_macro_f1(y_true, sq_arr)
                raw_f1 = compute_3class_macro_f1(y_true, raw_arr)
                acc = float(np.mean(y_true == sq_arr))

                run_records[pol_name].append({
                    "seed": seed,
                    "workload": w_name,
                    "macro_f1": sq_f1,
                    "raw_macro_f1": raw_f1,
                    "accuracy": acc,
                    "mean_cost": float(np.mean(costs)),
                    "mean_latency_ms": float(np.mean(latencies)),
                    "on_time_rate": float(np.mean(on_times)),
                    "budget_violation_rate": float(np.mean(b_violations)),
                    "acceptance_rate": float(np.mean(accepts)),
                    "useful_on_time_rate": float(np.mean(useful_ots)),
                    "num_samples": N,
                })

    summary_metrics: Dict[str, Any] = {}
    table_rows = []
    for pol_name, runs in run_records.items():
        df_runs = pd.DataFrame(runs)
        sq_f1_mean = float(df_runs["macro_f1"].mean())
        sq_f1_std = float(df_runs["macro_f1"].std())
        raw_f1_mean = float(df_runs["raw_macro_f1"].mean())

        pol_traces = [t for t in all_query_traces if t["policy"] == pol_name]
        pooled_y_true = np.array([t["true_label"] for t in pol_traces])
        pooled_y_sq = np.array([t["service_quality_prediction"] for t in pol_traces])
        secondary_f1_pooled = compute_3class_macro_f1(pooled_y_true, pooled_y_sq)

        per_workload_breakdown = {}
        for w_name in WORKLOAD_NAMES:
            w_runs = df_runs[df_runs["workload"] == w_name]
            per_workload_breakdown[w_name] = {
                "service_quality_macro_f1": float(w_runs["macro_f1"].mean()),
                "raw_macro_f1": float(w_runs["raw_macro_f1"].mean()),
                "mean_cost": float(w_runs["mean_cost"].mean()),
                "mean_latency_ms": float(w_runs["mean_latency_ms"].mean()),
                "on_time_rate": float(w_runs["on_time_rate"].mean()),
                "useful_on_time_rate": float(w_runs["useful_on_time_rate"].mean()),
            }

        summary_metrics[pol_name] = {
            "primary_estimand": {
                "metric": "unweighted_mean_per_workload_seed_service_quality_macro_f1",
                "mean": sq_f1_mean,
                "std": sq_f1_std,
            },
            "raw_macro_f1_mean": raw_f1_mean,
            "secondary_estimand": {
                "metric": "pooled_confusion_macro_f1",
                "score": secondary_f1_pooled,
            },
            "mean_cost": float(df_runs["mean_cost"].mean()),
            "mean_latency_ms": float(df_runs["mean_latency_ms"].mean()),
            "mean_accuracy": float(df_runs["accuracy"].mean()),
            "on_time_rate": float(df_runs["on_time_rate"].mean()),
            "budget_violation_rate": float(df_runs["budget_violation_rate"].mean()),
            "acceptance_rate": float(df_runs["acceptance_rate"].mean()),
            "useful_on_time_rate": float(df_runs["useful_on_time_rate"].mean()),
            "num_runs": len(runs),
            "total_queries_evaluated": len(pol_traces),
            "per_workload_breakdown": per_workload_breakdown,
        }

        table_rows.append({
            "Policy": pol_name,
            "Primary_SQ_Macro_F1": sq_f1_mean,
            "Raw_Macro_F1": raw_f1_mean,
            "F1_Std": sq_f1_std,
            "Secondary_Pooled_F1": secondary_f1_pooled,
            "Mean_Cost": float(df_runs["mean_cost"].mean()),
            "Mean_Latency_ms": float(df_runs["mean_latency_ms"].mean()),
            "On_Time_Rate": float(df_runs["on_time_rate"].mean()),
            "Useful_On_Time_Rate": float(df_runs["useful_on_time_rate"].mean()),
            "Budget_Violations": float(df_runs["budget_violation_rate"].mean()),
            "Acceptance_Rate": float(df_runs["acceptance_rate"].mean()),
        })

    expected_grid = {
        "schema_version": "13.2",
        "timestamp_utc": utc_now,
        "split": "val",
        "num_samples_per_cell": N,
        "num_video_clusters": len(set(video_ids)),
        "seeds": SEEDS,
        "workloads": WORKLOAD_NAMES,
        "policies": list(run_records.keys()),
        "expected_total_rows": len(run_records) * len(WORKLOAD_NAMES) * len(SEEDS) * N,
        "deadline_stress_design_note": (
            "HETEROGENEOUS_DEADLINE_STRESS cycles paired scenarios [(15.0ms, 1.0), (30.0ms, 2.2), (50.0ms, 3.5)] "
            "via i % 3; NOT a 3x3 Cartesian product. Since 15ms > nominal 14ms TAV latency, "
            "deadline-only stress does not exceed nominal simulation time on clean latencies."
        ),
    }
    with open(os.path.join(DEV_DIR, "expected_grid.json"), "w", encoding="utf-8") as f:
        json.dump(expected_grid, f, indent=2)

    with open(os.path.join(DEV_DIR, "policy_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    df_table = pd.DataFrame(table_rows).sort_values(by="Primary_SQ_Macro_F1", ascending=False)
    df_table.to_csv(os.path.join(DEV_DIR, "matched_budget_frontiers.csv"), index=False)

    with open(os.path.join(DEV_DIR, "query_traces.json"), "w", encoding="utf-8") as f:
        json.dump(all_query_traces, f)

    logger.info(f"=== Development Evaluation Complete: {len(all_query_traces):,} total queries evaluated ===")
    logger.info(f"\n{df_table.to_string(index=False)}")
    return summary_metrics


if __name__ == "__main__":
    evaluate_development()
