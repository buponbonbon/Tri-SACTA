"""
scripts/v12_1_evaluate_policies.py: Exhaustive Matched-Budget Benchmark for Tri-SACTA V12.1.

Evaluates all 10 candidate and baseline policies across 7 realistic workloads and 3 seeds:
- Baseline policies: majority_class, text_only_lr, text_only_mlp, audio_only_mlp,
  visual_only_mlp, always_on_fusion, multimodal_concat_mlp, fixed_fallback,
  confidence_routing, historical_v12_brmo.
- Core contribution: orchestrated_brmo (Tri-SACTA V12.1).
- Scientific ablations: no_reliability, matched_budget_random, no_budget, no_conformal.

Outputs:
- results/v12_1_systems/policies/policy_metrics.json
- results/v12_1_systems/policies/pareto_points.csv
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.service.app import _load_fail_closed_router
from src.service.router import OrchestratorConfig

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

POLICIES = [
    "majority_class",
    "text_only_lr",
    "multimodal_concat_lr",
    "text_only_mlp",
    "audio_only_mlp",
    "visual_only_mlp",
    "always_on_fusion",
    "multimodal_concat_mlp",
    "fixed_fallback",
    "confidence_routing",
    "historical_v12_brmo",
    "orchestrated_brmo",
]

ABLATIONS = [
    "ablation_no_reliability",
    "ablation_matched_budget_random",
    "ablation_no_budget_constraint",
    "ablation_no_conformal_abstention",
]


def evaluate_all():
    logger.info("Initializing fail-closed BRMO router engine...")
    router = _load_fail_closed_router()
    logger.info(f"Router loaded successfully on {router.device}.")

    workloads_dir = os.path.join(repo_root, "results", "v12_1_systems", "workloads")
    out_dir = os.path.join(repo_root, "results", "v12_1_systems", "policies")
    os.makedirs(out_dir, exist_ok=True)

    all_metrics: Dict[str, Any] = {}
    pareto_rows: List[Dict[str, Any]] = []

    for w_name in WORKLOADS:
        w_path = os.path.join(workloads_dir, f"{w_name}.pt")
        if not os.path.exists(w_path):
            raise FileNotFoundError(f"Missing workload file: {w_path}")

        w_data = torch.load(w_path, map_location="cpu", weights_only=False)
        N = len(w_data["labels"])
        labels = w_data["labels"]
        text_feats = w_data["text"]
        audio_feats = w_data["audio"]
        visual_feats = w_data["visual"]
        mod_mask = w_data["modality_mask"]
        deadlines = w_data["deadline_ms"]
        costs = w_data["cost_budget"]

        all_metrics[w_name] = {}
        logger.info(f"--- Evaluating Workload: {w_name} ({N} samples) ---")

        # Combine standard policies and ablations
        eval_run_list = [(p, False, False, False, False) for p in POLICIES]
        eval_run_list.append(("orchestrated_brmo", True, False, False, False))  # no_reliability
        eval_run_list.append(("orchestrated_brmo", False, True, False, False))  # matched_budget_random
        eval_run_list.append(("orchestrated_brmo", False, False, True, False))  # no_budget
        eval_run_list.append(("orchestrated_brmo", False, False, False, True))  # no_conformal

        for p_spec in eval_run_list:
            policy_name, dis_rel, match_rand, dis_bud, dis_conf = p_spec
            display_name = policy_name
            if dis_rel:
                display_name = "ablation_no_reliability"
            elif match_rand:
                display_name = "ablation_matched_budget_random"
            elif dis_bud:
                display_name = "ablation_no_budget_constraint"
            elif dis_conf:
                display_name = "ablation_no_conformal_abstention"

            per_seed_results = []

            for seed in SEEDS:
                # Configure router ablations
                router.config.disable_reliability = dis_rel
                router.config.matched_budget_random = match_rand
                router.config.disable_budget = dis_bud
                router.config.disable_conformal = dis_conf

                preds_list = []
                abstained_list = []
                costs_list = []
                latencies_list = []
                deadlines_met_list = []
                budgets_met_list = []
                fast_path_count = 0

                for i in range(N):
                    f_i = {
                        "text": text_feats[i] if mod_mask["text"][i] else None,
                        "audio": audio_feats[i] if mod_mask["audio"][i] else None,
                        "visual": visual_feats[i] if mod_mask["visual"][i] else None,
                    }
                    m_mask_i = {
                        "text": bool(mod_mask["text"][i]),
                        "audio": bool(mod_mask["audio"][i]),
                        "visual": bool(mod_mask["visual"][i]),
                    }
                    d_i = deadlines[i]
                    c_i = costs[i]

                    res = router.route_and_predict(
                        features=f_i,
                        modality_mask=m_mask_i,
                        deadline_ms=d_i,
                        cost_budget=c_i,
                        policy=policy_name,
                        seed=seed,
                    )

                    preds_list.append(res["predicted_class_id"])
                    abstained_list.append(res["abstained"])
                    costs_list.append(res["compute_cost"])
                    latencies_list.append(res["latency_ms"])
                    deadlines_met_list.append(res["deadline_met"])
                    budgets_met_list.append(res["budget_met"])
                    if res["routing_decision"] in ["text_fast_path", "historical_v12_text_fast_path", "confidence_routing_fast_text"]:
                        fast_path_count += 1

                preds_np = np.array(preds_list)
                abst_np = np.array(abstained_list)
                acc = float(accuracy_score(labels, preds_np))
                f1_macro = float(f1_score(labels, preds_np, average="macro", zero_division=0))
                f1_weighted = float(f1_score(labels, preds_np, average="weighted", zero_division=0))

                # Effective metrics on retained (non-abstained) samples
                retained_mask = ~abst_np
                eff_acc = float(accuracy_score(labels[retained_mask], preds_np[retained_mask])) if retained_mask.any() else 0.0
                eff_f1 = float(f1_score(labels[retained_mask], preds_np[retained_mask], average="macro", zero_division=0)) if retained_mask.any() else 0.0

                abst_rate = float(np.mean(abst_np))
                fast_path_rate = float(fast_path_count / N)
                mean_cost = float(np.mean(costs_list))
                mean_latency = float(np.mean(latencies_list))
                dl_compliance = float(np.mean(deadlines_met_list))
                bud_compliance = float(np.mean(budgets_met_list))

                per_seed_results.append({
                    "seed": seed,
                    "accuracy": acc,
                    "macro_f1": f1_macro,
                    "weighted_f1": f1_weighted,
                    "effective_accuracy": eff_acc,
                    "effective_macro_f1": eff_f1,
                    "abstention_rate": abst_rate,
                    "fast_path_rate": fast_path_rate,
                    "mean_cost": mean_cost,
                    "mean_latency_ms": mean_latency,
                    "deadline_compliance": dl_compliance,
                    "budget_compliance": bud_compliance,
                })

                pareto_rows.append({
                    "workload": w_name,
                    "policy": display_name,
                    "seed": seed,
                    "macro_f1": f1_macro,
                    "effective_macro_f1": eff_f1,
                    "accuracy": acc,
                    "abstention_rate": abst_rate,
                    "fast_path_rate": fast_path_rate,
                    "compute_cost": mean_cost,
                    "latency_ms": mean_latency,
                    "deadline_compliance": dl_compliance,
                    "budget_compliance": bud_compliance,
                })

            # Multi-seed aggregates
            accs = [r["accuracy"] for r in per_seed_results]
            f1s = [r["macro_f1"] for r in per_seed_results]
            eff_f1s = [r["effective_macro_f1"] for r in per_seed_results]
            costs_m = [r["mean_cost"] for r in per_seed_results]
            lats = [r["mean_latency_ms"] for r in per_seed_results]
            absts = [r["abstention_rate"] for r in per_seed_results]
            fp_rates = [r["fast_path_rate"] for r in per_seed_results]

            all_metrics[w_name][display_name] = {
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s)),
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "effective_macro_f1_mean": float(np.mean(eff_f1s)),
                "abstention_rate_mean": float(np.mean(absts)),
                "fast_path_rate_mean": float(np.mean(fp_rates)),
                "compute_cost_mean": float(np.mean(costs_m)),
                "latency_ms_mean": float(np.mean(lats)),
                "per_seed": per_seed_results,
            }

            logger.info(
                f"  [{display_name:32s}] F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} | "
                f"Cost: {np.mean(costs_m):.2f} | Lat: {np.mean(lats):.2f}ms | FastPath: {np.mean(fp_rates)*100:.1f}% | Abst: {np.mean(absts)*100:.1f}%"
            )

    # Save comprehensive metrics JSON
    with open(os.path.join(out_dir, "policy_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Save Pareto CSV
    df = pd.DataFrame(pareto_rows)
    df.to_csv(os.path.join(out_dir, "pareto_points.csv"), index=False)

    logger.info(f"Policy evaluation complete. Saved metrics to {out_dir}")


if __name__ == "__main__":
    evaluate_all()
