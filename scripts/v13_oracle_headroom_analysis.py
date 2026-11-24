"""
scripts/v13_oracle_headroom_analysis.py: Rigorous Oracle Headroom Analysis for Tri-SACTA V13.

Performs exhaustive Oracle analysis across all 7 modality subsets:
{T}, {A}, {V}, {TA}, {TV}, {AV}, {TAV}

Evaluates:
1. Static Subsets Performance: Accuracy, Macro-F1, Compute Cost for each subset.
2. Unconstrained Oracle Upper Bound: Picks the minimal-cost correct subset per sample if one exists.
3. Cost-Constrained Oracle Frontiers: Oracle performance under explicit budget constraints B in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5].
4. Subset Error Complementarity: Rescue probability P(S_rescuer correct | S_primary wrong) across all pairs.
5. Headroom Gate Evaluation: Headroom = F1(Oracle) - max_S F1(S).
6. Clustered Video-Grouped Block Bootstrap: 1,000 resamples clustered by video_id to calculate genuine 95% CIs.

Outputs:
- results/v13_utility/headroom/headroom_summary.json
- results/v13_utility/headroom/complementarity_matrix.csv
- results/v13_utility/headroom/oracle_frontier.csv
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 100, 2026]
SUBSET_KEYS = ["T", "A", "V", "TA", "TV", "AV", "TAV"]
SUBSET_COSTS = {
    "T": 1.0,
    "A": 1.2,
    "V": 1.3,
    "TA": 2.2,
    "TV": 2.3,
    "AV": 2.5,
    "TAV": 3.5,
}
NUM_CLASSES = 3


def video_grouped_bootstrap_indices(video_ids: List[str], rng: np.random.Generator) -> List[int]:
    """Resamples unique video clusters with replacement and returns utterance indices."""
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


def run_headroom_analysis(split_name: str = "val"):
    logger.info(f"Running Oracle Headroom Analysis on split: '{split_name}' across seeds {SEEDS}...")
    pred_dir = os.path.join(repo_root, "results", "v13_utility", "predictions")
    out_dir = os.path.join(repo_root, "results", "v13_utility", "headroom")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load predictions for all subsets across all seeds
    data_by_seed: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        data_by_seed[seed] = {}
        for s_key in SUBSET_KEYS:
            pred_file = os.path.join(pred_dir, f"preds_{s_key}_seed{seed}.json")
            if not os.path.exists(pred_file):
                raise FileNotFoundError(f"Missing prediction file: {pred_file}. Run training script first.")
            with open(pred_file, "r", encoding="utf-8") as f:
                rec = json.load(f)
            data_by_seed[seed][s_key] = rec[split_name]

    # Verify ground truth labels and video_ids consistency
    ref_labels = np.array(data_by_seed[42]["T"]["labels"])
    ref_video_ids = data_by_seed[42]["T"]["video_ids"]
    N = len(ref_labels)

    # Summary metrics collector
    analysis_results: Dict[str, Any] = {
        "split": split_name,
        "sample_count": N,
        "unique_video_clusters": len(np.unique(ref_video_ids)),
        "subsets": {},
        "oracle": {},
        "headroom": {},
    }

    # 2. Evaluate static subsets across seeds
    logger.info("--- Evaluating Static Modality Subsets ---")
    static_summary: Dict[str, Dict[str, List[float]]] = {s: {"acc": [], "f1": []} for s in SUBSET_KEYS}

    for s_key in SUBSET_KEYS:
        cost = SUBSET_COSTS[s_key]
        for seed in SEEDS:
            preds = np.array(data_by_seed[seed][s_key]["predictions"])
            acc = float(accuracy_score(ref_labels, preds))
            f1 = float(f1_score(ref_labels, preds, average="macro", zero_division=0))
            static_summary[s_key]["acc"].append(acc)
            static_summary[s_key]["f1"].append(f1)

        analysis_results["subsets"][s_key] = {
            "cost": cost,
            "accuracy_mean": float(np.mean(static_summary[s_key]["acc"])),
            "accuracy_std": float(np.std(static_summary[s_key]["acc"])),
            "macro_f1_mean": float(np.mean(static_summary[s_key]["f1"])),
            "macro_f1_std": float(np.std(static_summary[s_key]["f1"])),
        }
        logger.info(f"Subset {s_key:4s} (Cost {cost:.1f}): Acc = {analysis_results['subsets'][s_key]['accuracy_mean']:.4f} ± {analysis_results['subsets'][s_key]['accuracy_std']:.4f} | "
                    f"Macro-F1 = {analysis_results['subsets'][s_key]['macro_f1_mean']:.4f} ± {analysis_results['subsets'][s_key]['macro_f1_std']:.4f}")

    best_static_subset = max(SUBSET_KEYS, key=lambda k: analysis_results["subsets"][k]["macro_f1_mean"])
    best_static_f1 = analysis_results["subsets"][best_static_subset]["macro_f1_mean"]
    best_static_acc = analysis_results["subsets"][best_static_subset]["accuracy_mean"]

    # 3. Evaluate Unconstrained Oracle
    logger.info("--- Evaluating Unconstrained Oracle ---")
    oracle_accs = []
    oracle_f1s = []
    oracle_costs = []
    oracle_subset_choices: Dict[int, List[str]] = {s: [] for s in SEEDS}

    for seed in SEEDS:
        oracle_preds = np.zeros(N, dtype=np.int64)
        oracle_sample_costs = np.zeros(N, dtype=np.float32)

        for i in range(N):
            y_i = ref_labels[i]
            # Find all subsets that correctly classify sample i
            correct_subsets = [s for s in SUBSET_KEYS if data_by_seed[seed][s]["predictions"][i] == y_i]

            if len(correct_subsets) > 0:
                # Pick minimal cost correct subset
                best_sub = min(correct_subsets, key=lambda s: SUBSET_COSTS[s])
                oracle_preds[i] = y_i
                oracle_sample_costs[i] = SUBSET_COSTS[best_sub]
                oracle_subset_choices[seed].append(best_sub)
            else:
                # If no subset is correct, fallback to T (minimal cost 1.0)
                oracle_preds[i] = data_by_seed[seed]["T"]["predictions"][i]
                oracle_sample_costs[i] = SUBSET_COSTS["T"]
                oracle_subset_choices[seed].append("T")

        acc = float(accuracy_score(ref_labels, oracle_preds))
        f1 = float(f1_score(ref_labels, oracle_preds, average="macro", zero_division=0))
        mean_c = float(np.mean(oracle_sample_costs))

        oracle_accs.append(acc)
        oracle_f1s.append(f1)
        oracle_costs.append(mean_c)

    analysis_results["oracle"]["unconstrained"] = {
        "accuracy_mean": float(np.mean(oracle_accs)),
        "accuracy_std": float(np.std(oracle_accs)),
        "macro_f1_mean": float(np.mean(oracle_f1s)),
        "macro_f1_std": float(np.std(oracle_f1s)),
        "cost_mean": float(np.mean(oracle_costs)),
        "cost_std": float(np.std(oracle_costs)),
    }

    # 4. Compute Headroom Metrics
    headroom_acc = analysis_results["oracle"]["unconstrained"]["accuracy_mean"] - best_static_acc
    headroom_f1 = analysis_results["oracle"]["unconstrained"]["macro_f1_mean"] - best_static_f1
    cost_delta = analysis_results["oracle"]["unconstrained"]["cost_mean"] - SUBSET_COSTS[best_static_subset]

    analysis_results["headroom"] = {
        "best_static_subset": best_static_subset,
        "best_static_f1": best_static_f1,
        "best_static_acc": best_static_acc,
        "oracle_f1": analysis_results["oracle"]["unconstrained"]["macro_f1_mean"],
        "oracle_acc": analysis_results["oracle"]["unconstrained"]["accuracy_mean"],
        "oracle_cost": analysis_results["oracle"]["unconstrained"]["cost_mean"],
        "headroom_macro_f1": headroom_f1,
        "headroom_accuracy": headroom_acc,
        "cost_delta": cost_delta,
        "meaningful_headroom_exists": bool(headroom_f1 >= 0.05),
    }

    logger.info(f"=== HEADROOM VERDICT ===")
    logger.info(f"Best Static Subset: '{best_static_subset}' (F1: {best_static_f1:.4f}, Acc: {best_static_acc:.4f}, Cost: {SUBSET_COSTS[best_static_subset]:.1f})")
    logger.info(f"Oracle Upper Bound: (F1: {analysis_results['oracle']['unconstrained']['macro_f1_mean']:.4f}, "
                f"Acc: {analysis_results['oracle']['unconstrained']['accuracy_mean']:.4f}, Cost: {analysis_results['oracle']['unconstrained']['cost_mean']:.2f})")
    logger.info(f"Headroom (F1 Gain): {headroom_f1 * 100:.2f}% ({headroom_f1:+.4f}) | Headroom (Acc Gain): {headroom_acc * 100:.2f}%")
    logger.info(f"Headroom >= 5.0 F1 points? {analysis_results['headroom']['meaningful_headroom_exists']}")

    # 5. Cost-Constrained Oracle Frontiers
    logger.info("--- Evaluating Cost-Constrained Oracle Frontier ---")
    budget_grid = [1.0, 1.5, 2.0, 2.2, 2.5, 3.0, 3.5]
    frontier_rows: List[Dict[str, Any]] = []

    for B in budget_grid:
        feasible_subsets = [s for s in SUBSET_KEYS if SUBSET_COSTS[s] <= B]
        b_accs, b_f1s, b_costs = [], [], []

        for seed in SEEDS:
            preds_b = np.zeros(N, dtype=np.int64)
            costs_b = np.zeros(N, dtype=np.float32)

            for i in range(N):
                y_i = ref_labels[i]
                correct_feasible = [s for s in feasible_subsets if data_by_seed[seed][s]["predictions"][i] == y_i]
                if len(correct_feasible) > 0:
                    best_s = min(correct_feasible, key=lambda s: SUBSET_COSTS[s])
                    preds_b[i] = y_i
                    costs_b[i] = SUBSET_COSTS[best_s]
                else:
                    # Fallback to minimal cost feasible subset
                    min_s = min(feasible_subsets, key=lambda s: SUBSET_COSTS[s])
                    preds_b[i] = data_by_seed[seed][min_s]["predictions"][i]
                    costs_b[i] = SUBSET_COSTS[min_s]

            b_accs.append(accuracy_score(ref_labels, preds_b))
            b_f1s.append(f1_score(ref_labels, preds_b, average="macro", zero_division=0))
            b_costs.append(np.mean(costs_b))

        frontier_rows.append({
            "budget_limit": B,
            "oracle_macro_f1_mean": float(np.mean(b_f1s)),
            "oracle_macro_f1_std": float(np.std(b_f1s)),
            "oracle_accuracy_mean": float(np.mean(b_accs)),
            "oracle_cost_mean": float(np.mean(b_costs)),
        })

    df_frontier = pd.DataFrame(frontier_rows)
    frontier_csv = os.path.join(out_dir, "oracle_frontier.csv")
    df_frontier.to_csv(frontier_csv, index=False)
    analysis_results["oracle"]["constrained_frontier"] = frontier_rows

    # 6. Error Complementarity & Rescue Matrix (Pairwise)
    logger.info("--- Computing Pairwise Error Complementarity Matrix ---")
    comp_matrix = np.zeros((len(SUBSET_KEYS), len(SUBSET_KEYS)), dtype=np.float64)

    for i_idx, s_primary in enumerate(SUBSET_KEYS):
        for j_idx, s_rescuer in enumerate(SUBSET_KEYS):
            rescue_rates = []
            for seed in SEEDS:
                prim_preds = np.array(data_by_seed[seed][s_primary]["predictions"])
                resc_preds = np.array(data_by_seed[seed][s_rescuer]["predictions"])

                prim_wrong = (prim_preds != ref_labels)
                if np.sum(prim_wrong) > 0:
                    resc_correct_given_wrong = np.mean(resc_preds[prim_wrong] == ref_labels[prim_wrong])
                    rescue_rates.append(resc_correct_given_wrong)
                else:
                    rescue_rates.append(0.0)
            comp_matrix[i_idx, j_idx] = np.mean(rescue_rates)

    df_comp = pd.DataFrame(comp_matrix, index=SUBSET_KEYS, columns=SUBSET_KEYS)
    comp_csv = os.path.join(out_dir, "complementarity_matrix.csv")
    df_comp.to_csv(comp_csv)
    analysis_results["complementarity_matrix"] = df_comp.to_dict()

    logger.info(f"Error Rescue Rates when Text (T) is Wrong:\n{df_comp.loc['T']}")

    # 7. Clustered Video-Grouped Bootstrap for Oracle Headroom
    logger.info("--- Executing Clustered Video-Grouped Block Bootstrap (B=1000) ---")
    rng = np.random.default_rng(2026)
    B_BOOT = 1000
    boot_headrooms = []

    for b in range(B_BOOT):
        sample_indices = video_grouped_bootstrap_indices(ref_video_ids, rng)
        boot_y = ref_labels[sample_indices]

        # Best static on resample
        static_f1s_b = []
        for s_key in SUBSET_KEYS:
            # Average predictions across seeds on this resample
            seed_f1s = []
            for seed in SEEDS:
                p = np.array(data_by_seed[seed][s_key]["predictions"])[sample_indices]
                seed_f1s.append(f1_score(boot_y, p, average="macro", zero_division=0))
            static_f1s_b.append(np.mean(seed_f1s))
        best_static_f1_b = max(static_f1s_b)

        # Oracle on resample
        seed_oracle_f1s = []
        for seed in SEEDS:
            oracle_preds_b = np.zeros(len(sample_indices), dtype=np.int64)
            for local_idx, orig_i in enumerate(sample_indices):
                y_i = ref_labels[orig_i]
                corr = [s for s in SUBSET_KEYS if data_by_seed[seed][s]["predictions"][orig_i] == y_i]
                if len(corr) > 0:
                    oracle_preds_b[local_idx] = y_i
                else:
                    oracle_preds_b[local_idx] = data_by_seed[seed]["T"]["predictions"][orig_i]
            seed_oracle_f1s.append(f1_score(boot_y, oracle_preds_b, average="macro", zero_division=0))

        oracle_f1_b = np.mean(seed_oracle_f1s)
        boot_headrooms.append(oracle_f1_b - best_static_f1_b)

    ci_lower = float(np.percentile(boot_headrooms, 2.5))
    ci_upper = float(np.percentile(boot_headrooms, 97.5))
    analysis_results["headroom"]["bootstrap_95ci"] = [ci_lower, ci_upper]
    analysis_results["headroom"]["bootstrap_mean"] = float(np.mean(boot_headrooms))
    analysis_results["headroom"]["bootstrap_std"] = float(np.std(boot_headrooms))

    logger.info(f"Video-Grouped Bootstrap Headroom: Mean = {np.mean(boot_headrooms):.4f} | 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Save summary json
    summary_path = os.path.join(out_dir, "headroom_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)

    logger.info(f"Headroom analysis complete. Saved to {summary_path}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    run_headroom_analysis(split)
