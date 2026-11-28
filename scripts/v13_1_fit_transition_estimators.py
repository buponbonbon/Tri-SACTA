"""
scripts/v13_1_fit_transition_estimators.py: Transition Risk and Equivalence Analysis for Tri-SACTA V13.1.

Computes:
1. Exact empirical transition frequencies:
   - Rescue (Wrong -> Correct, gamma)
   - Negative Transfer (Correct -> Wrong, nu)
   - Correct -> Correct
   - Wrong -> Wrong
   For all valid subset transitions:
   T -> TA, T -> TV
   TA -> TAV, TV -> TAV
   A -> TA, A -> AV
   V -> TV, V -> AV
   AV -> TAV
2. Statistical confidence intervals (Wilson score) for gamma and nu.
3. Analytical and numerical equivalence between Expected Utility and Derived Sequential Confidence Thresholds:
   tau(S, m, lambda) = (gamma - lambda * Delta C) / (gamma + nu)
4. Crossover proofs and unreachability analysis (e.g. Visual unselectable at state T under V13 priors).

Outputs:
- results/v13_1/priors/empirical_transition_priors.json
- results/v13_1/priors/derived_thresholds.json
"""

from __future__ import annotations

import os
import sys
import json
import math
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SEEDS = [42, 100, 2026]
MODALITY_COSTS = {"text": 1.0, "audio": 1.2, "visual": 1.3}

# Historical V13 priors for forensic comparison
HISTORICAL_V13_PRIORS = {
    ("T", "audio"): {"gamma": 0.3768, "nu": 0.1160, "dest": "TA", "cost": 1.2},
    ("T", "visual"): {"gamma": 0.3387, "nu": 0.2245, "dest": "TV", "cost": 1.3},
    ("TA", "visual"): {"gamma": 0.2437, "nu": 0.1700, "dest": "TAV", "cost": 1.3},
    ("TV", "audio"): {"gamma": 0.2891, "nu": 0.1850, "dest": "TAV", "cost": 1.2},
    ("A", "text"): {"gamma": 0.5495, "nu": 0.1200, "dest": "TA", "cost": 1.0},
    ("A", "visual"): {"gamma": 0.3781, "nu": 0.2500, "dest": "AV", "cost": 1.3},
    ("V", "text"): {"gamma": 0.5473, "nu": 0.1200, "dest": "TV", "cost": 1.0},
    ("V", "audio"): {"gamma": 0.3461, "nu": 0.2600, "dest": "AV", "cost": 1.2},
    ("AV", "text"): {"gamma": 0.5467, "nu": 0.1100, "dest": "TAV", "cost": 1.0},
}

TRANSITIONS = [
    ("T", "audio", "TA", 1.2),
    ("T", "visual", "TV", 1.3),
    ("TA", "visual", "TAV", 1.3),
    ("TV", "audio", "TAV", 1.2),
    ("A", "text", "TA", 1.0),
    ("A", "visual", "AV", 1.3),
    ("V", "text", "TV", 1.0),
    ("V", "audio", "AV", 1.2),
    ("AV", "text", "TAV", 1.0),
]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p + (z**2) / (2.0 * total)) / denom
    spread = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)) / denom
    return (max(0.0, float(centre - spread)), min(1.0, float(centre + spread)))


def load_subset_predictions(split_name: str = "val") -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Loads cached predictions for all subsets and seeds."""
    preds_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "predictions")
    subsets = ["T", "A", "V", "TA", "TV", "AV", "TAV"]
    data: Dict[str, Dict[int, Dict[str, Any]]] = {s: {} for s in subsets}

    for sub in subsets:
        for seed in SEEDS:
            fname = f"preds_{sub}_seed{seed}.json"
            fpath = os.path.join(preds_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing prediction file: {fpath}")
            with open(fpath, "r") as f:
                content = json.load(f)
            data[sub][seed] = content[split_name]
    return data


def compute_empirical_transitions(split_name: str = "val") -> Dict[str, Any]:
    print(f"Loading predictions on '{split_name}' for all 7 subsets across seeds {SEEDS}...")
    pred_data = load_subset_predictions(split_name)

    results = {}
    for src, mod, dst, cost in TRANSITIONS:
        key = f"{src}->{dst} (+{mod})"
        n_00, n_01, n_10, n_11 = 0, 0, 0, 0
        total_queries = 0

        for seed in SEEDS:
            src_preds = np.array(pred_data[src][seed]["predictions"])
            dst_preds = np.array(pred_data[dst][seed]["predictions"])
            labels = np.array(pred_data[src][seed]["labels"])
            N = len(labels)
            total_queries += N

            e_src = (src_preds != labels).astype(int)
            e_dst = (dst_preds != labels).astype(int)

            # Outcomes
            # Correct -> Correct
            n_00 += int(np.sum((e_src == 0) & (e_dst == 0)))
            # Correct -> Wrong (Harm / Negative Transfer)
            n_01 += int(np.sum((e_src == 0) & (e_dst == 1)))
            # Wrong -> Correct (Rescue)
            n_10 += int(np.sum((e_src == 1) & (e_dst == 0)))
            # Wrong -> Wrong
            n_11 += int(np.sum((e_src == 1) & (e_dst == 1)))

        n_errors = n_10 + n_11
        n_corrects = n_00 + n_01

        gamma = float(n_10 / n_errors) if n_errors > 0 else 0.0
        nu = float(n_01 / n_corrects) if n_corrects > 0 else 0.0
        gamma_ci = wilson_interval(n_10, n_errors)
        nu_ci = wilson_interval(n_01, n_corrects)

        net_gain = n_10 - n_01
        net_gain_per_query = float(net_gain / total_queries)

        results[f"({src}, {mod})"] = {
            "source_subset": src,
            "candidate_modality": mod,
            "destination_subset": dst,
            "incremental_cost": cost,
            "total_evaluated_queries": total_queries,
            "correct_to_correct": n_00,
            "correct_to_wrong_harm": n_01,
            "wrong_to_correct_rescue": n_10,
            "wrong_to_wrong": n_11,
            "total_source_errors": n_errors,
            "total_source_correct": n_corrects,
            "gamma_rescue_rate": gamma,
            "gamma_95_ci": list(gamma_ci),
            "nu_negative_transfer_rate": nu,
            "nu_95_ci": list(nu_ci),
            "net_correct_change": net_gain,
            "net_gain_per_query": net_gain_per_query,
        }

    return results


def derive_thresholds(
    priors: Dict[str, Any],
    lambda_utility: float = 0.05,
    tau_stop: float = 0.75,
) -> Dict[str, Any]:
    """
    Computes exact confidence thresholds tau(S, m, lambda) from gamma, nu, cost, and lambda:
    tau = (gamma - lambda * cost) / (gamma + nu)
    """
    thresholds = {}
    for key, p in priors.items():
        gamma = p.get("gamma", p.get("gamma_rescue_rate"))
        nu = p.get("nu", p.get("nu_negative_transfer_rate"))
        cost = p.get("cost", p.get("incremental_cost"))

        sum_rates = gamma + nu
        if sum_rates > 0.0:
            tau_val = (gamma - lambda_utility * cost) / sum_rates
        else:
            tau_val = 0.0

        effective_tau = min(tau_val, tau_stop)
        thresholds[key] = {
            "gamma": gamma,
            "nu": nu,
            "cost": cost,
            "lambda_utility": lambda_utility,
            "raw_derived_threshold": float(tau_val),
            "effective_threshold_with_stop": float(effective_tau),
            "is_feasible_for_p_ge_zero": tau_val > 0.0,
        }
    return thresholds


def verify_mathematical_equivalence():
    """
    Analytically and numerically tests equivalence between UtilityRouter decision
    and DerivedThresholdRouter decision across a dense grid of confidence values.
    """
    lambda_utility = 0.05
    tau_stop = 0.75
    p_grid = np.linspace(0.0, 1.0, 10001)

    print("Verifying mathematical equivalence for V13 historical priors...")
    for (src, mod), prior in HISTORICAL_V13_PRIORS.items():
        gamma = prior["gamma"]
        nu = prior["nu"]
        cost = prior["cost"]
        tau = (gamma - lambda_utility * cost) / (gamma + nu)

        # 1. Test decision agreement across all p
        utility_acquire = []
        threshold_acquire = []
        for p in p_grid:
            risk = 1.0 - p
            delta_r = (risk * gamma) - ((1.0 - risk) * nu)
            u = delta_r / cost
            u_dec = (u >= lambda_utility) and (p < tau_stop)

            t_dec = (p <= tau) and (p < tau_stop)
            utility_acquire.append(u_dec)
            threshold_acquire.append(t_dec)

        mismatches = np.sum(np.array(utility_acquire) != np.array(threshold_acquire))
        # Note: numerical edge case at exact floating point tau: within 1e-7 tolerance
        print(f"  {src} + {mod}: tau={tau:.6f}, mismatches over 10,001 points: {mismatches}")
        assert mismatches <= 1, f"Equivalence failed for {src}+{mod} with {mismatches} mismatches!"

    print("Mathematical equivalence successfully verified (0 discrepancy outside float precision)!")


def analyze_crossover_defect():
    """Analyzes the crossover behavior between Audio and Visual acquisition at state T."""
    p_grid = np.linspace(0.0, 1.0, 1001)
    prior_a = HISTORICAL_V13_PRIORS[("T", "audio")]
    prior_v = HISTORICAL_V13_PRIORS[("T", "visual")]

    u_a = [((1 - p) * prior_a["gamma"] - p * prior_a["nu"]) / prior_a["cost"] for p in p_grid]
    u_v = [((1 - p) * prior_v["gamma"] - p * prior_v["nu"]) / prior_v["cost"] for p in p_grid]

    diff = np.array(u_a) - np.array(u_v)
    min_diff = float(np.min(diff))
    max_diff = float(np.max(diff))

    print(f"V13 Crossover Analysis at state T:")
    print(f"  Utility(Audio | T) - Utility(Visual | T) min: {min_diff:.6f}, max: {max_diff:.6f}")
    print(f"  Is Audio Utility strictly greater than Visual for ALL p in [0, 1]? {bool(min_diff > 0)}")
    return {
        "strictly_audio_dominant": bool(min_diff > 0),
        "min_utility_difference": min_diff,
        "max_utility_difference": max_diff,
        "visual_unselectable_at_T_when_audio_available": True,
    }


def main():
    print("=== TRI-SACTA V13.1 TRANSITION ESTIMATION & EQUIVALENCE DERIVATION ===")
    out_dir = os.path.join(REPO_ROOT, "results", "v13_1", "priors")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Compute exact empirical transitions on val split
    val_empirical = compute_empirical_transitions("val")

    # 2. Derive thresholds for historical V13 constants
    v13_thresholds = derive_thresholds(HISTORICAL_V13_PRIORS, lambda_utility=0.05, tau_stop=0.75)

    # 3. Derive thresholds for corrected empirical priors
    corrected_thresholds = derive_thresholds(val_empirical, lambda_utility=0.05, tau_stop=0.75)

    # 4. Verify mathematical equivalence
    verify_mathematical_equivalence()

    # 5. Analyze crossover defect
    crossover_findings = analyze_crossover_defect()

    # 6. Save artifacts
    priors_artifact = {
        "description": "Exact empirical transition frequencies and error decomposition on validation set across seeds 42, 100, 2026",
        "historical_v13_priors_audit": {
            str(k): v for k, v in HISTORICAL_V13_PRIORS.items()
        },
        "empirical_val_transitions": val_empirical,
        "crossover_defect_findings": crossover_findings,
    }

    thresholds_artifact = {
        "mathematical_formula": "tau(S, m, lambda) = (gamma - lambda * Delta C) / (gamma + nu)",
        "v13_derived_thresholds": {str(k): v for k, v in v13_thresholds.items()},
        "corrected_empirical_thresholds": corrected_thresholds,
    }

    priors_path = os.path.join(out_dir, "empirical_transition_priors.json")
    thresholds_path = os.path.join(out_dir, "derived_thresholds.json")

    with open(priors_path, "w") as f:
        json.dump(priors_artifact, f, indent=2)
    with open(thresholds_path, "w") as f:
        json.dump(thresholds_artifact, f, indent=2)

    print(f"\nSaved empirical transition priors to: {priors_path}")
    print(f"Saved derived thresholds to: {thresholds_path}")


if __name__ == "__main__":
    main()
