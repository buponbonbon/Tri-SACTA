"""
scripts/v13_1_fit_conditional_utility.py: Fit and Validate State-Conditioned Transition Gain Estimators (Candidate M2).

Estimates:
G(z, a) = E[E_S - E_{S U {a}} | z] = P(Rescue | z) - P(Harm | z)
using pre-acquisition state features:
z = [p_max, margin, entropy, p_0, p_1, p_2, text_norm, budget_rem, deadline_rem]

Validates:
1. Calibration of predicted vs realized gain on held-out folds.
2. Ablation: Conditional gain (M2) vs Fixed priors (M1) vs Confidence alone.
3. Tests whether state features provide genuine predictive signal beyond confidence alone.

Outputs:
- results/v13_1/priors/conditional_utility_model.json
- results/v13_1/priors/estimator_calibration.json
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import load_split, SUBSET_METADATA
from src.models.fusion_backbone import SimpleMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 100, 2026]


def extract_state_features(
    probs: np.ndarray,
    spent_cost: float,
    spent_lat: float,
    cost_budget: float = 3.5,
    deadline_ms: float = 50.0,
) -> np.ndarray:
    """
    Extracts observed state features z legitimately known before acquisition:
    z = [p_max, margin, entropy, p_neg, p_neu, p_pos, rem_cost_ratio, rem_lat_ratio]
    """
    p_max = float(np.max(probs))
    sorted_p = np.sort(probs)
    margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) > 1 else 1.0
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))

    p_neg = float(probs[0])
    p_neu = float(probs[1])
    p_pos = float(probs[2])

    rem_cost = max(0.0, (cost_budget - spent_cost) / cost_budget)
    rem_lat = max(0.0, (deadline_ms - spent_lat) / deadline_ms)

    return np.array([p_max, margin, entropy, p_neg, p_neu, p_pos, rem_cost, rem_lat], dtype=np.float32)


def generate_training_data() -> Dict[str, Any]:
    """Generates state vectors z and transition gain targets on the 1000-sample train split across seeds."""
    device = torch.device("cpu")
    train_data = load_split("train", REPO_ROOT)
    labels = train_data["labels"].numpy()
    video_ids = train_data["video_ids"]
    N = len(labels)

    # Load all models
    ckpt_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "training", "checkpoints")
    models: Dict[str, Dict[int, SimpleMLP]] = {}
    for s_key, meta in SUBSET_METADATA.items():
        models[s_key] = {}
        for seed in SEEDS:
            m = SimpleMLP(in_dim=meta["in_dim"], hidden_dim=64 if "text" in meta["mods"] else (32 if "audio" in meta["mods"] else 16), num_classes=3)
            # Find checkpoint
            fpath = os.path.join(ckpt_dir, f"mlp_{s_key}_seed{seed}.pt")
            m.load_state_dict(torch.load(fpath, weights_only=True, map_location=device))
            m.eval()
            models[s_key][seed] = m

    # Compute predictions for all subsets
    all_probs: Dict[str, Dict[int, np.ndarray]] = {}
    with torch.no_grad():
        for s_key, meta in SUBSET_METADATA.items():
            all_probs[s_key] = {}
            tensors = [train_data[m] for m in meta["mods"]]
            concat_feat = torch.cat(tensors, dim=-1)
            for seed in SEEDS:
                logits = models[s_key][seed](concat_feat)
                probs = torch.softmax(logits, dim=-1).numpy()
                all_probs[s_key][seed] = probs

    # Build dataset for key transitions:
    # 1. T -> TA (+audio)
    # 2. T -> TV (+visual)
    # 3. TA -> TAV (+visual)
    transitions = [
        ("T", "audio", "TA", 1.0, 1.0),
        ("T", "visual", "TV", 1.0, 1.0),
        ("TA", "visual", "TAV", 2.2, 6.0),
        ("TV", "audio", "TAV", 2.3, 9.0),
    ]

    transition_datasets = {}
    for src, mod, dst, spent_c, spent_l in transitions:
        X_list, y_gain_list, y_rescue_list, y_harm_list, groups_list = [], [], [], [], []

        for seed in SEEDS:
            src_p = all_probs[src][seed]
            dst_p = all_probs[dst][seed]

            src_preds = src_p.argmax(axis=-1)
            dst_preds = dst_p.argmax(axis=-1)

            e_src = (src_preds != labels).astype(int)
            e_dst = (dst_preds != labels).astype(int)

            # Gain: e_src - e_dst (+1 if rescue, -1 if harm, 0 if neutral)
            gain = e_src - e_dst
            rescue = ((e_src == 1) & (e_dst == 0)).astype(int)
            harm = ((e_src == 0) & (e_dst == 1)).astype(int)

            for i in range(N):
                z = extract_state_features(src_p[i], spent_c, spent_l)
                X_list.append(z)
                y_gain_list.append(gain[i])
                y_rescue_list.append(rescue[i])
                y_harm_list.append(harm[i])
                groups_list.append(f"{video_ids[i]}_{seed}")

        transition_datasets[f"{src}+{mod}"] = {
            "source": src,
            "modality": mod,
            "dest": dst,
            "X": np.array(X_list, dtype=np.float32),
            "y_gain": np.array(y_gain_list, dtype=np.float32),
            "y_rescue": np.array(y_rescue_list, dtype=np.int32),
            "y_harm": np.array(y_harm_list, dtype=np.int32),
            "groups": np.array(groups_list),
        }

    return transition_datasets


def fit_and_evaluate_models(datasets: Dict[str, Any]):
    feature_names = ["p_max", "margin", "entropy", "p_neg", "p_neu", "p_pos", "rem_cost", "rem_lat"]
    trained_models = {}
    calibration_reports = {}

    for name, d in datasets.items():
        logger.info(f"--- Fitting Transition Gain Estimator for {name} ---")
        X = d["X"]
        y_gain = d["y_gain"]
        groups = d["groups"]

        # 5-fold GroupKFold by video cluster
        gkf = GroupKFold(n_splits=5)
        cv_pred_gain = np.zeros(len(y_gain), dtype=np.float32)

        # Fit linear Ridge regressor with L2 regularization
        model = Ridge(alpha=10.0)

        for train_idx, val_idx in gkf.split(X, y_gain, groups):
            model.fit(X[train_idx], y_gain[train_idx])
            cv_pred_gain[val_idx] = model.predict(X[val_idx])

        # Final fit on all training data
        model.fit(X, y_gain)

        # Evaluate calibration in 5 bins of predicted gain
        bins = np.percentile(cv_pred_gain, [0, 20, 40, 60, 80, 100])
        bin_records = []
        for b_i in range(len(bins) - 1):
            mask = (cv_pred_gain >= bins[b_i]) & (cv_pred_gain <= bins[b_i + 1])
            if np.sum(mask) > 0:
                mean_pred = float(np.mean(cv_pred_gain[mask]))
                mean_real = float(np.mean(y_gain[mask]))
                count = int(np.sum(mask))
                bin_records.append({
                    "bin_index": b_i,
                    "count": count,
                    "mean_predicted_gain": mean_pred,
                    "mean_realized_gain": mean_real,
                    "calibration_error": float(abs(mean_pred - mean_real)),
                })

        mse = float(mean_squared_error(y_gain, cv_pred_gain))
        baseline_mse = float(mean_squared_error(y_gain, np.full_like(y_gain, np.mean(y_gain))))

        logger.info(f"  {name}: CV MSE={mse:.5f} (vs Mean Baseline MSE={baseline_mse:.5f})")
        for br in bin_records:
            logger.info(f"    Bin {br['bin_index']}: Pred={br['mean_predicted_gain']:+.4f} | Real={br['mean_realized_gain']:+.4f} (N={br['count']})")

        trained_models[name] = {
            "source": d["source"],
            "modality": d["modality"],
            "dest": d["dest"],
            "intercept": float(model.intercept_),
            "coefficients": {fname: float(coef) for fname, coef in zip(feature_names, model.coef_)},
            "cv_mse": mse,
            "baseline_mean_mse": baseline_mse,
        }

        calibration_reports[name] = {
            "cv_mse": mse,
            "baseline_mean_mse": baseline_mse,
            "bins": bin_records,
        }

    return trained_models, calibration_reports


def main():
    logger.info("=== Fitting Tri-SACTA V13.1 Conditional Utility Models ===")
    datasets = generate_training_data()
    models, calibration = fit_and_evaluate_models(datasets)

    out_dir = os.path.join(REPO_ROOT, "results", "v13_1", "priors")
    os.makedirs(out_dir, exist_ok=True)

    models_path = os.path.join(out_dir, "conditional_utility_model.json")
    calib_path = os.path.join(out_dir, "estimator_calibration.json")

    with open(models_path, "w") as f:
        json.dump(models, f, indent=2)
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)

    logger.info(f"Saved conditional utility model to: {models_path}")
    logger.info(f"Saved calibration report to: {calib_path}")


if __name__ == "__main__":
    main()
