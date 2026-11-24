"""
scripts/v13_calibrate_conformal.py: Policy-Matched Split Conformal Calibration for Tri-SACTA V13.

Calibrates conformal prediction risk on the exact deployed sequential acquisition trajectory
using the held-out calibration split (N=250, 46 video clusters).

Rectifies Defect 3 from Forensic Audit (eliminates distribution shift between calibration and deployment).
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.service.utility_router import UtilityRouter, SUBSET_METADATA
from src.models.fusion_backbone import SimpleMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEEDS = [42, 100, 2026]
SUBSET_CONFIGS = {
    "T":   {"in_dim": 768, "hidden_dim": 64},
    "A":   {"in_dim": 74, "hidden_dim": 32},
    "V":   {"in_dim": 35, "hidden_dim": 16},
    "TA":  {"in_dim": 842, "hidden_dim": 64},
    "TV":  {"in_dim": 803, "hidden_dim": 64},
    "AV":  {"in_dim": 109, "hidden_dim": 32},
    "TAV": {"in_dim": 877, "hidden_dim": 64},
}
SUBSET_KEYS = list(SUBSET_CONFIGS.keys())


def load_predictor_bank(seed: int, device: torch.device) -> Dict[str, nn.Module]:
    """Loads all 7 task-matched subset MLPs for a given seed."""
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


def run_conformal_calibration(alpha: float = 0.10):
    logger.info(f"Starting Policy-Matched Conformal Calibration on 'cal' split (target alpha={alpha})...")
    data_dir = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", "cal")
    out_dir = os.path.join(repo_root, "results", "v13_utility", "conformal")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load clean cal split features
    t_feat = torch.load(os.path.join(data_dir, "text.pt"), map_location=device, weights_only=False).float()
    a_feat = torch.load(os.path.join(data_dir, "audio.pt"), map_location=device, weights_only=False).float()
    v_feat = torch.load(os.path.join(data_dir, "visual.pt"), map_location=device, weights_only=False).float()
    meta = torch.load(os.path.join(data_dir, "meta.pt"), map_location="cpu", weights_only=False)

    raw_sentiment = np.array([m["raw_sentiment"] for m in meta], dtype=np.float32)
    cal_labels = np.zeros(len(raw_sentiment), dtype=np.int64)
    cal_labels[raw_sentiment < 0.0] = 0
    cal_labels[raw_sentiment == 0.0] = 1
    cal_labels[raw_sentiment > 0.0] = 2

    cal_video_ids = [m["video_id"] for m in meta]
    N = len(cal_labels)
    logger.info(f"Calibration set loaded: N={N} samples, {len(np.unique(cal_video_ids))} unique video clusters.")

    calibration_records: Dict[str, Any] = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "cal_samples": N,
        "cal_video_clusters": len(np.unique(cal_video_ids)),
        "seeds": {},
    }

    quantiles_by_seed: Dict[int, float] = {}

    for seed in SEEDS:
        logger.info(f"--- Calibrating on Deployed Policy Trajectory for Seed {seed} ---")
        bank = load_predictor_bank(seed, device)
        router = UtilityRouter(
            predictor_bank=bank,
            device=device,
            lambda_utility=0.05,
            tau_stop_confidence=0.75,
            conformal_quantile=None,
        )

        nonconformity_scores = []
        routing_subsets = []

        for i in range(N):
            sample_feats = {
                "text": t_feat[i : i + 1],
                "audio": a_feat[i : i + 1],
                "visual": v_feat[i : i + 1],
            }
            mask = {"text": True, "audio": True, "visual": True}

            route_res = router.route_single_query(
                features=sample_feats,
                modality_mask=mask,
                deadline_ms=50.0,
                cost_budget=3.5,
            )

            p_dist = route_res["probabilities"]
            y_true = cal_labels[i]
            # Softmax residual nonconformity score: s = 1 - p(y_true)
            score = 1.0 - float(p_dist[y_true])
            nonconformity_scores.append(score)
            routing_subsets.append(route_res["final_subset"])

        scores_arr = np.array(nonconformity_scores)

        # Standard split conformal quantile with finite sample correction: ceil((N + 1) * (1 - alpha)) / N
        q_level = float(np.ceil((N + 1) * (1.0 - alpha)) / N)
        q_level = min(1.0, max(0.0, q_level))
        calibrated_q = float(np.quantile(scores_arr, q_level, method="higher"))

        quantiles_by_seed[seed] = calibrated_q

        # Verify empirical calibration on cal split itself
        covered = (scores_arr <= calibrated_q).astype(float)
        empirical_cov = float(np.mean(covered))
        # Subset routing distribution on cal split
        unique_subs, sub_counts = np.unique(routing_subsets, return_counts=True)
        sub_dist = {str(k): int(v) for k, v in zip(unique_subs, sub_counts)}

        calibration_records["seeds"][str(seed)] = {
            "calibrated_quantile": calibrated_q,
            "q_level": q_level,
            "mean_nonconformity_score": float(np.mean(scores_arr)),
            "std_nonconformity_score": float(np.std(scores_arr)),
            "routing_distribution": sub_dist,
        }
        logger.info(f"Seed {seed}: Calibrated Conformal Quantile q_hat = {calibrated_q:.4f} (Routing Dist: {sub_dist})")

    mean_q = float(np.mean(list(quantiles_by_seed.values())))
    calibration_records["mean_calibrated_quantile"] = mean_q
    logger.info(f"Mean Calibrated Quantile across seeds: {mean_q:.4f}")

    out_file = os.path.join(out_dir, "conformal_calibration.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calibration_records, f, indent=2)

    logger.info(f"Conformal calibration saved to: {out_file}")
    return calibration_records


if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    run_conformal_calibration(alpha)
