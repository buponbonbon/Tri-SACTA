"""
scripts/v14_research/run_v14_diagnostics.py: Development-Only V14 Counterfactual Headroom,
Oracle Upper-Bound, and Group-Cross-Fitted Learnability Diagnostic.
Uses ONLY 'val' (N=250, 29 videos) and frozen checkpoints; ZERO Phase II holdout access.
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import load_split, SUBSET_METADATA
from scripts.v13_2_evaluate_development import load_predictor_banks, compute_3class_macro_f1
from scripts.v13_2_run_phase2_holdout import build_workloads_from_tensors, WORKLOAD_NAMES

OUT_DIR = os.path.join(REPO_ROOT, "results", "v14", "research_diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

SUBSET_COSTS = {"T": 1.0, "A": 1.2, "V": 1.3, "TA": 2.2, "TV": 2.3, "AV": 2.5, "TAV": 3.5}
SUBSET_LAT_SEQ = {"T": 1.0, "A": 5.0, "V": 8.0, "TA": 6.0, "TV": 9.0, "AV": 13.0, "TAV": 14.0}
SUBSET_LAT_PAR = {"T": 1.0, "A": 5.0, "V": 8.0, "TA": 6.0, "TV": 9.0, "AV": 9.0, "TAV": 9.0}


def _eval_subset(model: torch.nn.Module, feat_dict: Dict[str, torch.Tensor], mods: List[str]) -> np.ndarray:
    x = torch.cat([feat_dict[m] for m in mods], dim=-1).float()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    return probs


def run_diagnostics() -> Dict[str, Any]:
    val = load_split("val", REPO_ROOT)
    y_true = val["labels"].numpy()
    video_ids = np.array(val["video_ids"])
    N = len(y_true)

    workloads = build_workloads_from_tensors(val["text"], val["audio"], val["visual"])
    banks = load_predictor_banks(torch.device("cpu"))
    bank42 = banks[42]

    # 1. Counterfactual Oracle Headroom across all 7 workloads (seed 42, val N=250)
    oracle_results = {}
    transition_counts = {"T_to_TA": {"rescue": 0, "harm": 0, "total": 0}, "T_to_TV": {"rescue": 0, "harm": 0, "total": 0}, "T_to_TAV": {"rescue": 0, "harm": 0, "total": 0}}
    probe_X, probe_y_res_ta, probe_y_harm_ta, probe_y_res_tav, probe_y_harm_tav, probe_groups = [], [], [], [], [], []

    for w_name in WORKLOAD_NAMES:
        wl = workloads[w_name]
        feats = {"text": wl["text"], "audio": wl["audio"], "visual": wl["visual"]}
        probs_by_sub = {s_key: _eval_subset(bank42[s_key], feats, meta["mods"]) for s_key, meta in SUBSET_METADATA.items()}
        preds_by_sub = {s_key: np.argmax(p, axis=-1) for s_key, p in probs_by_sub.items()}

        # Track T -> TA/TV/TAV transitions
        c_T = (preds_by_sub["T"] == y_true)
        c_TA = (preds_by_sub["TA"] == y_true)
        c_TV = (preds_by_sub["TV"] == y_true)
        c_TAV = (preds_by_sub["TAV"] == y_true)

        transition_counts["T_to_TA"]["rescue"] += int(np.sum((~c_T) & c_TA))
        transition_counts["T_to_TA"]["harm"] += int(np.sum(c_T & (~c_TA)))
        transition_counts["T_to_TA"]["total"] += N

        transition_counts["T_to_TV"]["rescue"] += int(np.sum((~c_T) & c_TV))
        transition_counts["T_to_TV"]["harm"] += int(np.sum(c_T & (~c_TV)))
        transition_counts["T_to_TV"]["total"] += N

        transition_counts["T_to_TAV"]["rescue"] += int(np.sum((~c_T) & c_TAV))
        transition_counts["T_to_TAV"]["harm"] += int(np.sum(c_T & (~c_TAV)))
        transition_counts["T_to_TAV"]["total"] += N

        if w_name in ("CLEAN_MULTIMODAL", "TEXT_EMBEDDING_NOISE", "INTERMITTENT_MISSINGNESS"):
            pT = probs_by_sub["T"]
            maxp = np.max(pT, axis=-1, keepdims=True)
            ent = -np.sum(pT * np.log(np.clip(pT, 1e-8, 1.0)), axis=-1, keepdims=True)
            sorted_p = np.sort(pT, axis=-1)
            margin = (sorted_p[:, -1:] - sorted_p[:, -2:-1])
            X_w = np.hstack([maxp, ent, margin, pT])
            probe_X.append(X_w)
            probe_y_res_ta.append(((~c_T) & c_TA).astype(int))
            probe_y_harm_ta.append((c_T & (~c_TA)).astype(int))
            probe_y_res_tav.append(((~c_T) & c_TAV).astype(int))
            probe_y_harm_tav.append((c_T & (~c_TAV)).astype(int))
            probe_groups.append(video_ids)

        # Compute 5 non-deployable oracles
        unres_preds, reach_preds, budg_preds, dead_preds, joint_par_preds = [], [], [], [], []
        for i in range(N):
            avail = {m: bool(wl["modality_mask"][m][i]) for m in ("text", "audio", "visual")}
            d_ms = float(wl["deadline_ms"][i])
            b_cost = float(wl["cost_budget"][i])
            yt = int(y_true[i])

            def pick_oracle(cands):
                if not cands:
                    return -1
                for s in cands:
                    if preds_by_sub[s][i] == yt:
                        return int(preds_by_sub[s][i])
                return int(preds_by_sub[cands[0]][i])

            all_subs = list(SUBSET_METADATA.keys())
            reach_subs = [s for s in all_subs if all(avail[m] for m in SUBSET_METADATA[s]["mods"])]
            budg_subs = [s for s in reach_subs if SUBSET_COSTS[s] <= b_cost]
            dead_subs = [s for s in reach_subs if SUBSET_LAT_SEQ[s] <= d_ms]
            joint_subs = [s for s in reach_subs if SUBSET_COSTS[s] <= b_cost and SUBSET_LAT_PAR[s] <= d_ms]

            unres_preds.append(pick_oracle(all_subs))
            reach_preds.append(pick_oracle(reach_subs))
            budg_preds.append(pick_oracle(budg_subs))
            dead_preds.append(pick_oracle(dead_subs))
            joint_par_preds.append(pick_oracle(joint_subs))

        oracle_results[w_name] = {
            "static_T_f1": compute_3class_macro_f1(y_true, preds_by_sub["T"]),
            "static_TAV_f1": compute_3class_macro_f1(y_true, preds_by_sub["TAV"]),
            "oracle_unrestricted_f1": compute_3class_macro_f1(y_true, np.array(unres_preds)),
            "oracle_reachable_f1": compute_3class_macro_f1(y_true, np.array(reach_preds)),
            "oracle_budget_feasible_f1": compute_3class_macro_f1(y_true, np.array(budg_preds)),
            "oracle_deadline_feasible_f1": compute_3class_macro_f1(y_true, np.array(dead_preds)),
            "oracle_joint_seq_par_feasible_f1": compute_3class_macro_f1(y_true, np.array(joint_par_preds)),
        }

    # 2. Group-Cross-Fitted Learnability Diagnostic (5-fold GroupKFold by video_id)
    X_all = np.vstack(probe_X)
    g_all = np.concatenate(probe_groups)
    gkf = GroupKFold(n_splits=5)

    learnability = {}
    targets = {
        "rescue_T_to_TA": np.concatenate(probe_y_res_ta),
        "harm_T_to_TA": np.concatenate(probe_y_harm_ta),
        "rescue_T_to_TAV": np.concatenate(probe_y_res_tav),
        "harm_T_to_TAV": np.concatenate(probe_y_harm_tav),
    }
    for t_name, y_t in targets.items():
        oof_const = np.zeros(len(y_t))
        oof_conf = np.zeros(len(y_t))
        oof_state = np.zeros(len(y_t))
        for tr, va in gkf.split(X_all, y_t, g_all):
            oof_const[va] = float(np.mean(y_t[tr]))
            m_conf = LogisticRegression(max_iter=500).fit(X_all[tr, :1], y_t[tr])
            oof_conf[va] = m_conf.predict_proba(X_all[va, :1])[:, 1]
            m_state = LogisticRegression(max_iter=500).fit(X_all[tr], y_t[tr])
            oof_state[va] = m_state.predict_proba(X_all[va])[:, 1]
        learnability[t_name] = {
            "positive_rate": float(np.mean(y_t)),
            "positive_count": int(np.sum(y_t)),
            "total_count": int(len(y_t)),
            "brier_constant": float(brier_score_loss(y_t, oof_const)),
            "brier_confidence_only": float(brier_score_loss(y_t, oof_conf)),
            "brier_observable_state": float(brier_score_loss(y_t, oof_state)),
            "auroc_confidence_only": float(roc_auc_score(y_t, oof_conf)),
            "auroc_observable_state": float(roc_auc_score(y_t, oof_state)),
        }

    summary = {
        "dataset": "val_development_only",
        "num_samples": N,
        "num_video_clusters": len(set(val["video_ids"])),
        "oracle_headroom_by_workload": oracle_results,
        "transition_counts": transition_counts,
        "group_cross_fitted_learnability": learnability,
    }
    with open(os.path.join(OUT_DIR, "v14_diagnostics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("V14_DIAGNOSTICS_OK")
    return summary


if __name__ == "__main__":
    run_diagnostics()
