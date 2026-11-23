"""
scripts/audit_rq1_spectral_detection.py: Independent Audit and Evaluation for Research Question 1.

RQ1: Does the proposed spectral reliability detector contain useful information,
and does it identify reliable, moderately shifted, and corrupted modalities better
than simpler baseline statistics?
"""

import os
import sys
import json
import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.spectral_detector import SpectralShiftDetector
from src.data.dataset import get_dataloaders
import yaml

def evaluate_rq1():
    with open("configs/mosei_real.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load source statistics
    detector_path = "checkpoints/cmu_mosei/real/seed42/source_spectral_stats.pt"
    if not os.path.exists(detector_path):
        raise FileNotFoundError(f"Missing spectral detector stats: {detector_path}")
    detector = SpectralShiftDetector(
        feature_dims={"text": 768, "audio": 74, "visual": 35, "prosody": 88},
        gamma=config.get("tta", {}).get("gamma_eig", 1.0),
        top_k=config.get("tta", {}).get("top_k_eig", 32),
    )
    detector.load_source_stats(detector_path)

    loaders = get_dataloaders(config, data_mode="real", seed=42)
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # Collect clean test batches
    clean_batches = []
    for b in test_loader:
        clean_batches.append(b)

    results = {"modalities": {}, "noise_severities": {}, "baseline_comparisons": {}}

    # For each modality, test detection across severities
    # Ground truth: 0 = unshifted (clean), 1 = shifted
    for mod in ["text", "audio", "visual"]:
        y_true_list = []
        spectral_scores = []
        mean_drift_scores = []
        var_ratio_scores = []

        # A. Clean batches (Label = 0)
        for b in clean_batches:
            feats = {mod: b["features"][mod]}
            mask = {mod: b["modality_mask"][mod]}
            
            # Spectral detector score
            s_score = detector.compute_shift_scores(feats, mask)[mod]
            spectral_scores.append(s_score)
            y_true_list.append(0)

            # Baseline 1: Euclidean mean drift from source mean
            src_mean = detector.source_means[mod]
            batch_mean = feats[mod].mean(dim=0)
            m_drift = float(torch.norm(batch_mean - src_mean).item())
            mean_drift_scores.append(m_drift)

            # Baseline 2: Variance ratio from source std
            src_var = (detector.source_stds[mod] ** 2).mean()
            batch_var = feats[mod].var(dim=0).mean()
            v_ratio = float(abs(torch.log(batch_var / (src_var + 1e-6))).item())
            var_ratio_scores.append(v_ratio)

        # B. Shifted batches (Label = 1) with varying severities
        severities = [0.1, 0.25, 0.5, 0.8, 1.0]
        mod_sev_results = {}
        for sev in severities:
            sev_scores = []
            for b in clean_batches:
                orig_feat = b["features"][mod].clone()
                # Apply noise proportional to severity
                noise = torch.randn_like(orig_feat) * (sev * 1.5)
                shifted_feat = orig_feat + noise
                feats = {mod: shifted_feat}
                mask = {mod: b["modality_mask"][mod]}

                s_score = detector.compute_shift_scores(feats, mask)[mod]
                spectral_scores.append(s_score)
                sev_scores.append(s_score)
                y_true_list.append(1)

                src_mean = detector.source_means[mod]
                batch_mean = shifted_feat.mean(dim=0)
                mean_drift_scores.append(float(torch.norm(batch_mean - src_mean).item()))

                src_var = (detector.source_stds[mod] ** 2).mean()
                batch_var = shifted_feat.var(dim=0).mean()
                var_ratio_scores.append(float(abs(torch.log(batch_var / (src_var + 1e-6))).item()))

            mod_sev_results[str(sev)] = float(np.mean(sev_scores))

        y_true = np.array(y_true_list)
        spec_s = np.array(spectral_scores)
        mean_s = np.array(mean_drift_scores)
        var_s = np.array(var_ratio_scores)

        # Clean null scores
        clean_scores = spec_s[y_true == 0]
        p95_clean = float(np.percentile(clean_scores, 95))
        fpr_at_p95 = float((clean_scores > p95_clean).mean())

        # AUROC & AUPRC
        auroc_spec = float(roc_auc_score(y_true, spec_s))
        auprc_spec = float(average_precision_score(y_true, spec_s))

        auroc_mean = float(roc_auc_score(y_true, mean_s))
        auprc_mean = float(average_precision_score(y_true, mean_s))

        auroc_var = float(roc_auc_score(y_true, var_s))
        auprc_var = float(average_precision_score(y_true, var_s))

        results["modalities"][mod] = {
            "auroc_spectral": auroc_spec,
            "auprc_spectral": auprc_spec,
            "auroc_mean_drift": auroc_mean,
            "auprc_mean_drift": auprc_mean,
            "auroc_var_ratio": auroc_var,
            "auprc_var_ratio": auprc_var,
            "clean_mean": float(np.mean(clean_scores)),
            "clean_p95": p95_clean,
            "clean_fpr": fpr_at_p95,
            "prevalence_shifted": float(y_true.mean()),
            "severity_curve": mod_sev_results,
        }

    os.makedirs("results/v10_audit/diagnostics", exist_ok=True)
    out_json = "results/v10_audit/diagnostics/rq1_spectral_detection_audit.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=== RQ1 Spectral Shift Detection Audit ===")
    for m, d in results["modalities"].items():
        print(f"Modality [{m}]:")
        print(f"  AUROC: Spectral={d['auroc_spectral']:.4f} vs MeanDrift={d['auroc_mean_drift']:.4f} vs VarRatio={d['auroc_var_ratio']:.4f}")
        print(f"  AUPRC: Spectral={d['auprc_spectral']:.4f} vs MeanDrift={d['auprc_mean_drift']:.4f} vs VarRatio={d['auprc_var_ratio']:.4f}")
        print(f"  Clean FPR: {d['clean_fpr']*100:.1f}%, Clean Mean Score: {d['clean_mean']:.4f}")

if __name__ == "__main__":
    evaluate_rq1()
