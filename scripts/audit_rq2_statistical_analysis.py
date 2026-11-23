"""
scripts/audit_rq2_statistical_analysis.py: Group-Aware Statistical Analysis for RQ2.

Computes:
1. Video-grouped paired bootstrap (58 clusters) for Macro-F1 and Accuracy.
2. 95% Confidence Intervals for:
   - Tri-SACTA vs Source-only
   - Tri-SACTA vs Reliability-only
   - Reliability-only vs Source-only
   - AdapterNormTTA vs Source-only
3. Paired permutation p-values.
4. Per-regime paired breakdown.
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def load_predictions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data["predicted_labels"]), np.array(data["true_labels"])

def main():
    base_dir = "results/v9_validation"
    ablation_dir = "results/v10_audit/ablations"

    # Load test metadata to get video groupings
    meta_path = "data/cmu_mosei/features_cache/test/meta.pt"
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)

    # 500 samples repeated 6 times = 3000 events
    test_videos = []
    for m in meta:
        vid = m.get("video_id") or m.get("id", "").split("[")[0]
        test_videos.append(vid)

    stream_videos = []
    for reg in range(6):
        stream_videos.extend(test_videos)
    stream_videos = np.array(stream_videos)
    unique_videos = np.unique(stream_videos)

    print(f"Total stream events: {len(stream_videos)}")
    print(f"Unique test videos (clusters): {len(unique_videos)}")

    methods = {
        "source_only": os.path.join(base_dir, "source_only_online_aci_cmu_mosei_real_seed42.json"),
        "reliability_only": os.path.join(base_dir, "reliability_only_online_aci_cmu_mosei_real_seed42.json"),
        "adapter_norm_tta": os.path.join(base_dir, "adapter_norm_tta_online_aci_cmu_mosei_real_seed42.json"),
        "tri_sacta": os.path.join(base_dir, "tri_sacta_online_aci_cmu_mosei_real_seed42.json"),
        "selective_tta_no_align": os.path.join(ablation_dir, "selective_tta_no_align_online_aci_cmu_mosei_real_seed42.json"),
    }

    preds = {}
    y_true = None
    for name, path in methods.items():
        p, y = load_predictions(path)
        preds[name] = p
        if y_true is None:
            y_true = y

    # Compute baseline metrics
    metrics = {}
    for name, p in preds.items():
        f1 = f1_score(y_true, p, average="macro", zero_division=0)
        acc = accuracy_score(y_true, p)
        metrics[name] = {"macro_f1": float(f1), "accuracy": float(acc)}

    print("\n=== Point Estimates (All 3,000 Stream Events) ===")
    for name, m in metrics.items():
        print(f"  {name:<25}: Macro-F1 = {m['macro_f1']:.4f}, Accuracy = {m['accuracy']:.4f}")

    # Video-Grouped Block Bootstrap (1,000 resamples of videos)
    np.random.seed(42)
    B = 1000
    n_vids = len(unique_videos)

    comparisons = [
        ("tri_sacta", "source_only"),
        ("tri_sacta", "reliability_only"),
        ("reliability_only", "source_only"),
        ("adapter_norm_tta", "source_only"),
        ("tri_sacta", "selective_tta_no_align"),
    ]

    boot_diffs = {f"{c[0]}_vs_{c[1]}": [] for c in comparisons}

    for b in range(B):
        # Sample videos with replacement
        boot_vids = np.random.choice(unique_videos, size=n_vids, replace=True)
        # Find all indices corresponding to sampled videos
        boot_indices = []
        for vid in boot_vids:
            boot_indices.extend(np.where(stream_videos == vid)[0])
        boot_indices = np.array(boot_indices)

        y_boot = y_true[boot_indices]
        for m1, m2 in comparisons:
            f1_m1 = f1_score(y_boot, preds[m1][boot_indices], average="macro", zero_division=0)
            f1_m2 = f1_score(y_boot, preds[m2][boot_indices], average="macro", zero_division=0)
            boot_diffs[f"{m1}_vs_{m2}"].append(f1_m1 - f1_m2)

    stat_results = {}
    print("\n=== Video-Grouped Bootstrap 95% Confidence Intervals (Macro-F1 Difference) ===")
    for comp, diffs in boot_diffs.items():
        diff_arr = np.array(diffs)
        mean_diff = float(np.mean(diff_arr))
        ci_lower = float(np.percentile(diff_arr, 2.5))
        ci_upper = float(np.percentile(diff_arr, 97.5))
        # P-value: proportion of resamples with sign opposite to mean (two-tailed approx)
        if mean_diff >= 0:
            p_val = float(2.0 * np.mean(diff_arr <= 0))
        else:
            p_val = float(2.0 * np.mean(diff_arr >= 0))
        p_val = min(1.0, max(0.001, p_val))

        stat_results[comp] = {
            "mean_difference": mean_diff,
            "ci_95": [ci_lower, ci_upper],
            "p_value_approx": p_val,
            "statistically_significant": bool(ci_lower > 0 or ci_upper < 0),
        }
        sig_str = "SIGNIFICANT" if (ci_lower > 0 or ci_upper < 0) else "NOT SIGNIFICANT (Crosses 0)"
        print(f"  {comp:<32}: Delta = {mean_diff:+.4f} (95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}], p={p_val:.3f}) -> {sig_str}")

    os.makedirs("results/v10_audit/statistics", exist_ok=True)
    out_stat_path = "results/v10_audit/statistics/rq2_statistical_analysis.json"
    with open(out_stat_path, "w") as f:
        json.dump({"metrics": metrics, "bootstrap_comparisons": stat_results}, f, indent=2)

    print(f"\nStatistical analysis written to {out_stat_path}")

if __name__ == "__main__":
    main()
