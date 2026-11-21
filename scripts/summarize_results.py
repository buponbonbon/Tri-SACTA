import json
import glob
import os

methods = [
    ("Source-Only + Static CP", "results/source_only_static_cp_mosei_synthetic_seed42.json"),
    ("TENT + Online ACI", "results/tent_online_aci_mosei_synthetic_seed42.json"),
    ("SAR + Online ACI", "results/sar_online_aci_mosei_synthetic_seed42.json"),
    ("READ + Online ACI", "results/read_online_aci_mosei_synthetic_seed42.json"),
    ("Tri-SACTA + Online ACI (Ours)", "results/tri_sacta_online_aci_mosei_synthetic_seed42.json"),
]

print("\n### Per-Regime Macro-F1 Breakdown")
print("| Method | R1 (Clean) | R2 (Audio Shift) | R3 (ASR Drift) | R4 (Visual+Miss) | R5 (Severe Compound) | Overall F1 |")
print("|---|---|---|---|---|---|---|")
for name, p in methods:
    if not os.path.exists(p):
        continue
    with open(p) as f:
        d = json.load(f)
    rm = d["regime_metrics"]
    f1s = [f"{rm[f'regime_{i}']['metrics']['macro_f1']:.4f}" for i in range(1, 6)]
    overall = f"{d['overall_metrics']['macro_f1']:.4f}"
    print(f"| {name} | " + " | ".join(f1s) + f" | **{overall}** |")

print("\n### Per-Regime Conformal Coverage Breakdown (Target = 90.0%)")
print("| Method | R1 (Clean) | R2 (Audio Shift) | R3 (ASR Drift) | R4 (Visual+Miss) | R5 (Severe Compound) | Overall Cov | Cov Gap | Avg Set Size |")
print("|---|---|---|---|---|---|---|---|---|")
for name, p in methods:
    if not os.path.exists(p):
        continue
    with open(p) as f:
        d = json.load(f)
    rm = d["regime_metrics"]
    covs = [f"{rm[f'regime_{i}']['metrics']['empirical_coverage']*100:.2f}%" for i in range(1, 6)]
    om = d["overall_metrics"]
    print(f"| {name} | " + " | ".join(covs) + f" | **{om['empirical_coverage']*100:.2f}%** | {om['coverage_gap']*100:.2f}% | {om['avg_set_size']:.2f} |")
