"""
scripts/phase3_network/generate_figures.py: Generate publication-grade vector (.svg) and
high-resolution (.png) figures from verified Phase II & Phase III artifacts.
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "publication_audit", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    # Figure B: Phase II Quality-Resource Trade-off
    with open(os.path.join(REPO_ROOT, "results", "v13_2", "publication_audit", "recomputed_metrics.json"), "r", encoding="utf-8") as f:
        p2 = json.load(f)["tcp_policies"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for pol, m in p2.items():
        ax.scatter(m["mean_app_bytes"], m["sq_macro_f1"], s=80, label=pol)
        ax.annotate(pol, (m["mean_app_bytes"], m["sq_macro_f1"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean Application-Protocol Bytes / Request (Confirmatory, N=4,131)")
    ax.set_ylabel("Service-Quality Macro-F1")
    ax.set_title("Phase II Confirmatory Quality-Resource Trade-off (TCP Loopback)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figB_phase2_quality_bytes.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "figB_phase2_quality_bytes.svg"))
    plt.close(fig)

    # Figure C: Network RTT vs Latency (Regime B)
    df_agg = pd.read_csv(os.path.join(REPO_ROOT, "results", "v13_2", "phase3_network", "network_aggregated_metrics.csv"))
    reg_b = df_agg[df_agg["condition_id"].isin(["RTT0_UNRES_B", "RTT10_UNRES_B", "RTT30_UNRES_B", "RTT80_UNRES_B"])]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for pol, g in reg_b.groupby("policy"):
        g = g.sort_values("target_rtt_ms")
        ax.plot(g["target_rtt_ms"], g["mean_wall_ms"], marker="o", label=pol)
    ax.set_xlabel("Target Injected RTT (ms) — Controlled TCP Loopback Emulation (Regime B)")
    ax.set_ylabel("Mean Wall-Clock Latency (ms)")
    ax.set_title("Regime B (250 ms Deadline): RTT Scaling Across Policies")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figC_rtt_vs_latency_regimeB.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "figC_rtt_vs_latency_regimeB.svg"))
    plt.close(fig)

    # Figure E: Analytical vs Observed Latency
    with open(os.path.join(REPO_ROOT, "results", "v13_2", "phase3_network", "analytical_model_validation.json"), "r", encoding="utf-8") as f:
        val = json.load(f)
    preds = [p["predicted_ms"] for p in val["predictions"]]
    obs = [p["observed_ms"] for p in val["predictions"]]

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.scatter(obs, preds, s=60, alpha=0.8)
    mx = max(max(obs), max(preds)) * 1.05
    ax.plot([0, mx], [0, mx], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Observed Mean Latency (ms)")
    ax.set_ylabel("Analytical Predicted Latency (ms)")
    ax.set_title(f"Analytical Latency Model Validation (MAE = {val['mae_ms']:.2f} ms)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "figE_analytical_vs_observed.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "figE_analytical_vs_observed.svg"))
    plt.close(fig)

    print("FIGURES_GENERATED_OK")


if __name__ == "__main__":
    main()
