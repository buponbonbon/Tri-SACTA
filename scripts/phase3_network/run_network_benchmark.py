"""
scripts/phase3_network/run_network_benchmark.py: Phase III Controlled Network-Aware Systems
Evaluation (Regime A & Regime B), Analytical Latency Model Validation, and Operating-Region Map.
Uses ONLY development ('val') data; NEVER touches Phase II confirmatory holdout.
"""

from __future__ import annotations

import os
import sys
import gzip
import json
import time
import random
import platform
from datetime import datetime, timezone
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import load_split, SUBSET_METADATA
from src.models.fusion_backbone import SimpleMLP
from src.service.distributed_providers import ModalitySocketServer
from src.service.distributed_coordinator import DistributedCoordinator
from scripts.v13_2_evaluate_development import compute_3class_macro_f1
from scripts.v13_2_run_phase2_holdout import build_workloads_from_tensors, _get_free_port
from scripts.phase3_network.network_emulator import NetworkCondition, emulate_rtt_and_bandwidth, calibrate_transport

OUT_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "phase3_network")
os.makedirs(OUT_DIR, exist_ok=True)

POLICIES = [
    ("sequential_confidence", "SequentialConfidence_0.55_0.45"),
    ("confidence_routing_055", "OneShot_ConfidenceRouting_0.55"),
    ("always_on_parallel", "AlwaysOn_Direct_Parallel"),
    ("always_on_serial", "AlwaysOn_Direct_Serial"),
    ("static_ta", "Static_TA_Direct"),
    ("unimodal_text", "Unimodal_Text"),
]

WORKLOADS = [
    "CLEAN_MULTIMODAL",
    "TEXT_EMBEDDING_NOISE",
    "INTERMITTENT_MISSINGNESS",
    "HETEROGENEOUS_DEADLINE_STRESS",
]

CONDITIONS = [
    NetworkCondition("RTT0_UNRES_A", 0.0, None, 0.0, "A"),
    NetworkCondition("RTT10_UNRES_A", 10.0, None, 0.0, "A"),
    NetworkCondition("RTT30_UNRES_A", 30.0, None, 0.0, "A"),
    NetworkCondition("RTT80_UNRES_A", 80.0, None, 0.0, "A"),
    NetworkCondition("RTT0_UNRES_B", 0.0, None, 0.0, "B"),
    NetworkCondition("RTT10_UNRES_B", 10.0, None, 0.0, "B"),
    NetworkCondition("RTT30_UNRES_B", 30.0, None, 0.0, "B"),
    NetworkCondition("RTT80_UNRES_B", 80.0, None, 0.0, "B"),
    NetworkCondition("RTT30_BW10_B", 30.0, 10.0, 0.0, "B"),
    NetworkCondition("RTT30_BW1_B", 30.0, 1.0, 0.0, "B"),
    NetworkCondition("RTT30_JIT5_B", 30.0, None, 5.0, "B"),
]


def load_seed42_bank() -> Dict[str, torch.nn.Module]:
    ckpt_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "training", "checkpoints")
    bank = {}
    for s_key, meta in SUBSET_METADATA.items():
        h_dim = 64 if "text" in meta["mods"] else (32 if "audio" in meta["mods"] else 16)
        m = SimpleMLP(meta["in_dim"], h_dim, 3)
        m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"mlp_{s_key}_seed42.pt"), weights_only=True, map_location="cpu"))
        m.eval()
        bank[s_key] = m
    return bank


def main():
    t_start = datetime.now(timezone.utc).isoformat()
    val_data = load_split("val", REPO_ROOT)
    # Deterministic whole-video-group-aware subset (first 5 video clusters in sorted order)
    all_vids = val_data["video_ids"]
    selected_vids = sorted(set(all_vids))[:5]
    sel_idx = [i for i, v in enumerate(all_vids) if v in selected_vids]
    N_sub = len(sel_idx)

    with open(os.path.join(OUT_DIR, "data_selection_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "split": "val",
            "selection_rule": "first_5_sorted_video_clusters",
            "num_video_clusters": len(selected_vids),
            "video_ids": selected_vids,
            "num_utterances": N_sub,
            "sample_indices": sel_idx,
        }, f, indent=2)

    calib = calibrate_transport(CONDITIONS)
    with open(os.path.join(OUT_DIR, "transport_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    t_ten = val_data["text"][sel_idx]
    a_ten = val_data["audio"][sel_idx]
    v_ten = val_data["visual"][sel_idx]
    y_true = val_data["labels"].numpy()[sel_idx]
    vids = [all_vids[i] for i in sel_idx]

    workloads = build_workloads_from_tensors(t_ten, a_ten, v_ten)
    bank = load_seed42_bank()
    device = torch.device("cpu")

    a_port = _get_free_port()
    v_port = _get_free_port()
    a_srv = ModalitySocketServer("audio", host="127.0.0.1", port=a_port, features_tensor=a_ten.float())
    v_srv = ModalitySocketServer("visual", host="127.0.0.1", port=v_port, features_tensor=v_ten.float())
    a_srv.start()
    v_srv.start()
    time.sleep(0.15)

    rows: List[Dict[str, Any]] = []
    rng = random.Random(1320)
    try:
        for cond in CONDITIONS:
            for w_name in WORKLOADS:
                wl = workloads[w_name]
                a_srv.update_features(wl["audio"])
                v_srv.update_features(wl["visual"])
                for pol_mode, pol_label in POLICIES:
                    coord = DistributedCoordinator(bank, device, audio_port=a_port, visual_port=v_port, persistent_connections=True)
                    try:
                        for i in range(N_sub):
                            s_mask = {m: bool(wl["modality_mask"][m][i]) for m in ("text", "audio", "visual")}
                            d_ms = float(wl["deadline_ms"][i]) if cond.regime == "A" else 250.0
                            b_cost = float(wl["cost_budget"][i])
                            t0 = time.perf_counter()
                            res = coord.process_query(
                                sample_idx=i,
                                text_feature=wl["text"][i : i + 1],
                                policy_mode=pol_mode,
                                tau_stage1=0.55,
                                tau_stage2=0.45,
                                deadline_ms=d_ms,
                                cost_budget=b_cost,
                                modality_mask=s_mask,
                            )
                            # Emulate critical-path network rounds on top of loopback measurement
                            rounds = 0 if res["final_subset"] == "T" else (1 if pol_mode in ("always_on_parallel", "confidence_routing_055", "static_ta") else (2 if res["final_subset"] == "TAV" else 1))
                            net_timed_out = False
                            for _ in range(rounds):
                                rem_s = max(0.0, (d_ms / 1000.0) - (time.perf_counter() - t0))
                                em = emulate_rtt_and_bandwidth(int(res["application_protocol_bytes"] / max(1, rounds)), cond, rng=rng, deadline_remaining_s=rem_s)
                                if em["timed_out"]:
                                    net_timed_out = True
                                    break
                            total_ms = (time.perf_counter() - t0) * 1000.0
                            on_time = int(total_ms <= d_ms and not net_timed_out and res["on_time"])
                            sq_pred = int(res["prediction"]) if (int(res["prediction"]) in (0, 1, 2) and on_time == 1) else -1
                            rows.append({
                                "sample_idx": sel_idx[i],
                                "video_id": vids[i],
                                "workload": w_name,
                                "condition_id": cond.condition_id,
                                "regime": cond.regime,
                                "target_rtt_ms": cond.target_rtt_ms,
                                "bandwidth_mbps": cond.bandwidth_mbps if cond.bandwidth_mbps is not None else -1.0,
                                "jitter_ms": cond.jitter_ms,
                                "policy": pol_label,
                                "true_label": int(y_true[i]),
                                "prediction": int(res["prediction"]),
                                "service_quality_prediction": sq_pred,
                                "final_subset": res["final_subset"],
                                "rounds": rounds,
                                "application_protocol_bytes": int(res["application_protocol_bytes"]),
                                "wall_ms": total_ms,
                                "deadline_ms": d_ms,
                                "on_time": on_time,
                                "useful_on_time": int(sq_pred in (0, 1, 2)),
                                "budget_violated": int(res["nominal_cost"] > b_cost),
                            })
                    finally:
                        coord.close()
    finally:
        a_srv.stop()
        v_srv.stop()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "network_benchmark_traces.csv.gz"), index=False, compression="gzip")

    # Aggregate & Operating Regions
    agg = []
    for (cid, reg, rtt, pol), g in df.groupby(["condition_id", "regime", "target_rtt_ms", "policy"]):
        sq_f1s = [compute_3class_macro_f1(wg["true_label"].to_numpy(), wg["service_quality_prediction"].to_numpy()) for _, wg in g.groupby("workload")]
        agg.append({
            "condition_id": cid,
            "regime": reg,
            "target_rtt_ms": float(rtt),
            "policy": pol,
            "sq_macro_f1": float(np.mean(sq_f1s)),
            "mean_app_bytes": float(g["application_protocol_bytes"].mean()),
            "mean_wall_ms": float(g["wall_ms"].mean()),
            "p95_wall_ms": float(np.percentile(g["wall_ms"].to_numpy(), 95)),
            "useful_on_time_rate": float(g["useful_on_time"].mean()),
            "budget_violation_rate": float(g["budget_violated"].mean()),
            "meets_guardrails": bool(g["useful_on_time"].mean() >= 0.95 and g["budget_violated"].mean() == 0.0),
        })
    df_agg = pd.DataFrame(agg)
    df_agg.to_csv(os.path.join(OUT_DIR, "network_aggregated_metrics.csv"), index=False)

    # Analytical Model Validation (fit H_p and R_p on RTT0_UNRES_B, predict RTT10/30/80_UNRES_B)
    base_b = df_agg[df_agg["condition_id"] == "RTT0_UNRES_B"].set_index("policy")
    r_means = df[df["condition_id"] == "RTT0_UNRES_B"].groupby("policy")["rounds"].mean()
    val_errs = []
    for cid in ("RTT10_UNRES_B", "RTT30_UNRES_B", "RTT80_UNRES_B"):
        sub = df_agg[df_agg["condition_id"] == cid]
        for _, row in sub.iterrows():
            pol = row["policy"]
            pred_ms = float(base_b.loc[pol, "mean_wall_ms"]) + float(r_means.loc[pol]) * float(row["target_rtt_ms"])
            err = abs(pred_ms - float(row["mean_wall_ms"]))
            val_errs.append({"condition_id": cid, "policy": pol, "predicted_ms": pred_ms, "observed_ms": float(row["mean_wall_ms"]), "abs_err_ms": err})

    mae_ms = float(np.mean([e["abs_err_ms"] for e in val_errs]))
    medae_ms = float(np.median([e["abs_err_ms"] for e in val_errs]))
    with open(os.path.join(OUT_DIR, "analytical_model_validation.json"), "w", encoding="utf-8") as f:
        json.dump({"mae_ms": mae_ms, "medae_ms": medae_ms, "predictions": val_errs}, f, indent=2)

    with open(os.path.join(OUT_DIR, "operating_regions.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_start": t_start,
            "timestamp_end": datetime.now(timezone.utc).isoformat(),
            "environment": {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__},
            "rows": len(df),
            "summary": agg,
        }, f, indent=2)

    print(f"PHASE3_NET_OK | rows={len(df)} | analytical_MAE_ms={mae_ms:.2f}")


if __name__ == "__main__":
    main()
