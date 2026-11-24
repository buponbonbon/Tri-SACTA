"""
scripts/v13_2_benchmark_distributed.py: Real Distributed Systems Benchmarking for Tri-SACTA V13.2 (Phase I-D).

Features:
- Spawns independent TCP socket servers for Audio (port 8765) and Visual (port 8766).
- Tests 250 queries from the validation split across all 4 transport policies:
  1. AlwaysOn (direct TAV, no redundant Text pass)
  2. Static_TA (direct TA, no redundant Text pass)
  3. SequentialConfidence_0.55_0.45 (pays intermediate inference passes)
  4. Unimodal_Text (local baseline, 0 wire bytes)
- Warm-up phase before timing measurement.
- Real application protocol wire byte accounting and latency statistics (p50, p95, p99).
- Clean process-safe server lifecycle.

Outputs:
- results/v13_2/systems/distributed_summary.json
- results/v13_2/systems/distributed_benchmark.csv
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import load_split, SUBSET_METADATA
from src.models.fusion_backbone import SimpleMLP
from src.service.distributed_providers import ModalitySocketServer
from src.service.distributed_coordinator import DistributedCoordinator

LOG_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "logs")
SYS_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "systems")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SYS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "distributed_benchmark.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def compute_3class_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in [0, 1, 2]:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def run_distributed_benchmark():
    logger.info("=== Starting Tri-SACTA V13.2 Real Distributed Systems Benchmark ===")
    device = torch.device("cpu")

    audio_port = 8765
    visual_port = 8766
    audio_server = ModalitySocketServer("audio", host="127.0.0.1", port=audio_port, split_name="val")
    visual_server = ModalitySocketServer("visual", host="127.0.0.1", port=visual_port, split_name="val")

    audio_server.start()
    visual_server.start()
    time.sleep(0.5)

    try:
        # Load predictor bank for seed 42
        ckpt_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "training", "checkpoints")
        bank = {}
        for s_key, meta in SUBSET_METADATA.items():
            in_dim = meta["in_dim"]
            h_dim = 64 if "text" in meta["mods"] else (32 if "audio" in meta["mods"] else 16)
            model = SimpleMLP(in_dim=in_dim, hidden_dim=h_dim, num_classes=3).to(device)
            fpath = os.path.join(ckpt_dir, f"mlp_{s_key}_seed42.pt")
            model.load_state_dict(torch.load(fpath, weights_only=True, map_location=device))
            model.eval()
            bank[s_key] = model

        coordinator = DistributedCoordinator(
            bank,
            device,
            audio_host="127.0.0.1",
            audio_port=audio_port,
            visual_host="127.0.0.1",
            visual_port=visual_port,
        )

        val_data = load_split("val", REPO_ROOT)
        text_tensors = val_data["text"]
        y_true = val_data["labels"].numpy()
        N = len(y_true)
        logger.info(f"Loaded validation split: {N} samples.")

        # Warm-up phase (10 queries)
        for i in range(10):
            coordinator.process_query(i, text_tensors[i : i + 1], policy_mode="always_on")

        policies = [
            ("always_on", "AlwaysOn_Direct_TAV"),
            ("static_ta", "Static_TA_Direct"),
            ("sequential_confidence", "SequentialConfidence_0.55_0.45"),
            ("unimodal_text", "Unimodal_Text"),
        ]

        all_results = []
        summary_records = {}

        for pol_mode, pol_label in policies:
            logger.info(f"Benchmarking distributed policy: '{pol_label}' across {N} queries...")
            preds = []
            wall_times = []
            app_bytes_list = []
            pay_bytes_list = []
            rtt_times = []
            inf_times = []

            for i in range(N):
                t_feat = text_tensors[i : i + 1]
                res = coordinator.process_query(
                    sample_idx=i,
                    text_feature=t_feat,
                    policy_mode=pol_mode,
                    tau_stage1=0.55,
                    tau_stage2=0.45,
                    deadline_ms=50.0,
                    cost_budget=3.5,
                )

                preds.append(res["prediction"])
                wall_times.append(res["wall_ms"])
                app_bytes_list.append(res["application_protocol_bytes"])
                pay_bytes_list.append(res["payload_bytes"])
                rtt_times.append(res["network_rtt_ms"])
                inf_times.append(res["inference_ms"])

                all_results.append({
                    "sample_idx": i,
                    "policy": pol_label,
                    "policy_mode": pol_mode,
                    "true_label": int(y_true[i]),
                    "prediction": res["prediction"],
                    "confidence": res["confidence"],
                    "final_subset": res["final_subset"],
                    "application_protocol_bytes": res["application_protocol_bytes"],
                    "payload_bytes": res["payload_bytes"],
                    "network_rtt_ms": res["network_rtt_ms"],
                    "inference_ms": res["inference_ms"],
                    "wall_ms": res["wall_ms"],
                    "num_network_hops": res["num_network_hops"],
                    "on_time": res["on_time"],
                })

            y_pred_arr = np.array(preds)
            macro_f1 = compute_3class_macro_f1(y_true, y_pred_arr)
            wall_arr = np.array(wall_times)

            summary_records[pol_label] = {
                "policy_mode": pol_mode,
                "macro_f1": float(macro_f1),
                "total_application_protocol_bytes": int(np.sum(app_bytes_list)),
                "mean_application_protocol_bytes": float(np.mean(app_bytes_list)),
                "total_payload_bytes": int(np.sum(pay_bytes_list)),
                "mean_wall_ms": float(np.mean(wall_arr)),
                "p50_wall_ms": float(np.percentile(wall_arr, 50)),
                "p95_wall_ms": float(np.percentile(wall_arr, 95)),
                "p99_wall_ms": float(np.percentile(wall_arr, 99)),
                "mean_network_rtt_ms": float(np.mean(rtt_times)),
                "mean_inference_ms": float(np.mean(inf_times)),
                "on_time_rate": float(np.mean([t <= 50.0 for t in wall_times])),
                "num_samples": N,
            }

        # Calculate relative comparisons vs Always-On
        always_bytes = summary_records["AlwaysOn_Direct_TAV"]["total_application_protocol_bytes"]
        always_wall = summary_records["AlwaysOn_Direct_TAV"]["mean_wall_ms"]
        seq_bytes = summary_records["SequentialConfidence_0.55_0.45"]["total_application_protocol_bytes"]
        seq_wall = summary_records["SequentialConfidence_0.55_0.45"]["mean_wall_ms"]

        bytes_reduction_pct = ((always_bytes - seq_bytes) / always_bytes * 100.0) if always_bytes > 0 else 0.0
        latency_reduction_pct = ((always_wall - seq_wall) / always_wall * 100.0) if always_wall > 0 else 0.0

        distributed_summary = {
            "benchmark_version": "13.2",
            "transport": "Real TCP Stream Sockets (ports 8765, 8766)",
            "network_topology": "Localhost loopback with physical serialization and OS network stack",
            "sample_count": N,
            "seed": 42,
            "baseline_fairness": {
                "always_on_direct_tav": True,
                "static_ta_direct": True,
                "sequential_pays_intermediate_overhead": True,
            },
            "bandwidth_reduction": {
                "always_on_application_protocol_bytes": always_bytes,
                "sequential_application_protocol_bytes": seq_bytes,
                "bytes_saved": always_bytes - seq_bytes,
                "bandwidth_savings_percentage": bytes_reduction_pct,
                "latency_reduction_percentage": latency_reduction_pct,
            },
            "policies": summary_records,
        }

        # Save artifacts
        out_summary_json = os.path.join(SYS_DIR, "distributed_summary.json")
        with open(out_summary_json, "w", encoding="utf-8") as f:
            json.dump(distributed_summary, f, indent=2)

        df_bench = pd.DataFrame(all_results)
        out_csv = os.path.join(SYS_DIR, "distributed_benchmark.csv")
        df_bench.to_csv(out_csv, index=False)

        logger.info(f"=== Distributed Benchmark Complete ===")
        logger.info(f"Always-On App Bytes: {always_bytes:,} | Sequential App Bytes: {seq_bytes:,}")
        logger.info(f"Measured App Wire Bytes Reduction: {bytes_reduction_pct:.2f}%")
        logger.info(f"Measured Mean Latency Reduction: {latency_reduction_pct:.2f}% (AlwaysOn {always_wall:.2f}ms vs Sequential {seq_wall:.2f}ms)")
        logger.info(f"Artifacts saved to {SYS_DIR}")
        return distributed_summary

    finally:
        audio_server.stop()
        visual_server.stop()


if __name__ == "__main__":
    run_distributed_benchmark()
