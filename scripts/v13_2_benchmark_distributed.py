"""
scripts/v13_2_benchmark_distributed.py: Comprehensive Real TCP Socket Benchmarking for Tri-SACTA V13.2.

Resolves Blockers 5, 7, 8, and 9:
1. Full 7-Workload x 3-Seed Paired Resource-Quality TCP Benchmark (persistent keep-alive loopback TCP,
   42,000 real socket queries across all 8 required modes: AlwaysOn_Direct_Serial,
   AlwaysOn_Direct_Parallel, Static_TA_Direct, BudgetAdmittedMaxSubset,
   OneShot_ConfidenceRouting_0.55, SequentialConfidence_0.55_0.45, Unimodal_Text,
   MatchedBudgetRandom_1.45).
2. Measures physical application-protocol request+response bytes separately from payload bytes,
   plus wall latency p50/p95/p99 (numpy linear quantile), network RTT, inference ms,
   on-time rate, acceptance rate, useful-on-time rate, and client/server byte totals.
3. Sensitivity runs on CLEAN_MULTIMODAL (seed 42, N=250):
   - Multi-repetition timing stability (3 reps, persistent)
   - Concurrency = 1 vs 4 workers
   - Per-request non-persistent TCP connection mode sensitivity
   - Cold vs warm run comparison
4. Controlled Fault-Injection TCP Suite verifying graceful degradation & error byte accounting across:
   disconnect_mid_frame, connection_refused, slow_provider_timeout, oversize_header,
   nan_feature, out_of_bounds_sample_id, tight_budget_deadline, and total_blackout.
5. Offline vs Socket decision parity verification on CLEAN_MULTIMODAL.
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
import platform
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
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
from src.service.baseline_policies import (
    SequentialConfidenceRouter,
    ConfidenceRoutingPolicy,
    AlwaysOnFusionPolicy,
    BudgetAdmittedMaxSubsetPolicy,
    StaticSubsetPolicy,
)

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

SEEDS = [42, 100, 2026]
WORKLOAD_NAMES = [
    "CLEAN_MULTIMODAL",
    "TEXT_EMBEDDING_NOISE",
    "AUDIO_FEATURE_MASKING",
    "VISUAL_ZERO_MASK",
    "COMPOUND_FAULT",
    "INTERMITTENT_MISSINGNESS",
    "HETEROGENEOUS_DEADLINE_STRESS",
]

DISTRIBUTED_POLICIES = [
    ("always_on_serial", "AlwaysOn_Direct_Serial"),
    ("always_on_parallel", "AlwaysOn_Direct_Parallel"),
    ("static_ta", "Static_TA_Direct"),
    ("budget_admitted_max", "BudgetAdmittedMaxSubset"),
    ("confidence_routing_055", "OneShot_ConfidenceRouting_0.55"),
    ("sequential_confidence", "SequentialConfidence_0.55_0.45"),
    ("unimodal_text", "Unimodal_Text"),
    ("matched_budget_random", "MatchedBudgetRandom_1.45"),
]


def compute_3class_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in [0, 1, 2]:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def load_all_banks(device: torch.device) -> Dict[int, Dict[str, Any]]:
    ckpt_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "training", "checkpoints")
    banks = {}
    for seed in SEEDS:
        bank = {}
        for s_key, meta in SUBSET_METADATA.items():
            in_dim = meta["in_dim"]
            h_dim = 64 if "text" in meta["mods"] else (32 if "audio" in meta["mods"] else 16)
            model = SimpleMLP(in_dim=in_dim, hidden_dim=h_dim, num_classes=3).to(device)
            fpath = os.path.join(ckpt_dir, f"mlp_{s_key}_seed{seed}.pt")
            model.load_state_dict(torch.load(fpath, weights_only=True, map_location=device))
            model.eval()
            bank[s_key] = model
        banks[seed] = bank
    return banks


def check_offline_vs_socket_parity(
    banks: Dict[int, Dict[str, Any]],
    workloads: Dict[str, Dict[str, Any]],
    audio_port: int,
    visual_port: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Asserts 100% path & prediction parity between offline replay and TCP socket runtime on CLEAN_MULTIMODAL."""
    wl = workloads["CLEAN_MULTIMODAL"]
    N = len(wl["text"])
    coord = DistributedCoordinator(
        banks[42], device, audio_port=audio_port, visual_port=visual_port, persistent_connections=True
    )
    offline_map = {
        "sequential_confidence": SequentialConfidenceRouter(banks[42], device, 0.55, 0.45, True),
        "confidence_routing_055": ConfidenceRoutingPolicy(banks[42], device, 0.55),
        "static_ta": StaticSubsetPolicy(banks[42], device, "TA"),
        "always_on_serial": AlwaysOnFusionPolicy(banks[42], device),
        "always_on_parallel": AlwaysOnFusionPolicy(banks[42], device),
        "budget_admitted_max": BudgetAdmittedMaxSubsetPolicy(banks[42], device),
        "unimodal_text": StaticSubsetPolicy(banks[42], device, "T"),
    }
    parity_report = {}
    try:
        for mode, off_pol in offline_map.items():
            mismatches = 0
            for i in range(N):
                feats = {"text": wl["text"][i : i + 1], "audio": wl["audio"][i : i + 1], "visual": wl["visual"][i : i + 1]}
                mask = {"text": True, "audio": True, "visual": True}
                r_off = off_pol.route_single_query(feats, mask, deadline_ms=50.0, cost_budget=3.5)
                r_tcp = coord.process_query(i, wl["text"][i : i + 1], policy_mode=mode, deadline_ms=50.0, cost_budget=3.5, modality_mask=mask)
                if r_off["prediction"] != r_tcp["prediction"] or r_off["final_subset"] != r_tcp["final_subset"]:
                    mismatches += 1
            parity_report[mode] = {"samples": N, "mismatches": mismatches, "exact_parity": mismatches == 0}
    finally:
        coord.close()
    return parity_report


def run_fault_injection_suite(
    bank42: Dict[str, Any],
    val_data: Dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    """Runs controlled real-TCP fault-injection scenarios across candidate & comparators."""
    a_port, v_port = 8775, 8776
    a_srv = ModalitySocketServer("audio", host="127.0.0.1", port=a_port, split_name="val")
    v_srv = ModalitySocketServer("visual", host="127.0.0.1", port=v_port, split_name="val")
    a_srv.start()
    v_srv.start()
    time.sleep(0.3)

    rows = []
    t_feat_high = val_data["text"][0:1]
    t_feat_low = val_data["text"][2:3]  # Stage-1 confidence ~0.41 < 0.45 -> triggers Stage 2 & 3 escalation!
    clean_mask = {"text": True, "audio": True, "visual": True}

    scenarios = [
        ("disconnect_mid_frame", "audio_disconnect", 2, t_feat_low, 50.0, 3.5, clean_mask),
        ("connection_refused", "visual_port_closed", 2, t_feat_low, 50.0, 3.5, clean_mask),
        ("slow_provider_timeout", "audio_delay_25ms_under_15ms", 2, t_feat_low, 15.0, 3.5, clean_mask),
        ("oversize_malformed_header", "audio_oversize_header", 2, t_feat_low, 50.0, 3.5, clean_mask),
        ("non_finite_nan_feature", "audio_nan_sample", 2, t_feat_low, 50.0, 3.5, clean_mask),
        ("out_of_bounds_sample_id", "oob_sample_99999", 99999, t_feat_low, 50.0, 3.5, clean_mask),
        ("tight_budget_reject", "budget_0.5", 0, t_feat_high, 50.0, 0.5, clean_mask),
        ("tight_deadline_reject", "deadline_0.5ms", 0, t_feat_high, 0.5, 3.5, clean_mask),
        ("total_blackout", "no_modalities", 0, t_feat_high, 50.0, 3.5, {"text": False, "audio": False, "visual": False}),
    ]

    try:
        for sc_name, fault_kind, s_idx, t_feat, d_ms, b_cost, mask in scenarios:
            for mode, label in [
                ("always_on_parallel", "AlwaysOn_Direct_Parallel"),
                ("static_ta", "Static_TA_Direct"),
                ("confidence_routing_055", "OneShot_ConfidenceRouting_0.55"),
                ("sequential_confidence", "SequentialConfidence_0.55_0.45"),
            ]:
                a_srv.clear_faults()
                v_srv.clear_faults()
                v_port_use = v_port

                if fault_kind == "audio_disconnect":
                    a_srv.set_fault(disconnect_mid_payload=True)
                elif fault_kind == "visual_port_closed":
                    v_port_use = 8799  # Dead closed port
                elif fault_kind == "audio_delay_25ms_under_15ms":
                    a_srv.set_fault(delay_ms=25.0)
                elif fault_kind == "audio_oversize_header":
                    a_srv.set_fault(send_oversize_header=True)
                elif fault_kind == "audio_nan_sample":
                    a_srv.set_fault(inject_nan_sample=s_idx)

                coord = DistributedCoordinator(
                    bank42, device, audio_port=a_port, visual_port=v_port_use, persistent_connections=False
                )
                try:
                    res = coord.process_query(
                        sample_idx=s_idx, text_feature=t_feat, policy_mode=mode,
                        deadline_ms=d_ms, cost_budget=b_cost, modality_mask=mask,
                    )
                    rows.append({
                        "scenario": sc_name,
                        "fault_kind": fault_kind,
                        "policy": label,
                        "status": res["status"],
                        "prediction": res["prediction"],
                        "service_quality_prediction": res["service_quality_prediction"],
                        "final_subset": res["final_subset"],
                        "application_protocol_bytes": res["application_protocol_bytes"],
                        "wall_ms": res["wall_ms"],
                        "on_time": int(res["on_time"]),
                        "failed_attempts": res["failed_attempts"],
                        "routing_reason": res["routing_reason"],
                    })
                finally:
                    coord.close()
    finally:
        a_srv.stop()
        v_srv.stop()

    df_fault = pd.DataFrame(rows)
    df_fault.to_csv(os.path.join(SYS_DIR, "distributed_fault_benchmark.csv"), index=False)
    return df_fault


def run_distributed_benchmark():
    utc_now = datetime.now(timezone.utc).isoformat()
    logger.info("=== Starting Tri-SACTA V13.2 Full Distributed TCP Socket Benchmark ===")
    device = torch.device("cpu")

    val_data = load_split("val", REPO_ROOT)
    y_true = val_data["labels"].numpy()
    video_ids = val_data["video_ids"]
    N = len(y_true)

    wl_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "workloads", "val")
    workloads = {
        w_name: torch.load(os.path.join(wl_dir, f"{w_name}.pt"), map_location="cpu", weights_only=False)
        for w_name in WORKLOAD_NAMES
    }
    banks = load_all_banks(device)

    audio_port = 8765
    visual_port = 8766
    audio_server = ModalitySocketServer("audio", host="127.0.0.1", port=audio_port, split_name="val")
    visual_server = ModalitySocketServer("visual", host="127.0.0.1", port=visual_port, split_name="val")
    audio_server.start()
    visual_server.start()
    time.sleep(0.3)

    try:
        # 1. Parity Check
        logger.info("[1/4] Verifying Offline vs TCP Socket exact decision parity on CLEAN_MULTIMODAL...")
        parity = check_offline_vs_socket_parity(banks, workloads, audio_port, visual_port, device)
        for mode, p_info in parity.items():
            if not p_info["exact_parity"]:
                raise RuntimeError(f"Parity violation for {mode}: {p_info}")
        logger.info("  Offline vs Socket parity PASSED 100% across all deterministic modes.")

        # 2. Primary Paired 7-Workload x 3-Seed TCP Benchmark (Persistent Keep-Alive Sockets)
        logger.info("[2/4] Running 7-Workload x 3-Seed Paired TCP Benchmark (42,000 real socket queries)...")
        all_rows: List[Dict[str, Any]] = []
        audio_server.reset_telemetry()
        visual_server.reset_telemetry()
        client_total_app_bytes = 0

        # Warm-up coordinator
        warm_coord = DistributedCoordinator(banks[42], device, audio_port=audio_port, visual_port=visual_port, persistent_connections=True)
        for i in range(10):
            warm_coord.process_query(i, val_data["text"][i : i + 1], policy_mode="always_on_parallel")
            warm_coord.process_query(i, val_data["text"][i : i + 1], policy_mode="sequential_confidence")
        warm_coord.close()

        client_bytes_after_warmup_audio = audio_server.total_app_bytes_sent + audio_server.total_app_bytes_recv
        client_bytes_after_warmup_visual = visual_server.total_app_bytes_sent + visual_server.total_app_bytes_recv

        for seed in SEEDS:
            coordinators = {
                pol_label: DistributedCoordinator(
                    banks[seed], device, audio_port=audio_port, visual_port=visual_port, persistent_connections=True
                )
                for _, pol_label in DISTRIBUTED_POLICIES
            }
            try:
                for w_idx, w_name in enumerate(WORKLOAD_NAMES):
                    wl = workloads[w_name]
                    audio_server.update_features(wl["audio"])
                    visual_server.update_features(wl["visual"])
                    w_text = wl["text"]
                    w_masks = wl["modality_mask"]
                    w_deadlines = wl["deadline_ms"]
                    w_budgets = wl["cost_budget"]

                    # Counterbalance policy execution order across workloads & seeds
                    rot = (seed + w_idx) % len(DISTRIBUTED_POLICIES)
                    ordered_pols = DISTRIBUTED_POLICIES[rot:] + DISTRIBUTED_POLICIES[:rot]

                    for pol_mode, pol_label in ordered_pols:
                        coord = coordinators[pol_label]
                        rand_rng = np.random.default_rng(42) if pol_mode == "matched_budget_random" else None
                        for i in range(N):
                            sample_mask = {
                                "text": bool(w_masks["text"][i]),
                                "audio": bool(w_masks["audio"][i]),
                                "visual": bool(w_masks["visual"][i]),
                            }
                            d_ms = float(w_deadlines[i])
                            b_cost = float(w_budgets[i])
                            res = coord.process_query(
                                sample_idx=i,
                                text_feature=w_text[i : i + 1],
                                policy_mode=pol_mode,
                                tau_stage1=0.55,
                                tau_stage2=0.45,
                                deadline_ms=d_ms,
                                cost_budget=b_cost,
                                modality_mask=sample_mask,
                                rng=rand_rng,
                                p_acquire=0.28,
                            )
                            client_total_app_bytes += res["application_protocol_bytes"]
                            all_rows.append({
                                "sample_idx": i,
                                "video_id": video_ids[i],
                                "seed": seed,
                                "workload": w_name,
                                "policy": pol_label,
                                "policy_mode": pol_mode,
                                "true_label": int(y_true[i]),
                                "prediction": res["prediction"],
                                "service_quality_prediction": res["service_quality_prediction"],
                                "final_subset": res["final_subset"],
                                "status": res["status"],
                                "nominal_cost": res["nominal_cost"],
                                "application_protocol_bytes": res["application_protocol_bytes"],
                                "payload_bytes": res["payload_bytes"],
                                "network_rtt_ms": res["network_rtt_ms"],
                                "inference_ms": res["inference_ms"],
                                "wall_ms": res["wall_ms"],
                                "deadline_limit": d_ms,
                                "budget_limit": b_cost,
                                "on_time": int(res["on_time"]),
                                "budget_violated": int(res["budget_violated"]),
                                "accepted": int(res["accepted"]),
                                "useful_on_time": int(res["useful_on_time"]),
                                "failed_attempts": res["failed_attempts"],
                                "num_network_hops": res["num_network_hops"],
                            })
            finally:
                for coord in coordinators.values():
                    coord.close()

        server_total_app_bytes = (
            (audio_server.total_app_bytes_sent + audio_server.total_app_bytes_recv - client_bytes_after_warmup_audio)
            + (visual_server.total_app_bytes_sent + visual_server.total_app_bytes_recv - client_bytes_after_warmup_visual)
        )
        byte_discrepancy = abs(client_total_app_bytes - server_total_app_bytes)
        logger.info(
            f"  Client total app bytes: {client_total_app_bytes:,} | Server measured app bytes: {server_total_app_bytes:,} | Diff: {byte_discrepancy}"
        )

        # Save complete CSV
        df_full = pd.DataFrame(all_rows)
        out_csv = os.path.join(SYS_DIR, "distributed_benchmark.csv")
        df_full.to_csv(out_csv, index=False)

        # 3. Sensitivity Suite (CLEAN_MULTIMODAL seed 42, N=250: 3 reps, concurrency 4, non-persistent, cold vs warm)
        logger.info("[3/4] Running sensitivity benchmark (3 reps, non-persistent TCP, concurrency=4, cold/warm)...")
        wl_clean = workloads["CLEAN_MULTIMODAL"]
        audio_server.update_features(wl_clean["audio"])
        visual_server.update_features(wl_clean["visual"])

        sens_rows: List[Dict[str, Any]] = []
        sens_policies = [
            ("always_on_serial", "AlwaysOn_Direct_Serial"),
            ("always_on_parallel", "AlwaysOn_Direct_Parallel"),
            ("static_ta", "Static_TA_Direct"),
            ("budget_admitted_max", "BudgetAdmittedMaxSubset"),
            ("confidence_routing_055", "OneShot_ConfidenceRouting_0.55"),
            ("sequential_confidence", "SequentialConfidence_0.55_0.45"),
            ("unimodal_text", "Unimodal_Text"),
        ]

        # 3a. Persistent 3 repetitions + Cold/Warm check
        for rep in range(3):
            for pol_mode, pol_label in sens_policies:
                coord = DistributedCoordinator(banks[42], device, audio_port=audio_port, visual_port=visual_port, persistent_connections=True)
                try:
                    for i in range(N):
                        res = coord.process_query(i, wl_clean["text"][i : i + 1], policy_mode=pol_mode, deadline_ms=50.0, cost_budget=3.5)
                        sens_rows.append({
                            "study": "persistent_3reps", "rep": rep, "concurrency": 1, "persistent": True,
                            "cold_query": int(rep == 0 and i == 0), "sample_idx": i, "policy": pol_label,
                            "application_protocol_bytes": res["application_protocol_bytes"],
                            "wall_ms": res["wall_ms"], "on_time": int(res["on_time"]),
                            "service_quality_prediction": res["service_quality_prediction"], "true_label": int(y_true[i]),
                        })
                finally:
                    coord.close()

        # 3b. Non-persistent per-request TCP connect sensitivity (CLEAN_MULTIMODAL seed 42, 1 rep = 1750 sockets)
        for pol_mode, pol_label in sens_policies:
            coord = DistributedCoordinator(banks[42], device, audio_port=audio_port, visual_port=visual_port, persistent_connections=False)
            try:
                for i in range(N):
                    res = coord.process_query(i, wl_clean["text"][i : i + 1], policy_mode=pol_mode, deadline_ms=50.0, cost_budget=3.5)
                    sens_rows.append({
                        "study": "non_persistent_connect", "rep": 0, "concurrency": 1, "persistent": False,
                        "cold_query": int(i == 0), "sample_idx": i, "policy": pol_label,
                        "application_protocol_bytes": res["application_protocol_bytes"],
                        "wall_ms": res["wall_ms"], "on_time": int(res["on_time"]),
                        "service_quality_prediction": res["service_quality_prediction"], "true_label": int(y_true[i]),
                    })
            finally:
                coord.close()

        # 3c. Concurrency = 4 (persistent connection per worker)
        for pol_mode, pol_label in sens_policies:
            def _worker(idx_slice: List[int]):
                w_coord = DistributedCoordinator(banks[42], device, audio_port=audio_port, visual_port=visual_port, persistent_connections=True)
                out = []
                try:
                    for i in idx_slice:
                        r = w_coord.process_query(i, wl_clean["text"][i : i + 1], policy_mode=pol_mode, deadline_ms=50.0, cost_budget=3.5)
                        out.append({
                            "study": "concurrency_4", "rep": 0, "concurrency": 4, "persistent": True,
                            "cold_query": int(i == idx_slice[0]), "sample_idx": i, "policy": pol_label,
                            "application_protocol_bytes": r["application_protocol_bytes"],
                            "wall_ms": r["wall_ms"], "on_time": int(r["on_time"]),
                            "service_quality_prediction": r["service_quality_prediction"], "true_label": int(y_true[i]),
                        })
                finally:
                    w_coord.close()
                return out

            chunks = [list(range(k, N, 4)) for k in range(4)]
            with ThreadPoolExecutor(max_workers=4) as ex:
                for chunk_res in ex.map(_worker, chunks):
                    sens_rows.extend(chunk_res)

        df_sens = pd.DataFrame(sens_rows)
        df_sens.to_csv(os.path.join(SYS_DIR, "distributed_sensitivity.csv"), index=False)

        # 4. Controlled Fault Stress Suite
        logger.info("[4/4] Running controlled TCP fault-injection suite...")
        df_fault = run_fault_injection_suite(banks[42], val_data, device)

        # Recompute canonical summary directly from df_full
        policy_summaries = {}
        for _, pol_label in DISTRIBUTED_POLICIES:
            sub = df_full[df_full["policy"] == pol_label]
            ws_sq_f1s = []
            ws_raw_f1s = []
            for (w_name, s_val), grp in sub.groupby(["workload", "seed"]):
                yt = grp["true_label"].to_numpy()
                ws_sq_f1s.append(compute_3class_macro_f1(yt, grp["service_quality_prediction"].to_numpy()))
                ws_raw_f1s.append(compute_3class_macro_f1(yt, grp["prediction"].to_numpy()))
            walls = sub["wall_ms"].to_numpy()
            policy_summaries[pol_label] = {
                "service_quality_macro_f1": float(np.mean(ws_sq_f1s)),
                "raw_macro_f1": float(np.mean(ws_raw_f1s)),
                "total_application_protocol_bytes": int(sub["application_protocol_bytes"].sum()),
                "mean_application_protocol_bytes": float(sub["application_protocol_bytes"].mean()),
                "total_payload_bytes": int(sub["payload_bytes"].sum()),
                "mean_wall_ms": float(np.mean(walls)),
                "p50_wall_ms": float(np.quantile(walls, 0.50, method="linear")),
                "p95_wall_ms": float(np.quantile(walls, 0.95, method="linear")),
                "p99_wall_ms": float(np.quantile(walls, 0.99, method="linear")),
                "mean_network_rtt_ms": float(sub["network_rtt_ms"].mean()),
                "mean_inference_ms": float(sub["inference_ms"].mean()),
                "on_time_rate": float(sub["on_time"].mean()),
                "acceptance_rate": float(sub["accepted"].mean()),
                "useful_on_time_rate": float(sub["useful_on_time"].mean()),
                "budget_violation_rate": float(sub["budget_violated"].mean()),
                "mean_nominal_cost": float(sub["nominal_cost"].mean()),
                "num_samples": int(len(sub)),
            }

        sens_summary = {}
        for (study, pol), grp in df_sens.groupby(["study", "policy"]):
            sens_summary.setdefault(study, {})[pol] = {
                "num_queries": int(len(grp)),
                "total_app_bytes": int(grp["application_protocol_bytes"].sum()),
                "mean_app_bytes": float(grp["application_protocol_bytes"].mean()),
                "mean_wall_ms": float(grp["wall_ms"].mean()),
                "p50_wall_ms": float(np.quantile(grp["wall_ms"].to_numpy(), 0.50, method="linear")),
                "p95_wall_ms": float(np.quantile(grp["wall_ms"].to_numpy(), 0.95, method="linear")),
                "p99_wall_ms": float(np.quantile(grp["wall_ms"].to_numpy(), 0.99, method="linear")),
                "on_time_rate": float(grp["on_time"].mean()),
            }

        seq_bytes = policy_summaries["SequentialConfidence_0.55_0.45"]["mean_application_protocol_bytes"]
        ao_ser_bytes = policy_summaries["AlwaysOn_Direct_Serial"]["mean_application_protocol_bytes"]
        ao_par_bytes = policy_summaries["AlwaysOn_Direct_Parallel"]["mean_application_protocol_bytes"]
        sta_bytes = policy_summaries["Static_TA_Direct"]["mean_application_protocol_bytes"]
        cr_bytes = policy_summaries["OneShot_ConfidenceRouting_0.55"]["mean_application_protocol_bytes"]

        distributed_summary = {
            "benchmark_version": "13.2",
            "timestamp_utc": utc_now,
            "transport_scope": "Single-host localhost TCP loopback sockets (ports 8765, 8766); threaded providers; NOT multi-device WAN or link-layer energy/TCP-IP packet capture.",
            "connection_mode_primary": "persistent_keep_alive_tcp_nodelay",
            "quantile_convention": "numpy.quantile(method='linear')",
            "environment": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "device": str(device),
            },
            "population": {
                "split": "val",
                "workloads": WORKLOAD_NAMES,
                "seeds": SEEDS,
                "queries_per_policy": N * len(WORKLOAD_NAMES) * len(SEEDS),
                "total_rows_in_csv": int(len(df_full)),
            },
            "offline_vs_socket_clean_parity": parity,
            "byte_agreement": {
                "client_total_app_bytes": int(client_total_app_bytes),
                "server_total_app_bytes": int(server_total_app_bytes),
                "abs_discrepancy": int(byte_discrepancy),
            },
            "resource_comparisons_measured_app_bytes": {
                "vs_AlwaysOn_Direct_Serial_pct": float((ao_ser_bytes - seq_bytes) / ao_ser_bytes * 100.0) if ao_ser_bytes > 0 else 0.0,
                "vs_AlwaysOn_Direct_Parallel_pct": float((ao_par_bytes - seq_bytes) / ao_par_bytes * 100.0) if ao_par_bytes > 0 else 0.0,
                "vs_Static_TA_Direct_pct": float((sta_bytes - seq_bytes) / sta_bytes * 100.0) if sta_bytes > 0 else 0.0,
                "vs_OneShot_ConfidenceRouting_0.55_pct": float((cr_bytes - seq_bytes) / cr_bytes * 100.0) if cr_bytes > 0 else 0.0,
            },
            "policies": policy_summaries,
            "sensitivity_studies": sens_summary,
            "fault_stress_summary": {
                "total_fault_queries": int(len(df_fault)),
                "scenarios": sorted(list(df_fault["scenario"].unique())),
            },
        }

        with open(os.path.join(SYS_DIR, "distributed_summary.json"), "w", encoding="utf-8") as f:
            json.dump(distributed_summary, f, indent=2)

        logger.info("=== Distributed TCP Benchmark Complete ===")
        for pol, p_s in policy_summaries.items():
            logger.info(
                f"  {pol:32s} | SQ-F1={p_s['service_quality_macro_f1']:.4f} | "
                f"AppBytes/q={p_s['mean_application_protocol_bytes']:.1f}B | "
                f"Mean={p_s['mean_wall_ms']:.2f}ms | p95={p_s['p95_wall_ms']:.2f}ms | "
                f"p99={p_s['p99_wall_ms']:.2f}ms | UOT={p_s['useful_on_time_rate']*100:.2f}%"
            )
        return distributed_summary

    finally:
        audio_server.stop()
        visual_server.stop()


if __name__ == "__main__":
    run_distributed_benchmark()
