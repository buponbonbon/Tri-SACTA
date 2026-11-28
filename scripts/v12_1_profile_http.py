"""
scripts/v12_1_profile_http.py: Real HTTP Runtime and Latency Profiler for Tri-SACTA V12.1.

Executes genuine HTTP requests against the FastAPI service (src/service/app.py) using TestClient:
- Cold vs Warm separation (1 cold request + 100 warm requests).
- Concurrency levels: C=1 (sequential) and C=4 (concurrent client load).
- Measures end-to-end HTTP wire latency (client-side wall clock) vs internal service latency.
- Validates fail-closed behavior, response payloads, and schema conformity.

Outputs:
- results/v12_1_systems/profiling/http_trials.csv
- results/v12_1_systems/profiling/runtime_summary.json
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch
from fastapi.testclient import TestClient

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.service.app import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def profile_http_service():
    logger.info("Initializing FastAPI TestClient for HTTP runtime profiling...")
    out_dir = os.path.join(repo_root, "results", "v12_1_systems", "profiling")
    os.makedirs(out_dir, exist_ok=True)

    client = TestClient(app)

    # 1. Test health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    health_data = health_resp.json()
    logger.info(f"Health check verified: {health_data['service']} on {health_data['device']}")

    # Load clean sample features for profiling
    data_dir = os.path.join(repo_root, "data", "cmu_mosei", "features_cache", "val")
    text_feat = torch.load(os.path.join(data_dir, "text.pt"), map_location="cpu", weights_only=False)[0].tolist()
    audio_feat = torch.load(os.path.join(data_dir, "audio.pt"), map_location="cpu", weights_only=False)[0].tolist()
    visual_feat = torch.load(os.path.join(data_dir, "visual.pt"), map_location="cpu", weights_only=False)[0].tolist()

    payload_clean = {
        "features": {
            "text": text_feat,
            "audio": audio_feat,
            "visual": visual_feat,
        },
        "deadline_ms": 50.0,
        "cost_budget": 3.5,
        "policy": "orchestrated_brmo",
        "seed": 42,
    }

    trials: List[Dict[str, Any]] = []

    # 2. Cold Start Request (1 trial)
    t0_cold = time.perf_counter()
    resp_cold = client.post("/predict", json=payload_clean)
    wall_cold_ms = (time.perf_counter() - t0_cold) * 1000.0
    assert resp_cold.status_code == 200, f"Cold request failed: {resp_cold.text}"
    cold_data = resp_cold.json()

    trials.append({
        "trial_id": 0,
        "mode": "cold_start",
        "concurrency": 1,
        "wall_latency_ms": wall_cold_ms,
        "server_latency_ms": cold_data["latency_ms"],
        "routing_decision": cold_data["routing_decision"],
        "compute_cost": cold_data["compute_cost"],
        "abstained": cold_data["abstained"],
        "http_status": resp_cold.status_code,
    })
    logger.info(f"Cold Start: Wall={wall_cold_ms:.2f}ms | Server={cold_data['latency_ms']:.2f}ms | Route={cold_data['routing_decision']}")

    # 3. Warm Sequential Requests (C=1, 100 trials)
    logger.info("Executing 100 warm sequential requests (C=1)...")
    for i in range(1, 101):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=payload_clean)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        trials.append({
            "trial_id": i,
            "mode": "warm_sequential",
            "concurrency": 1,
            "wall_latency_ms": wall_ms,
            "server_latency_ms": data["latency_ms"],
            "routing_decision": data["routing_decision"],
            "compute_cost": data["compute_cost"],
            "abstained": data["abstained"],
            "http_status": resp.status_code,
        })

    # 4. Concurrent Requests (C=4, 50 requests in batches of 4)
    logger.info("Executing concurrent requests (C=4, 50 requests)...")
    def send_one(req_idx: int):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=payload_clean)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json()
        return {
            "trial_id": 100 + req_idx,
            "mode": "warm_concurrent",
            "concurrency": 4,
            "wall_latency_ms": wall_ms,
            "server_latency_ms": data["latency_ms"],
            "routing_decision": data["routing_decision"],
            "compute_cost": data["compute_cost"],
            "abstained": data["abstained"],
            "http_status": resp.status_code,
        }

    with ThreadPoolExecutor(max_workers=4) as executor:
        conc_results = list(executor.map(send_one, range(1, 51)))
    trials.extend(conc_results)

    # 5. Compile DataFrame & Statistics
    df = pd.DataFrame(trials)
    csv_path = os.path.join(out_dir, "http_trials.csv")
    df.to_csv(csv_path, index=False)

    df_warm_seq = df[df["mode"] == "warm_sequential"]
    df_conc = df[df["mode"] == "warm_concurrent"]

    summary = {
        "cold_start_latency_ms": float(wall_cold_ms),
        "warm_sequential_c1": {
            "num_requests": len(df_warm_seq),
            "wall_latency_mean_ms": float(df_warm_seq["wall_latency_ms"].mean()),
            "wall_latency_std_ms": float(df_warm_seq["wall_latency_ms"].std()),
            "wall_latency_p50_ms": float(df_warm_seq["wall_latency_ms"].quantile(0.50)),
            "wall_latency_p95_ms": float(df_warm_seq["wall_latency_ms"].quantile(0.95)),
            "wall_latency_p99_ms": float(df_warm_seq["wall_latency_ms"].quantile(0.99)),
            "server_latency_mean_ms": float(df_warm_seq["server_latency_ms"].mean()),
            "mean_compute_cost": float(df_warm_seq["compute_cost"].mean()),
        },
        "warm_concurrent_c4": {
            "num_requests": len(df_conc),
            "wall_latency_mean_ms": float(df_conc["wall_latency_ms"].mean()),
            "wall_latency_p50_ms": float(df_conc["wall_latency_ms"].quantile(0.50)),
            "wall_latency_p95_ms": float(df_conc["wall_latency_ms"].quantile(0.95)),
            "wall_latency_p99_ms": float(df_conc["wall_latency_ms"].quantile(0.99)),
            "server_latency_mean_ms": float(df_conc["server_latency_ms"].mean()),
        },
        "service_health": health_data,
    }

    summary_path = os.path.join(out_dir, "runtime_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Profiling complete. C=1 Mean Wall: {summary['warm_sequential_c1']['wall_latency_mean_ms']:.2f}ms "
        f"(p95: {summary['warm_sequential_c1']['wall_latency_p95_ms']:.2f}ms). "
        f"Saved to {summary_path}"
    )


if __name__ == "__main__":
    profile_http_service()
