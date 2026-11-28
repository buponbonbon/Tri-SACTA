"""
scripts/publication_audit/verify_phase2_integrity.py: Standalone Read-Only Phase II Artifact
Integrity & Raw-Trace Recomputation Script.
"""

from __future__ import annotations

import os
import sys
import gzip
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.v13_2_evaluate_development import compute_3class_macro_f1
from scripts.v13_2_cluster_statistics import paired_cluster_bootstrap_on_df


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1_lf_normalized(path: str) -> Dict[str, Any]:
    raw = open(path, "rb").read()
    norm = raw.replace(b"\r\n", b"\n")
    header = f"blob {len(norm)}\0".encode("utf-8")
    blob_sha1 = hashlib.sha1(header + norm).hexdigest()
    return {
        "raw_bytes": len(raw),
        "lf_normalized_bytes": len(norm),
        "crlf_line_count_diff": len(raw) - len(norm),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "lf_normalized_sha256": hashlib.sha256(norm).hexdigest(),
        "computed_git_blob_sha1": blob_sha1,
    }


def run_stage_a_audit() -> Dict[str, Any]:
    conf_dir = os.path.join(REPO_ROOT, "results", "v13_2", "confirmatory")
    out_dir = os.path.join(REPO_ROOT, "results", "v13_2", "publication_audit")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Artifact Integrity & CRLF/LF Git Blob Verification
    csv_path = os.path.join(conf_dir, "distributed_benchmark.csv")
    gz_path = os.path.join(conf_dir, "query_traces.json.gz")
    csv_info = git_blob_sha1_lf_normalized(csv_path)
    git_blob = subprocess.check_output(
        ["git", "ls-tree", "HEAD", "results/v13_2/confirmatory/distributed_benchmark.csv"],
        cwd=REPO_ROOT, text=True,
    ).strip().split()[2]
    csv_info["committed_git_blob_sha1"] = git_blob
    csv_info["git_blob_exact_match"] = (csv_info["computed_git_blob_sha1"] == git_blob)

    gz_sha256 = sha256_file(gz_path)
    gz_bytes = os.path.getsize(gz_path)

    with open(os.path.join(REPO_ROOT, "results", "v13_2", "provenance", "holdout_access_ledger.json"), "r", encoding="utf-8") as f:
        ledger = json.load(f)

    integrity_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": ledger.get("phase"),
        "fresh_holdout_inference_executed": ledger.get("fresh_holdout_inference_executed"),
        "distributed_benchmark_csv": csv_info,
        "query_traces_json_gz": {"bytes": gz_bytes, "sha256": gz_sha256},
        "final_decision_json_sha256": sha256_file(os.path.join(conf_dir, "final_decision.json")),
        "cluster_bootstrap_summary_json_sha256": sha256_file(os.path.join(conf_dir, "cluster_bootstrap_summary.json")),
        "policy_evaluation_summary_json_sha256": sha256_file(os.path.join(conf_dir, "policy_evaluation_summary.json")),
    }
    with open(os.path.join(out_dir, "artifact_integrity_report.json"), "w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2)

    # 2. Independent Raw-Trace Recomputation
    df_tcp = pd.read_csv(csv_path)
    tcp_rows = len(df_tcp)
    tcp_unique = len(df_tcp.drop_duplicates(subset=["sample_idx", "seed", "workload", "policy"]))
    tcp_policies = {}
    for pol, g in df_tcp.groupby("policy"):
        sq_f1s, raw_f1s = [], []
        for (_, _), wg in g.groupby(["workload", "seed"]):
            yt = wg["true_label"].to_numpy()
            sq_f1s.append(compute_3class_macro_f1(yt, wg["service_quality_prediction"].to_numpy()))
            raw_f1s.append(compute_3class_macro_f1(yt, wg["prediction"].to_numpy()))
        lat = g["wall_ms"].to_numpy()
        tcp_policies[str(pol)] = {
            "sq_macro_f1": float(np.mean(sq_f1s)),
            "raw_macro_f1": float(np.mean(raw_f1s)),
            "mean_app_bytes": float(g["application_protocol_bytes"].mean()),
            "mean_wall_ms": float(np.mean(lat)),
            "p50_wall_ms": float(np.percentile(lat, 50)),
            "p95_wall_ms": float(np.percentile(lat, 95)),
            "p99_wall_ms": float(np.percentile(lat, 99)),
            "useful_on_time_rate": float(g["useful_on_time"].mean()),
            "acceptance_rate": float(g["prediction"].isin([0, 1, 2]).mean()),
            "budget_violation_rate": float(g["budget_violated"].mean()),
        }

    boot = paired_cluster_bootstrap_on_df(
        df_tcp,
        "SequentialConfidence_0.55_0.45",
        ["AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel", "Static_TA_Direct", "OneShot_ConfidenceRouting_0.55", "Unimodal_Text"],
        "service_quality_prediction",
        "application_protocol_bytes",
        B=2000,
        seed=1320,
    )

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        off_traces = json.load(f)
    df_off = pd.DataFrame(off_traces)
    off_rows = len(df_off)
    off_policies = {}
    for pol, g in df_off.groupby("policy"):
        sq_f1s = [
            compute_3class_macro_f1(wg["true_label"].to_numpy(), wg["service_quality_prediction"].to_numpy())
            for (_, _), wg in g.groupby(["workload", "seed"])
        ]
        off_policies[str(pol)] = {
            "sq_macro_f1": float(np.mean(sq_f1s)),
            "mean_nominal_cost": float(g["spent_cost"].mean()),
            "useful_on_time_rate": float(g["useful_on_time"].mean()),
        }

    recomputed = {
        "tcp_rows": tcp_rows,
        "tcp_unique_keys": tcp_unique,
        "tcp_grid_exact": (tcp_rows == 694008 and tcp_unique == 694008),
        "offline_rows": off_rows,
        "offline_policies_count": len(off_policies),
        "tcp_policies": tcp_policies,
        "tcp_bootstrap_primary": boot,
        "offline_policies": off_policies,
    }
    with open(os.path.join(out_dir, "recomputed_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(recomputed, f, indent=2)

    print(f"STAGE_A_OK | blob_match={csv_info['git_blob_exact_match']} | tcp={tcp_rows} | off={off_rows}")
    return recomputed


if __name__ == "__main__":
    run_stage_a_audit()
