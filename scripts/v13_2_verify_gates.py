"""
scripts/v13_2_verify_gates.py: Strict Fail-Closed Gate Verifier for Tri-SACTA V13.2.

Enforces complete, independent recomputation from raw artifacts (zero trusting summaries):
- G1: Provenance, physical asset SHA-256, row identity, speaker proxy schema, eligibility list uniqueness & disjointness.
- G2: Full expected run grid, unique composite keys, cross-policy pairing, domain checks,
      poisoned/mask/blackout checks, and recomputed metrics for ALL policies.
- G3: Dual-endpoint (primary measured bytes + secondary nominal cost) cluster bootstrap recomputation
      from raw CSV/JSON, joint intersection check across ALL primary comparators (AlwaysOn & Static_TA),
      and operational guardrails.
- G4: Distributed CSV completeness & recomputation of all metrics (bytes, quantiles via linear method,
      F1, useful-on-time, budget violations), client/server byte check, fault CSV check, and test logs.
- G5: Versioned freeze manifest strict schema, ALL 21 checkpoint byte hashes, eligibility digest,
      tracked code/config/dependency lock hashes, detached digest, and optional external pinned digest.
- G6: Confirmatory boundary & holdout isolation audit (filesystem scan + access ledger + approval state).
- G7: Novelty status validation and generated-report consistency check.

CLI returns nonzero on ANY integrity/preflight failure, while distinguishing a valid
negative scientific outcome (exit code 0 or 2? Note: integrity failure returns exit code 2;
valid negative effect returns exit code 0 with scientific_status=DEVELOPMENT_EFFECT_NOT_SUPPORTED).
"""

from __future__ import annotations

import os
import sys
import json
import math
import glob
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import SUBSET_METADATA
from scripts.v13_2_cluster_statistics import paired_cluster_bootstrap_on_df

EXPECTED_MOSEI_PKL_SHA256 = "90701ad5bfa8767ca253af729c1d346e9a8d259ed24f16de113c845ffb60d90e"
CANONICAL_WORKLOADS = [
    "CLEAN_MULTIMODAL",
    "TEXT_EMBEDDING_NOISE",
    "AUDIO_FEATURE_MASKING",
    "VISUAL_ZERO_MASK",
    "COMPOUND_FAULT",
    "INTERMITTENT_MISSINGNESS",
    "HETEROGENEOUS_DEADLINE_STRESS",
]
CANONICAL_SEEDS = [42, 100, 2026]
REQUIRED_TCP_POLICIES = [
    "AlwaysOn_Direct_Serial",
    "AlwaysOn_Direct_Parallel",
    "Static_TA_Direct",
    "BudgetAdmittedMaxSubset",
    "OneShot_ConfidenceRouting_0.55",
    "SequentialConfidence_0.55_0.45",
    "Unimodal_Text",
]
REQUIRED_FAULT_SCENARIOS = [
    "disconnect_mid_frame",
    "connection_refused",
    "slow_provider_timeout",
    "oversize_malformed_header",
    "non_finite_nan_feature",
    "out_of_bounds_sample_id",
    "tight_budget_reject",
    "tight_deadline_reject",
    "total_blackout",
]


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_3class_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in [0, 1, 2]:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def _finite(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def verify_gates(
    base_dir: Optional[str] = None,
    repo_root: Optional[str] = None,
    pinned_manifest_sha256: Optional[str] = None,
    expected_raw_sha256: str = EXPECTED_MOSEI_PKL_SHA256,
    verify_code_inventory: bool = True,
    verify_docs_consistency: bool = True,
    min_bootstrap_b: int = 2000,
    write_output: bool = True,
) -> Dict[str, Any]:
    r_root = repo_root if repo_root is not None else REPO_ROOT
    res_root = base_dir if base_dir is not None else os.path.join(r_root, "results", "v13_2")
    utc_now = datetime.now(timezone.utc).isoformat()

    log_dir = os.path.join(res_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("v13_2_verifier")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    gate_results: Dict[str, Any] = {}
    unresolved_blockers: List[str] = []

    # -------------------------------------------------------------------------
    # G1: Provenance, Identities, Speaker Proxy Schema & Eligibility Manifest
    # -------------------------------------------------------------------------
    prov_path = os.path.join(res_root, "provenance", "provenance_audit.json")
    exp_ledger_path = os.path.join(res_root, "provenance", "exposure_ledger.json")
    elig_path = os.path.join(res_root, "provenance", "eligible_holdout_manifest.json")
    meta_log_path = os.path.join(res_root, "provenance", "metadata_access_log.json")

    if not all(os.path.exists(p) for p in [prov_path, exp_ledger_path, elig_path, meta_log_path]):
        gate_results["G1_Provenance"] = {
            "status": "BLOCKED",
            "reason": "Missing one or more required provenance artifacts (provenance_audit, exposure_ledger, eligible_holdout_manifest, metadata_access_log).",
        }
        unresolved_blockers.append("G1: Missing provenance artifacts")
    else:
        try:
            with open(prov_path, "r", encoding="utf-8") as f:
                prov = json.load(f)
            with open(elig_path, "r", encoding="utf-8") as f:
                elig = json.load(f)

            raw_rel = prov.get("raw_assets", {}).get("mosei.pkl", {}).get("path", "data/cmu_mosei/raw/mosei.pkl")
            raw_abs = os.path.join(r_root, raw_rel)
            if not os.path.exists(raw_abs):
                raise FileNotFoundError(f"Physical raw asset missing: {raw_abs}")
            on_disk_raw_sha = compute_file_sha256(raw_abs)
            if on_disk_raw_sha != expected_raw_sha256:
                raise ValueError(f"Raw asset SHA mismatch: disk={on_disk_raw_sha} vs expected={expected_raw_sha256}")

            # Verify speaker schema across all partitions (F07)
            split_lin = prov.get("split_lineage_and_exposure", {})
            if not split_lin:
                raise ValueError("Missing split_lineage_and_exposure")
            for sp_name, sp_info in split_lin.items():
                spk = sp_info.get("speaker_schema", {})
                if spk.get("speaker_id_verified") is not None:
                    raise ValueError(f"Split {sp_name} improperly asserts non-null speaker_id_verified (F07).")
                if "UNKNOWN" not in str(spk.get("speaker_overlap_status", "")):
                    raise ValueError(f"Split {sp_name} speaker_overlap_status must explicitly remain UNKNOWN (F07).")

            # Verify row identity checks (F06)
            row_chk = prov.get("row_identity_verification", {})
            if not row_chk:
                raise ValueError("Missing row_identity_verification")
            for sp_name, r_info in row_chk.items():
                if r_info.get("raw_row_mismatches", -1) != 0 or r_info.get("labels_7class_mismatches", -1) != 0:
                    raise ValueError(f"Split {sp_name} has raw row/label mismatches.")
                dist = r_info.get("target_3class_counts", {})
                if sum(dist.values()) != r_info.get("count"):
                    raise ValueError(f"Split {sp_name} 3-class count sum != count.")

            # Verify unexposed pool & eligibility manifest (F05, F06)
            unexp = prov.get("unexposed_pool_audit", {})
            unexp_cnt = int(unexp.get("unexposed_available_utterances", -1))
            dist_unexp = unexp.get("class_distribution_3class", {})
            if sum(dist_unexp.values()) != unexp_cnt:
                raise ValueError(f"Unexposed class sum {sum(dist_unexp.values())} != count {unexp_cnt} (F06).")
            if int(unexp.get("cross_split_video_overlap_with_train_val_cal_test", -1)) != 0:
                raise ValueError("Cross-split video overlap != 0.")
            if int(unexp.get("duplicate_canonical_keys", -1)) != 0:
                raise ValueError("Duplicate canonical keys detected.")

            actual_elig_sha = compute_file_sha256(elig_path)
            if actual_elig_sha != unexp.get("eligible_manifest_sha256"):
                raise ValueError("Eligible manifest SHA256 mismatch vs provenance_audit.")

            samples = elig.get("eligible_samples", [])
            if len(samples) != unexp_cnt or len(samples) != int(elig.get("eligible_utterances", -1)):
                raise ValueError(f"Eligible samples count mismatch: list={len(samples)} vs expected={unexp_cnt} (F05).")

            cids = [s.get("canonical_id") for s in samples]
            if len(set(cids)) != len(samples):
                raise ValueError("Duplicate canonical_id detected in eligible_holdout_manifest.json (F05).")
            raw_indices = [int(s.get("raw_test_index", -1)) for s in samples]
            raw_pool_sz = int(unexp.get("raw_test_pool_size", 0))
            if any(idx < 500 or idx >= raw_pool_sz for idx in raw_indices):
                raise ValueError("Eligible raw_test_index out of valid unexposed bounds [500, pool_size) (F05).")
            if int(elig.get("verified_video_overlap_with_all_exposed", -1)) != 0:
                raise ValueError("Eligible pool video overlap with exposed splits != 0 (F05).")

            gate_results["G1_Provenance"] = {
                "status": "PASS",
                "reason": f"Raw SHA256 verified, speaker proxy limitation enforced, {len(samples)} unique eligible IDs verified.",
                "on_disk_raw_sha256": on_disk_raw_sha,
                "eligible_manifest_sha256": actual_elig_sha,
                "eligible_count": len(samples),
            }
        except Exception as e:
            gate_results["G1_Provenance"] = {"status": "FAIL", "reason": str(e)}
            unresolved_blockers.append(f"G1: {e}")

    # -------------------------------------------------------------------------
    # G5 Schema Check Early Helper (to cross-check expected grid in G2 for E08)
    # -------------------------------------------------------------------------
    manifest_path = os.path.join(res_root, "protocol", "freeze_manifest.json")
    manifest_data: Optional[Dict[str, Any]] = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = None

    # -------------------------------------------------------------------------
    # G2: Evaluator Grid Completeness, Uniqueness, Pairing & All-Policy Recomputation
    # -------------------------------------------------------------------------
    grid_path = os.path.join(res_root, "development", "expected_grid.json")
    metrics_path = os.path.join(res_root, "development", "policy_metrics.json")
    traces_path = os.path.join(res_root, "development", "query_traces.json")
    guardrails_path = os.path.join(res_root, "development", "boundary_guardrails.json")

    if not all(os.path.exists(p) for p in [grid_path, metrics_path, traces_path, guardrails_path]):
        gate_results["G2_Evaluator_Correctness"] = {
            "status": "BLOCKED",
            "reason": "Missing development artifacts (expected_grid, policy_metrics, query_traces, or boundary_guardrails).",
        }
        unresolved_blockers.append("G2: Missing development artifacts")
    else:
        try:
            with open(grid_path, "r", encoding="utf-8") as f:
                grid = json.load(f)
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            with open(traces_path, "r", encoding="utf-8") as f:
                traces = json.load(f)
            with open(guardrails_path, "r", encoding="utf-8") as f:
                guard = json.load(f)

            if not guard.get("all_passed", False):
                raise ValueError("Boundary guardrail suite did not pass.")

            # Validate expected grid against manifest protocol if available (E08)
            exp_policies = grid.get("policies", [])
            exp_workloads = grid.get("workloads", [])
            exp_seeds = grid.get("seeds", [])
            n_per_cell = int(grid.get("num_samples_per_cell", 0))
            exp_total = len(exp_policies) * len(exp_workloads) * len(exp_seeds) * n_per_cell
            if exp_total <= 0 or exp_total != int(grid.get("expected_total_rows", -1)):
                raise ValueError("Invalid expected_grid dimensions.")

            if manifest_data and "evaluator_grid" in manifest_data:
                m_grid = manifest_data["evaluator_grid"]
                if set(m_grid.get("workload_families", [])) != set(exp_workloads):
                    raise ValueError("Workload list in expected_grid differs from freeze_manifest (E08).")
                if set(m_grid.get("seeds", [])) != set(exp_seeds):
                    raise ValueError("Seed list in expected_grid differs from freeze_manifest (E08).")
                if "expected_policies" in m_grid and set(m_grid["expected_policies"]) != set(exp_policies):
                    raise ValueError("Policy list in expected_grid differs from freeze_manifest (E08).")
                if "expected_samples_per_cell" in m_grid and int(m_grid["expected_samples_per_cell"]) != n_per_cell:
                    raise ValueError("Sample count per cell differs from freeze_manifest (E08).")

            if len(traces) != exp_total:
                raise ValueError(f"Trace row count {len(traces)} != expected_total_rows {exp_total} (E01/E02).")

            # Uniqueness, domain, safety traps, and cross-policy pairing
            seen_keys = set()
            pairing_ref: Dict[Tuple[str, int, int], Tuple[int, str, Any, float, float]] = {}
            stress_budgets = set()
            by_pol_ws: Dict[str, Dict[Tuple[str, int], Dict[str, List[Any]]]] = {p: {} for p in exp_policies}

            for t in traces:
                pol = t["policy"]
                if pol not in by_pol_ws:
                    raise ValueError(f"Unregistered policy '{pol}' found in traces (E02).")
                w_name = t["workload"]
                s_val = int(t["seed"])
                idx = int(t["sample_idx"])
                if w_name not in exp_workloads or s_val not in exp_seeds:
                    raise ValueError(f"Unexpected workload/seed ({w_name}, {s_val}) (E02).")

                ckey = (w_name, s_val, pol, idx)
                if ckey in seen_keys:
                    raise ValueError(f"Duplicate composite key detected: {ckey} (E02).")
                seen_keys.add(ckey)

                yt = int(t["true_label"])
                yp = int(t["prediction"])
                ysq = int(t["service_quality_prediction"])
                sc = t["spent_cost"]
                sl = t["spent_latency_ms"]
                bl = t["budget_limit"]
                dl = t["deadline_limit"]
                mask = t["modality_mask"]
                f_sub = t["final_subset"]

                if yt not in (0, 1, 2):
                    raise ValueError(f"Invalid true_label {yt} (E04).")
                if yp not in (-1, 0, 1, 2) or ysq not in (-1, 0, 1, 2):
                    raise ValueError(f"Invalid prediction class {yp}/{ysq} (E04).")
                if not (_finite(sc) and _finite(sl) and _finite(bl) and _finite(dl)):
                    raise ValueError("Non-finite float in traces (E04).")
                if sc < 0.0 or sl < 0.0:
                    raise ValueError(f"Negative cost or latency: cost={sc}, lat={sl} (E04).")

                # Pairing check across policies (E03)
                p_key = (w_name, s_val, idx)
                mask_tup = (bool(mask["text"]), bool(mask["audio"]), bool(mask["visual"]))
                cur_pair = (yt, str(t["video_id"]), mask_tup, float(bl), float(dl))
                if p_key not in pairing_ref:
                    pairing_ref[p_key] = cur_pair
                elif pairing_ref[p_key] != cur_pair:
                    raise ValueError(f"Cross-policy pairing mismatch at {p_key}: {pairing_ref[p_key]} vs {cur_pair} (E03).")

                if w_name == "HETEROGENEOUS_DEADLINE_STRESS":
                    stress_budgets.add(round(float(bl), 2))

                # Mask & Blackout / Budget guardrails (E05, E06)
                avail_mods = [m for m in ("text", "audio", "visual") if mask[m]]
                if not avail_mods:
                    if yp != -1 or sc != 0.0:
                        raise ValueError(f"Blackout sample returned fake prediction/cost: pred={yp}, cost={sc} (E05).")
                else:
                    min_avail_cost = min(SUBSET_METADATA[k]["cost"] for k in ("T", "A", "V") if SUBSET_METADATA[k]["mods"][0] in avail_mods)
                    if pol != "AlwaysOnFusion" and bl + 1e-9 < min_avail_cost:
                        if yp != -1 or sc != 0.0:
                            raise ValueError(f"Infeasible budget sample did not reject cleanly: pol={pol}, bl={bl}, pred={yp} (E05).")
                if f_sub != "NONE":
                    for mod_used in SUBSET_METADATA[f_sub]["mods"]:
                        if not mask[mod_used]:
                            raise ValueError(f"Policy {pol} accessed masked modality '{mod_used}' (E05).")
                if pol != "AlwaysOnFusion" and (sc > bl + 1e-6 or int(t["budget_violated"]) != 0):
                    raise ValueError(f"Constrained policy {pol} violated budget: spent={sc} > limit={bl} (E06).")

                ws_key = (w_name, s_val)
                cell = by_pol_ws[pol].setdefault(ws_key, {"yt": [], "yp": [], "ysq": [], "cost": [], "acpt": [], "uot": [], "bv": []})
                cell["yt"].append(yt)
                cell["yp"].append(yp)
                cell["ysq"].append(ysq)
                cell["cost"].append(float(sc))
                cell["acpt"].append(int(t["accepted"]))
                cell["uot"].append(int(t["useful_on_time"]))
                cell["bv"].append(int(t["budget_violated"]))

            if "HETEROGENEOUS_DEADLINE_STRESS" in exp_workloads and len(stress_budgets) < 2:
                raise ValueError(f"HETEROGENEOUS_DEADLINE_STRESS lacks per-request varied budget_limit: {stress_budgets} (E06).")

            # Verify EVERY policy's recomputed metrics vs policy_metrics.json (E07)
            for pol in exp_policies:
                if pol not in metrics_data:
                    raise ValueError(f"Policy '{pol}' missing from policy_metrics.json.")
                if len(by_pol_ws[pol]) != len(exp_workloads) * len(exp_seeds):
                    raise ValueError(f"Policy '{pol}' missing (workload, seed) cells (E01).")
                sq_f1s, raw_f1s, costs, acpts, uots, bvs = [], [], [], [], [], []
                for ws_key, cell in by_pol_ws[pol].items():
                    if len(cell["yt"]) != n_per_cell:
                        raise ValueError(f"Policy '{pol}' cell {ws_key} has {len(cell['yt'])} != {n_per_cell} rows (E01).")
                    yt_arr = np.array(cell["yt"])
                    sq_f1s.append(compute_3class_macro_f1(yt_arr, np.array(cell["ysq"])))
                    raw_f1s.append(compute_3class_macro_f1(yt_arr, np.array(cell["yp"])))
                    costs.append(float(np.mean(cell["cost"])))
                    acpts.append(float(np.mean(cell["acpt"])))
                    uots.append(float(np.mean(cell["uot"])))
                    bvs.append(float(np.mean(cell["bv"])))

                rec_sq = metrics_data[pol]["primary_estimand"]["mean"]
                rec_cost = metrics_data[pol]["mean_cost"]
                if not (_finite(rec_sq) and _finite(rec_cost)):
                    raise ValueError(f"Non-finite summary metrics for {pol} (E04).")
                if abs(float(np.mean(sq_f1s)) - float(rec_sq)) > 1e-5:
                    raise ValueError(f"Recomputed SQ-F1 mismatch for {pol}: {np.mean(sq_f1s):.6f} vs {rec_sq:.6f} (E07).")
                if abs(float(np.mean(costs)) - float(rec_cost)) > 1e-5:
                    raise ValueError(f"Recomputed cost mismatch for {pol}: {np.mean(costs):.6f} vs {rec_cost:.6f} (E07).")

            cand_sq_f1 = float(metrics_data["SequentialConfidence_0.55_0.45"]["primary_estimand"]["mean"])
            gate_results["G2_Evaluator_Correctness"] = {
                "status": "PASS",
                "reason": f"Verified full {exp_total:,}-row grid across all {len(exp_policies)} policies, exact pairing, domain, and zero discrepancies.",
                "total_rows_verified": exp_total,
                "recomputed_candidate_sq_f1": cand_sq_f1,
            }
        except Exception as e:
            gate_results["G2_Evaluator_Correctness"] = {"status": "FAIL", "reason": str(e)}
            unresolved_blockers.append(f"G2: {e}")

    # -------------------------------------------------------------------------
    # G4: Runtime, Real TCP CSV Recomputation, Test Execution Logs & Fault Suite
    # -------------------------------------------------------------------------
    sys_json_path = os.path.join(res_root, "systems", "distributed_summary.json")
    sys_csv_path = os.path.join(res_root, "systems", "distributed_benchmark.csv")
    fault_csv_path = os.path.join(res_root, "systems", "distributed_fault_benchmark.csv")
    test_log_path = os.path.join(res_root, "validation", "test_execution_summary.json")

    if not all(os.path.exists(p) for p in [sys_json_path, sys_csv_path, fault_csv_path, test_log_path]):
        gate_results["G4_Systems_Telemetry"] = {
            "status": "BLOCKED",
            "reason": "Missing distributed_summary.json, distributed_benchmark.csv, distributed_fault_benchmark.csv, or test_execution_summary.json (R02).",
        }
        unresolved_blockers.append("G4: Missing systems telemetry or test execution logs")
    else:
        try:
            with open(sys_json_path, "r", encoding="utf-8") as f:
                sys_json = json.load(f)
            with open(test_log_path, "r", encoding="utf-8") as f:
                test_summary = json.load(f)

            if int(test_summary.get("exit_code", -1)) != 0 or int(test_summary.get("failed", -1)) != 0 or int(test_summary.get("passed", 0)) <= 0:
                raise ValueError(f"Engineering tests failed or incomplete: {test_summary} (R02).")

            df_sys = pd.read_csv(sys_csv_path)
            df_fault = pd.read_csv(fault_csv_path)

            # Check required policies and scenarios (T05, T06)
            present_pols = set(df_sys["policy"].unique().tolist())
            missing_pols = [p for p in REQUIRED_TCP_POLICIES if p not in present_pols]
            if missing_pols:
                raise ValueError(f"Missing required distributed baselines: {missing_pols} (T05/T06).")

            present_faults = set(df_fault["scenario"].unique().tolist())
            missing_faults = [sc for sc in REQUIRED_FAULT_SCENARIOS if sc not in present_faults]
            if missing_faults:
                raise ValueError(f"Missing required fault-injection scenarios: {missing_faults} (T06).")

            # Domain & completeness checks on CSV (T02, E04)
            pop_q = int(sys_json.get("population", {}).get("queries_per_policy", 0))
            if pop_q <= 0:
                raise ValueError("Invalid population.queries_per_policy in distributed_summary.json")
            if (df_sys["application_protocol_bytes"] < 0).any() or (df_sys["wall_ms"] < 0).any():
                raise ValueError("Negative bytes or wall_ms in distributed_benchmark.csv (E04).")
            if not np.all(np.isfinite(df_sys[["application_protocol_bytes", "wall_ms"]].to_numpy())):
                raise ValueError("NaN/Inf in distributed_benchmark.csv (E04).")

            # Check client/server discrepancy bound
            ba = sys_json.get("byte_agreement", {})
            c_bytes = int(ba.get("client_total_app_bytes", -1))
            s_bytes = int(ba.get("server_total_app_bytes", -2))
            total_failed = int(df_sys["failed_attempts"].sum()) if "failed_attempts" in df_sys.columns else 0
            if c_bytes < 0 or abs(c_bytes - s_bytes) > max(0, total_failed * 4096):
                raise ValueError(f"Client/server byte discrepancy {abs(c_bytes - s_bytes)} exceeds bounded timeout allowance (failed={total_failed}).")

            # Recompute EVERY policy summary directly from CSV (T01, T02)
            for pol in present_pols:
                sub = df_sys[df_sys["policy"] == pol]
                if len(sub) != pop_q:
                    raise ValueError(f"Policy '{pol}' has {len(sub)} rows in CSV != expected {pop_q} (dropped timeouts? T02).")
                if pol not in sys_json.get("policies", {}):
                    raise ValueError(f"Policy '{pol}' missing from distributed_summary.json.")
                rec = sys_json["policies"][pol]

                ws_f1s = [
                    compute_3class_macro_f1(grp["true_label"].to_numpy(), grp["service_quality_prediction"].to_numpy())
                    for _, grp in sub.groupby(["workload", "seed"])
                ]
                re_sq = float(np.mean(ws_f1s))
                re_bytes = int(sub["application_protocol_bytes"].sum())
                re_mean_b = float(sub["application_protocol_bytes"].mean())
                walls = sub["wall_ms"].to_numpy()
                re_p50 = float(np.quantile(walls, 0.50, method="linear"))
                re_p95 = float(np.quantile(walls, 0.95, method="linear"))
                re_p99 = float(np.quantile(walls, 0.99, method="linear"))
                re_uot = float(sub["useful_on_time"].mean())
                re_acpt = float(sub["accepted"].mean())
                re_bv = float(sub["budget_violated"].mean())

                if not all(_finite(rec.get(k)) for k in ("service_quality_macro_f1", "total_application_protocol_bytes", "p95_wall_ms", "useful_on_time_rate")):
                    raise ValueError(f"Non-finite field in distributed_summary.json for {pol} (E04).")

                if (
                    abs(re_sq - float(rec["service_quality_macro_f1"])) > 1e-4
                    or re_bytes != int(rec["total_application_protocol_bytes"])
                    or abs(re_mean_b - float(rec["mean_application_protocol_bytes"])) > 1e-3
                    or abs(re_p50 - float(rec["p50_wall_ms"])) > 1e-3
                    or abs(re_p95 - float(rec["p95_wall_ms"])) > 1e-3
                    or abs(re_p99 - float(rec["p99_wall_ms"])) > 1e-3
                    or abs(re_uot - float(rec["useful_on_time_rate"])) > 1e-4
                    or abs(re_acpt - float(rec["acceptance_rate"])) > 1e-4
                ):
                    raise ValueError(f"CSV vs distributed_summary.json mismatch for '{pol}' (T01).")

                if pol not in ("AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel") and re_bv > 0.0:
                    raise ValueError(f"Constrained TCP policy '{pol}' has budget violations ({re_bv}) (S06).")

            gate_results["G4_Systems_Telemetry"] = {
                "status": "PASS",
                "reason": "Recomputed all distributed CSV policy metrics, client/server byte agreement, fault scenarios, and test execution logs.",
                "client_vs_server_byte_diff": abs(c_bytes - s_bytes),
                "total_csv_rows_verified": int(len(df_sys)),
                "candidate_mean_app_bytes": float(sys_json["policies"]["SequentialConfidence_0.55_0.45"]["mean_application_protocol_bytes"]),
            }
        except Exception as e:
            gate_results["G4_Systems_Telemetry"] = {"status": "FAIL", "reason": str(e)}
            unresolved_blockers.append(f"G4: {e}")

    # -------------------------------------------------------------------------
    # G3: Dual-Endpoint Statistical Recomputation & Joint Effect Evaluation
    # -------------------------------------------------------------------------
    stat_path = os.path.join(res_root, "statistics", "paired_bootstrap_summary.json")
    g3_integrity_status = "FAIL"
    g3_effect_status = "BLOCKED"

    if not os.path.exists(stat_path) or not os.path.exists(sys_csv_path):
        gate_results["G3_Statistical_Headroom"] = {
            "status": "BLOCKED",
            "integrity_status": "BLOCKED",
            "effect_status": "BLOCKED",
            "reason": "Missing paired_bootstrap_summary.json or distributed_benchmark.csv.",
        }
        unresolved_blockers.append("G3: Missing statistics artifacts")
    else:
        try:
            with open(stat_path, "r", encoding="utf-8") as f:
                stat_data = json.load(f)
            pm = stat_data.get("primary_measured_endpoint", {})
            if pm.get("resource_unit") != "application_protocol_bytes" or pm.get("prediction_column") != "service_quality_prediction":
                raise ValueError("Primary statistical endpoint must use measured application_protocol_bytes and service_quality_prediction (S05).")
            B_val = int(pm.get("num_bootstrap_resamples", 0))
            if B_val < min_bootstrap_b:
                raise ValueError(f"Bootstrap resamples B={B_val} < required {min_bootstrap_b}.")

            df_sys = pd.read_csv(sys_csv_path)
            csv_videos = sorted(df_sys["video_id"].unique().tolist())
            if pm.get("ordered_video_clusters") != csv_videos or int(pm.get("num_video_clusters", -1)) != len(csv_videos):
                raise ValueError("Bootstrap video cluster list misaligned with raw CSV data (S04).")

            # Recompute primary endpoint bootstrap directly from CSV! (S03, S04)
            re_boot = paired_cluster_bootstrap_on_df(
                df_sys,
                candidate=pm.get("candidate_policy", "SequentialConfidence_0.55_0.45"),
                comparators=list(pm.get("comparisons", {}).keys()),
                pred_col="service_quality_prediction",
                resource_col="application_protocol_bytes",
                B=B_val,
                seed=int(pm.get("rng_seed", 1320)),
            )

            primary_comps = ["AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel", "Static_TA_Direct"]
            for p_comp in primary_comps:
                if p_comp not in pm.get("comparisons", {}):
                    raise ValueError(f"Primary comparator '{p_comp}' missing from bootstrap comparisons (S01/S02).")
                rec_c = pm["comparisons"][p_comp]
                act_c = re_boot["comparisons"][p_comp]
                rec_f1_ci = rec_c["delta_macro_f1_95_ci"]
                act_f1_ci = act_c["delta_macro_f1_95_ci"]
                rec_s_ci = rec_c["resource_saving_pct_95_ci"]
                act_s_ci = act_c["resource_saving_pct_95_ci"]
                if not (all(_finite(x) for x in rec_f1_ci) and rec_s_ci is not None and all(_finite(x) for x in rec_s_ci)):
                    raise ValueError(f"Non-finite CI values for {p_comp} (E04).")
                if (
                    abs(rec_f1_ci[0] - act_f1_ci[0]) > 1e-4
                    or abs(rec_f1_ci[1] - act_f1_ci[1]) > 1e-4
                    or abs(rec_s_ci[0] - act_s_ci[0]) > 1e-3
                    or abs(rec_s_ci[1] - act_s_ci[1]) > 1e-3
                ):
                    raise ValueError(f"Bootstrap CI mismatch vs raw CSV recomputation for {p_comp} (S03).")

            g3_integrity_status = "PASS"

            # Now evaluate scientific effect & guardrails WITHOUT altering integrity PASS! (S01, S02, S06, R03)
            eff_failures = []
            for p_comp in primary_comps:
                act_c = re_boot["comparisons"][p_comp]
                if act_c["delta_macro_f1_95_ci"][0] < -0.010:
                    eff_failures.append(f"{p_comp} F1 lower CI {act_c['delta_macro_f1_95_ci'][0]:.4f} < -0.010")
                if not act_c["saving_pct_defined"] or act_c["resource_saving_pct_95_ci"][0] < 15.0:
                    val_s = act_c["resource_saving_pct_95_ci"][0] if act_c["saving_pct_defined"] else "undefined"
                    eff_failures.append(f"{p_comp} byte-saving lower CI {val_s} < 15.0%")

            cand_sub = df_sys[df_sys["policy"] == "SequentialConfidence_0.55_0.45"]
            cand_uot = float(cand_sub["useful_on_time"].mean())
            cand_acpt = float(cand_sub["accepted"].mean())
            cand_bv = float(cand_sub["budget_violated"].mean())
            if cand_uot < 0.95:
                eff_failures.append(f"useful_on_time_rate {cand_uot:.4f} < 0.95 (S06)")
            if cand_acpt < 0.98:
                eff_failures.append(f"acceptance_rate {cand_acpt:.4f} < 0.98 (S06)")
            if cand_bv > 0.0:
                eff_failures.append(f"budget_violation_rate {cand_bv:.4f} > 0.0 (S06)")

            if eff_failures:
                g3_effect_status = "NOT_SUPPORTED"
                gate_status = "FAIL"
                reason = "Integrity verified, but scientific effect/guardrails failed: " + "; ".join(eff_failures)
            else:
                g3_effect_status = "SUPPORTED"
                gate_status = "PASS"
                reason = "Bootstrap recomputation verified; joint non-inferiority & >=15% measured byte savings vs AlwaysOn (serial & parallel) and Static_TA_Direct PASS with guardrails."

            gate_results["G3_Statistical_Headroom"] = {
                "status": gate_status,
                "integrity_status": g3_integrity_status,
                "effect_status": g3_effect_status,
                "reason": reason,
                "primary_resource_unit": "application_protocol_bytes",
                "primary_comparators": {
                    k: {
                        "delta_f1_ci": re_boot["comparisons"][k]["delta_macro_f1_95_ci"],
                        "app_bytes_saved_ci_pct": re_boot["comparisons"][k]["resource_saving_pct_95_ci"],
                    }
                    for k in primary_comps
                },
                "guardrails": {"useful_on_time_rate": cand_uot, "acceptance_rate": cand_acpt, "budget_violation_rate": cand_bv},
            }
        except Exception as e:
            g3_integrity_status = "FAIL"
            g3_effect_status = "BLOCKED"
            gate_results["G3_Statistical_Headroom"] = {
                "status": "FAIL",
                "integrity_status": "FAIL",
                "effect_status": "BLOCKED",
                "reason": str(e),
            }
            unresolved_blockers.append(f"G3: {e}")

    # -------------------------------------------------------------------------
    # G5: Strict Frozen Protocol Manifest & Inventory Verification
    # -------------------------------------------------------------------------
    detached_path = os.path.join(res_root, "protocol", "freeze_manifest.sha256")
    if not os.path.exists(manifest_path):
        gate_results["G5_Frozen_Protocol"] = {"status": "BLOCKED", "reason": f"Missing freeze manifest: {manifest_path}"}
        unresolved_blockers.append("G5: Missing freeze_manifest.json")
    else:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m_data = json.load(f)

            required_top = [
                "manifest_version",
                "scientific_source_commit_sha",
                "candidate_policy",
                "primary_hypothesis",
                "primary_estimand",
                "evaluator_grid",
                "checkpoints_inventory",
                "eligible_holdout_pool",
                "code_inventory",
                "dependencies_lock",
                "executable_protocol",
                "phase_i_contract",
            ]
            missing_top = [k for k in required_top if not m_data.get(k)]
            if missing_top:
                raise ValueError(f"Freeze manifest schema missing or empty required keys: {missing_top} (F01).")
            if m_data["manifest_version"] != "13.2":
                raise ValueError(f"Unknown manifest_version: {m_data['manifest_version']}")

            cp = m_data["candidate_policy"]
            if cp.get("name") != "SequentialConfidence_0.55_0.45" or cp.get("tau_stage1") != 0.55 or cp.get("tau_stage2") != 0.45 or cp.get("audio_first") is not True:
                raise ValueError("Frozen candidate_policy thresholds/order tampered (F03).")

            ph = m_data["primary_hypothesis"]
            if ph.get("non_inferiority_margin") != -0.01 or ph.get("resource_reduction_threshold_pct") != 15.0:
                raise ValueError("Primary hypothesis margins tampered (F03).")
            if set(["AlwaysOnFusion", "Static_TA"]) - set(ph.get("primary_comparators", [])):
                raise ValueError("Primary comparators incomplete (F03).")

            # Verify ALL 21 checkpoints on disk (F02)
            ckpts = m_data["checkpoints_inventory"]
            if len(ckpts) != 21:
                raise ValueError(f"Expected 21 checkpoints in freeze manifest, got {len(ckpts)} (F02).")
            for cname, info in ckpts.items():
                cpath = os.path.join(r_root, info["path"])
                if not os.path.exists(cpath):
                    raise ValueError(f"Frozen checkpoint missing on disk: {info['path']} (F02).")
                actual_csha = compute_file_sha256(cpath)
                if actual_csha != info["sha256"]:
                    raise ValueError(f"Checkpoint byte hash mismatch for {cname} (F02).")
                expected_name = f"mlp_{info['subset']}_seed{info['seed']}.pt"
                if cname != expected_name or not os.path.basename(cpath) == expected_name:
                    raise ValueError(f"Checkpoint seed/subset mapping mismatch: {cname} vs {expected_name} (F02).")

            # Verify eligibility manifest digest (F05)
            elig_pool = m_data["eligible_holdout_pool"]
            if not os.path.exists(elig_path) or compute_file_sha256(elig_path) != elig_pool.get("eligible_manifest_sha256"):
                raise ValueError("Freeze manifest eligibility SHA256 does not match eligible_holdout_manifest.json (F05).")

            # Verify dependency lock & tracked code inventory on disk (F03)
            dep = m_data["dependencies_lock"]
            dep_abs = os.path.join(r_root, dep.get("path", ""))
            if not os.path.exists(dep_abs) or compute_file_sha256(dep_abs) != dep.get("sha256"):
                raise ValueError("Dependency lock hash mismatch or missing (F03).")

            if verify_code_inventory:
                code_inv = m_data["code_inventory"]
                if not code_inv:
                    raise ValueError("Empty code_inventory (F01/F03).")
                for rel_p, info in code_inv.items():
                    abs_p = os.path.join(r_root, rel_p)
                    if not os.path.exists(abs_p):
                        raise ValueError(f"Tracked source/script missing on disk: {rel_p} (F03).")
                    if compute_file_sha256(abs_p) != info["sha256"]:
                        raise ValueError(f"Tracked source/script modified after freeze: {rel_p} (F03).")

            actual_manifest_sha = compute_file_sha256(manifest_path)
            if not os.path.exists(detached_path):
                raise ValueError("Missing detached freeze_manifest.sha256 file.")
            with open(detached_path, "r", encoding="utf-8") as f:
                detached_sha = f.read().strip().split()[0]
            if detached_sha != actual_manifest_sha:
                raise ValueError(f"Detached digest mismatch: {detached_sha} != {actual_manifest_sha} (F04).")

            if pinned_manifest_sha256 is not None and actual_manifest_sha != pinned_manifest_sha256:
                raise ValueError(f"Manifest digest {actual_manifest_sha} != externally pinned digest {pinned_manifest_sha256} (F04).")

            gate_results["G5_Frozen_Protocol"] = {
                "status": "PASS",
                "reason": "Strict schema, 21 checkpoint hashes, code inventory, dependency lock, and detached digest verified.",
                "manifest_sha256": actual_manifest_sha,
                "scientific_source_commit_sha": m_data["scientific_source_commit_sha"],
            }
        except Exception as e:
            gate_results["G5_Frozen_Protocol"] = {"status": "FAIL", "reason": str(e)}
            unresolved_blockers.append(f"G5: {e}")

    # -------------------------------------------------------------------------
    # G6: Confirmatory Boundary & Holdout Isolation Check
    # -------------------------------------------------------------------------
    ledger_path = os.path.join(res_root, "provenance", "holdout_access_ledger.json")
    approval_path = os.path.join(res_root, "protocol", "approval_record.json")
    if not os.path.exists(ledger_path):
        gate_results["G6_Confirmatory_Boundary"] = {
            "status": "BLOCKED",
            "confirmatory_execution_status": "BLOCKED",
            "reason": "Missing holdout_access_ledger.json",
        }
        unresolved_blockers.append("G6: Missing holdout_access_ledger.json")
    else:
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                ledger = json.load(f)

            prohibited_patterns = [
                os.path.join(r_root, "data", "cmu_mosei", "features_cache", "holdout*"),
                os.path.join(r_root, "results", "v13_utility", "workloads", "holdout*"),
                os.path.join(res_root, "confirmatory", "*"),
            ]
            unexpected_artifacts = []
            for pat in prohibited_patterns:
                unexpected_artifacts.extend(glob.glob(pat))

            if unexpected_artifacts:
                raise ValueError(f"BOUNDARY VIOLATION (H03): Unexpected holdout artifacts present in Phase I: {unexpected_artifacts}")

            if (
                ledger.get("fresh_holdout_features_extracted") is not False
                or ledger.get("fresh_holdout_inference_executed") is not False
                or ledger.get("fresh_holdout_predictions_written") is not False
            ):
                raise ValueError("Holdout access ledger indicates fresh holdout was already touched in Phase I (H03).")

            if os.path.exists(approval_path):
                with open(approval_path, "r", encoding="utf-8") as f:
                    app = json.load(f)
                if app.get("approved") is True:
                    # In Phase I check that an active approval isn't mismatched/forged without execution
                    m_sha = gate_results.get("G5_Frozen_Protocol", {}).get("manifest_sha256")
                    if app.get("manifest_sha256") != m_sha:
                        raise ValueError("Stale or mismatched approval_record.json detected (H02).")

            gate_results["G6_Confirmatory_Boundary"] = {
                "status": "PASS",
                "confirmatory_execution_status": "NOT_RUN",
                "phase_ii_authorization": "PENDING_AUTHOR_APPROVAL",
                "reason": "Verified holdout_access_ledger.json and filesystem scan: zero fresh holdout features/predictions exist (NOT_RUN).",
            }
        except Exception as e:
            gate_results["G6_Confirmatory_Boundary"] = {
                "status": "FAIL",
                "confirmatory_execution_status": "BOUNDARY_VIOLATION",
                "phase_ii_authorization": "BLOCKED",
                "reason": str(e),
            }
            unresolved_blockers.append(f"G6: {e}")

    # -------------------------------------------------------------------------
    # G7: Novelty Assessment & Generated-Report Consistency Check
    # -------------------------------------------------------------------------
    novelty_path = os.path.join(res_root, "validation", "novelty_assessment.json")
    novelty_status = "UNRESOLVED"
    if not os.path.exists(novelty_path):
        gate_results["G7_Claims_and_Novelty"] = {
            "status": "BLOCKED",
            "novelty_status": "UNRESOLVED",
            "reason": "Missing novelty_assessment.json",
        }
        unresolved_blockers.append("G7: Missing novelty_assessment.json")
    else:
        try:
            with open(novelty_path, "r", encoding="utf-8") as f:
                nov = json.load(f)
            novelty_status = nov.get("novelty_status", "UNRESOLVED")
            if novelty_status not in ("UNRESOLVED", "SUPPORTED_WITH_SCOPE", "OVERLAP_IDENTIFIED"):
                raise ValueError(f"Invalid novelty_status: {novelty_status}")
            if len(nov.get("prior_art_comparisons", [])) < 4:
                raise ValueError("Prior art comparison matrix incomplete (<4 works).")

            # Report consistency check (R01)
            any_integrity_failed = any(
                gate_results[g]["status"] != "PASS"
                for g in ["G1_Provenance", "G2_Evaluator_Correctness", "G4_Systems_Telemetry", "G5_Frozen_Protocol", "G6_Confirmatory_Boundary"]
            ) or g3_integrity_status != "PASS" or g3_effect_status != "SUPPORTED"

            if verify_docs_consistency and any_integrity_failed:
                for doc_rel in ["README.md", "docs/V13_2_08_RELEASE_VERDICT.md", "docs/HANDOVER_V13_2.md"]:
                    d_abs = os.path.join(r_root, doc_rel)
                    if os.path.exists(d_abs):
                        with open(d_abs, "r", encoding="utf-8") as df:
                            txt = df.read()
                        if "ALL GATES PASS" in txt or "READY_FOR_HOLDOUT_REVIEW" in txt:
                            raise ValueError(
                                f"Report consistency failure (R01): '{doc_rel}' claims READY_FOR_HOLDOUT_REVIEW/ALL GATES PASS while a gate failed."
                            )

            gate_results["G7_Claims_and_Novelty"] = {
                "status": "PASS",
                "novelty_status": novelty_status,
                "reason": f"Prior art verified ({len(nov['prior_art_comparisons'])} works); honest novelty_status={novelty_status}.",
            }
        except Exception as e:
            gate_results["G7_Claims_and_Novelty"] = {
                "status": "FAIL",
                "novelty_status": "UNRESOLVED",
                "reason": str(e),
            }
            unresolved_blockers.append(f"G7: {e}")

    # -------------------------------------------------------------------------
    # Independent Synthesis of Output Verdicts (Section 8)
    # -------------------------------------------------------------------------
    eng_integrity = all(
        gate_results[k]["status"] == "PASS"
        for k in ["G1_Provenance", "G2_Evaluator_Correctness", "G4_Systems_Telemetry"]
    )
    all_integrity = (
        eng_integrity
        and g3_integrity_status == "PASS"
        and gate_results["G5_Frozen_Protocol"]["status"] == "PASS"
        and gate_results["G6_Confirmatory_Boundary"]["status"] == "PASS"
        and gate_results["G7_Claims_and_Novelty"]["status"] == "PASS"
    )

    if not eng_integrity:
        eng_merge_rec = "DO_NOT_MERGE"
    elif all_integrity:
        eng_merge_rec = "RELEASE_CANDIDATE_REVIEWABLE"
    else:
        eng_merge_rec = "REVIEWABLE_DEVELOPMENT_PROTOTYPE"

    if gate_results["G6_Confirmatory_Boundary"]["status"] != "PASS" or not all_integrity:
        sci_status = "BLOCKED_BEFORE_HOLDOUT"
    elif g3_effect_status == "NOT_SUPPORTED":
        sci_status = "DEVELOPMENT_EFFECT_NOT_SUPPORTED"
    else:
        sci_status = "READY_FOR_HOLDOUT_REVIEW"

    verdict = {
        "verifier_version": "13.2",
        "timestamp_utc": utc_now,
        "engineering_merge_recommendation": eng_merge_rec,
        "merge_recommendation": eng_merge_rec,
        "scientific_status": sci_status,
        "development_effect_status": g3_effect_status,
        "novelty_status": novelty_status,
        "phase_ii_authorization": gate_results["G6_Confirmatory_Boundary"].get("phase_ii_authorization", "PENDING_AUTHOR_APPROVAL"),
        "confirmatory_execution_status": gate_results["G6_Confirmatory_Boundary"].get("confirmatory_execution_status", "NOT_RUN"),
        "reasons": {k: v["reason"] for k, v in gate_results.items()},
        "unresolved_blockers": unresolved_blockers,
        "gates": gate_results,
    }

    if write_output:
        val_dir = os.path.join(res_root, "validation")
        os.makedirs(val_dir, exist_ok=True)
        out_json = os.path.join(val_dir, "final_decision.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(verdict, f, indent=2)
        with open(os.path.join(log_dir, "gate_verification.log"), "w", encoding="utf-8") as lf:
            lf.write(json.dumps(verdict, indent=2))

    logger.info(
        f"Verification complete | Engineering={eng_merge_rec} | Scientific={sci_status} | "
        f"DevEffect={g3_effect_status} | Novelty={novelty_status} | Confirmatory={verdict['confirmatory_execution_status']}"
    )
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tri-SACTA V13.2 Fail-Closed Gate Verifier")
    parser.add_argument("--pinned-manifest-sha256", default=None, help="Optional externally pinned manifest digest")
    parser.add_argument("--skip-docs-consistency", action="store_true", help="Skip docs consistency check before doc regen")
    args = parser.parse_args()

    res = verify_gates(
        pinned_manifest_sha256=args.pinned_manifest_sha256,
        verify_docs_consistency=not args.skip_docs_consistency,
    )
    # Nonzero exit on integrity/preflight block, but NOT on valid negative scientific outcome (R03)
    integrity_failed = res["engineering_merge_recommendation"] == "DO_NOT_MERGE" or res["scientific_status"] == "BLOCKED_BEFORE_HOLDOUT"
    sys.exit(2 if integrity_failed else 0)
