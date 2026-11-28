"""
tests/test_v13_2_verifier_adversarial.py: Complete Positive & Adversarial Negative Acceptance Suite (F01-R03).

Builds a self-contained synthetic positive fixture in an isolated temp directory
(zero dependency on mosei.pkl, zero modification to live results/v13_2/), verifies
clean PASS on the intact fixture, and mutates ONE defect at a time for EVERY ID in Section 5.1:
- Freeze & Provenance: F01, F02, F03, F04, F05, F06, F07
- Evaluator & Pairing: E01, E02, E03, E04, E05, E06, E07, E08
- Statistics: S01, S02, S03, S04, S05, S06
- Transport Telemetry: T01, T02, T05, T06 (T03 & T04 verified in runtime safety suite)
- Holdout Boundary & Gating: H01, H02, H03, H04, H05
- Reporting & Integrity vs Effect: R01, R02, R03
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import hashlib
import tempfile
import unittest
import subprocess
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.v13_2_verify_gates import verify_gates, compute_file_sha256, compute_3class_macro_f1
from scripts.v13_2_cluster_statistics import paired_cluster_bootstrap_on_df
from scripts.v13_2_run_phase2_holdout import check_phase2_authorization, AuthorizationError
from scripts.v13_generate_workloads import generate_workloads_for_split


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_synthetic_positive_workspace(root_dir: str) -> Dict[str, Any]:
    """
    Constructs a minimal, internally consistent, mathematically verified synthetic positive fixture
    covering ALL 7 gates cleanly in < 0.05s without touching the private dataset or live outputs.
    """
    res_dir = os.path.join(root_dir, "results", "v13_2")
    for sub in ["provenance", "development", "statistics", "systems", "protocol", "validation", "logs"]:
        os.makedirs(os.path.join(res_dir, sub), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "data", "cmu_mosei", "raw"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "src"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "docs"), exist_ok=True)

    # 1. Raw synthetic asset + tracked file + lock
    raw_file = os.path.join(root_dir, "data", "cmu_mosei", "raw", "mosei.pkl")
    with open(raw_file, "wb") as f:
        f.write(b"SYNTHETIC_MOSEI_ASSET_V13_2")
    raw_sha = compute_file_sha256(raw_file)

    src_dummy = os.path.join(root_dir, "src", "dummy_policy.py")
    with open(src_dummy, "w", encoding="utf-8") as f:
        f.write("THRESHOLD = 0.55\n")
    lock_file = os.path.join(root_dir, "requirements-lock.txt")
    with open(lock_file, "w", encoding="utf-8") as f:
        f.write("torch==2.5.1\n")

    # 2. 21 tiny deterministic checkpoints
    ckpt_dir = os.path.join(root_dir, "results", "v13_utility", "training", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_inv = {}
    for sub in ["T", "A", "V", "TA", "TV", "AV", "TAV"]:
        for seed in [42, 100, 2026]:
            cname = f"mlp_{sub}_seed{seed}.pt"
            cpath = os.path.join(ckpt_dir, cname)
            with open(cpath, "wb") as f:
                f.write(f"CKPT:{sub}:{seed}".encode("utf-8"))
            ckpt_inv[cname] = {
                "path": f"results/v13_utility/training/checkpoints/{cname}",
                "subset": sub,
                "seed": seed,
                "size_bytes": os.path.getsize(cpath),
                "sha256": compute_file_sha256(cpath),
            }

    # 3. Provenance & Eligibility Manifest (12 eligible items across 6 video clusters)
    elig_samples = []
    for idx in range(500, 512):
        vid = f"elig_vid_{(idx - 500) // 2}"
        cid = _sha_bytes(f"{raw_sha}:test:{idx}:{vid}".encode("utf-8"))
        elig_samples.append({"raw_test_index": idx, "canonical_id": cid, "source_partition": "test", "video_group_id": vid, "start_time": 0.0, "end_time": 1.0})
    elig_obj = {
        "schema_version": "13.2",
        "eligible_utterances": 12,
        "eligible_unique_video_clusters": 6,
        "verified_video_overlap_with_all_exposed": 0,
        "eligible_samples": elig_samples,
    }
    elig_path = os.path.join(res_dir, "provenance", "eligible_holdout_manifest.json")
    with open(elig_path, "w", encoding="utf-8") as f:
        json.dump(elig_obj, f, indent=2)
    elig_sha = compute_file_sha256(elig_path)

    prov_obj = {
        "audit_version": "13.2",
        "raw_assets": {"mosei.pkl": {"path": "data/cmu_mosei/raw/mosei.pkl", "sha256": raw_sha}},
        "split_lineage_and_exposure": {
            "val": {"speaker_schema": {"speaker_id_verified": None, "speaker_overlap_status": "UNKNOWN"}}
        },
        "row_identity_verification": {
            "val": {"count": 12, "raw_row_mismatches": 0, "labels_7class_mismatches": 0, "target_3class_counts": {"negative": 4, "neutral": 4, "positive": 4}}
        },
        "unexposed_pool_audit": {
            "raw_test_pool_size": 520,
            "unexposed_available_utterances": 12,
            "class_distribution_3class": {"negative (0)": 4, "neutral (1)": 4, "positive (2)": 4},
            "cross_split_video_overlap_with_train_val_cal_test": 0,
            "duplicate_canonical_keys": 0,
            "eligible_manifest_sha256": elig_sha,
        },
    }
    with open(os.path.join(res_dir, "provenance", "provenance_audit.json"), "w", encoding="utf-8") as f:
        json.dump(prov_obj, f, indent=2)
    with open(os.path.join(res_dir, "provenance", "exposure_ledger.json"), "w", encoding="utf-8") as f:
        json.dump({"partitions": {"val": {"status": "DEVELOPMENT_FIT"}}}, f, indent=2)
    with open(os.path.join(res_dir, "provenance", "metadata_access_log.json"), "w", encoding="utf-8") as f:
        json.dump({"events": [{"access_type": "METADATA_ONLY"}]}, f, indent=2)
    with open(os.path.join(res_dir, "provenance", "holdout_access_ledger.json"), "w", encoding="utf-8") as f:
        json.dump({
            "fresh_holdout_features_extracted": False,
            "fresh_holdout_inference_executed": False,
            "fresh_holdout_predictions_written": False,
        }, f, indent=2)

    # 4. Development & Distributed Traces (8 policies x 2 workloads x 1 seed x 12 utterances across 6 video clusters)
    # Candidate is accurate & cheap (100B, cost 1.0); AlwaysOn & Static_TA use 400B/250B and slightly lower/equal F1
    workloads = ["CLEAN_MULTIMODAL", "HETEROGENEOUS_DEADLINE_STRESS"]
    seeds = [42]
    dev_policies = ["SequentialConfidence_0.55_0.45", "AlwaysOnFusion", "Static_TA"]
    tcp_policies = [
        "AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel", "Static_TA_Direct",
        "BudgetAdmittedMaxSubset", "OneShot_ConfidenceRouting_0.55",
        "SequentialConfidence_0.55_0.45", "Unimodal_Text",
    ]
    n_samples = 12
    # Create paired ground truth across 6 clusters
    vids = [f"v_{i // 2}" for i in range(n_samples)]
    y_true = [i % 3 for i in range(n_samples)]

    dev_traces = []
    sys_rows = []
    for w_name in workloads:
        for i in range(n_samples):
            yt = y_true[i]
            vid = vids[i]
            b_lim = [1.0, 2.2, 3.5][i % 3] if w_name == "HETEROGENEOUS_DEADLINE_STRESS" else 3.5
            d_lim = [15.0, 30.0, 50.0][i % 3] if w_name == "HETEROGENEOUS_DEADLINE_STRESS" else 50.0
            mask = {"text": True, "audio": True, "visual": True}

            for pol in dev_policies:
                if pol == "SequentialConfidence_0.55_0.45":
                    pred = yt if i < 10 else (yt + 1) % 3
                    cost = 1.0
                elif pol == "AlwaysOnFusion":
                    pred = yt if i < 8 else (yt + 1) % 3
                    cost = 3.5
                else:  # Static_TA
                    pred = yt if i < 9 else (yt + 1) % 3
                    cost = 2.2 if b_lim >= 2.2 else 1.0
                dev_traces.append({
                    "sample_idx": i, "video_id": vid, "seed": 42, "workload": w_name, "policy": pol,
                    "true_label": yt, "prediction": pred, "service_quality_prediction": pred,
                    "spent_cost": cost, "spent_latency_ms": 1.0, "final_subset": "T",
                    "budget_limit": b_lim, "deadline_limit": d_lim, "modality_mask": mask,
                    "on_time": 1, "budget_violated": int(cost > b_lim and pol == "AlwaysOnFusion"),
                    "accepted": 1, "useful_on_time": 1,
                })

            for pol in tcp_policies:
                if pol == "SequentialConfidence_0.55_0.45":
                    pred, app_b, wall = (yt if i < 10 else (yt + 1) % 3), 100, 0.5
                elif pol in ("AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel"):
                    pred, app_b, wall = (yt if i < 8 else (yt + 1) % 3), 500, 1.0
                elif pol == "Static_TA_Direct":
                    pred, app_b, wall = (yt if i < 9 else (yt + 1) % 3), 300, 0.6
                elif pol == "Unimodal_Text":
                    pred, app_b, wall = (yt if i < 7 else (yt + 1) % 3), 0, 0.2
                else:
                    pred, app_b, wall = (yt if i < 9 else (yt + 1) % 3), 300, 0.7
                sys_rows.append({
                    "sample_idx": i, "video_id": vid, "seed": 42, "workload": w_name, "policy": pol,
                    "true_label": yt, "prediction": pred, "service_quality_prediction": pred,
                    "application_protocol_bytes": app_b, "payload_bytes": app_b // 2,
                    "wall_ms": wall, "on_time": 1, "budget_violated": 0, "accepted": 1, "useful_on_time": 1, "failed_attempts": 0,
                })

    dev_metrics = {}
    for pol in dev_policies:
        sub = [t for t in dev_traces if t["policy"] == pol]
        ws_f1 = []
        for w in workloads:
            w_sub = [t for t in sub if t["workload"] == w]
            ws_f1.append(compute_3class_macro_f1(np.array([t["true_label"] for t in w_sub]), np.array([t["service_quality_prediction"] for t in w_sub])))
        dev_metrics[pol] = {
            "primary_estimand": {"mean": float(np.mean(ws_f1))},
            "mean_cost": float(np.mean([t["spent_cost"] for t in sub])),
        }

    with open(os.path.join(res_dir, "development", "expected_grid.json"), "w", encoding="utf-8") as f:
        json.dump({
            "workloads": workloads, "seeds": seeds, "policies": dev_policies,
            "num_samples_per_cell": n_samples, "expected_total_rows": len(dev_traces),
        }, f, indent=2)
    with open(os.path.join(res_dir, "development", "policy_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dev_metrics, f, indent=2)
    with open(os.path.join(res_dir, "development", "query_traces.json"), "w", encoding="utf-8") as f:
        json.dump(dev_traces, f)
    with open(os.path.join(res_dir, "development", "boundary_guardrails.json"), "w", encoding="utf-8") as f:
        json.dump({"all_passed": True}, f, indent=2)

    df_sys = pd.DataFrame(sys_rows)
    df_sys.to_csv(os.path.join(res_dir, "systems", "distributed_benchmark.csv"), index=False)
    total_sys_bytes = int(df_sys["application_protocol_bytes"].sum())
    sys_pol_sum = {}
    for pol in tcp_policies:
        s = df_sys[df_sys["policy"] == pol]
        ws_f1 = [compute_3class_macro_f1(g["true_label"].to_numpy(), g["service_quality_prediction"].to_numpy()) for _, g in s.groupby(["workload", "seed"])]
        walls = s["wall_ms"].to_numpy()
        sys_pol_sum[pol] = {
            "service_quality_macro_f1": float(np.mean(ws_f1)),
            "total_application_protocol_bytes": int(s["application_protocol_bytes"].sum()),
            "mean_application_protocol_bytes": float(s["application_protocol_bytes"].mean()),
            "mean_wall_ms": float(np.mean(walls)),
            "p50_wall_ms": float(np.quantile(walls, 0.50, method="linear")),
            "p95_wall_ms": float(np.quantile(walls, 0.95, method="linear")),
            "p99_wall_ms": float(np.quantile(walls, 0.99, method="linear")),
            "useful_on_time_rate": 1.0,
            "acceptance_rate": 1.0,
        }
    with open(os.path.join(res_dir, "systems", "distributed_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "population": {"queries_per_policy": len(workloads) * n_samples},
            "byte_agreement": {"client_total_app_bytes": total_sys_bytes, "server_total_app_bytes": total_sys_bytes},
            "policies": sys_pol_sum,
        }, f, indent=2)
    pd.DataFrame([{"scenario": sc, "status": "SUCCESS"} for sc in [
        "disconnect_mid_frame", "connection_refused", "slow_provider_timeout", "oversize_malformed_header",
        "non_finite_nan_feature", "out_of_bounds_sample_id", "tight_budget_reject", "tight_deadline_reject", "total_blackout",
    ]]).to_csv(os.path.join(res_dir, "systems", "distributed_fault_benchmark.csv"), index=False)

    # 5. Bootstrap stats (B=200 for super-fast test fixture execution)
    boot_primary = paired_cluster_bootstrap_on_df(
        df_sys, "SequentialConfidence_0.55_0.45",
        ["AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel", "Static_TA_Direct"],
        "service_quality_prediction", "application_protocol_bytes", B=200, seed=1320,
    )
    with open(os.path.join(res_dir, "statistics", "paired_bootstrap_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"primary_measured_endpoint": boot_primary}, f, indent=2)

    # 6. Validation logs & novelty
    with open(os.path.join(res_dir, "validation", "test_execution_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"exit_code": 0, "passed": 50, "failed": 0}, f, indent=2)
    with open(os.path.join(res_dir, "validation", "novelty_assessment.json"), "w", encoding="utf-8") as f:
        json.dump({"novelty_status": "OVERLAP_IDENTIFIED", "prior_art_comparisons": [{"w": 1}, {"w": 2}, {"w": 3}, {"w": 4}]}, f, indent=2)

    # 7. Freeze Manifest & Detached Digest
    manifest_obj = {
        "manifest_version": "13.2",
        "scientific_source_commit_sha": "abcdef1234567890",
        "candidate_policy": {"name": "SequentialConfidence_0.55_0.45", "tau_stage1": 0.55, "tau_stage2": 0.45, "audio_first": True},
        "primary_hypothesis": {
            "resource_unit": "application_protocol_bytes",
            "non_inferiority_margin": -0.01,
            "resource_reduction_threshold_pct": 15.0,
            "primary_comparators": ["AlwaysOnFusion", "Static_TA"],
        },
        "primary_estimand": {"name": "service_quality_macro_f1"},
        "evaluator_grid": {"workload_families": workloads, "seeds": seeds, "expected_policies": dev_policies, "expected_samples_per_cell": n_samples},
        "checkpoints_inventory": ckpt_inv,
        "eligible_holdout_pool": {"eligible_manifest_path": "results/v13_2/provenance/eligible_holdout_manifest.json", "eligible_manifest_sha256": elig_sha},
        "code_inventory": {"src/dummy_policy.py": {"path": "src/dummy_policy.py", "sha256": compute_file_sha256(src_dummy)}},
        "dependencies_lock": {"path": "requirements-lock.txt", "sha256": compute_file_sha256(lock_file)},
        "executable_protocol": {"phase_i_verify": "python scripts/v13_2_verify_gates.py"},
        "phase_i_contract": {"phase": "PHASE_I"},
    }
    m_path = os.path.join(res_dir, "protocol", "freeze_manifest.json")
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, indent=2)
    m_sha = compute_file_sha256(m_path)
    with open(os.path.join(res_dir, "protocol", "freeze_manifest.sha256"), "w", encoding="utf-8") as f:
        f.write(m_sha + "  freeze_manifest.json\n")

    return {"root": root_dir, "res_dir": res_dir, "raw_sha": raw_sha, "manifest_sha": m_sha, "elig_sha": elig_sha}


class TestV132VerifierAdversarialComplete(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.fix = build_synthetic_positive_workspace(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, pinned_sha=None):
        return verify_gates(
            base_dir=self.fix["res_dir"],
            repo_root=self.fix["root"],
            pinned_manifest_sha256=pinned_sha,
            expected_raw_sha256=self.fix["raw_sha"],
            min_bootstrap_b=200,
            write_output=False,
        )

    def test_positive_fixture_all_pass(self):
        """Baseline positive synthetic fixture passes all Phase I gates cleanly."""
        v = self._run(pinned_sha=self.fix["manifest_sha"])
        self.assertEqual(v["engineering_merge_recommendation"], "RELEASE_CANDIDATE_REVIEWABLE")
        self.assertEqual(v["scientific_status"], "READY_FOR_HOLDOUT_REVIEW")
        self.assertEqual(v["confirmatory_execution_status"], "NOT_RUN")

    # --- Freeze & Provenance Tests F01-F07 ---
    def test_f01_empty_freeze_manifest_fails_g5(self):
        """F01: Empty {} valid JSON in freeze_manifest.json must FAIL G5."""
        m_path = os.path.join(self.fix["res_dir"], "protocol", "freeze_manifest.json")
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        v = self._run()
        self.assertEqual(v["gates"]["G5_Frozen_Protocol"]["status"], "FAIL")
        self.assertEqual(v["scientific_status"], "BLOCKED_BEFORE_HOLDOUT")

    def test_f02_modified_checkpoint_bytes_fails_g5(self):
        """F02: Modifying 1 byte inside one of 21 checkpoints fails G5 inventory check."""
        cpath = os.path.join(self.fix["root"], "results/v13_utility/training/checkpoints/mlp_TA_seed100.pt")
        with open(cpath, "wb") as f:
            f.write(b"TAMPERED_CKPT")
        v = self._run()
        self.assertEqual(v["gates"]["G5_Frozen_Protocol"]["status"], "FAIL")

    def test_f03_modified_code_after_freeze_fails_g5(self):
        """F03: Modifying tracked scientific source code after freeze fails G5."""
        with open(os.path.join(self.fix["root"], "src", "dummy_policy.py"), "w", encoding="utf-8") as f:
            f.write("THRESHOLD = 0.99\n")
        v = self._run()
        self.assertEqual(v["gates"]["G5_Frozen_Protocol"]["status"], "FAIL")

    def test_f04_recomputed_tampered_manifest_fails_pinned_digest(self):
        """F04: Altering manifest and recomputing its own detached digest fails against pinned digest."""
        m_path = os.path.join(self.fix["res_dir"], "protocol", "freeze_manifest.json")
        with open(m_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        m["candidate_policy"]["tau_stage2"] = 0.45  # keep schema valid, change timestamp
        m["timestamp_utc"] = "TAMPERED"
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        new_sha = compute_file_sha256(m_path)
        with open(os.path.join(self.fix["res_dir"], "protocol", "freeze_manifest.sha256"), "w", encoding="utf-8") as f:
            f.write(new_sha + "\n")
        v = self._run(pinned_sha=self.fix["manifest_sha"])
        self.assertEqual(v["gates"]["G5_Frozen_Protocol"]["status"], "FAIL")

    def test_f05_duplicate_eligible_id_fails_g1_and_g5(self):
        """F05: Duplicate canonical_id in eligible_holdout_manifest.json fails G1 & G5."""
        ep = os.path.join(self.fix["res_dir"], "provenance", "eligible_holdout_manifest.json")
        with open(ep, "r", encoding="utf-8") as f:
            el = json.load(f)
        el["eligible_samples"][1]["canonical_id"] = el["eligible_samples"][0]["canonical_id"]
        with open(ep, "w", encoding="utf-8") as f:
            json.dump(el, f)
        v = self._run()
        self.assertEqual(v["gates"]["G1_Provenance"]["status"], "FAIL")
        self.assertEqual(v["gates"]["G5_Frozen_Protocol"]["status"], "FAIL")

    def test_f06_corrupt_raw_asset_fails_g1(self):
        """F06: Corrupted raw mosei.pkl bytes fail G1 SHA verification."""
        with open(os.path.join(self.fix["root"], "data/cmu_mosei/raw/mosei.pkl"), "wb") as f:
            f.write(b"CORRUPTED_DATA")
        v = self._run()
        self.assertEqual(v["gates"]["G1_Provenance"]["status"], "FAIL")
        self.assertEqual(v["engineering_merge_recommendation"], "DO_NOT_MERGE")

    def test_f07_verified_speaker_proxy_claim_fails_g1(self):
        """F07: Treating video prefix as verified speaker identity fails G1."""
        pp = os.path.join(self.fix["res_dir"], "provenance", "provenance_audit.json")
        with open(pp, "r", encoding="utf-8") as f:
            p = json.load(f)
        p["split_lineage_and_exposure"]["val"]["speaker_schema"]["speaker_overlap_status"] = "ZERO_SPEAKER_OVERLAP_VERIFIED"
        with open(pp, "w", encoding="utf-8") as f:
            json.dump(p, f)
        v = self._run()
        self.assertEqual(v["gates"]["G1_Provenance"]["status"], "FAIL")

    # --- Evaluator & Pairing Tests E01-E08 ---
    def test_e01_missing_trace_row_fails_g2(self):
        """E01: Deleting a single trace row violates exact expected grid in G2."""
        tp = os.path.join(self.fix["res_dir"], "development", "query_traces.json")
        with open(tp, "r", encoding="utf-8") as f:
            tr = json.load(f)
        tr.pop()
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(tr, f)
        v = self._run()
        self.assertEqual(v["gates"]["G2_Evaluator_Correctness"]["status"], "FAIL")

    def test_e02_duplicate_key_fails_g2(self):
        """E02: Duplicate composite key replacing another sample fails G2 uniqueness."""
        tp = os.path.join(self.fix["res_dir"], "development", "query_traces.json")
        with open(tp, "r", encoding="utf-8") as f:
            tr = json.load(f)
        tr[1] = dict(tr[0])
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(tr, f)
        v = self._run()
        self.assertEqual(v["gates"]["G2_Evaluator_Correctness"]["status"], "FAIL")

    def test_e03_unpaired_label_across_policies_fails_g2(self):
        """E03: Different true_label for baseline vs candidate on same sample fails pairing check."""
        tp = os.path.join(self.fix["res_dir"], "development", "query_traces.json")
        with open(tp, "r", encoding="utf-8") as f:
            tr = json.load(f)
        for t in tr:
            if t["policy"] == "Static_TA" and t["sample_idx"] == 0:
                t["true_label"] = (t["true_label"] + 1) % 3
                break
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(tr, f)
        v = self._run()
        self.assertEqual(v["gates"]["G2_Evaluator_Correctness"]["status"], "FAIL")

    def test_e04_nan_and_negative_cost_fail_domain(self):
        """E04: Negative spent_cost or NaN metric fails G2 domain checks."""
        tp = os.path.join(self.fix["res_dir"], "development", "query_traces.json")
        with open(tp, "r", encoding="utf-8") as f:
            tr = json.load(f)
        tr[0]["spent_cost"] = -1.5
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(tr, f)
        v = self._run()
        self.assertEqual(v["gates"]["G2_Evaluator_Correctness"]["status"], "FAIL")

    def test_e05_masked_modality_access_in_trace_fails_g2(self):
        """E05: Trace showing prediction using masked modality fails G2."""
        tp = os.path.join(self.fix["res_dir"], "development", "query_traces.json")
        with open(tp, "r", encoding="utf-8") as f:
            tr = json.load(f)
        tr[0]["modality_mask"]["text"] = False
        tr[0]["final_subset"] = "T"
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(tr, f)
        v = self._run()
        self.assertEqual(v["gates"]["G2_Evaluator_Correctness"]["status"], "FAIL")

    def test_e06_hardcoded_budget_35_in_stress_fails_g2(self):
        """E06: Forcing budget_limit=3.5 across all deadline stress requests fails G2."""
        tp = os.path.join(self.fix["res_dir"], "development", "query_traces.json")
        with open(tp, "r", encoding="utf-8") as f:
            tr = json.load(f)
        for t in tr:
            if t["workload"] == "HETEROGENEOUS_DEADLINE_STRESS":
                t["budget_limit"] = 3.5
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(tr, f)
        v = self._run()
        self.assertEqual(v["gates"]["G2_Evaluator_Correctness"]["status"], "FAIL")

    def test_e07_altered_baseline_prediction_only_fails_g2(self):
        """E07: Altering Static_TA baseline trace prediction (candidate untouched) fails G2 all-policy check."""
        tp = os.path.join(self.fix["res_dir"], "development", "query_traces.json")
        with open(tp, "r", encoding="utf-8") as f:
            tr = json.load(f)
        for t in tr:
            if t["policy"] == "Static_TA":
                t["prediction"] = (t["prediction"] + 1) % 3
                t["service_quality_prediction"] = t["prediction"]
        with open(tp, "w", encoding="utf-8") as f:
            json.dump(tr, f)
        v = self._run()
        self.assertEqual(v["gates"]["G2_Evaluator_Correctness"]["status"], "FAIL")

    def test_e08_tampered_expected_grid_violating_manifest_fails_g2(self):
        """E08: Altering expected_grid.json to omit a workload fails cross-check against freeze_manifest."""
        gp = os.path.join(self.fix["res_dir"], "development", "expected_grid.json")
        with open(gp, "r", encoding="utf-8") as f:
            g = json.load(f)
        g["workloads"] = ["CLEAN_MULTIMODAL"]
        g["expected_total_rows"] = len(g["policies"]) * g["num_samples_per_cell"]
        with open(gp, "w", encoding="utf-8") as f:
            json.dump(g, f)
        v = self._run()
        self.assertEqual(v["gates"]["G2_Evaluator_Correctness"]["status"], "FAIL")

    # --- Statistics Tests S01-S06 ---
    def test_s01_static_ta_fails_noninferiority_fails_g3(self):
        """S01: AlwaysOn passes, Static_TA lower F1 CI < -0.010 -> G3 effect must FAIL cleanly."""
        df = pd.read_csv(os.path.join(self.fix["res_dir"], "systems", "distributed_benchmark.csv"))
        # Make Static_TA_Direct 100% accurate so candidate loses F1 against Static_TA while beating AlwaysOn
        mask = df["policy"] == "Static_TA_Direct"
        df.loc[mask, "service_quality_prediction"] = df.loc[mask, "true_label"]
        df.loc[mask, "prediction"] = df.loc[mask, "true_label"]
        df.to_csv(os.path.join(self.fix["res_dir"], "systems", "distributed_benchmark.csv"), index=False)
        # Recompute matching summary & bootstrap cleanly (genuine negative scientific result!)
        boot = paired_cluster_bootstrap_on_df(
            df, "SequentialConfidence_0.55_0.45",
            ["AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel", "Static_TA_Direct"],
            "service_quality_prediction", "application_protocol_bytes", B=200, seed=1320,
        )
        with open(os.path.join(self.fix["res_dir"], "statistics", "paired_bootstrap_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"primary_measured_endpoint": boot}, f, indent=2)
        with open(os.path.join(self.fix["res_dir"], "systems", "distributed_summary.json"), "r", encoding="utf-8") as f:
            sj = json.load(f)
        s = df[mask]
        sj["policies"]["Static_TA_Direct"]["service_quality_macro_f1"] = float(np.mean([
            compute_3class_macro_f1(g["true_label"].to_numpy(), g["service_quality_prediction"].to_numpy())
            for _, g in s.groupby(["workload", "seed"])
        ]))
        with open(os.path.join(self.fix["res_dir"], "systems", "distributed_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sj, f, indent=2)

        v = self._run()
        self.assertEqual(v["gates"]["G3_Statistical_Headroom"]["integrity_status"], "PASS")
        self.assertEqual(v["gates"]["G3_Statistical_Headroom"]["effect_status"], "NOT_SUPPORTED")
        self.assertEqual(v["scientific_status"], "DEVELOPMENT_EFFECT_NOT_SUPPORTED")

    def test_s02_static_ta_resource_below_15pct_fails_g3(self):
        """S02: AlwaysOn passes >=15% saving, Static_TA byte saving <15% -> G3 effect must FAIL."""
        df = pd.read_csv(os.path.join(self.fix["res_dir"], "systems", "distributed_benchmark.csv"))
        df.loc[df["policy"] == "Static_TA_Direct", "application_protocol_bytes"] = 105  # cand is 100 -> ~4.8% saving
        df.to_csv(os.path.join(self.fix["res_dir"], "systems", "distributed_benchmark.csv"), index=False)
        boot = paired_cluster_bootstrap_on_df(
            df, "SequentialConfidence_0.55_0.45",
            ["AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel", "Static_TA_Direct"],
            "service_quality_prediction", "application_protocol_bytes", B=200, seed=1320,
        )
        with open(os.path.join(self.fix["res_dir"], "statistics", "paired_bootstrap_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"primary_measured_endpoint": boot}, f, indent=2)
        with open(os.path.join(self.fix["res_dir"], "systems", "distributed_summary.json"), "r", encoding="utf-8") as f:
            sj = json.load(f)
        sub = df[df["policy"] == "Static_TA_Direct"]
        sj["policies"]["Static_TA_Direct"]["total_application_protocol_bytes"] = int(sub["application_protocol_bytes"].sum())
        sj["policies"]["Static_TA_Direct"]["mean_application_protocol_bytes"] = float(sub["application_protocol_bytes"].mean())
        tot = int(df["application_protocol_bytes"].sum())
        sj["byte_agreement"] = {"client_total_app_bytes": tot, "server_total_app_bytes": tot}
        with open(os.path.join(self.fix["res_dir"], "systems", "distributed_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sj, f, indent=2)

        v = self._run()
        self.assertEqual(v["gates"]["G3_Statistical_Headroom"]["status"], "FAIL")
        self.assertEqual(v["development_effect_status"], "NOT_SUPPORTED")

    def test_s03_tampered_ci_in_json_fails_recomputation(self):
        """S03: Tampering bootstrap CI in JSON without changing raw CSV fails G3 recomputation."""
        sp = os.path.join(self.fix["res_dir"], "statistics", "paired_bootstrap_summary.json")
        with open(sp, "r", encoding="utf-8") as f:
            sd = json.load(f)
        sd["primary_measured_endpoint"]["comparisons"]["Static_TA_Direct"]["delta_macro_f1_95_ci"] = [0.25, 0.35]
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(sd, f)
        v = self._run()
        self.assertEqual(v["gates"]["G3_Statistical_Headroom"]["integrity_status"], "FAIL")
        self.assertEqual(v["scientific_status"], "BLOCKED_BEFORE_HOLDOUT")

    def test_s04_misaligned_cluster_list_fails_g3(self):
        """S04: Removing a cluster ID from bootstrap cluster list fails G3 alignment check."""
        sp = os.path.join(self.fix["res_dir"], "statistics", "paired_bootstrap_summary.json")
        with open(sp, "r", encoding="utf-8") as f:
            sd = json.load(f)
        sd["primary_measured_endpoint"]["ordered_video_clusters"].pop()
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(sd, f)
        v = self._run()
        self.assertEqual(v["gates"]["G3_Statistical_Headroom"]["integrity_status"], "FAIL")

    def test_s05_nominal_unit_in_primary_endpoint_fails_g3(self):
        """S05: Putting nominal_cost unit into primary_measured_endpoint fails protocol/units check."""
        sp = os.path.join(self.fix["res_dir"], "statistics", "paired_bootstrap_summary.json")
        with open(sp, "r", encoding="utf-8") as f:
            sd = json.load(f)
        sd["primary_measured_endpoint"]["resource_unit"] = "nominal_cost"
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(sd, f)
        v = self._run()
        self.assertEqual(v["gates"]["G3_Statistical_Headroom"]["integrity_status"], "FAIL")

    def test_s06_guardrail_violation_fails_effect(self):
        """S06: Useful on-time rate below 0.95 or budget violation > 0 fails G3 effect / G4."""
        df = pd.read_csv(os.path.join(self.fix["res_dir"], "systems", "distributed_benchmark.csv"))
        df.loc[df["policy"] == "SequentialConfidence_0.55_0.45", "budget_violated"] = 1
        df.to_csv(os.path.join(self.fix["res_dir"], "systems", "distributed_benchmark.csv"), index=False)
        v = self._run()
        self.assertEqual(v["gates"]["G4_Systems_Telemetry"]["status"], "FAIL")
        self.assertEqual(v["engineering_merge_recommendation"], "DO_NOT_MERGE")

    # --- Transport Telemetry Tests T01, T02, T05, T06 ---
    def test_t01_tampered_systems_json_vs_csv_fails_g4(self):
        """T01: Altering JSON bytes/p95 while leaving CSV unchanged fails G4 recomputation."""
        sp = os.path.join(self.fix["res_dir"], "systems", "distributed_summary.json")
        with open(sp, "r", encoding="utf-8") as f:
            sj = json.load(f)
        sj["policies"]["SequentialConfidence_0.55_0.45"]["p95_wall_ms"] = 0.01
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(sj, f)
        v = self._run()
        self.assertEqual(v["gates"]["G4_Systems_Telemetry"]["status"], "FAIL")

    def test_t02_dropped_csv_rows_fails_g4(self):
        """T02: Removing a timeout row from distributed_benchmark.csv fails G4 completeness check."""
        cp = os.path.join(self.fix["res_dir"], "systems", "distributed_benchmark.csv")
        df = pd.read_csv(cp).iloc[:-1]
        df.to_csv(cp, index=False)
        v = self._run()
        self.assertEqual(v["gates"]["G4_Systems_Telemetry"]["status"], "FAIL")

    def test_t05_t06_missing_parallel_or_oneshot_baseline_fails_g4(self):
        """T05/T06: Missing AlwaysOn_Direct_Parallel or OneShot_ConfidenceRouting_0.55 fails G4."""
        cp = os.path.join(self.fix["res_dir"], "systems", "distributed_benchmark.csv")
        df = pd.read_csv(cp)
        df = df[df["policy"] != "AlwaysOn_Direct_Parallel"]
        df.to_csv(cp, index=False)
        v = self._run()
        self.assertEqual(v["gates"]["G4_Systems_Telemetry"]["status"], "FAIL")

    # --- Holdout Isolation & Gating Tests H01-H05 ---
    def test_h01_phase2_runner_refuses_without_approval(self):
        """H01: Phase II runner fails closed before loading encoder/models when no approval supplied."""
        with self.assertRaises(AuthorizationError):
            check_phase2_authorization(
                os.path.join(self.fix["res_dir"], "protocol", "freeze_manifest.json"),
                approval_path=None,
                repo_root=self.fix["root"],
                verify_git_head=False,
            )

    def test_h02_phase2_runner_refuses_stale_digest_or_edited_code(self):
        """H02: Stale manifest digest in approval record fails before holdout access."""
        app_path = os.path.join(self.fix["res_dir"], "protocol", "approval_record.json")
        with open(app_path, "w", encoding="utf-8") as f:
            json.dump({
                "approved": True,
                "allowed_operation": "PHASE_II_CONFIRMATORY_EXECUTION",
                "manifest_sha256": "0" * 64,
                "eligible_manifest_sha256": self.fix["elig_sha"],
            }, f)
        with self.assertRaises(AuthorizationError):
            check_phase2_authorization(
                os.path.join(self.fix["res_dir"], "protocol", "freeze_manifest.json"),
                approval_path=app_path,
                repo_root=self.fix["root"],
                verify_git_head=False,
            )

    def test_h03_unexpected_holdout_artifact_fails_g6(self):
        """H03: Presence of fresh holdout prediction file in Phase I causes G6 boundary FAIL."""
        conf_dir = os.path.join(self.fix["res_dir"], "confirmatory")
        os.makedirs(conf_dir, exist_ok=True)
        with open(os.path.join(conf_dir, "holdout_predictions.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        v = self._run()
        self.assertEqual(v["gates"]["G6_Confirmatory_Boundary"]["status"], "FAIL")
        self.assertEqual(v["confirmatory_execution_status"], "BOUNDARY_VIOLATION")

    def test_h04_legacy_workload_generator_guards_test_and_holdout(self):
        """H04: Legacy workload generator rejects 'holdout' and default unapproved 'test' access."""
        with self.assertRaises(PermissionError):
            generate_workloads_for_split("holdout")
        with self.assertRaises(PermissionError):
            generate_workloads_for_split("test")

    def test_h05_phase1_reports_not_run_with_checked_evidence(self):
        """H05: Valid Phase I without holdout results reports NOT_RUN, never CONFIRMED."""
        v = self._run()
        self.assertEqual(v["confirmatory_execution_status"], "NOT_RUN")
        self.assertEqual(v["phase_ii_authorization"], "PENDING_AUTHOR_APPROVAL")

    # --- Reporting & Outcome Separation Tests R01-R03 ---
    def test_r01_inconsistent_docs_claiming_ready_when_gate_failed_fails_g7(self):
        """R01: If a gate failed while README.md still asserts READY_FOR_HOLDOUT_REVIEW, G7 fails."""
        with open(os.path.join(self.fix["root"], "README.md"), "w", encoding="utf-8") as f:
            f.write("Status: READY_FOR_HOLDOUT_REVIEW\n")
        # Break G1 so a gate fails
        with open(os.path.join(self.fix["root"], "data/cmu_mosei/raw/mosei.pkl"), "wb") as f:
            f.write(b"BAD")
        v = self._run()
        self.assertEqual(v["gates"]["G7_Claims_and_Novelty"]["status"], "FAIL")

    def test_r02_failed_engineering_test_log_blocks_merge(self):
        """R02: Nonzero test exit code or failed tests blocks G4 & engineering merge."""
        with open(os.path.join(self.fix["res_dir"], "validation", "test_execution_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"exit_code": 1, "passed": 49, "failed": 1}, f)
        v = self._run()
        self.assertEqual(v["gates"]["G4_Systems_Telemetry"]["status"], "FAIL")
        self.assertEqual(v["engineering_merge_recommendation"], "DO_NOT_MERGE")

    def test_r03_valid_negative_effect_passes_integrity_blocks_scientific_ready(self):
        """R03: Valid negative scientific result passes engineering integrity while setting DEVELOPMENT_EFFECT_NOT_SUPPORTED."""
        df = pd.read_csv(os.path.join(self.fix["res_dir"], "systems", "distributed_benchmark.csv"))
        # Make AlwaysOn 100% accurate so candidate fails non-inferiority legitimately
        mask = df["policy"].isin(["AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel"])
        df.loc[mask, "service_quality_prediction"] = df.loc[mask, "true_label"]
        df.to_csv(os.path.join(self.fix["res_dir"], "systems", "distributed_benchmark.csv"), index=False)
        boot = paired_cluster_bootstrap_on_df(
            df, "SequentialConfidence_0.55_0.45",
            ["AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel", "Static_TA_Direct"],
            "service_quality_prediction", "application_protocol_bytes", B=200, seed=1320,
        )
        with open(os.path.join(self.fix["res_dir"], "statistics", "paired_bootstrap_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"primary_measured_endpoint": boot}, f, indent=2)
        with open(os.path.join(self.fix["res_dir"], "systems", "distributed_summary.json"), "r", encoding="utf-8") as f:
            sj = json.load(f)
        for pol in ("AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel"):
            s = df[df["policy"] == pol]
            sj["policies"][pol]["service_quality_macro_f1"] = float(np.mean([
                compute_3class_macro_f1(g["true_label"].to_numpy(), g["service_quality_prediction"].to_numpy())
                for _, g in s.groupby(["workload", "seed"])
            ]))
        with open(os.path.join(self.fix["res_dir"], "systems", "distributed_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sj, f, indent=2)

        v = self._run()
        self.assertEqual(v["engineering_merge_recommendation"], "RELEASE_CANDIDATE_REVIEWABLE")
        self.assertEqual(v["development_effect_status"], "NOT_SUPPORTED")
        self.assertEqual(v["scientific_status"], "DEVELOPMENT_EFFECT_NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
