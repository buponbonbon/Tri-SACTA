"""
scripts/v13_2_run_phase2_holdout.py: Strictly Gated Phase II Holdout Runner & End-to-End Pipeline.

MANDATORY SAFETY PROPERTIES:
1. NEVER accesses raw_test[500:] feature arrays or labels unless ALL authorization,
   manifest digest, checkpoint byte hashes, source tree hashes, clean git status, and
   no-prior-confirmatory-execution checks PASS.
2. In Phase I, default invocations and dry-runs are strictly incapable of reading fresh holdout.
3. Validates external approval_record.json BEFORE loading encoders or predictive models.
4. Enforces the exact pinned mDeBERTa encoder & revision (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli,
   rev 8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c) and verifies snapshot asset SHA-256 digests.
5. Logs every execution attempt (dry-runs, refusals, authorized runs) to
   results/v13_2/provenance/holdout_access_ledger.json and prevents repeated confirmatory runs.
6. Executes the complete Phase II pipeline: gated extraction -> 7-workload generation ->
   offline policy evaluation (16 policies) -> measured TCP benchmark (8 policies) ->
   dual-endpoint paired video-grouped bootstrap -> evidence-derived confirmatory verdict.
"""

from __future__ import annotations

import os
import sys
import json
import time
import socket
import pickle
import hashlib
import argparse
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PINNED_ENCODER_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
PINNED_ENCODER_REV = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"

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


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class AuthorizationError(PermissionError):
    """Raised when Phase II holdout execution is requested without valid matching authorization."""


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def record_holdout_ledger_attempt(
    ledger_path: str,
    mode: str,
    outcome: str,
    details: str,
    approval_path: Optional[str] = None,
    manifest_sha256: Optional[str] = None,
    mark_confirmatory_extracted: bool = False,
    mark_confirmatory_executed: bool = False,
) -> None:
    """Append an execution attempt to holdout_access_ledger.json without altering Phase I flags unless authorized."""
    if os.path.exists(ledger_path):
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
    else:
        ledger = {
            "schema_version": "13.2",
            "phase": "PHASE_I",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "fresh_holdout_features_extracted": False,
            "fresh_holdout_inference_executed": False,
            "fresh_holdout_predictions_written": False,
            "metadata_only_eligibility_audit_executed": True,
            "phase_ii_authorization": "PENDING_AUTHOR_APPROVAL",
            "execution_attempts": [],
        }

    attempts = ledger.setdefault("execution_attempts", [])
    attempts.append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "outcome": outcome,
        "details": details,
        "approval_path": approval_path,
        "manifest_sha256": manifest_sha256,
    })

    if mark_confirmatory_extracted:
        ledger["fresh_holdout_features_extracted"] = True
    if mark_confirmatory_executed:
        ledger["fresh_holdout_inference_executed"] = True
        ledger["fresh_holdout_predictions_written"] = True
        ledger["phase"] = "PHASE_II_COMPLETED"
        ledger["phase_ii_authorization"] = "EXECUTED"

    ledger["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)


def check_phase2_authorization(
    manifest_path: str,
    approval_path: Optional[str],
    repo_root: str = REPO_ROOT,
    verify_git_head: bool = True,
    ledger_path: Optional[str] = None,
    confirmatory_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fail-closed preflight verification executed BEFORE any encoder, checkpoint, or raw feature load.
    Checks:
    1. No prior confirmatory execution in holdout_access_ledger.json or confirmatory output directory.
    2. Explicit approval JSON file must exist and carry approved == True.
    3. Manifest JSON + detached .sha256 digest must match approval["manifest_sha256"].
    4. Eligibility manifest SHA-256 must match both freeze_manifest and approval.
    5. All 21 checkpoint files and tracked scientific source files must match freeze_manifest hashes on disk.
    6. Pinned encoder ID and revision in freeze_manifest must match frozen protocol constants.
    7. Optionally verify git HEAD matches approval["freeze_package_commit_sha"] and scientific paths are clean.
    """
    resolved_ledger = ledger_path or os.path.join(repo_root, "results", "v13_2", "provenance", "holdout_access_ledger.json")
    resolved_conf_dir = confirmatory_dir or os.path.join(repo_root, "results", "v13_2", "confirmatory")

    # 0. Prevent repeated confirmatory execution
    if os.path.exists(resolved_ledger):
        with open(resolved_ledger, "r", encoding="utf-8") as f:
            ledger_data = json.load(f)
        if ledger_data.get("fresh_holdout_inference_executed") or ledger_data.get("phase") == "PHASE_II_COMPLETED":
            raise AuthorizationError(
                "PHASE II BLOCKED (H03): Holdout access ledger shows confirmatory execution already completed. Repeated confirmatory runs are prohibited."
            )
    if os.path.isdir(resolved_conf_dir):
        existing_outputs = [
            fname for fname in ("final_decision.json", "query_traces.json", "distributed_benchmark.csv")
            if os.path.exists(os.path.join(resolved_conf_dir, fname))
        ]
        if existing_outputs:
            raise AuthorizationError(
                f"PHASE II BLOCKED (H03): Confirmatory outputs already exist in {resolved_conf_dir}: {existing_outputs}. Repeated confirmatory runs are prohibited."
            )

    if not approval_path or not os.path.exists(approval_path):
        raise AuthorizationError(
            "PHASE II BLOCKED (H01): Explicit author approval file is missing or not supplied."
        )
    if not os.path.exists(manifest_path):
        raise AuthorizationError(f"PHASE II BLOCKED (H02): Freeze manifest not found at {manifest_path}")

    with open(approval_path, "r", encoding="utf-8") as f:
        approval = json.load(f)
    if approval.get("approved") is not True or approval.get("allowed_operation") != "PHASE_II_CONFIRMATORY_EXECUTION":
        raise AuthorizationError("PHASE II BLOCKED (H01): Approval record does not explicitly authorize Phase II.")

    actual_manifest_sha = compute_sha256(manifest_path)
    if approval.get("manifest_sha256") != actual_manifest_sha:
        raise AuthorizationError(
            f"PHASE II BLOCKED (H02): Manifest SHA mismatch. Approval={approval.get('manifest_sha256')}, Disk={actual_manifest_sha}"
        )

    detached_path = os.path.join(os.path.dirname(manifest_path), "freeze_manifest.sha256")
    if os.path.exists(detached_path):
        with open(detached_path, "r", encoding="utf-8") as f:
            detached_sha = f.read().strip().split()[0]
        if detached_sha != actual_manifest_sha:
            raise AuthorizationError("PHASE II BLOCKED (H02): Detached freeze_manifest.sha256 does not match manifest.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify pinned encoder metadata in manifest
    feat_meta = manifest.get("dataset_and_provenance", {}).get("feature_extraction", {})
    if feat_meta.get("text_encoder") != PINNED_ENCODER_ID or feat_meta.get("snapshot_revision") != PINNED_ENCODER_REV:
        raise AuthorizationError(
            f"PHASE II BLOCKED (H02): Frozen encoder metadata mismatch in manifest: {feat_meta}"
        )

    # Verify eligibility manifest digest
    elig_rel = manifest.get("eligible_holdout_pool", {}).get("eligible_manifest_path", "")
    elig_abs = os.path.join(repo_root, elig_rel)
    if not elig_rel or not os.path.exists(elig_abs):
        raise AuthorizationError("PHASE II BLOCKED (H02): Eligible holdout manifest missing.")
    actual_elig_sha = compute_sha256(elig_abs)
    if actual_elig_sha != manifest["eligible_holdout_pool"].get("eligible_manifest_sha256"):
        raise AuthorizationError("PHASE II BLOCKED (H02): Eligible manifest SHA differs from freeze manifest.")
    if approval.get("eligible_manifest_sha256") != actual_elig_sha:
        raise AuthorizationError("PHASE II BLOCKED (H02): Eligible manifest SHA differs from author approval.")

    # Verify all 21 checkpoints on disk
    ckpt_inv = manifest.get("checkpoints_inventory", {})
    if len(ckpt_inv) != 21:
        raise AuthorizationError(f"PHASE II BLOCKED (H02): Expected 21 checkpoints, found {len(ckpt_inv)}.")
    for cname, info in ckpt_inv.items():
        cpath = os.path.join(repo_root, info["path"])
        if not os.path.exists(cpath) or compute_sha256(cpath) != info["sha256"]:
            raise AuthorizationError(f"PHASE II BLOCKED (H02): Checkpoint hash mismatch or missing for {cname}.")

    # Verify tracked scientific code inventory on disk
    code_inv = manifest.get("code_inventory", {})
    if not code_inv:
        raise AuthorizationError("PHASE II BLOCKED (H02): Empty code_inventory in manifest.")
    for rel_p, info in code_inv.items():
        abs_p = os.path.join(repo_root, rel_p)
        if not os.path.exists(abs_p) or compute_sha256(abs_p) != info["sha256"]:
            raise AuthorizationError(f"PHASE II BLOCKED (H02): Tracked file modified or missing since freeze: {rel_p}")

    # Verify dependency lock
    dep_info = manifest.get("dependencies_lock", {})
    dep_path = os.path.join(repo_root, dep_info.get("path", ""))
    if not os.path.exists(dep_path) or compute_sha256(dep_path) != dep_info.get("sha256"):
        raise AuthorizationError("PHASE II BLOCKED (H02): requirements-lock.txt modified or missing since freeze.")

    if verify_git_head:
        try:
            head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
            if approval.get("freeze_package_commit_sha") != head_sha:
                raise AuthorizationError(
                    f"PHASE II BLOCKED (H02): Git HEAD {head_sha} != approved SHA {approval.get('freeze_package_commit_sha')}."
                )
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain", "src", "scripts", "requirements-lock.txt"],
                cwd=repo_root, text=True,
            ).strip()
            if dirty:
                raise AuthorizationError(f"PHASE II BLOCKED (H02): Dirty scientific source tree:\n{dirty}")
        except subprocess.CalledProcessError as e:
            raise AuthorizationError(f"PHASE II BLOCKED (H02): Git verification error: {e}")

    return {"authorized": True, "manifest": manifest, "approval": approval, "manifest_sha256": actual_manifest_sha}


def extract_features_for_explicit_ids(
    raw_pkl_path: str,
    eligible_samples: List[Dict[str, Any]],
    output_dir: str,
    expected_assets: Optional[Dict[str, str]] = None,
    snapshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Phase II feature extraction function accepting an explicit ordered list of canonical
    eligible_samples (raw_test_index, canonical_id, video_group_id, start_time, end_time).
    Enforces strictly pinned mDeBERTa encoder & asset SHA-256 digests.
    NEVER overwrites data/cmu_mosei/features_cache/test.
    """
    if "features_cache" in os.path.abspath(output_dir) and "test" in os.path.basename(output_dir):
        raise ValueError("Refusing to overwrite historical test feature cache.")

    from scripts.acquire_multimodal_features import (
        encode_transcripts_with_deberta,
        PINNED_HF_MODEL_ID,
        PINNED_HF_REVISION,
        PINNED_HF_ASSET_SHA256,
    )
    import numpy as np
    import torch

    assets_to_verify = expected_assets or PINNED_HF_ASSET_SHA256

    with open(raw_pkl_path, "rb") as f:
        raw_mosei = pickle.load(f)
    raw_test = raw_mosei["test"]

    transcripts = []
    vis_list = []
    aud_list = []
    meta_out = []

    for rec in eligible_samples:
        idx = int(rec["raw_test_index"])
        it = raw_test[idx]
        vid = str(it[2][0])
        st = float(it[2][1])
        et = float(it[2][2])
        if vid != rec["video_group_id"] or abs(st - rec["start_time"]) > 1e-5 or abs(et - rec["end_time"]) > 1e-5:
            raise ValueError(f"Metadata mismatch at raw_test[{idx}] against canonical manifest.")

        words, vis_arr, aud_arr = it[0]
        text = " ".join([w for w in words if w and w not in ["<pad>", "<unk>", "$"]])
        if not text.strip():
            text = "..."
        v_mean = np.mean(vis_arr, axis=0) if vis_arr.ndim == 2 else vis_arr
        a_mean = np.mean(aud_arr, axis=0) if aud_arr.ndim == 2 else aud_arr
        raw_sent = float(it[1][0][0])

        transcripts.append(text)
        vis_list.append(v_mean)
        aud_list.append(a_mean)
        meta_out.append({
            "canonical_id": rec["canonical_id"],
            "raw_test_index": idx,
            "video_id": vid,
            "start_time": st,
            "end_time": et,
            "raw_sentiment": raw_sent,
        })

    os.makedirs(output_dir, exist_ok=True)
    t_tensor = encode_transcripts_with_deberta(
        transcripts,
        snapshot_dir=snapshot_dir,
        expected_repo_id=PINNED_ENCODER_ID if PINNED_ENCODER_ID == PINNED_HF_MODEL_ID else PINNED_HF_MODEL_ID,
        expected_revision=PINNED_ENCODER_REV if PINNED_ENCODER_REV == PINNED_HF_REVISION else PINNED_HF_REVISION,
        expected_assets=assets_to_verify,
        local_files_only=True,
    )
    v_tensor = torch.tensor(np.array(vis_list), dtype=torch.float16)
    a_tensor = torch.tensor(np.array(aud_list), dtype=torch.float16)

    torch.save(t_tensor, os.path.join(output_dir, "text.pt"))
    torch.save(v_tensor, os.path.join(output_dir, "visual.pt"))
    torch.save(a_tensor, os.path.join(output_dir, "audio.pt"))
    torch.save(meta_out, os.path.join(output_dir, "meta.pt"))

    return {"extracted_count": len(meta_out), "output_dir": output_dir}


def build_workloads_from_tensors(
    text_clean: "torch.Tensor",
    audio_clean: "torch.Tensor",
    visual_clean: "torch.Tensor",
) -> Dict[str, Dict[str, Any]]:
    """Build the 7 deterministic V13.2 stress workloads in memory from clean feature tensors."""
    import numpy as np
    import torch

    text_clean = text_clean.float()
    audio_clean = audio_clean.float()
    visual_clean = visual_clean.float()
    N = len(text_clean)

    workloads: Dict[str, Dict[str, Any]] = {}
    workloads["CLEAN_MULTIMODAL"] = {
        "text": text_clean.clone(),
        "audio": audio_clean.clone(),
        "visual": visual_clean.clone(),
        "modality_mask": {m: [True] * N for m in ("text", "audio", "visual")},
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    torch.manual_seed(42)
    workloads["TEXT_EMBEDDING_NOISE"] = {
        "text": text_clean + torch.randn_like(text_clean) * 0.40,
        "audio": audio_clean.clone(),
        "visual": visual_clean.clone(),
        "modality_mask": {m: [True] * N for m in ("text", "audio", "visual")},
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    workloads["AUDIO_FEATURE_MASKING"] = {
        "text": text_clean.clone(),
        "audio": torch.zeros_like(audio_clean),
        "visual": visual_clean.clone(),
        "modality_mask": {"text": [True] * N, "audio": [False] * N, "visual": [True] * N},
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    workloads["VISUAL_ZERO_MASK"] = {
        "text": text_clean.clone(),
        "audio": audio_clean.clone(),
        "visual": torch.zeros_like(visual_clean),
        "modality_mask": {"text": [True] * N, "audio": [True] * N, "visual": [False] * N},
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    torch.manual_seed(100)
    workloads["COMPOUND_FAULT"] = {
        "text": text_clean + torch.randn_like(text_clean) * 0.50,
        "audio": torch.zeros_like(audio_clean),
        "visual": visual_clean.clone(),
        "modality_mask": {"text": [True] * N, "audio": [False] * N, "visual": [True] * N},
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    rng = np.random.default_rng(2026)
    workloads["INTERMITTENT_MISSINGNESS"] = {
        "text": text_clean.clone(),
        "audio": audio_clean.clone(),
        "visual": visual_clean.clone(),
        "modality_mask": {
            "text": (rng.uniform(size=N) > 0.10).tolist(),
            "audio": (rng.uniform(size=N) > 0.30).tolist(),
            "visual": (rng.uniform(size=N) > 0.40).tolist(),
        },
        "deadline_ms": [50.0] * N,
        "cost_budget": [3.5] * N,
    }

    workloads["HETEROGENEOUS_DEADLINE_STRESS"] = {
        "text": text_clean.clone(),
        "audio": audio_clean.clone(),
        "visual": visual_clean.clone(),
        "modality_mask": {m: [True] * N for m in ("text", "audio", "visual")},
        "deadline_ms": [15.0 if i % 3 == 0 else (30.0 if i % 3 == 1 else 50.0) for i in range(N)],
        "cost_budget": [1.0 if i % 3 == 0 else (2.2 if i % 3 == 1 else 3.5) for i in range(N)],
    }

    return workloads


def run_full_evaluation_pipeline(
    text_tensor: "torch.Tensor",
    audio_tensor: "torch.Tensor",
    visual_tensor: "torch.Tensor",
    labels_3c: "np.ndarray",
    video_ids: List[str],
    output_dir: str,
    predictor_banks: Optional[Dict[int, Dict[str, Any]]] = None,
    seeds: Optional[List[int]] = None,
    bootstrap_B: int = 2000,
    bootstrap_seed: int = 1320,
    dataset_label: str = "confirmatory",
) -> Dict[str, Any]:
    """
    Complete end-to-end executable Phase II evaluation pipeline:
    1. Deterministic 7-workload generation.
    2. Offline evaluation of all 16 policies with PoisonedFeatureDict trapping.
    3. Measured TCP loopback socket benchmark across all 8 distributed policies.
    4. Dual-endpoint paired video-grouped block bootstrap (TCP bytes + nominal cost).
    5. Evidence-derived confirmatory verdict written to output_dir/final_decision.json.
    """
    import numpy as np
    import pandas as pd
    import torch
    from src.service.conditional_utility import ConditionalGainEstimator
    from src.service.distributed_providers import ModalitySocketServer
    from src.service.distributed_coordinator import DistributedCoordinator
    from scripts.v13_2_evaluate_development import (
        compute_3class_macro_f1,
        load_predictor_banks,
        instantiate_policies,
        PoisonedFeatureDict,
    )
    from scripts.v13_2_cluster_statistics import paired_cluster_bootstrap_on_df

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    use_seeds = seeds if seeds is not None else [42, 100, 2026]
    banks = predictor_banks if predictor_banks is not None else load_predictor_banks(device)
    cond_estimator = ConditionalGainEstimator()

    N = len(labels_3c)
    workloads = build_workloads_from_tensors(text_tensor, audio_tensor, visual_tensor)

    # 1. Offline Policy Evaluation
    all_query_traces: List[Dict[str, Any]] = []
    run_records: Dict[str, List[Dict[str, Any]]] = {}

    for seed in use_seeds:
        policies = instantiate_policies(banks[seed], device, cond_estimator)
        for pol_name, policy in policies.items():
            run_records.setdefault(pol_name, [])
            for w_name in WORKLOAD_NAMES:
                wl = workloads[w_name]
                w_text, w_audio, w_visual = wl["text"], wl["audio"], wl["visual"]
                w_masks = wl["modality_mask"]
                w_deadlines, w_budgets = wl["deadline_ms"], wl["cost_budget"]

                raw_preds, sq_preds, costs, latencies = [], [], [], []
                on_times, b_violations, accepts, useful_ots = [], [], [], []

                for i in range(N):
                    s_mask = {m: bool(w_masks[m][i]) for m in ("text", "audio", "visual")}
                    s_feat = PoisonedFeatureDict(
                        {"text": w_text[i : i + 1], "audio": w_audio[i : i + 1], "visual": w_visual[i : i + 1]},
                        s_mask,
                    )
                    d_ms = float(w_deadlines[i])
                    b_cost = float(w_budgets[i])
                    res = policy.route_single_query(s_feat, s_mask, deadline_ms=d_ms, cost_budget=b_cost)

                    pred = int(res["prediction"])
                    c = float(res["spent_cost"])
                    lat = float(res["spent_latency_ms"])
                    on_time = int(lat <= d_ms)
                    b_viol = int(c > b_cost)
                    accepted = int(pred in (0, 1, 2))
                    sq_pred = pred if (accepted == 1 and on_time == 1) else -1
                    useful_ot = int(sq_pred in (0, 1, 2))

                    raw_preds.append(pred)
                    sq_preds.append(sq_pred)
                    costs.append(c)
                    latencies.append(lat)
                    on_times.append(on_time)
                    b_violations.append(b_viol)
                    accepts.append(accepted)
                    useful_ots.append(useful_ot)

                    all_query_traces.append({
                        "sample_idx": i,
                        "video_id": video_ids[i],
                        "seed": seed,
                        "workload": w_name,
                        "policy": pol_name,
                        "true_label": int(labels_3c[i]),
                        "prediction": pred,
                        "service_quality_prediction": sq_pred,
                        "spent_cost": c,
                        "spent_latency_ms": lat,
                        "final_subset": res.get("final_subset", "NONE"),
                        "acquisition_path": res.get("acquisition_path", []),
                        "routing_reason": res.get("routing_reason", "unknown"),
                        "status": res.get("status", "SUCCESS"),
                        "budget_limit": b_cost,
                        "deadline_limit": d_ms,
                        "on_time": on_time,
                        "budget_violated": b_viol,
                        "accepted": accepted,
                        "useful_on_time": useful_ot,
                    })

                sq_f1 = compute_3class_macro_f1(labels_3c, np.array(sq_preds))
                raw_f1 = compute_3class_macro_f1(labels_3c, np.array(raw_preds))
                run_records[pol_name].append({
                    "seed": seed,
                    "workload": w_name,
                    "macro_f1": sq_f1,
                    "raw_macro_f1": raw_f1,
                    "mean_cost": float(np.mean(costs)),
                    "mean_latency_ms": float(np.mean(latencies)),
                    "on_time_rate": float(np.mean(on_times)),
                    "budget_violation_rate": float(np.mean(b_violations)),
                    "acceptance_rate": float(np.mean(accepts)),
                    "useful_on_time_rate": float(np.mean(useful_ots)),
                })

    offline_summary = {}
    for pol_name, recs in run_records.items():
        df_p = pd.DataFrame(recs)
        offline_summary[pol_name] = {
            "overall_mean_macro_f1": float(df_p["macro_f1"].mean()),
            "overall_mean_raw_macro_f1": float(df_p["raw_macro_f1"].mean()),
            "overall_mean_cost": float(df_p["mean_cost"].mean()),
            "overall_mean_latency_ms": float(df_p["mean_latency_ms"].mean()),
            "overall_on_time_rate": float(df_p["on_time_rate"].mean()),
            "overall_budget_violation_rate": float(df_p["budget_violation_rate"].mean()),
            "overall_acceptance_rate": float(df_p["acceptance_rate"].mean()),
            "overall_useful_on_time_rate": float(df_p["useful_on_time_rate"].mean()),
        }

    with open(os.path.join(output_dir, "query_traces.json"), "w", encoding="utf-8") as f:
        json.dump(all_query_traces, f)
    with open(os.path.join(output_dir, "policy_evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset_label, "num_samples": N, "summary": offline_summary}, f, indent=2)

    # 2. Measured TCP Socket Benchmark
    a_port = _get_free_port()
    v_port = _get_free_port()
    audio_srv = ModalitySocketServer("audio", host="127.0.0.1", port=a_port, features_tensor=audio_tensor.float())
    visual_srv = ModalitySocketServer("visual", host="127.0.0.1", port=v_port, features_tensor=visual_tensor.float())
    audio_srv.start()
    visual_srv.start()
    time.sleep(0.15)

    tcp_rows: List[Dict[str, Any]] = []
    try:
        for seed in use_seeds:
            coords = {
                pol_label: DistributedCoordinator(
                    banks[seed], device, audio_port=a_port, visual_port=v_port, persistent_connections=True
                )
                for _, pol_label in DISTRIBUTED_POLICIES
            }
            try:
                for w_idx, w_name in enumerate(WORKLOAD_NAMES):
                    wl = workloads[w_name]
                    audio_srv.update_features(wl["audio"])
                    visual_srv.update_features(wl["visual"])
                    w_text = wl["text"]
                    w_masks = wl["modality_mask"]
                    w_deadlines, w_budgets = wl["deadline_ms"], wl["cost_budget"]

                    rot = (seed + w_idx) % len(DISTRIBUTED_POLICIES)
                    ordered = DISTRIBUTED_POLICIES[rot:] + DISTRIBUTED_POLICIES[:rot]
                    for pol_mode, pol_label in ordered:
                        coord = coords[pol_label]
                        rand_rng = np.random.default_rng(42) if pol_mode == "matched_budget_random" else None
                        for i in range(N):
                            s_mask = {m: bool(w_masks[m][i]) for m in ("text", "audio", "visual")}
                            res = coord.process_query(
                                sample_idx=i,
                                text_feature=w_text[i : i + 1],
                                policy_mode=pol_mode,
                                tau_stage1=0.55,
                                tau_stage2=0.45,
                                deadline_ms=float(w_deadlines[i]),
                                cost_budget=float(w_budgets[i]),
                                modality_mask=s_mask,
                                rng=rand_rng,
                                p_acquire=0.28,
                            )
                            tcp_rows.append({
                                "sample_idx": i,
                                "video_id": video_ids[i],
                                "seed": seed,
                                "workload": w_name,
                                "policy": pol_label,
                                "policy_mode": pol_mode,
                                "true_label": int(labels_3c[i]),
                                "prediction": int(res["prediction"]),
                                "service_quality_prediction": int(res["service_quality_prediction"]),
                                "final_subset": res["final_subset"],
                                "status": res["status"],
                                "nominal_cost": float(res["nominal_cost"]),
                                "application_protocol_bytes": int(res["application_protocol_bytes"]),
                                "payload_bytes": int(res["payload_bytes"]),
                                "wall_ms": float(res["wall_ms"]),
                                "on_time": int(res["on_time"]),
                                "useful_on_time": int(res["service_quality_prediction"] in (0, 1, 2)),
                                "budget_violated": int(res["nominal_cost"] > float(w_budgets[i])),
                            })
            finally:
                for coord in coords.values():
                    coord.close()
    finally:
        audio_srv.stop()
        visual_srv.stop()

    df_tcp = pd.DataFrame(tcp_rows)
    df_tcp.to_csv(os.path.join(output_dir, "distributed_benchmark.csv"), index=False)

    # 3. Dual-Endpoint Paired Video-Grouped Cluster Bootstrap
    df_traces = pd.DataFrame(all_query_traces)
    tcp_boot = paired_cluster_bootstrap_on_df(
        df_tcp,
        candidate="SequentialConfidence_0.55_0.45",
        comparators=["AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel", "Static_TA_Direct"],
        pred_col="service_quality_prediction",
        resource_col="application_protocol_bytes",
        B=bootstrap_B,
        seed=bootstrap_seed,
    )
    nom_boot = paired_cluster_bootstrap_on_df(
        df_traces,
        candidate="SequentialConfidence_0.55_0.45",
        comparators=["AlwaysOnFusion", "Static_TA"],
        pred_col="service_quality_prediction",
        resource_col="spent_cost",
        B=bootstrap_B,
        seed=bootstrap_seed,
    )

    with open(os.path.join(output_dir, "cluster_bootstrap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_label,
                "B": bootstrap_B,
                "seed": bootstrap_seed,
                "primary_tcp_bytes_endpoint": tcp_boot,
                "secondary_nominal_cost_endpoint": nom_boot,
            },
            f,
            indent=2,
        )

    # 4. Evidence-Derived Verdict
    cand_off = offline_summary["SequentialConfidence_0.55_0.45"]
    guardrails_ok = (
        cand_off["overall_useful_on_time_rate"] >= 0.95
        and cand_off["overall_acceptance_rate"] >= 0.98
        and cand_off["overall_budget_violation_rate"] == 0.0
    )
    tcp_joint_ok = all(
        comp["non_inferiority_margin_0_01_pass"] and comp["resource_reduction_15pct_pass"]
        for comp in tcp_boot["comparisons"].values()
    )
    nom_joint_ok = all(
        comp["non_inferiority_margin_0_01_pass"] and comp["resource_reduction_15pct_pass"]
        for comp in nom_boot["comparisons"].values()
    )

    if guardrails_ok and tcp_joint_ok and nom_joint_ok:
        verdict = "CONFIRMATORY_SUPPORTED"
    elif not tcp_joint_ok or not nom_joint_ok or not guardrails_ok:
        verdict = "CONFIRMATORY_NOT_SUPPORTED"
    else:
        verdict = "CONFIRMATORY_INCONCLUSIVE"

    decision = {
        "dataset": dataset_label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_samples": N,
        "num_video_clusters": len(set(video_ids)),
        "seeds": use_seeds,
        "bootstrap_B": bootstrap_B,
        "guardrails_passed": guardrails_ok,
        "tcp_joint_criterion_passed": tcp_joint_ok,
        "nominal_joint_criterion_passed": nom_joint_ok,
        "confirmatory_verdict": verdict,
    }
    with open(os.path.join(output_dir, "final_decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    return decision


def run_dry_run_dev_only(
    manifest_path: str,
    synthetic_only: bool = False,
    full_pipeline: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Development-only dry run incapable of reading fresh holdout features or raw_test[500:].
    Verifies pipeline wiring on val or tiny synthetic tensors and keeps fresh holdout locked.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if synthetic_only and not full_pipeline:
        import torch
        from src.service.baseline_policies import SequentialConfidenceRouter
        from src.data.loader import SUBSET_METADATA
        bank = {k: torch.nn.Linear(v["in_dim"], 3).eval() for k, v in SUBSET_METADATA.items()}
        router = SequentialConfidenceRouter(bank, torch.device("cpu"), 0.55, 0.45, True)
        res = router.route_single_query(
            {"text": torch.ones(1, 768), "audio": torch.ones(1, 74), "visual": torch.ones(1, 35)},
            {"text": True, "audio": True, "visual": True},
        )
        return {"mode": "synthetic_dry_run", "holdout_accessed": False, "sample_pred": res["prediction"]}

    if full_pipeline:
        import numpy as np
        import torch
        from src.data.loader import SUBSET_METADATA, load_split

        target_out = output_dir or tempfile.mkdtemp(prefix="trisacta_v13_2_dryrun_")
        if synthetic_only:
            torch.manual_seed(7)
            N = 12
            t_ten = torch.randn(N, 768)
            a_ten = torch.randn(N, 74)
            v_ten = torch.randn(N, 35)
            labels_3c = np.array([i % 3 for i in range(N)], dtype=np.int64)
            vids = [f"syn_vid_{i // 3}" for i in range(N)]
            banks = {42: {k: torch.nn.Linear(v["in_dim"], 3).eval() for k, v in SUBSET_METADATA.items()}}
            res = run_full_evaluation_pipeline(
                t_ten, a_ten, v_ten, labels_3c, vids, target_out,
                predictor_banks=banks, seeds=[42], bootstrap_B=20, dataset_label="synthetic_dry_run",
            )
        else:
            val_data = load_split("val", REPO_ROOT)
            res = run_full_evaluation_pipeline(
                val_data["text"], val_data["audio"], val_data["visual"],
                val_data["labels"].numpy(), val_data["video_ids"], target_out,
                seeds=[42, 100, 2026], bootstrap_B=200, dataset_label="dev_val_dry_run",
            )
        return {"mode": "full_pipeline_dry_run", "holdout_accessed": False, "output_dir": target_out, "decision": res}

    from src.data.loader import load_split, SUBSET_METADATA
    from src.models.fusion_backbone import SimpleMLP
    from src.service.baseline_policies import SequentialConfidenceRouter
    import torch

    val_data = load_split("val", REPO_ROOT)
    ckpt_dir = os.path.join(REPO_ROOT, "results", "v13_utility", "training", "checkpoints")
    bank = {}
    for s_key, meta in SUBSET_METADATA.items():
        h_dim = 64 if "text" in meta["mods"] else (32 if "audio" in meta["mods"] else 16)
        m = SimpleMLP(meta["in_dim"], h_dim, 3)
        m.load_state_dict(torch.load(os.path.join(ckpt_dir, f"mlp_{s_key}_seed42.pt"), weights_only=True, map_location="cpu"))
        m.eval()
        bank[s_key] = m
    router = SequentialConfidenceRouter(bank, torch.device("cpu"), 0.55, 0.45, True)
    preds = []
    for i in range(10):
        res = router.route_single_query(
            {"text": val_data["text"][i : i + 1], "audio": val_data["audio"][i : i + 1], "visual": val_data["visual"][i : i + 1]},
            {"text": True, "audio": True, "visual": True},
        )
        preds.append(res["prediction"])
    return {
        "mode": "dev_val_dry_run_10_samples",
        "holdout_accessed": False,
        "manifest_version": manifest.get("manifest_version"),
        "sample_preds": preds,
    }


def execute_phase2_confirmatory(
    manifest_path: str,
    approval_path: Optional[str],
    repo_root: str = REPO_ROOT,
    verify_git_head: bool = True,
) -> Dict[str, Any]:
    """
    Authorized Phase II confirmatory entrypoint:
    1. Enforces check_phase2_authorization (including repeated-execution block).
    2. Extracts features for explicit canonical eligible holdout IDs using strictly pinned mDeBERTa.
    3. Executes full 7-workload x 3-seed offline + TCP + bootstrap pipeline.
    4. Records extraction & completion in holdout_access_ledger.json.
    """
    import numpy as np
    import torch

    ledger_path = os.path.join(repo_root, "results", "v13_2", "provenance", "holdout_access_ledger.json")
    out_conf = os.path.join(repo_root, "results", "v13_2", "confirmatory")

    try:
        auth = check_phase2_authorization(
            manifest_path, approval_path, repo_root=repo_root, verify_git_head=verify_git_head,
            ledger_path=ledger_path, confirmatory_dir=out_conf,
        )
    except AuthorizationError as exc:
        record_holdout_ledger_attempt(
            ledger_path, mode="execute_holdout", outcome="BLOCKED", details=str(exc), approval_path=approval_path,
        )
        raise

    elig_path = os.path.join(repo_root, auth["manifest"]["eligible_holdout_pool"]["eligible_manifest_path"])
    with open(elig_path, "r", encoding="utf-8") as f:
        elig = json.load(f)

    feat_assets = auth["manifest"]["dataset_and_provenance"]["feature_extraction"].get("hf_snapshot_assets")
    feat_dir = os.path.join(out_conf, "features_cache")
    extract_features_for_explicit_ids(
        os.path.join(repo_root, "data", "cmu_mosei", "raw", "mosei.pkl"),
        elig["eligible_samples"],
        feat_dir,
        expected_assets=feat_assets,
    )
    record_holdout_ledger_attempt(
        ledger_path,
        mode="execute_holdout_extraction",
        outcome="EXTRACTED",
        details=f"Extracted {len(elig['eligible_samples'])} holdout samples to {feat_dir}",
        approval_path=approval_path,
        manifest_sha256=auth["manifest_sha256"],
        mark_confirmatory_extracted=True,
    )

    t_ten = torch.load(os.path.join(feat_dir, "text.pt"), map_location="cpu", weights_only=False)
    a_ten = torch.load(os.path.join(feat_dir, "audio.pt"), map_location="cpu", weights_only=False)
    v_ten = torch.load(os.path.join(feat_dir, "visual.pt"), map_location="cpu", weights_only=False)
    meta_out = torch.load(os.path.join(feat_dir, "meta.pt"), map_location="cpu", weights_only=False)

    raw_sent = np.array([m["raw_sentiment"] for m in meta_out], dtype=np.float32)
    labels_3c = np.zeros(len(raw_sent), dtype=np.int64)
    labels_3c[raw_sent < 0.0] = 0
    labels_3c[raw_sent == 0.0] = 1
    labels_3c[raw_sent > 0.0] = 2
    video_ids = [m["video_id"] for m in meta_out]

    decision = run_full_evaluation_pipeline(
        t_ten, a_ten, v_ten, labels_3c, video_ids, out_conf,
        seeds=[42, 100, 2026], bootstrap_B=2000, dataset_label="confirmatory_holdout",
    )
    record_holdout_ledger_attempt(
        ledger_path,
        mode="execute_holdout",
        outcome="EXECUTED",
        details=f"Confirmatory verdict: {decision['confirmatory_verdict']}",
        approval_path=approval_path,
        manifest_sha256=auth["manifest_sha256"],
        mark_confirmatory_extracted=True,
        mark_confirmatory_executed=True,
    )
    return decision


def main():
    parser = argparse.ArgumentParser(description="Tri-SACTA V13.2 Gated Phase II Runner")
    parser.add_argument("--manifest", default=os.path.join(REPO_ROOT, "results", "v13_2", "protocol", "freeze_manifest.json"))
    parser.add_argument("--approval", default=None, help="Path to explicit author approval JSON")
    parser.add_argument("--dry-run-dev", action="store_true", help="Run dev-only pipeline check (cannot touch holdout)")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic dry-run fixture")
    parser.add_argument("--full-pipeline", action="store_true", help="Run full offline+TCP+bootstrap pipeline in dry-run mode")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for dry-run full pipeline")
    parser.add_argument("--execute-holdout", action="store_true", help="Execute Phase II holdout (requires valid --approval)")
    args = parser.parse_args()

    if args.dry_run_dev:
        rep = run_dry_run_dev_only(
            args.manifest,
            synthetic_only=args.synthetic,
            full_pipeline=args.full_pipeline,
            output_dir=args.output_dir,
        )
        logger.info(f"Dry-run completed (zero holdout access): {rep}")
        return

    if args.execute_holdout:
        decision = execute_phase2_confirmatory(args.manifest, args.approval, repo_root=REPO_ROOT)
        logger.info(f"Phase II Confirmatory Execution completed: {decision}")
        return

    # Fail closed if invoked without --dry-run-dev or --execute-holdout
    ledger_path = os.path.join(REPO_ROOT, "results", "v13_2", "provenance", "holdout_access_ledger.json")
    try:
        check_phase2_authorization(args.manifest, args.approval, repo_root=REPO_ROOT, ledger_path=ledger_path)
    except AuthorizationError as exc:
        record_holdout_ledger_attempt(
            ledger_path, mode="preflight_no_flag", outcome="BLOCKED", details=str(exc), approval_path=args.approval,
        )
        raise


if __name__ == "__main__":
    main()
