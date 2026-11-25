"""
scripts/v13_2_freeze_protocol.py: Constructs the Complete Immutable Freeze Package (Commit 2).

MUST be executed strictly AFTER committing the immutable Scientific Source Commit (Commit 1),
with zero subsequent edits to tracked source code, checkpoints, or dev/stats/systems evidence.
"""

from __future__ import annotations

import os
import sys
import json
import glob
import hashlib
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PROTO_DIR = os.path.join(REPO_ROOT, "results", "v13_2", "protocol")
os.makedirs(PROTO_DIR, exist_ok=True)


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_freeze_package() -> Dict[str, Any]:
    utc_now = datetime.now(timezone.utc).isoformat()
    sci_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty_sci = subprocess.check_output(
        ["git", "status", "--porcelain", "src", "scripts", "tests", "requirements-lock.txt"],
        cwd=REPO_ROOT, text=True,
    ).strip()
    if dirty_sci:
        raise RuntimeError(
            f"Refusing to freeze: scientific code/tests/lock tree has uncommitted changes:\n{dirty_sci}\n"
            "Commit Scientific Source Commit FIRST."
        )

    with open(os.path.join(REPO_ROOT, "results", "v13_2", "provenance", "provenance_audit.json"), "r", encoding="utf-8") as f:
        prov = json.load(f)
    with open(os.path.join(REPO_ROOT, "results", "v13_2", "provenance", "eligible_holdout_manifest.json"), "r", encoding="utf-8") as f:
        elig = json.load(f)
    with open(os.path.join(REPO_ROOT, "results", "v13_2", "protocol", "checkpoint_hashes.json"), "r", encoding="utf-8") as f:
        ckpt_inv = json.load(f)["checkpoints"]
    with open(os.path.join(REPO_ROOT, "results", "v13_2", "protocol", "environment_lock.json"), "r", encoding="utf-8") as f:
        env_lock = json.load(f)

    # Tracked scientific code inventory
    code_inventory: Dict[str, Dict[str, Any]] = {}
    tracked_patterns = ["src/**/*.py", "scripts/**/*.py", "tests/**/*.py"]
    for pat in tracked_patterns:
        for abs_p in sorted(glob.glob(os.path.join(REPO_ROOT, pat), recursive=True)):
            rel_p = os.path.relpath(abs_p, REPO_ROOT).replace("\\", "/")
            role = "source" if rel_p.startswith("src/") else ("script" if rel_p.startswith("scripts/") else "test")
            code_inventory[rel_p] = {
                "path": rel_p,
                "role": role,
                "size_bytes": os.path.getsize(abs_p),
                "sha256": compute_sha256(abs_p),
            }

    lock_path = os.path.join(REPO_ROOT, "requirements-lock.txt")
    elig_path = os.path.join(REPO_ROOT, "results", "v13_2", "provenance", "eligible_holdout_manifest.json")
    prov_path = os.path.join(REPO_ROOT, "results", "v13_2", "provenance", "provenance_audit.json")
    exp_path = os.path.join(REPO_ROOT, "results", "v13_2", "provenance", "exposure_ledger.json")

    manifest_obj = {
        "manifest_version": "13.2",
        "timestamp_utc": utc_now,
        "audited_base_sha": "1cd239e2b4fc1ef7e5cdd8c213cd71acf5ac022e",
        "scientific_source_commit_sha": sci_head,
        "active_branch": "research/v13.2-final-systems-validation-release-candidate",
        "candidate_policy": {
            "name": "SequentialConfidence_0.55_0.45",
            "class": "SequentialConfidenceRouter",
            "tau_stage1": 0.55,
            "tau_stage2": 0.45,
            "audio_first": True,
            "conformal_secondary_note": "Split conformal calibrated on cal split is secondary only; finite-sample exchangeability breaks under covariate/fault shift (coverage degrades ~75.9%); operational rejections (-1) are not conformal abstentions.",
        },
        "primary_hypothesis": {
            "protocol_amendment": "AMENDED_BEFORE_HOLDOUT_SECTION_3_5",
            "statement": (
                "Under identical task-matched predictor banks and 7 heterogeneous workloads, "
                "SequentialConfidence_0.55_0.45 achieves >= 15.0% reduction in measured application-protocol bytes "
                "per submitted request with non-inferior Service-Quality Macro-F1 (lower 95% clustered CI >= -0.010) "
                "jointly against AlwaysOnFusion (serial & parallel) and Static_TA_Direct."
            ),
            "resource_unit": "application_protocol_bytes",
            "secondary_simulation_resource_unit": "nominal_cost_units",
            "non_inferiority_margin": -0.01,
            "resource_reduction_threshold_pct": 15.0,
            "primary_comparators": ["AlwaysOnFusion", "Static_TA"],
            "tcp_primary_comparators": ["AlwaysOn_Direct_Serial", "AlwaysOn_Direct_Parallel", "Static_TA_Direct"],
            "joint_criterion": "INTERSECTION_ALL_PRIMARY_COMPARATORS_AND_BOTH_ENDPOINTS",
            "guardrails_point_thresholds": {
                "min_useful_on_time_rate": 0.95,
                "min_acceptance_rate": 0.98,
                "max_budget_violation_rate_constrained": 0.0,
            },
        },
        "primary_estimand": {
            "name": "unweighted_mean_per_workload_seed_service_quality_macro_f1",
            "label_set": [0, 1, 2],
            "rejection_and_late_handling": "Prediction=-1 or on_time==0 counts as rejection (FN for true class); raw_macro_f1 reported separately.",
        },
        "secondary_estimand": {"name": "pooled_confusion_macro_f1"},
        "evaluator_grid": {
            "seeds": [42, 100, 2026],
            "workload_families": [
                "CLEAN_MULTIMODAL",
                "TEXT_EMBEDDING_NOISE",
                "AUDIO_FEATURE_MASKING",
                "VISUAL_ZERO_MASK",
                "COMPOUND_FAULT",
                "INTERMITTENT_MISSINGNESS",
                "HETEROGENEOUS_DEADLINE_STRESS",
            ],
            "expected_policies": [
                "SequentialConfidence_0.55_0.45",
                "SequentialConfidence_0.65_0.55",
                "SequentialConfidence_VisualFirst_0.55_0.45",
                "ConfidenceRouting_0.55",
                "ConfidenceRouting_0.65",
                "AlwaysOnFusion",
                "BudgetAdmittedMaxSubset",
                "Static_T",
                "Static_TA",
                "Static_TV",
                "Static_AV",
                "Static_TAV",
                "FixedFallback",
                "MatchedBudgetRandom_1.45",
                "DerivedThreshold_0.05",
                "ConditionalUtility_0.05",
            ],
            "expected_samples_per_cell": 250,
            "deadline_stress_paired_scenarios": [(15.0, 1.0), (30.0, 2.2), (50.0, 3.5)],
            "paired_scenarios_not_cartesian": True,
            "bootstrap": {"B": 2000, "rng_seed": 1320, "ci_method": "two_sided_percentile_2.5_97.5"},
        },
        "dataset_and_provenance": {
            "raw_mosei_pkl_sha256": prov["raw_assets"]["mosei.pkl"]["sha256"],
            "provenance_audit_sha256": compute_sha256(prov_path),
            "exposure_ledger_sha256": compute_sha256(exp_path),
            "target_class_mapping": "raw_sentiment < 0 -> 0 (negative), == 0 -> 1 (neutral), > 0 -> 2 (positive)",
            "cached_labels_pt_note": "7-class legacy labels.pt; loader uses 3-class target from raw_sentiment.",
            "feature_extraction": {
                "text_encoder": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                "snapshot_revision": "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c",
                "audio": "COVAREP_74d_mean_pool_fp16",
                "visual": "FACET_35d_mean_pool_fp16",
            },
        },
        "eligible_holdout_pool": {
            "eligible_manifest_path": "results/v13_2/provenance/eligible_holdout_manifest.json",
            "eligible_manifest_sha256": compute_sha256(elig_path),
            "eligible_utterances": elig["eligible_utterances"],
            "eligible_unique_video_clusters": elig["eligible_unique_video_clusters"],
            "selection_rule": elig["selection_rule"],
            "speaker_overlap_disclosure": elig["speaker_overlap_status"],
        },
        "checkpoints_inventory": ckpt_inv,
        "code_inventory": code_inventory,
        "dependencies_lock": {
            "path": "requirements-lock.txt",
            "sha256": compute_sha256(lock_path),
        },
        "environment_lock_sha256": compute_sha256(os.path.join(PROTO_DIR, "environment_lock.json")),
        "executable_protocol": {
            "metadata_preflight": "python scripts/v13_2_audit_provenance.py",
            "development_evaluation": "python scripts/v13_2_evaluate_development.py",
            "distributed_benchmark": "python scripts/v13_2_benchmark_distributed.py",
            "paired_statistics": "python scripts/v13_2_cluster_statistics.py",
            "gate_verification": "python scripts/v13_2_verify_gates.py",
            "phase_ii_dry_run_dev": "python scripts/v13_2_run_phase2_holdout.py --dry-run-dev",
            "phase_ii_confirmatory": "python scripts/v13_2_run_phase2_holdout.py --execute-holdout --approval <AUTHOR_APPROVAL_JSON>",
        },
        "failure_handling_policy": (
            "Failed model or transport requests are retained as observed failures in full denominators (never dropped). "
            "No selective restarts, no holdout tuning, no threshold updates."
        ),
        "phase_i_contract": {
            "phase": "PHASE_I",
            "phase_ii_authorization": "PENDING_AUTHOR_APPROVAL",
            "boundary_rule": "STOP before extracting features or running inference on eligible holdout.",
        },
    }

    m_path = os.path.join(PROTO_DIR, "freeze_manifest.json")
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, indent=2)
    m_sha = compute_sha256(m_path)

    detached_path = os.path.join(PROTO_DIR, "freeze_manifest.sha256")
    with open(detached_path, "w", encoding="utf-8") as f:
        f.write(f"{m_sha}  freeze_manifest.json\n")

    approval_schema = {
        "schema_version": "13.2",
        "approved": False,
        "allowed_operation": "NONE_PENDING_AUTHOR_APPROVAL",
        "scientific_source_commit_sha": sci_head,
        "freeze_package_commit_sha": "<VERIFY_GITHUB_FREEZE_PACKAGE_COMMIT_SHA>",
        "manifest_sha256": m_sha,
        "eligible_manifest_sha256": manifest_obj["eligible_holdout_pool"]["eligible_manifest_sha256"],
        "instructions": "To authorize Phase II, repository author replaces approved=true, allowed_operation='PHASE_II_CONFIRMATORY_EXECUTION', and sets freeze_package_commit_sha after reviewing on GitHub.",
    }
    with open(os.path.join(PROTO_DIR, "approval_schema.json"), "w", encoding="utf-8") as f:
        json.dump(approval_schema, f, indent=2)

    print(f"Freeze manifest created: {m_path}")
    print(f"Scientific source SHA:   {sci_head}")
    print(f"Freeze manifest SHA-256: {m_sha}")
    print(f"Tracked files locked:    {len(code_inventory)} source/script/test files + {len(ckpt_inv)} checkpoints")
    return manifest_obj


if __name__ == "__main__":
    build_freeze_package()
