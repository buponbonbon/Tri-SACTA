"""
tests/test_v13_2_phase2_pipeline.py: End-to-End Phase II Pipeline, Frozen Encoder Pinning,
Holdout Ledger Recording, and Repeated-Execution Prevention Tests for Tri-SACTA V13.2.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.acquire_multimodal_features import (
    PINNED_HF_MODEL_ID,
    PINNED_HF_REVISION,
    resolve_and_verify_pinned_deberta,
)
from scripts.v13_2_run_phase2_holdout import (
    AuthorizationError,
    check_phase2_authorization,
    record_holdout_ledger_attempt,
    run_dry_run_dev_only,
)


class TestV132Phase2PipelineAndPinning(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="v13_2_p2_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_synthetic_end_to_end_phase2_pipeline(self):
        manifest_path = os.path.join(REPO_ROOT, "results", "v13_2", "protocol", "freeze_manifest.json")
        out_dir = os.path.join(self.tmp_dir, "full_pipe_out")
        res = run_dry_run_dev_only(
            manifest_path,
            synthetic_only=True,
            full_pipeline=True,
            output_dir=out_dir,
        )
        self.assertFalse(res["holdout_accessed"])
        self.assertEqual(res["mode"], "full_pipeline_dry_run")
        for fname in (
            "query_traces.json",
            "policy_evaluation_summary.json",
            "distributed_benchmark.csv",
            "cluster_bootstrap_summary.json",
            "final_decision.json",
        ):
            fpath = os.path.join(out_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing expected pipeline output: {fname}")

        with open(os.path.join(out_dir, "final_decision.json"), "r", encoding="utf-8") as f:
            dec = json.load(f)
        self.assertIn(dec["confirmatory_verdict"], ("CONFIRMATORY_SUPPORTED", "CONFIRMATORY_NOT_SUPPORTED", "CONFIRMATORY_BLOCKED_INCOMPLETE_GRID"))
        self.assertTrue(dec["tcp_grid_complete"])
        self.assertIn("tcp_guardrails_passed", dec)
        self.assertIn("tcp_guardrails_metrics", dec)
        self.assertIn("nominal_joint_criterion_passed", dec)

    def test_frozen_encoder_pinning_refuses_mismatch_and_missing(self):
        with self.assertRaises(RuntimeError):
            resolve_and_verify_pinned_deberta(
                expected_repo_id="microsoft/deberta-v3-base",
                expected_revision=PINNED_HF_REVISION,
            )
        with self.assertRaises(RuntimeError):
            resolve_and_verify_pinned_deberta(
                expected_repo_id=PINNED_HF_MODEL_ID,
                expected_revision="0000000000000000000000000000000000000000",
            )
        with self.assertRaises(RuntimeError):
            resolve_and_verify_pinned_deberta(
                snapshot_dir=os.path.join(self.tmp_dir, "nonexistent_snapshot"),
                local_files_only=True,
            )

    def test_frozen_encoder_pinning_verifies_asset_hashes(self):
        snap = os.path.join(self.tmp_dir, "snap")
        os.makedirs(snap, exist_ok=True)
        asset_path = os.path.join(snap, "config.json")
        with open(asset_path, "wb") as f:
            f.write(b'{"model_type": "deberta-v2"}')
        with self.assertRaises(RuntimeError):
            resolve_and_verify_pinned_deberta(
                snapshot_dir=snap,
                expected_assets={"config.json": "0" * 64},
                local_files_only=True,
            )
        # Also verify dict-format asset metadata (as stored in freeze_manifest.json)
        import hashlib
        real_sha = hashlib.sha256(b'{"model_type": "deberta-v2"}').hexdigest()
        resolved = resolve_and_verify_pinned_deberta(
            snapshot_dir=snap,
            expected_assets={"config.json": {"size_bytes": 28, "sha256": real_sha}},
            local_files_only=True,
        )
        self.assertEqual(resolved, snap)

    def test_ledger_recording_and_repeated_execution_prevention(self):
        ledger_path = os.path.join(self.tmp_dir, "holdout_access_ledger.json")
        conf_dir = os.path.join(self.tmp_dir, "confirmatory")
        os.makedirs(conf_dir, exist_ok=True)

        # Record a blocked attempt -> fresh_holdout_* must stay False
        record_holdout_ledger_attempt(
            ledger_path,
            mode="execute_holdout",
            outcome="BLOCKED",
            details="Missing approval",
        )
        with open(ledger_path, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        self.assertFalse(ledger["fresh_holdout_inference_executed"])
        self.assertEqual(len(ledger["execution_attempts"]), 1)

        # Mark confirmatory started (simulating interrupted/partial run) -> must block repeated run
        record_holdout_ledger_attempt(
            ledger_path,
            mode="execute_holdout_start",
            outcome="STARTED",
            details="Interrupted run test",
            mark_confirmatory_started=True,
        )
        with self.assertRaises(AuthorizationError) as ctx_partial:
            check_phase2_authorization(
                os.path.join(REPO_ROOT, "results", "v13_2", "protocol", "freeze_manifest.json"),
                approval_path=None,
                repo_root=REPO_ROOT,
                verify_git_head=False,
                ledger_path=ledger_path,
                confirmatory_dir=conf_dir,
            )
        self.assertIn("H03", str(ctx_partial.exception))

        # Non-empty confirmatory directory (e.g., partial features_cache) must also block
        partial_ledger = os.path.join(self.tmp_dir, "clean_ledger.json")
        os.makedirs(os.path.join(conf_dir, "features_cache"), exist_ok=True)
        with self.assertRaises(AuthorizationError) as ctx_dir:
            check_phase2_authorization(
                os.path.join(REPO_ROOT, "results", "v13_2", "protocol", "freeze_manifest.json"),
                approval_path=None,
                repo_root=REPO_ROOT,
                verify_git_head=False,
                ledger_path=partial_ledger,
                confirmatory_dir=conf_dir,
            )
        self.assertIn("H03", str(ctx_dir.exception))

        # Mark confirmatory executed -> check_phase2_authorization must refuse repeated run
        record_holdout_ledger_attempt(
            ledger_path,
            mode="execute_holdout",
            outcome="EXECUTED",
            details="Completed",
            mark_confirmatory_extracted=True,
            mark_confirmatory_executed=True,
        )
        with self.assertRaises(AuthorizationError) as ctx:
            check_phase2_authorization(
                os.path.join(REPO_ROOT, "results", "v13_2", "protocol", "freeze_manifest.json"),
                approval_path=None,
                repo_root=REPO_ROOT,
                verify_git_head=False,
                ledger_path=ledger_path,
                confirmatory_dir=conf_dir,
            )
        self.assertIn("H03", str(ctx.exception))


def test_synthetic_end_to_end_phase2_pipeline():
    t = TestV132Phase2PipelineAndPinning()
    t.setUp()
    try:
        t.test_synthetic_end_to_end_phase2_pipeline()
    finally:
        t.tearDown()


def test_frozen_encoder_pinning_refuses_mismatch_and_missing():
    t = TestV132Phase2PipelineAndPinning()
    t.setUp()
    try:
        t.test_frozen_encoder_pinning_refuses_mismatch_and_missing()
    finally:
        t.tearDown()


def test_frozen_encoder_pinning_verifies_asset_hashes():
    t = TestV132Phase2PipelineAndPinning()
    t.setUp()
    try:
        t.test_frozen_encoder_pinning_verifies_asset_hashes()
    finally:
        t.tearDown()


def test_ledger_recording_and_repeated_execution_prevention():
    t = TestV132Phase2PipelineAndPinning()
    t.setUp()
    try:
        t.test_ledger_recording_and_repeated_execution_prevention()
    finally:
        t.tearDown()


if __name__ == "__main__":
    unittest.main()
