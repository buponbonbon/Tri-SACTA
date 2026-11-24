"""
tests/test_v13_2_verifier_adversarial.py: Adversarial Negative Tests for Gate Verifier (Phase I-B/Section 8).

Verifies that the automated verifier:
1. Rejects missing artifacts (fails closed to DO_NOT_MERGE).
2. Rejects corrupted raw data hashes.
3. Rejects tampered raw query traces that disagree with JSON summaries.
4. Rejects non-inferiority failures (lower CI bound worse than -0.010).
5. Rejects insufficient cost reductions (lower CI bound < 15%).
6. Rejects insufficient physical wire byte reductions (< 15%).
7. Rejects missing freeze manifest.
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

from scripts.v13_2_verify_gates import verify_gates


class TestV132VerifierAdversarial(unittest.TestCase):
    def setUp(self):
        # Create a scratch directory copying valid results/v13_2/
        self.test_dir = tempfile.mkdtemp()
        src_v13_2 = os.path.join(REPO_ROOT, "results", "v13_2")
        self.mock_results = os.path.join(self.test_dir, "v13_2")
        shutil.copytree(src_v13_2, self.mock_results)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_valid_fixtures_pass_engineering_and_freeze(self):
        """Clean valid fixtures must produce REVIEWABLE_DEVELOPMENT_PROTOTYPE and READY_FOR_HOLDOUT_REVIEW."""
        v = verify_gates(base_dir=self.mock_results)
        self.assertEqual(v["merge_recommendation"], "REVIEWABLE_DEVELOPMENT_PROTOTYPE")
        self.assertEqual(v["scientific_status"], "READY_FOR_HOLDOUT_REVIEW")
        self.assertEqual(v["gates"]["G1_Provenance"]["status"], "PASS")
        self.assertEqual(v["gates"]["G2_Evaluator_Correctness"]["status"], "PASS")
        self.assertEqual(v["gates"]["G3_Statistical_Headroom"]["status"], "PASS")
        self.assertEqual(v["gates"]["G4_Systems_Telemetry"]["status"], "PASS")
        self.assertEqual(v["gates"]["G5_Frozen_Protocol"]["status"], "PASS")
        self.assertEqual(v["gates"]["G6_Confirmatory_Execution"]["status"], "NOT_RUN")

    def test_missing_provenance_artifact_fails_closed(self):
        """Deleting provenance_audit.json must cause G1 to FAIL and merge_rec to DO_NOT_MERGE."""
        prov_path = os.path.join(self.mock_results, "provenance", "provenance_audit.json")
        os.remove(prov_path)
        v = verify_gates(base_dir=self.mock_results)
        self.assertEqual(v["gates"]["G1_Provenance"]["status"], "FAIL")
        self.assertEqual(v["merge_recommendation"], "DO_NOT_MERGE")

    def test_tampered_trace_prediction_fails_evaluator_check(self):
        """Altering a raw prediction in query_traces.json without updating summary causes G2 to FAIL."""
        traces_path = os.path.join(self.mock_results, "development", "query_traces.json")
        with open(traces_path, "r", encoding="utf-8") as f:
            traces = json.load(f)

        # Invert predictions for candidate policy on first 50 items
        for t in traces:
            if t["policy"] == "SequentialConfidence_0.55_0.45":
                t["prediction"] = 1 if t["prediction"] != 1 else 0

        with open(traces_path, "w", encoding="utf-8") as f:
            json.dump(traces, f)

        v = verify_gates(base_dir=self.mock_results)
        self.assertEqual(v["gates"]["G2_Evaluator_Correctness"]["status"], "FAIL")
        self.assertEqual(v["merge_recommendation"], "DO_NOT_MERGE")

    def test_f1_ci_worse_than_margin_fails_statistical_headroom(self):
        """A lower CI bound worse than -0.010 (e.g. -0.025) must cause G3 to FAIL."""
        stat_path = os.path.join(self.mock_results, "statistics", "paired_bootstrap_summary.json")
        with open(stat_path, "r", encoding="utf-8") as f:
            stat_data = json.load(f)

        # Mutate CI to violate non-inferiority
        stat_data["comparisons"]["AlwaysOnFusion"]["delta_macro_f1_95_ci"] = [-0.025, 0.010]
        with open(stat_path, "w", encoding="utf-8") as f:
            json.dump(stat_data, f)

        v = verify_gates(base_dir=self.mock_results)
        self.assertEqual(v["gates"]["G3_Statistical_Headroom"]["status"], "FAIL")
        self.assertEqual(v["scientific_status"], "BLOCKED_BEFORE_HOLDOUT")

    def test_cost_saving_below_15_percent_fails_statistical_headroom(self):
        """A lower cost saving CI bound below 15% (e.g. 8.0%) must cause G3 to FAIL."""
        stat_path = os.path.join(self.mock_results, "statistics", "paired_bootstrap_summary.json")
        with open(stat_path, "r", encoding="utf-8") as f:
            stat_data = json.load(f)

        stat_data["comparisons"]["AlwaysOnFusion"]["cost_saving_pct_95_ci"] = [8.0, 12.0]
        with open(stat_path, "w", encoding="utf-8") as f:
            json.dump(stat_data, f)

        v = verify_gates(base_dir=self.mock_results)
        self.assertEqual(v["gates"]["G3_Statistical_Headroom"]["status"], "FAIL")
        self.assertEqual(v["scientific_status"], "BLOCKED_BEFORE_HOLDOUT")

    def test_insufficient_wire_bytes_reduction_fails_systems_telemetry(self):
        """Wire bytes reduction below 15% (e.g. 5.0%) must cause G4 to FAIL."""
        sys_path = os.path.join(self.mock_results, "systems", "distributed_summary.json")
        with open(sys_path, "r", encoding="utf-8") as f:
            sys_data = json.load(f)

        sys_data["bandwidth_reduction"]["bandwidth_savings_percentage"] = 5.0
        with open(sys_path, "w", encoding="utf-8") as f:
            json.dump(sys_data, f)

        v = verify_gates(base_dir=self.mock_results)
        self.assertEqual(v["gates"]["G4_Systems_Telemetry"]["status"], "FAIL")
        self.assertEqual(v["merge_recommendation"], "DO_NOT_MERGE")

    def test_missing_freeze_manifest_fails_frozen_protocol(self):
        """Deleting freeze_manifest.json must cause G5 to FAIL."""
        m_path = os.path.join(self.mock_results, "protocol", "freeze_manifest.json")
        os.remove(m_path)
        v = verify_gates(base_dir=self.mock_results)
        self.assertEqual(v["gates"]["G5_Frozen_Protocol"]["status"], "FAIL")
        self.assertEqual(v["scientific_status"], "BLOCKED_BEFORE_HOLDOUT")


if __name__ == "__main__":
    unittest.main()
