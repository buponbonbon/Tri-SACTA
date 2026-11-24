"""
tests/test_v13_1_provenance.py: Data Provenance, Leakage, and Quarantine Unit Tests.
"""

import os
import sys
import json
import unittest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.v13_1_audit_provenance import compute_sha256


class TestV131Provenance(unittest.TestCase):

    def test_raw_manifest_exists_and_unaltered(self):
        manifest_path = os.path.join(REPO_ROOT, "data", "cmu_mosei", "data_manifest.json")
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.assertEqual(len(manifest["items"]), 2000)

    def test_historical_split_disjointness(self):
        manifest_path = os.path.join(REPO_ROOT, "data", "cmu_mosei", "data_manifest.json")
        with open(manifest_path, "r") as f:
            items = json.load(f)["items"]

        vids_by_split = {}
        for it in items:
            vids_by_split.setdefault(it["split"], set()).add(it["video_id"])

        # Train vs Val vs Cal vs Test must have 0 shared videos
        for s1 in ["train", "val", "cal", "test"]:
            for s2 in ["train", "val", "cal", "test"]:
                if s1 != s2:
                    overlap = vids_by_split[s1].intersection(vids_by_split[s2])
                    self.assertEqual(len(overlap), 0, f"Video leakage found between {s1} and {s2}: {overlap}")

    def test_provenance_audit_artifact_validity(self):
        audit_path = os.path.join(REPO_ROOT, "results", "v13_1", "provenance", "provenance_audit.json")
        self.assertTrue(os.path.exists(audit_path))
        with open(audit_path, "r") as f:
            audit = json.load(f)

        self.assertEqual(audit["gate_g1_verdict"]["status"], "PASS")
        self.assertGreaterEqual(audit["unexposed_pool_audit"]["unexposed_available_utterances"], 4000)
        self.assertEqual(audit["unexposed_pool_audit"]["overlap_with_historical_train_val_cal_test"], 0)


if __name__ == "__main__":
    unittest.main()
