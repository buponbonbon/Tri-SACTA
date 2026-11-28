"""
tests/test_phase3_audit_integrity.py: Regression tests for Stage A artifact integrity & CRLF/LF verification.
"""
import os
import sys
import json
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.publication_audit.verify_phase2_integrity import git_blob_sha1_lf_normalized


class TestPhase3AuditIntegrity(unittest.TestCase):
    def test_crlf_lf_git_blob_match(self):
        csv_path = os.path.join(REPO_ROOT, "results", "v13_2", "confirmatory", "distributed_benchmark.csv")
        if not os.path.exists(csv_path):
            self.skipTest("Confirmatory CSV not present")
        info = git_blob_sha1_lf_normalized(csv_path)
        self.assertEqual(info["lf_normalized_sha256"], "32ed7388af2e25abf2ad73eec1fe3cf064cab4aee423f540440dfec2ef2c4438")
        self.assertEqual(info["computed_git_blob_sha1"], "391f1990789d5faa01d369881f13663f2c897310")


if __name__ == "__main__":
    unittest.main()
