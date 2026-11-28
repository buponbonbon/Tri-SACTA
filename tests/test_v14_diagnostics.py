"""
tests/test_v14_diagnostics.py: Regression tests for V14 development-only diagnostics.
"""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.v14_research.run_v14_diagnostics import run_diagnostics


class TestV14Diagnostics(unittest.TestCase):
    def test_v14_diagnostics_runs_on_dev_without_holdout(self):
        summary = run_diagnostics()
        self.assertEqual(summary["dataset"], "val_development_only")
        self.assertEqual(summary["num_samples"], 250)
        self.assertIn("CLEAN_MULTIMODAL", summary["oracle_headroom_by_workload"])
        self.assertIn("rescue_T_to_TA", summary["group_cross_fitted_learnability"])


if __name__ == "__main__":
    unittest.main()
