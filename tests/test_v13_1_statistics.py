"""
tests/test_v13_1_statistics.py: Unit tests for statistical methods and fast Macro-F1.
"""

import os
import sys
import unittest
import numpy as np
from sklearn.metrics import f1_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.v13_1_cluster_statistics import fast_macro_f1


class TestV131Statistics(unittest.TestCase):

    def test_fast_macro_f1_matches_sklearn_exact(self):
        """Verifies fast_macro_f1 produces exact 0.000000 delta against sklearn across random trials."""
        rng = np.random.default_rng(1301)
        for _ in range(20):
            y_true = rng.integers(0, 3, size=250)
            y_pred = rng.integers(0, 3, size=250)

            sk_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            my_f1 = fast_macro_f1(y_true, y_pred, num_classes=3)
            self.assertAlmostEqual(sk_f1, my_f1, places=6)

    def test_video_grouped_indices_preserves_clusters(self):
        """Verifies that all utterances belonging to a video are sampled together."""
        from scripts.v13_1_reachable_oracle import video_grouped_bootstrap_indices

        video_ids = ["vid_A", "vid_A", "vid_B", "vid_B", "vid_B", "vid_C"]
        rng = np.random.default_rng(42)

        indices = video_grouped_bootstrap_indices(video_ids, rng)
        sampled_vids = [video_ids[idx] for idx in indices]

        # Check: for every video in sampled_vids, its count must be a multiple of its original count
        for v, orig_count in [("vid_A", 2), ("vid_B", 3), ("vid_C", 1)]:
            c = sampled_vids.count(v)
            self.assertEqual(c % orig_count, 0, f"Video {v} cluster fractured in bootstrap sample!")


if __name__ == "__main__":
    unittest.main()
