"""
tests/test_v13_2_evaluator_regression.py: Rigorous Regression Suite for Evaluator Correctness (Phase I-B).

Verifies Section 3.3 Requirements:
1. Propagated per-request budgets and deadlines (no hardcoded 3.5).
2. Infeasible initial modality handling (cost < 1.0 or deadline < 1.0) -> reject (-1), 0 cost.
3. Total blackout handling -> reject (-1), 0 cost.
4. Masked feature access trap (MaskedFeatureAccessViolation).
5. 3-class Macro-F1 with reject category (-1) exact parity with sklearn.
6. Budget-admitted max-subset vs Always-on unrestricted violation reporting.
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import SUBSET_METADATA
from src.service.baseline_policies import (
    BasePredictorPolicy,
    MaskedFeatureAccessViolation,
    SequentialConfidenceRouter,
    ConfidenceRoutingPolicy,
    FixedFallbackPolicy,
    AlwaysOnFusionPolicy,
    BudgetAdmittedMaxSubsetPolicy,
    StaticSubsetPolicy,
)
from src.service.conditional_router import ConditionalUtilityRouter


def dummy_predictor_bank() -> dict[str, nn.Module]:
    """Creates lightweight dummy linear classifiers for testing."""
    bank = {}
    for k, v in SUBSET_METADATA.items():
        in_dim = v["in_dim"]
        linear = nn.Linear(in_dim, 3)
        # Initialize with deterministic weights
        nn.init.constant_(linear.weight, 0.01)
        nn.init.constant_(linear.bias, 0.0)
        linear.eval()
        bank[k] = linear
    return bank


def dummy_features() -> dict[str, torch.Tensor]:
    """Creates deterministic dummy feature tensors."""
    return {
        "text": torch.ones(1, 768) * 0.5,
        "audio": torch.ones(1, 74) * 0.3,
        "visual": torch.ones(1, 35) * 0.2,
    }


def compute_3class_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact 3-class macro-F1 with explicit rejection (-1) handled as false negatives."""
    f1s = []
    for c in [0, 1, 2]:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


class TestV132EvaluatorRegression(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.bank = dummy_predictor_bank()
        self.features = dummy_features()
        self.clean_mask = {"text": True, "audio": True, "visual": True}

    def test_infeasible_initial_budget_returns_rejection(self):
        """Budget below text cost (1.0) must reject with -1 and zero cost."""
        router = SequentialConfidenceRouter(self.bank, self.device, tau_stage1=0.6, tau_stage2=0.5)
        res = router.route_single_query(self.features, self.clean_mask, deadline_ms=50.0, cost_budget=0.5)
        self.assertEqual(res["prediction"], -1)
        self.assertEqual(res["spent_cost"], 0.0)
        self.assertEqual(res["status"], "REJECTED_INFEASIBLE")

    def test_infeasible_initial_deadline_returns_rejection(self):
        """Deadline below text latency (1.0 ms) must reject with -1 and zero latency."""
        router = SequentialConfidenceRouter(self.bank, self.device, tau_stage1=0.6, tau_stage2=0.5)
        res = router.route_single_query(self.features, self.clean_mask, deadline_ms=0.5, cost_budget=3.5)
        self.assertEqual(res["prediction"], -1)
        self.assertEqual(res["spent_cost"], 0.0)
        self.assertEqual(res["status"], "REJECTED_INFEASIBLE")

    def test_total_blackout_returns_rejection(self):
        """Zero available modalities must reject with -1 and zero cost."""
        router = SequentialConfidenceRouter(self.bank, self.device)
        blackout_mask = {"text": False, "audio": False, "visual": False}
        res = router.route_single_query(self.features, blackout_mask, deadline_ms=50.0, cost_budget=3.5)
        self.assertEqual(res["prediction"], -1)
        self.assertEqual(res["spent_cost"], 0.0)
        self.assertEqual(res["status"], "REJECTED_BLACKOUT")

    def test_masked_feature_access_trap(self):
        """Attempting to forward pass an unavailable modality must throw MaskedFeatureAccessViolation."""
        base_policy = BasePredictorPolicy(self.bank, self.device)
        # Attempt to predict 'TA' when audio is masked
        mask_no_audio = {"text": True, "audio": False, "visual": True}
        with self.assertRaises(MaskedFeatureAccessViolation):
            base_policy._predict_subset("TA", self.features, modality_mask=mask_no_audio)

    def test_propagated_per_request_budget_enforcement(self):
        """Router must stop at Stage 1 if cost_budget=1.0 even if confidence is low."""
        # High threshold forces escalation if budget allowed
        router = SequentialConfidenceRouter(self.bank, self.device, tau_stage1=0.99, tau_stage2=0.99)
        res = router.route_single_query(self.features, self.clean_mask, deadline_ms=50.0, cost_budget=1.0)
        self.assertEqual(res["final_subset"], "T")
        self.assertLessEqual(res["spent_cost"], 1.0)

    def test_budget_admitted_max_subset_policy(self):
        """BudgetAdmittedMaxSubsetPolicy picks best feasible subset under budget."""
        policy = BudgetAdmittedMaxSubsetPolicy(self.bank, self.device)
        # Under budget 1.0 -> T
        r1 = policy.route_single_query(self.features, self.clean_mask, deadline_ms=50.0, cost_budget=1.0)
        self.assertEqual(r1["final_subset"], "T")
        self.assertEqual(r1["spent_cost"], 1.0)

        # Under budget 2.2 -> TA
        r2 = policy.route_single_query(self.features, self.clean_mask, deadline_ms=50.0, cost_budget=2.2)
        self.assertEqual(r2["final_subset"], "TA")
        self.assertEqual(r2["spent_cost"], 2.2)

        # Under budget 3.5 -> TAV
        r3 = policy.route_single_query(self.features, self.clean_mask, deadline_ms=50.0, cost_budget=3.5)
        self.assertEqual(r3["final_subset"], "TAV")
        self.assertEqual(r3["spent_cost"], 3.5)

    def test_always_on_unrestricted_violation_reporting(self):
        """AlwaysOnFusionPolicy flags budget_violated=True when cost_budget < 3.5."""
        policy = AlwaysOnFusionPolicy(self.bank, self.device)
        res = policy.route_single_query(self.features, self.clean_mask, deadline_ms=50.0, cost_budget=1.0)
        self.assertEqual(res["final_subset"], "TAV")
        self.assertEqual(res["spent_cost"], 3.5)
        self.assertTrue(res["budget_violated"])

    def test_fast_macro_f1_parity_with_sklearn_and_rejections(self):
        """compute_3class_macro_f1 with rejected predictions must match sklearn macro-F1 exactly."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 3, size=200)
        y_pred = rng.integers(-1, 3, size=200)  # Contains -1 rejections

        sk_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0.0)
        custom_f1 = compute_3class_macro_f1(y_true, y_pred)

        self.assertAlmostEqual(sk_f1, custom_f1, places=7)

    def test_cluster_bootstrap_cross_check_against_bruteforce_sklearn(self):
        """Paired linear-sufficiency cluster bootstrap matches brute-force sklearn resampling on a small fixture."""
        import pandas as pd
        from scripts.v13_2_cluster_statistics import paired_cluster_bootstrap_on_df

        # Hand-constructed small fixture: 3 video clusters, 2 workloads, 1 seed, 2 policies
        rows = []
        data = [
            ("v1", 0, 0, 0, 100.0, 200.0),
            ("v1", 1, 1, 0, 100.0, 200.0),
            ("v2", 2, 2, 2, 100.0, 200.0),
            ("v2", 0, -1, 0, 0.0, 200.0),
            ("v3", 1, 1, 1, 100.0, 200.0),
            ("v3", 2, 2, 1, 100.0, 200.0),
        ]
        for w in ("W1", "W2"):
            for i, (vid, yt, pc, pb, bc, bb) in enumerate(data):
                rows.append({"video_id": vid, "workload": w, "seed": 42, "policy": "Cand", "true_label": yt, "sq": pc, "bytes": bc})
                rows.append({"video_id": vid, "workload": w, "seed": 42, "policy": "Base", "true_label": yt, "sq": pb, "bytes": bb})
        df = pd.DataFrame(rows)

        res = paired_cluster_bootstrap_on_df(df, "Cand", ["Base"], "sq", "bytes", B=200, seed=1320)

        # Brute-force reference using exact same Multinomial draws + sklearn f1_score
        clusters = sorted(df["video_id"].unique().tolist())
        rng = np.random.default_rng(1320)
        w_res = rng.multinomial(len(clusters), [1.0 / len(clusters)] * len(clusters), size=200)
        bf_deltas = []
        for b in range(200):
            counts = w_res[b]
            sub_rows_c = []
            sub_rows_b = []
            for k_idx, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                vid = clusters[k_idx]
                for _ in range(int(cnt)):
                    sub_rows_c.append(df[(df["video_id"] == vid) & (df["policy"] == "Cand")])
                    sub_rows_b.append(df[(df["video_id"] == vid) & (df["policy"] == "Base")])
            dfc = pd.concat(sub_rows_c)
            dfb = pd.concat(sub_rows_b)
            f1_c = np.mean([f1_score(g["true_label"], g["sq"], labels=[0, 1, 2], average="macro", zero_division=0.0) for _, g in dfc.groupby("workload")])
            f1_b = np.mean([f1_score(g["true_label"], g["sq"], labels=[0, 1, 2], average="macro", zero_division=0.0) for _, g in dfb.groupby("workload")])
            bf_deltas.append(float(f1_c - f1_b))

        bf_ci = [float(np.percentile(bf_deltas, 2.5)), float(np.percentile(bf_deltas, 97.5))]
        lin_ci = res["comparisons"]["Base"]["delta_macro_f1_95_ci"]
        self.assertAlmostEqual(bf_ci[0], lin_ci[0], places=6)
        self.assertAlmostEqual(bf_ci[1], lin_ci[1], places=6)


if __name__ == "__main__":
    unittest.main()
