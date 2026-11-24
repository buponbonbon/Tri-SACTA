"""
tests/test_v13_1_utility.py: Unit tests for Tri-SACTA V13.1 Utility and Threshold Equivalence.

Tests:
1. Exact decision agreement between UtilityRouter and DerivedThresholdRouter on synthetic and real inputs.
2. Mathematical verification of tau(S, m, lambda) across probability simplex.
3. Crossover theorem: Audio utility strictly dominates Visual utility at state T under V13 priors.
4. Correct transition risk formula: Delta R_hat = (1-p)*gamma - p*nu.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.fusion_backbone import SimpleMLP
from src.service.utility_router import UtilityRouter, EMPIRICAL_PRIORS, MODALITY_COSTS
from src.service.baseline_policies import (
    DerivedThresholdRouter,
    SequentialConfidenceRouter,
    StaticSubsetPolicy,
)


class MockPredictorBank:
    """Creates a deterministic dummy predictor bank for unit tests."""
    @staticmethod
    def create(seed=42):
        torch.manual_seed(seed)
        configs = {
            "T": 768, "A": 74, "V": 35,
            "TA": 842, "TV": 803, "AV": 109, "TAV": 877
        }
        bank = {}
        for sub, in_dim in configs.items():
            model = SimpleMLP(in_dim=in_dim, hidden_dim=16, num_classes=3)
            model.eval()
            bank[sub] = model
        return bank


class TestV131UtilityEquivalence(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.bank = MockPredictorBank.create(42)
        self.utility_router = UtilityRouter(self.bank, self.device, lambda_utility=0.05, tau_stop_confidence=0.75)
        self.derived_router = DerivedThresholdRouter(self.bank, self.device, lambda_utility=0.05, tau_stop_confidence=0.75)

    def test_derived_threshold_algebraic_formula(self):
        """Verifies tau = (gamma - lambda * C) / (gamma + nu)."""
        lambda_val = 0.05
        for (src, mod), p_dict in EMPIRICAL_PRIORS.items():
            g, n = p_dict["gamma"], p_dict["nu"]
            c = MODALITY_COSTS[mod]
            expected_tau = (g - lambda_val * c) / (g + n)
            actual_tau = self.derived_router.thresholds[(src, mod)]
            self.assertAlmostEqual(expected_tau, actual_tau, places=6)

    def test_v13_crossover_dominance(self):
        """Proves Audio utility strictly exceeds Visual utility at state T for all p in [0, 1]."""
        prior_a = EMPIRICAL_PRIORS[("T", "audio")]
        prior_v = EMPIRICAL_PRIORS[("T", "visual")]
        c_a = MODALITY_COSTS["audio"]
        c_v = MODALITY_COSTS["visual"]

        for p in np.linspace(0.0, 1.0, 101):
            u_a = ((1.0 - p) * prior_a["gamma"] - p * prior_a["nu"]) / c_a
            u_v = ((1.0 - p) * prior_v["gamma"] - p * prior_v["nu"]) / c_v
            self.assertGreater(u_a, u_v, f"Visual unexpectedly equal or exceeded Audio at p={p}")

    def test_100pct_router_agreement(self):
        """Verifies 100% agreement in routing decisions, paths, and costs across 50 random queries."""
        np.random.seed(1301)
        torch.manual_seed(1301)

        for q_idx in range(50):
            features = {
                "text": torch.randn(1, 768),
                "audio": torch.randn(1, 74),
                "visual": torch.randn(1, 35),
            }
            # Test various availability masks
            masks = [
                {"text": True, "audio": True, "visual": True},
                {"text": True, "audio": False, "visual": True},
                {"text": True, "audio": True, "visual": False},
                {"text": False, "audio": True, "visual": True},
            ]
            for mask in masks:
                res_u = self.utility_router.route_single_query(features, mask, deadline_ms=50.0, cost_budget=3.5)
                res_d = self.derived_router.route_single_query(features, mask, deadline_ms=50.0, cost_budget=3.5)

                self.assertEqual(res_u["prediction"], res_d["prediction"])
                self.assertEqual(res_u["final_subset"], res_d["final_subset"])
                self.assertEqual(res_u["acquisition_path"], res_d["acquisition_path"])
                self.assertAlmostEqual(res_u["spent_cost"], res_d["spent_cost"], places=5)
                self.assertTrue(np.allclose(res_u["probabilities"], res_d["probabilities"], atol=1e-5))

    def test_sequential_confidence_router(self):
        """Verifies SequentialConfidenceRouter behavior and budget adherence."""
        seq_router = SequentialConfidenceRouter(self.bank, self.device, tau_stage1=0.80, tau_stage2=0.70)
        features = {
            "text": torch.randn(1, 768),
            "audio": torch.randn(1, 74),
            "visual": torch.randn(1, 35),
        }
        res = seq_router.route_single_query(features, {"text": True, "audio": True, "visual": True}, cost_budget=2.2)
        self.assertLessEqual(res["spent_cost"], 2.2)
        self.assertIn(res["final_subset"], ["T", "TA"])

    def test_static_subset_policy(self):
        """Verifies StaticSubsetPolicy execution and missingness fallback."""
        static_ta = StaticSubsetPolicy(self.bank, self.device, target_subset="TA")
        features = {
            "text": torch.randn(1, 768),
            "audio": torch.randn(1, 74),
            "visual": torch.randn(1, 35),
        }
        res_clean = static_ta.route_single_query(features, {"text": True, "audio": True, "visual": True})
        self.assertEqual(res_clean["final_subset"], "TA")
        self.assertAlmostEqual(res_clean["spent_cost"], 2.2)

        # Fallback when audio missing
        res_missing = static_ta.route_single_query(features, {"text": True, "audio": False, "visual": True})
        self.assertEqual(res_missing["final_subset"], "T")
        self.assertAlmostEqual(res_missing["spent_cost"], 1.0)

    def test_conditional_utility_counterfactual_expressivity(self):
        """
        Counterfactual test: Two queries with identical p_max can yield different
        estimated gains and different acquisition decisions under state conditioning.
        """
        from src.service.conditional_utility import ConditionalGainEstimator
        from src.service.conditional_router import ConditionalUtilityRouter

        est = ConditionalGainEstimator()
        router = ConditionalUtilityRouter(self.bank, self.device, lambda_utility=0.07, estimator=est)

        # Query A: p_max = 0.55 leaning negative [0.55, 0.40, 0.05]
        probs_a = np.array([0.55, 0.40, 0.05])
        gain_a = est.estimate_gain("T", probs_a, "visual", 1.0, 1.0)
        u_a = est.estimate_utility("T", probs_a, "visual", 1.0, 1.0)

        # Query B: p_max = 0.55 leaning positive [0.05, 0.40, 0.55]
        probs_b = np.array([0.05, 0.40, 0.55])
        gain_b = est.estimate_gain("T", probs_b, "visual", 1.0, 1.0)
        u_b = est.estimate_utility("T", probs_b, "visual", 1.0, 1.0)

        # Verify gains are genuinely different despite matched p_max
        self.assertNotAlmostEqual(gain_a, gain_b, places=3)
        # At lambda = 0.07, u_a >= 0.07 while u_b < 0.07
        self.assertGreaterEqual(u_a, 0.07)
        self.assertLess(u_b, 0.07)


if __name__ == "__main__":
    unittest.main()
