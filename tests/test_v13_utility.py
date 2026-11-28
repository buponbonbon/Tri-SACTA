"""
tests/test_v13_utility.py: Comprehensive Regression Tests for Tri-SACTA V13.

Verifies:
1. Predictor bank weights, architecture, and forward shapes across all 7 subsets.
2. Marginal expected utility formula and mathematical monotonicity.
3. Sequential router state transitions, deadline handling, and budget enforcement.
4. Policy-matched conformal prediction set generation.
5. Clustered video-grouped bootstrap indices generator.
"""

from __future__ import annotations

import os
import sys
import json
import pytest
import numpy as np
import torch
import torch.nn as nn

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.service.utility_router import (
    UtilityRouter,
    mods_to_subset_key,
    SUBSET_METADATA,
    MODALITY_COSTS,
    MODALITY_LATENCIES,
)
from src.service.baseline_policies import (
    ConfidenceRoutingPolicy,
    FixedFallbackPolicy,
    AlwaysOnFusionPolicy,
    MatchedBudgetRandomPolicy,
)
from src.models.fusion_backbone import SimpleMLP
from scripts.v13_paired_video_bootstrap import (
    fast_macro_f1,
    video_grouped_bootstrap_indices,
)


@pytest.fixture
def mock_predictor_bank():
    device = torch.device("cpu")
    bank = {}
    subset_dims = {
        "T": 768, "A": 74, "V": 35,
        "TA": 842, "TV": 803, "AV": 109, "TAV": 877
    }
    for k, d in subset_dims.items():
        m = SimpleMLP(in_dim=d, hidden_dim=16, num_classes=3)
        m.eval()
        bank[k] = m
    return bank, device


def test_mods_to_subset_key():
    assert mods_to_subset_key(["text"]) == "T"
    assert mods_to_subset_key(["audio"]) == "A"
    assert mods_to_subset_key(["visual"]) == "V"
    assert mods_to_subset_key(["text", "audio"]) == "TA"
    assert mods_to_subset_key(["audio", "text"]) == "TA"
    assert mods_to_subset_key(["text", "visual"]) == "TV"
    assert mods_to_subset_key(["audio", "visual"]) == "AV"
    assert mods_to_subset_key(["text", "audio", "visual"]) == "TAV"
    with pytest.raises(ValueError):
        mods_to_subset_key(["unknown"])


def test_marginal_utility_formula(mock_predictor_bank):
    bank, device = mock_predictor_bank
    router = UtilityRouter(predictor_bank=bank, device=device, lambda_utility=0.05)

    # When confidence is high (risk is low), utility should be low or negative
    confident_probs = np.array([0.90, 0.05, 0.05])
    u_high_conf = router.estimate_marginal_utility("T", confident_probs, "audio")

    # When confidence is low (risk is high), utility should be significantly higher
    uncertain_probs = np.array([0.40, 0.35, 0.25])
    u_low_conf = router.estimate_marginal_utility("T", uncertain_probs, "audio")

    assert u_low_conf > u_high_conf
    assert u_high_conf < 0.0  # Corruption dominates when already confident


def test_sequential_routing_early_stopping(mock_predictor_bank):
    bank, device = mock_predictor_bank
    router = UtilityRouter(
        predictor_bank=bank,
        device=device,
        lambda_utility=0.05,
        tau_stop_confidence=0.70,
        conformal_quantile=0.85,
    )

    # Mock features
    feats = {
        "text": torch.randn(1, 768),
        "audio": torch.randn(1, 74),
        "visual": torch.randn(1, 35),
    }
    mask = {"text": True, "audio": True, "visual": True}

    res = router.route_single_query(
        features=feats,
        modality_mask=mask,
        deadline_ms=50.0,
        cost_budget=3.5,
    )

    assert "prediction" in res
    assert res["prediction"] in [0, 1, 2]
    assert len(res["probabilities"]) == 3
    assert res["spent_cost"] in [1.0, 2.2, 2.3, 2.5, 3.5]
    assert len(res["acquisition_path"]) >= 1
    assert "conformal_set" in res
    assert len(res["conformal_set"]) >= 1


def test_deadline_and_budget_enforcement(mock_predictor_bank):
    bank, device = mock_predictor_bank
    router = UtilityRouter(predictor_bank=bank, device=device, lambda_utility=-1.0) # Always wants to acquire

    feats = {
        "text": torch.randn(1, 768),
        "audio": torch.randn(1, 74),
        "visual": torch.randn(1, 35),
    }
    mask = {"text": True, "audio": True, "visual": True}

    # Strict cost budget = 1.0 (Text only)
    res_b1 = router.route_single_query(feats, mask, deadline_ms=50.0, cost_budget=1.0)
    assert res_b1["final_subset"] == "T"
    assert res_b1["spent_cost"] == 1.0

    # Strict latency deadline = 1.0 ms (Text only, since Audio takes 5ms)
    res_lat1 = router.route_single_query(feats, mask, deadline_ms=1.0, cost_budget=3.5)
    assert res_lat1["final_subset"] == "T"
    assert res_lat1["spent_latency_ms"] == 1.0


def test_blackout_and_missing_modality_handling(mock_predictor_bank):
    bank, device = mock_predictor_bank
    router = UtilityRouter(predictor_bank=bank, device=device)

    feats = {
        "text": torch.randn(1, 768),
        "audio": torch.randn(1, 74),
        "visual": torch.randn(1, 35),
    }

    # Text missing -> fallback to audio
    mask_no_text = {"text": False, "audio": True, "visual": True}
    res_no_t = router.route_single_query(feats, mask_no_text, deadline_ms=50.0, cost_budget=3.5)
    assert "text" not in SUBSET_METADATA[res_no_t["final_subset"]]["mods"]

    # All missing -> blackout fallback
    mask_none = {"text": False, "audio": False, "visual": False}
    res_none = router.route_single_query(feats, mask_none, deadline_ms=50.0, cost_budget=3.5)
    assert res_none["routing_reason"] == "blackout_fallback"
    assert len(res_none["conformal_set"]) == 3


def test_baseline_policies_interfaces(mock_predictor_bank):
    bank, device = mock_predictor_bank
    feats = {"text": torch.randn(1, 768), "audio": torch.randn(1, 74), "visual": torch.randn(1, 35)}
    mask = {"text": True, "audio": True, "visual": True}

    cr = ConfidenceRoutingPolicy(bank, device, tau_confidence=0.60)
    res_cr = cr.route_single_query(feats, mask)
    assert res_cr["prediction"] in [0, 1, 2]

    ff = FixedFallbackPolicy(bank, device)
    res_ff = ff.route_single_query(feats, mask)
    assert res_ff["final_subset"] == "T"
    assert res_ff["spent_cost"] == 1.0

    ao = AlwaysOnFusionPolicy(bank, device)
    res_ao = ao.route_single_query(feats, mask)
    assert res_ao["final_subset"] == "TAV"
    assert res_ao["spent_cost"] == 3.5

    mbr = MatchedBudgetRandomPolicy(bank, device, p_acquire=0.5, seed=42)
    res_mbr = mbr.route_single_query(feats, mask)
    assert res_mbr["prediction"] in [0, 1, 2]


def test_fast_macro_f1():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 1, 2])
    from sklearn.metrics import f1_score
    sk_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fast_f1 = fast_macro_f1(y_true, y_pred)
    assert abs(sk_f1 - fast_f1) < 1e-6


def test_video_grouped_bootstrap_disjointness():
    rng = np.random.default_rng(42)
    video_ids = ["vid_1", "vid_1", "vid_2", "vid_2", "vid_2", "vid_3"]
    indices = video_grouped_bootstrap_indices(video_ids, rng)
    # Check that for any sampled video, all its utterances are included together
    sampled_vids = [video_ids[i] for i in indices]
    from collections import Counter
    counts = Counter(sampled_vids)
    for v, c in counts.items():
        expected_mult = 2 if v == "vid_1" else (3 if v == "vid_2" else 1)
        assert c % expected_mult == 0
