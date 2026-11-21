"""
Unit tests for SpectralShiftDetector (T01 - T04).
"""

import pytest
import torch
import numpy as np
from src.models.spectral_detector import SpectralShiftDetector


def test_spectral_clean_null():
    """T01: Clean holdout test batches should yield small excess shift scores."""
    torch.manual_seed(42)
    dims = {"visual": 512, "audio": 768}
    det = SpectralShiftDetector(feature_dims=dims)

    # Register source with 200 samples
    source_feats = {
        "visual": torch.randn(200, 512),
        "audio": torch.randn(200, 768),
    }
    det.register_source_stats(source_feats)

    # Test clean batches from the same distribution
    clean_batch = {
        "visual": torch.randn(32, 512),
        "audio": torch.randn(32, 768),
    }
    mask = {"visual": torch.ones(32), "audio": torch.ones(32)}
    scores = det.compute_shift_scores(clean_batch, mask)

    # Calibrated excess shift on clean samples should be close to 0 (< 1.5)
    assert scores["visual"] < 1.5, f"Clean visual excess shift too high: {scores['visual']}"
    assert scores["audio"] < 1.5, f"Clean audio excess shift too high: {scores['audio']}"


def test_spectral_corruptions():
    """T02: Distribution shift (mean shift and variance increase) must increase shift score."""
    torch.manual_seed(42)
    dims = {"visual": 512, "audio": 768}
    det = SpectralShiftDetector(feature_dims=dims)

    source_feats = {
        "visual": torch.randn(200, 512),
        "audio": torch.randn(200, 768),
    }
    det.register_source_stats(source_feats)

    clean_batch = {
        "visual": torch.randn(32, 512),
        "audio": torch.randn(32, 768),
    }
    corrupted_batch = {
        "visual": torch.randn(32, 512) * 3.0 + 2.5,
        "audio": torch.randn(32, 768),
    }
    mask = {"visual": torch.ones(32), "audio": torch.ones(32)}

    clean_scores = det.compute_shift_scores(clean_batch, mask)
    corrupt_scores = det.compute_shift_scores(corrupted_batch, mask)

    assert corrupt_scores["visual"] > clean_scores["visual"] + 3.0, (
        f"Corrupted score {corrupt_scores['visual']} not significantly higher than clean {clean_scores['visual']}"
    )


def test_stable_weights_extreme():
    """T03: Extreme shift scores (e.g. 1000.0) must not cause underflow or NaN/Inf."""
    dims = {"visual": 512, "audio": 768}
    det = SpectralShiftDetector(feature_dims=dims)

    extreme_scores = {"visual": 10000.0, "audio": 0.0}
    mask = {"visual": torch.ones(16), "audio": torch.ones(16)}

    weights = det.compute_reliability_weights(extreme_scores, mask, tau=1.0)

    for m in ["visual", "audio"]:
        w = weights[m]
        assert not torch.isnan(w).any(), f"NaN in weights for {m}"
        assert not torch.isinf(w).any(), f"Inf in weights for {m}"
        assert (w >= 0.0).all(), f"Negative weight for {m}"

    # Visual weight should be small but strictly non-negative
    assert weights["visual"].mean().item() < 0.01
    assert weights["audio"].mean().item() > 0.90


def test_detector_excludes_missing():
    """T04: Zero-padded rows corresponding to missing modalities must be excluded from stats."""
    torch.manual_seed(42)
    dims = {"visual": 512}
    det = SpectralShiftDetector(feature_dims=dims)

    source_feats = {"visual": torch.randn(200, 512)}
    det.register_source_stats(source_feats)

    # 16 clean samples + 16 missing samples (all zeros)
    feats = torch.randn(32, 512)
    feats[16:] = 0.0
    mask = {"visual": torch.cat([torch.ones(16), torch.zeros(16)])}

    # Score with mask excluding the zeros
    score_with_mask = det.compute_shift_scores({"visual": feats}, mask)["visual"]

    # Score on just the 16 clean samples directly
    clean_16 = feats[:16]
    score_clean_direct = det.compute_shift_scores({"visual": clean_16}, {"visual": torch.ones(16)})["visual"]

    # The two scores should be identical because missing zeros were excluded!
    assert abs(score_with_mask - score_clean_direct) < 1e-4, (
        f"Masked score {score_with_mask} differs from clean direct {score_clean_direct}!"
    )
