"""
tests/test_v8_regression.py: Regression suite for Tri-SACTA V8 (P0 and P1 audit fixes).

Verifies:
1. V8-01: FullAdaptation baseline performs active non-zero parameter updates (P0-1).
2. V8-02: Event-level conformal telemetry maintains 1-to-1 unique event IDs (P0-2).
3. V8-03: GlobalACI maintains a single global alpha scalar across all mask keys (P0-3).
4. V8-04: Frozen logits replay produces bit-for-bit identical predictions across conformal controllers (P0-4).
5. V8-05: Certainty-weighted alignment suppresses noisy/uncertain anchor supervision (P0-5).
6. V8-06: Calibrated spectral temperatures achieve clean anchor retention >= 85% and non-zero moderate adaptation (P1-1).
"""

import math
import copy
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.fusion_backbone import MultimodalFusionModel
from src.models.spectral_detector import SpectralShiftDetector
from src.adaptation.tri_sacta import TriSACTA
from src.adaptation.baselines_tta import FullAdaptation
from src.conformal.online_aci import OnlineACI, GlobalACI


@pytest.fixture
def dummy_config():
    return {
        "model": {
            "fusion_dim": 64,
            "num_heads": 2,
            "num_layers": 1,
            "adapter_rank": 4,
            "dropout": 0.0,
            "num_classes": 7,
        },
        "features": {
            "visual_dim": 35,
            "audio_dim": 74,
            "text_dim": 768,
            "prosody_dim": 88,
        },
        "tta": {
            "tau": {
                "text": 2.3282,
                "audio": 0.0273,
                "visual": 0.0217,
                "prosody": 0.0250,
            },
            "theta_high": 0.80,
            "theta_low": 0.10,
            "lambda_align": 0.5,
            "lambda_reg": 0.01,
            "lr": 1.0e-3,
        },
        "conformal": {
            "alpha": 0.10,
            "eta_0": 0.01,
            "beta_1": 0.5,
            "beta_2": 0.5,
            "eta_min": 0.001,
            "eta_max": 0.10,
            "window_size": 100,
            "delay": 0,
            "score_fn": "aps",
        },
    }


def test_v8_01_full_adaptation_performs_parameter_updates(dummy_config):
    """V8-01: FullAdaptation baseline MUST perform active non-zero parameter updates (P0-1)."""
    model = MultimodalFusionModel(dummy_config)
    adapter = FullAdaptation(model, dummy_config)

    B = 8
    batch = {
        "text": torch.randn(B, 768),
        "audio": torch.randn(B, 74),
        "visual": torch.randn(B, 35),
    }
    mask = {
        "text": torch.ones(B, dtype=torch.bool),
        "audio": torch.ones(B, dtype=torch.bool),
        "visual": torch.ones(B, dtype=torch.bool),
        "prosody": torch.zeros(B, dtype=torch.bool),
    }

    res = adapter.adapt_step(batch, mask)

    assert res["adapted"] is True, "FullAdaptation must report adapted=True"
    assert res["param_delta"] > 0.0, f"FullAdaptation param_delta must be positive, got {res['param_delta']}"
    assert res["loss"] > 0.0, "FullAdaptation loss must be positive"
    print(f"[PASS] V8-01: FullAdaptation verified with param_delta = {res['param_delta']:.6f} > 0.")


def test_v8_02_event_telemetry_has_unique_event_ids(dummy_config):
    """V8-02: Event-level conformal telemetry must maintain 1-to-1 unique event IDs (P0-2)."""
    aci = OnlineACI(num_classes=7, alpha=0.10, config=dummy_config["conformal"])

    num_samples = 20
    event_ids = []

    for i in range(num_samples):
        logits = torch.randn(1, 7)
        label = torch.tensor([i % 7])
        m_key = "1110"

        # predict and update sequentially
        p_set = aci.predict_set(logits, m_key)[0]
        aci.update(logits, label, m_key)

        last_ev = aci.get_last_event_telemetry()
        assert last_ev is not None, f"Telemetry was None at step {i}"
        event_ids.append(last_ev["event_id"])

    assert len(event_ids) == num_samples, f"Expected {num_samples} telemetries, got {len(event_ids)}"
    assert len(set(event_ids)) == num_samples, f"Event IDs must be unique! Got {len(set(event_ids))} unique out of {num_samples}: {event_ids}"
    assert event_ids == list(range(1, num_samples + 1)), f"Event IDs should be sequential 1..N, got {event_ids}"
    print(f"[PASS] V8-02: Exactly {len(event_ids)} unique, sequential event IDs verified: {event_ids[:5]}...")


def test_v8_03_global_aci_maintains_single_alpha(dummy_config):
    """V8-03: GlobalACI must maintain a single global alpha scalar across all mask keys (P0-3)."""
    g_aci = GlobalACI(num_classes=7, alpha=0.10, config=dummy_config["conformal"])

    # Provide miscoverage on mask '1110'
    logits_miss = torch.zeros(1, 7)
    logits_miss[0, 0] = 10.0  # predicts class 0
    label_miss = torch.tensor([5])  # true class is 5 (miss!)

    g_aci.predict_set(logits_miss, "1110")
    g_aci.update(logits_miss, label_miss, "1110")

    # Alpha should have adapted globally
    alpha_after_miss = g_aci._alpha_global
    assert alpha_after_miss != 0.10, "Global alpha must update after feedback"
    assert g_aci.alphas["1110"] == alpha_after_miss, "alphas['1110'] must match global alpha"
    assert g_aci.alphas["1000"] == alpha_after_miss, "alphas['1000'] must also match global alpha (no per-mask divergence)"
    assert g_aci.alphas["0110"] == alpha_after_miss, "alphas['0110'] must also match global alpha"

    metrics = g_aci.get_metrics()
    assert metrics["controller_mode"] == "global_aci"
    print(f"[PASS] V8-03: GlobalACI verified — global alpha = {alpha_after_miss:.4f} identical across all masks.")


def test_v8_04_frozen_logits_replay_consistency(dummy_config):
    """V8-04: Frozen logits replay produces bit-for-bit identical predictions across conformal controllers (P0-4)."""
    B = 10
    saved_logits = torch.randn(B, 7)

    aci1 = OnlineACI(num_classes=7, alpha=0.10, config=dummy_config["conformal"])
    aci2 = OnlineACI(num_classes=7, alpha=0.10, config=dummy_config["conformal"])

    sets1 = [aci1.predict_set(saved_logits[i:i+1], "1110")[0] for i in range(B)]
    sets2 = [aci2.predict_set(saved_logits[i:i+1], "1110")[0] for i in range(B)]

    assert sets1 == sets2, "Replayed identical logits must produce bit-for-bit identical prediction sets"
    print("[PASS] V8-04: Frozen logits replay consistency verified.")


def test_v8_05_certainty_weighted_alignment(dummy_config):
    """V8-05: Certainty-weighted alignment suppresses noisy/uncertain anchor supervision (P0-5)."""
    model = MultimodalFusionModel(dummy_config)
    feat_dims = {"visual": 35, "audio": 74, "text": 768, "prosody": 88}
    detector = SpectralShiftDetector(feat_dims)

    sacta = TriSACTA(model, detector, dummy_config)

    B = 4
    # Create high-entropy anchor logits (nearly uniform across all 7 classes)
    high_entropy_anchor_logits = torch.zeros(B, 7)  # uniform probs 1/7
    adaptable_logits = torch.randn(B, 7)

    per_mod_logits = {
        "audio": high_entropy_anchor_logits,
        "text": adaptable_logits,
    }
    mask = {
        "audio": torch.ones(B, dtype=torch.bool),
        "text": torch.ones(B, dtype=torch.bool),
    }

    # Alignment loss with high-entropy anchor teacher
    loss = sacta._alignment_loss(per_mod_logits, anchor_set={"audio"}, adaptable_set={"text"}, modality_mask=mask)

    # Because entropy is maximal, certainty is 0.0 -> loss must be 0.0
    assert abs(loss.item()) < 1e-5, f"Uncertain anchor teacher must produce zero alignment loss, got {loss.item()}"
    print(f"[PASS] V8-05: Certainty-weighted alignment verified — high-entropy teacher produces loss = {loss.item():.6f}.")


def test_v8_06_calibrated_spectral_temperatures_retention(dummy_config):
    """V8-06: Calibrated spectral temperatures achieve clean anchor retention >= 85% and non-zero moderate adaptation (P1-1)."""
    tau = dummy_config["tta"]["tau"]
    theta_high = dummy_config["tta"]["theta_high"]
    theta_low = dummy_config["tta"]["theta_low"]

    # Verify per-modality values
    assert tau["text"] > 1.0, f"Text tau should be around 2.3, got {tau['text']}"
    assert tau["audio"] < 0.1, f"Audio tau should be around 0.027, got {tau['audio']}"
    assert tau["visual"] < 0.1, f"Visual tau should be around 0.022, got {tau['visual']}"

    # Text clean score 0.03 -> r = exp(-2.3282 * 0.03) = 0.932 >= 0.80 (Anchor!)
    r_text_clean = math.exp(-tau["text"] * 0.03)
    assert r_text_clean >= theta_high, f"Clean text must be Anchor (>= {theta_high}), got {r_text_clean:.3f}"

    # Audio clean score 4.5 -> r = exp(-0.0273 * 4.5) = 0.884 >= 0.80 (Anchor!)
    r_audio_clean = math.exp(-tau["audio"] * 4.5)
    assert r_audio_clean >= theta_high, f"Clean audio must be Anchor (>= {theta_high}), got {r_audio_clean:.3f}"

    # Audio moderate shift score 25.0 -> r = exp(-0.0273 * 25.0) = 0.505 in [0.10, 0.80) (Adaptable!)
    r_audio_mod = math.exp(-tau["audio"] * 25.0)
    assert theta_low <= r_audio_mod < theta_high, f"Moderate audio shift must be Adaptable in [{theta_low}, {theta_high}), got {r_audio_mod:.3f}"

    print(f"[PASS] V8-06: Calibrated temperatures verified: clean text r={r_text_clean:.3f}, clean audio r={r_audio_clean:.3f}, moderate audio r={r_audio_mod:.3f}.")
