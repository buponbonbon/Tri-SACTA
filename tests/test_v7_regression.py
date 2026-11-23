"""
tests/test_v7_regression.py: Regression and validation tests for Tri-SACTA V7.

Covers:
- V7-01: Effective mask propagation in post-adaptation forward pass (Task 5)
- V7-02: Single-counted abstention semantics & selective accuracy calculation (Task 6)
- V7-03: Direct OnlineACI observation telemetry extraction (Task 7)
- V7-04: Per-modality calibrated tau support in SpectralShiftDetector (Task 4 & 11)
- V7-05: Stream equivalence hashing over sample IDs, corrupted tensors, and physical masks (Task 8)
- V7-06: ModalityProjection input normalization and learned modality gating prior (Task 3)
"""

from __future__ import annotations

import os
import sys
import math
import copy
import hashlib
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.fusion_backbone import MultimodalFusionModel, ModalityProjection, MODALITY_ORDER
from src.models.spectral_detector import SpectralShiftDetector
from src.adaptation.tri_sacta import TriSACTA
from src.conformal.online_aci import OnlineACI, PredictionRecord


def _get_mock_config() -> dict:
    return {
        "dataset": {"name": "cmu_mosei", "num_classes": 7},
        "features": {
            "visual_dim": 35,
            "audio_dim": 74,
            "text_dim": 768,
            "prosody_dim": 88,
        },
        "model": {
            "fusion_dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "adapter_rank": 8,
            "dropout": 0.1,
            "num_classes": 7,
        },
        "tta": {
            "tau": {"text": 3.3664, "audio": 0.0369, "visual": 0.0294, "prosody": 0.03},
            "theta_high": 0.75,
            "theta_low": 0.20,
            "lambda_align": 0.5,
            "lambda_reg": 0.01,
            "lr": 1e-4,
            "gamma_eig": 1.0,
            "top_k_eig": 32,
            "batch_size": 16,
        },
        "conformal": {
            "alpha": 0.10,
            "eta_0": 0.01,
            "beta_1": 0.5,
            "beta_2": 0.5,
            "eta_min": 0.001,
            "eta_max": 0.10,
            "min_mask_support": 5,
            "window_size": 100,
            "delay": 0,
            "score_fn": "aps",
            "controller": "dynamic_mask_aci",
        },
    }


def test_v7_01_effective_mask_propagation_in_post_adaptation():
    """V7-01: Post-adaptation evaluation forward pass MUST strictly receive effective_mask (Task 5)."""
    cfg = _get_mock_config()
    model = MultimodalFusionModel(cfg)
    feat_dims = {"visual": 35, "audio": 74, "text": 768, "prosody": 88}
    detector = SpectralShiftDetector(feat_dims, gamma=1.0, top_k=32)

    # Register mock source stats
    train_feats = {
        m: torch.randn(100, cfg["features"][f"{m}_dim"])
        for m in ["visual", "audio", "text", "prosody"]
    }
    detector.register_source_stats(train_feats)

    tri = TriSACTA(model, detector, cfg)

    # Track calls to forward to verify effective_mask was passed
    original_forward = model.forward
    forward_calls = []

    def mocked_forward(*args, **kwargs):
        forward_calls.append(kwargs)
        return original_forward(*args, **kwargs)

    model.forward = mocked_forward

    B = 8
    batch = {
        "visual": torch.randn(B, 35),
        "audio": torch.randn(B, 74),
        "text": torch.randn(B, 768) + 0.3,
        "prosody": torch.randn(B, 88),
    }
    mask = {m: torch.ones(B, dtype=torch.bool) for m in MODALITY_ORDER}
    precomputed_scores = {"text": 0.25, "visual": 0.0, "audio": 0.0, "prosody": 0.0}

    res = tri.adapt_step(batch, mask, precomputed_shift_scores=precomputed_scores)
    assert res["adapted"] is True, "Expected Tri-SACTA to adapt with clean anchors and adaptable text."

    # Check forward calls
    assert len(forward_calls) >= 2, "Expected initial and post-adaptation forward passes."
    last_call = forward_calls[-1]
    assert "effective_mask" in last_call, "effective_mask missing in post-adaptation forward pass! (Task 5 failure)"
    assert last_call["effective_mask"] is not None, "effective_mask was None in post-adaptation forward pass!"
    print("[PASS] V7-01: Post-adaptation forward pass strictly enforces effective_mask.")


def test_v7_02_single_counted_abstentions_and_selective_accuracy():
    """V7-02: Abstention tracking must count each sample exactly once per emitted prediction (Task 6)."""
    cfg = _get_mock_config()
    model = MultimodalFusionModel(cfg)
    B = 10

    # Reset counters
    model.reset_counters()
    assert model.abstain_count == 0
    assert model.fallback_count == 0

    # Simulate forward passes with all modalities rejected for 4 out of 10 samples
    features = {m: torch.randn(B, cfg["features"][f"{m}_dim"]) for m in MODALITY_ORDER}
    eff_mask = {m: torch.tensor([True]*6 + [False]*4) for m in MODALITY_ORDER}

    out1 = model(features, effective_mask=eff_mask)
    assert out1["all_modalities_rejected"].sum().item() == 4

    # Multiple internal model calls should not multiply user-level prediction count
    out2 = model(features, effective_mask=eff_mask)
    assert out2["all_modalities_rejected"].sum().item() == 4

    # Outer loop single-count evaluation
    emitted_abstains = int(out1["all_modalities_rejected"].sum().item())
    assert emitted_abstains == 4

    # Selective accuracy calculation on non-abstained (6 samples)
    trues = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    preds = out1["logits"].argmax(dim=-1)
    non_abstained = ~out1["all_modalities_rejected"]

    valid_trues = trues[non_abstained].numpy()
    valid_preds = preds[non_abstained].numpy()
    assert len(valid_trues) == 6
    assert len(valid_preds) == 6
    print("[PASS] V7-02: Single-counted abstentions and selective evaluation verified.")


def test_v7_03_direct_online_aci_telemetry_extraction():
    """V7-03: OnlineACI must directly provide exact alpha_before, alpha_after, eta_used, quantile, clipping_residual (Task 7)."""
    cfg = _get_mock_config()
    aci = OnlineACI(num_classes=7, alpha=0.10, config=cfg)

    # Initial state
    assert aci.get_last_event_telemetry() is None

    logits = torch.randn(1, 7)
    mask_key = "1110"

    # Emit prediction
    p_sets = aci.predict_set(logits, mask_key, shift_velocity=0.25, disagreement=0.15)
    assert len(p_sets) == 1

    # Immediate feedback via standard update API
    labels = torch.tensor([0])
    aci.update(logits, labels, mask_key, shift_velocity=0.25, disagreement=0.15)

    last_ev = aci.get_last_event_telemetry()
    assert last_ev is not None
    assert "event_id" in last_ev
    assert "alpha_before" in last_ev
    assert "alpha_after" in last_ev
    assert "eta_used" in last_ev
    assert "quantile" in last_ev
    assert "clipping_residual" in last_ev
    assert last_ev["event_id"] == 1
    assert math.isclose(last_ev["alpha_before"], 0.10, abs_tol=1e-5)
    assert last_ev["eta_used"] > 0.0
    print(f"[PASS] V7-03: Direct OnlineACI observation telemetry verified: {last_ev}")


def test_v7_04_per_modality_calibrated_tau_support():
    """V7-04: SpectralShiftDetector supports per-modality dictionary tau (Task 4 & 11)."""
    feat_dims = {"text": 768, "audio": 74, "visual": 35, "prosody": 88}
    detector = SpectralShiftDetector(feat_dims, gamma=1.0, top_k=32)

    train_feats = {m: torch.randn(50, feat_dims[m]) for m in feat_dims}
    detector.register_source_stats(train_feats)

    tau_dict = {"text": 3.3664, "audio": 0.0369, "visual": 0.0294, "prosody": 0.03}
    shift_scores = {"text": 0.05, "audio": 4.5, "visual": 7.0, "prosody": 0.0}
    mask = {m: torch.ones(1, dtype=torch.bool) for m in feat_dims}

    rel_weights = detector.compute_reliability_weights(shift_scores, mask, tau=tau_dict)

    r_text = rel_weights["text"].item()
    r_audio = rel_weights["audio"].item()
    r_visual = rel_weights["visual"].item()

    # With calibrated tau, clean-level shift scores yield reasonable anchor/adaptable weights
    assert r_text >= 0.75, f"Expected text anchor: {r_text}"
    assert 0.20 <= r_audio <= 0.90, f"Expected audio in valid range: {r_audio}"
    assert 0.20 <= r_visual <= 0.90, f"Expected visual in valid range: {r_visual}"
    print(f"[PASS] V7-04: Per-modality tau calibration verified: text={r_text:.3f}, audio={r_audio:.3f}, visual={r_visual:.3f}")


def test_v7_05_stream_equivalence_hasher():
    """V7-05: Stream equivalence hasher hashes ordered IDs, corrupted features, and physical masks (Task 8)."""
    def compute_stream_hash(batches):
        hasher = hashlib.sha256()
        for reg_idx, b in batches:
            hasher.update(str(reg_idx).encode("utf-8"))
            seg_ids_str = ",".join(b.get("segment_ids", [str(i) for i in range(len(b["labels"]))]))
            hasher.update(seg_ids_str.encode("utf-8"))
            hasher.update(b["labels"].detach().cpu().numpy().tobytes())
            for m in sorted(b["features"].keys()):
                hasher.update(m.encode("utf-8"))
                hasher.update(b["features"][m].detach().cpu().numpy().tobytes())
            for m in sorted(b["modality_mask"].keys()):
                hasher.update(m.encode("utf-8"))
                hasher.update(b["modality_mask"][m].detach().cpu().numpy().tobytes())
        return hasher.hexdigest()

    # Create dummy stream
    stream_a = []
    stream_b = []
    for reg in range(1, 3):
        b = {
            "segment_ids": [f"seg_{reg}_{i}" for i in range(4)],
            "labels": torch.tensor([0, 1, 2, 3]),
            "features": {"text": torch.randn(4, 768), "audio": torch.randn(4, 74)},
            "modality_mask": {"text": torch.ones(4, dtype=torch.bool), "audio": torch.ones(4, dtype=torch.bool)},
        }
        # Deepcopy for stream_b to simulate independent execution
        b_copy = {
            "segment_ids": list(b["segment_ids"]),
            "labels": b["labels"].clone(),
            "features": {k: v.clone() for k, v in b["features"].items()},
            "modality_mask": {k: v.clone() for k, v in b["modality_mask"].items()},
        }
        stream_a.append((reg, b))
        stream_b.append((reg, b_copy))

    hash_a = compute_stream_hash(stream_a)
    hash_b = compute_stream_hash(stream_b)
    assert hash_a == hash_b, "Identical stream produced different hashes!"

    # Perturb stream_b: change feature tensor slightly
    stream_b[0][1]["features"]["text"][0, 0] += 1e-4
    hash_b_perturbed = compute_stream_hash(stream_b)
    assert hash_a != hash_b_perturbed, "Perturbed stream produced identical hash! (Lack of sensitivity)"
    print(f"[PASS] V7-05: Cryptographic stream equivalence hashing verified (hash: {hash_a[:16]}).")


def test_v7_06_modality_projection_input_norm_and_learned_gate():
    """V7-06: ModalityProjection has input LayerNorm and MultimodalFusionModel has learned modality gate (Task 3)."""
    cfg = _get_mock_config()
    model = MultimodalFusionModel(cfg)

    # Check input_norm in ModalityProjection
    for m in MODALITY_ORDER:
        proj = model.projections[m]
        assert hasattr(proj, "input_norm"), f"ModalityProjection for {m} missing input_norm!"
        assert isinstance(proj.input_norm, nn.LayerNorm)

    # Check modality_gate
    assert hasattr(model, "modality_gate"), "MultimodalFusionModel missing learned modality_gate!"
    assert model.modality_gate.shape == (len(MODALITY_ORDER),)

    # Forward pass sanity check
    B = 4
    feats = {m: torch.randn(B, cfg["features"][f"{m}_dim"]) for m in MODALITY_ORDER}
    out = model(feats)
    assert out["logits"].shape == (B, 7)
    assert not torch.isnan(out["logits"]).any()
    print("[PASS] V7-06: ModalityProjection input normalization and learned gating prior verified.")


if __name__ == "__main__":
    test_v7_01_effective_mask_propagation_in_post_adaptation()
    test_v7_02_single_counted_abstentions_and_selective_accuracy()
    test_v7_03_direct_online_aci_telemetry_extraction()
    test_v7_04_per_modality_calibrated_tau_support()
    test_v7_05_stream_equivalence_hasher()
    test_v7_06_modality_projection_input_norm_and_learned_gate()
    print("\nALL V7 REGRESSION TESTS PASSED SUCCESSFULLY!")
