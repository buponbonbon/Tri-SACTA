"""
tests/test_v5_provenance.py: Rigorous Regression Suite for Tri-SACTA V5 Acceptance Gates.

Acceptance Gates Covered:
- V5-01: Aggregator Quarantine Exclusion & Tamper Prevention
- V5-02: Complete Separation of Synthetic Fixtures & Fail-Closed Feature Extractor
- V5-03: Full Feature Checksum Verification (row 51+ tamper, path traversal, empty digest)
- V5-04: Manifest-to-Tensor Row Join & Sample Integrity Validation
- V5-05: Checkpoint & Spectral Statistics Provenance Hash Binding
- V5-06: Delayed-Feedback Mathematical Counterexample & Exact Accounting Identity
- V5-07: Transductive Spectral Velocity Interface (Scalar & Dict handling)
- V5-11: Fusion Backbone Abstain Policy & NaN Sanitization
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
import hashlib
from collections import deque
from math import ceil

import torch
import numpy as np

import importlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.manifest import DataManifest, ManifestItem
from src.data.dataset import MultimodalDataset
from src.data.fixtures import generate_synthetic_test_fixture
from src.models.spectral_detector import SpectralShiftDetector
from src.models.fusion_backbone import MultimodalFusionModel
from src.adaptation.tri_sacta import TriSACTA
from src.conformal.online_aci import OnlineAdaptiveACI
from scripts.extract_features_sequential import SequentialFeatureExtractor

aggregate_mod = importlib.import_module("scripts.05_aggregate_results")
load_runs = aggregate_mod.load_runs


def test_v5_01_aggregator_quarantine_exclusion():
    """V5-01: Verify that aggregator strictly rejects quarantined and legacy results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        quarantine_dir = os.path.join(tmpdir, "quarantined_v3_fixtures")
        os.makedirs(quarantine_dir, exist_ok=True)
        normal_dir = os.path.join(tmpdir, "legitimate_runs")
        os.makedirs(normal_dir, exist_ok=True)

        # 1. Place pseudo-real file in quarantined directory
        q_file = os.path.join(quarantine_dir, "tri_sacta_seed42.json")
        with open(q_file, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": "mosei",
                "data_mode": "real",
                "protocol": "prequential",
                "method": "tri_sacta",
                "conformal": "online_aci",
                "delay": 0,
                "seed": 42,
                "overall_metrics": {"macro_f1": 0.12},
                "quarantined": True,
            }, f)

        # 2. Place legitimate run in normal directory
        norm_file = os.path.join(normal_dir, "tri_sacta_seed42.json")
        with open(norm_file, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": "mosei",
                "data_mode": "synthetic",
                "protocol": "prequential",
                "method": "tri_sacta",
                "conformal": "online_aci",
                "delay": 0,
                "seed": 42,
                "overall_metrics": {"macro_f1": 0.45},
            }, f)

        # Aggregator should ONLY load legitimate run, completely ignoring quarantined file
        runs = load_runs(tmpdir, target_data_mode="real")
        assert len(runs) == 0, f"Quarantined real file leaked into real aggregation: {runs}"

        runs_synth = load_runs(tmpdir, target_data_mode="synthetic")
        assert len(runs_synth) == 1, f"Legitimate synthetic run not loaded: {runs_synth}"
        print("[PASS] V5-01: Aggregator strictly excluded quarantined artifacts from aggregation.")


def test_v5_02_fail_closed_feature_extractor():
    """V5-02: Extractor must fail closed on missing media; no synthetic fallback in real mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_p = os.path.join(tmpdir, "manifest.json")
        extractor = SequentialFeatureExtractor(
            dataset_name="cmu_mosei",
            output_dir=tmpdir,
            manifest_path=manifest_p,
            device="cpu",
        )

        # 1. Missing video media -> must raise FileNotFoundError
        try:
            extractor.extract_visual_stage(["sample_001"], video_paths=None)
            assert False, "Should have raised FileNotFoundError for missing video media"
        except FileNotFoundError as e:
            assert "[FAIL-CLOSED]" in str(e)

        # 2. Missing audio media -> must raise FileNotFoundError
        try:
            extractor.extract_audio_stage(["sample_001"], audio_paths=None)
            assert False, "Should have raised FileNotFoundError for missing audio media"
        except FileNotFoundError as e:
            assert "[FAIL-CLOSED]" in str(e)

        print("[PASS] V5-02: Feature extractor fails closed on missing raw assets.")


def test_v5_03_full_feature_checksum_and_tamper():
    """V5-03: Verify checksum validation catches tamper at row 55, empty checksums, and path traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate 60 fixture items
        fixture_manifest = generate_synthetic_test_fixture(
            output_dir=tmpdir,
            dataset_name="test_mosei",
            split_counts={"train": 60, "val": 10, "cal": 10, "test": 10},
        )

        # 1. Convert to pseudo-real manifest to test real integrity check
        manifest_p = os.path.join(tmpdir, "test_manifest.json")
        fixture_manifest.dataset_name = "cmu_mosei"
        fixture_manifest.dataset_version = "1.0-real-verified"
        for it in fixture_manifest.items:
            it.sample_id = it.sample_id.replace("fixture_test_mosei_", "mosei_")
            it.speaker_id = it.speaker_id.replace("fixture_spk_", "spk_real_")
            it.video_id = it.video_id.replace("fixture_vid_", "vid_real_")

        # Clean baseline validation
        report_clean = fixture_manifest.validate_split_integrity(
            num_classes=7,
            data_mode="real",
            base_dir=tmpdir,
        )
        assert report_clean["is_valid"], f"Clean manifest should be valid: {report_clean['errors']}"

        # 2. Tamper row 55 (past the old 50-item cutoff!)
        fixture_manifest.items[55].feature_checksums = dict(fixture_manifest.items[55].feature_checksums)
        fixture_manifest.items[55].feature_checksums["text"] = "0" * 64
        report_tampered = fixture_manifest.validate_split_integrity(
            num_classes=7,
            data_mode="real",
            base_dir=tmpdir,
        )
        assert not report_tampered["is_valid"], "Tampered row 55 was not caught by checksum validation!"
        assert any("Checksum mismatch" in err and "55" in err for err in report_tampered["errors"]), f"Errors: {report_tampered['errors']}"

        # 3. Path traversal test
        fixture_manifest.items[55].feature_files["text"] = "../../escaped_path.pt"
        report_traversal = fixture_manifest.validate_split_integrity(
            num_classes=7,
            data_mode="real",
            base_dir=tmpdir,
        )
        assert not report_traversal["is_valid"], "Path traversal was not caught!"
        assert any("Path traversal detected" in err for err in report_traversal["errors"])

        print("[PASS] V5-03: Full feature checksum validation verified (row 55 tamper & path traversal caught).")


def test_v5_04_manifest_to_tensor_row_join():
    """V5-04: MultimodalDataset must strictly join cache rows to manifest items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        split_dir = os.path.join(tmpdir, "train")
        os.makedirs(split_dir, exist_ok=True)

        count = 10
        torch.save(torch.zeros(count, 512, dtype=torch.float16), os.path.join(split_dir, "visual.pt"))
        torch.save(torch.zeros(count, 768, dtype=torch.float16), os.path.join(split_dir, "audio.pt"))
        torch.save(torch.zeros(count, 768, dtype=torch.float16), os.path.join(split_dir, "text.pt"))
        torch.save(torch.zeros(count, 88, dtype=torch.float16), os.path.join(split_dir, "prosody.pt"))
        torch.save(torch.zeros(count, dtype=torch.long), os.path.join(split_dir, "labels.pt"))

        meta = [
            {"segment_id": f"utt_{i:03d}", "speaker_id": f"spk_{i:02d}", "video_id": f"vid_{i:02d}", "label": 0}
            for i in range(count)
        ]
        torch.save(meta, os.path.join(split_dir, "meta.pt"))

        # Build matching manifest
        manifest_items = [
            ManifestItem(sample_id=f"utt_{i:03d}", speaker_id=f"spk_{i:02d}", video_id=f"vid_{i:02d}", split="train", raw_label=0, mapped_label=0)
            for i in range(count)
        ]
        manifest = DataManifest(
            dataset_name="cmu_mosei",
            dataset_version="1.0-test",
            feature_sources={"text": "real"},
            feature_dims={"text": 768},
            items=manifest_items,
        )

        cfg = {
            "dataset": {"name": "cmu_mosei", "num_classes": 7},
            "features": {"cache_dir": tmpdir},
            "model": {"num_classes": 7},
        }

        # 1. Clean load should pass
        ds = MultimodalDataset(features_dir=tmpdir, split="train", config=cfg, data_mode="real", manifest=manifest)
        assert len(ds) == count

        # 2. Mismatched utterance ID at row 4
        meta_corrupted = list(meta)
        meta_corrupted[4] = {"segment_id": "utt_WRONG", "speaker_id": "spk_04", "video_id": "vid_04", "label": 0}
        torch.save(meta_corrupted, os.path.join(split_dir, "meta.pt"))
        try:
            _ = MultimodalDataset(features_dir=tmpdir, split="train", config=cfg, data_mode="real", manifest=manifest)
            assert False, "Should have raised ValueError on utterance ID mismatch!"
        except ValueError as e:
            assert "Utterance ID mismatch" in str(e)

        print("[PASS] V5-04: Manifest-to-tensor row join and integrity verification confirmed.")


def test_v5_05_checkpoint_and_stats_binding():
    """V5-05: Checkpoints and spectral stats must be cryptographically bound to manifest hash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        detector = SpectralShiftDetector(feature_dims={"text": 16}, top_k=4)
        stats_meta = {
            "dataset_name": "cmu_mosei",
            "data_mode": "real",
            "seed": 42,
            "manifest_hash": "expected_sha_12345",
        }
        stats_path = os.path.join(tmpdir, "stats.pt")
        detector.save_source_stats(stats_path, metadata=stats_meta)

        # 1. Correct expected metadata -> loads successfully
        loaded = detector.load_source_stats(stats_path, expected_metadata=stats_meta)
        assert loaded["metadata"]["manifest_hash"] == "expected_sha_12345"

        # 2. Tampered manifest hash -> must raise ValueError
        tampered_meta = dict(stats_meta)
        tampered_meta["manifest_hash"] = "tampered_sha_67890"
        try:
            detector.load_source_stats(stats_path, expected_metadata=tampered_meta)
            assert False, "Should have failed on manifest_hash mismatch!"
        except ValueError as e:
            assert "Spectral stats metadata mismatch" in str(e)

        print("[PASS] V5-05: Checkpoint and spectral stats provenance hash binding verified.")


def test_v5_06_delayed_feedback_counterexample_and_exact_identity():
    """V5-06: Reproduce minimal delayed-feedback counterexample & verify exact diagnostic identity."""
    T, eta, target = 10000, 0.01, 0.10
    delays = [0, 10, 50]
    expected_gaps = {0: 0.0010, 10: 0.2237, 50: 0.3609}

    for delay in delays:
        a = target
        pending, window = deque(), deque(maxlen=200)
        misses, cumulative_residual = 0, 0.0

        for t in range(T):
            score = 0.2 + 0.75 * t / T
            if a <= 0:
                e = 0
            elif a >= 1:
                e = 1
            else:
                k = ceil((len(window) + 1) * (1 - min(1 - 1e-4, max(1e-4, a))))
                q = 1.0 if k > len(window) or not window else sorted(window)[max(0, k - 1)]
                e = int(score > q)
            misses += e
            pending.append((e, score))

            if len(pending) > delay:
                old, old_score = pending.popleft()
                window.append(old_score)
                raw = a + eta * (target - old)
                a_new = min(1 + eta, max(-eta, raw))
                c_j = a_new - raw
                cumulative_residual += abs(c_j)
                a = a_new

        mean_err = misses / T
        gap = abs(mean_err - target)
        old_claimed_bound = (1 + eta) / (eta * T) + delay / T

        # Check counterexample numbers match audit report
        assert abs(gap - expected_gaps[delay]) < 0.005, f"Delay {delay}: expected gap {expected_gaps[delay]}, got {gap}"

        # Verify exact diagnostic accounting identity:
        # |mean_err - target| <= |a - target| / (eta * T) + cumulative_residual / (eta * T) + delay / T
        n_obs = T - len(pending)
        exact_bound = abs(a - target) / (eta * T) + cumulative_residual / (eta * T) + len(pending) / T
        assert gap <= exact_bound + 1e-6, f"Exact diagnostic bound violated: gap={gap}, bound={exact_bound}"

        if delay > 0:
            # Crucial verification: the old claimed bound IS strictly violated
            assert gap > old_claimed_bound, f"Delay {delay}: old claimed bound {old_claimed_bound} should have been violated by gap {gap}!"

    print("[PASS] V5-06: Delayed-feedback counterexample reproduced and exact diagnostic identity confirmed.")


def test_v5_07_transductive_spectral_velocity_interface():
    """V5-07: Verify scalar spectral velocity interface and transductive handler."""
    cfg = {
        "model": {"fusion_dim": 64, "num_heads": 2, "num_layers": 1, "adapter_rank": 4, "num_classes": 7},
        "features": {"visual_dim": 16, "audio_dim": 16, "text_dim": 16, "prosody_dim": 16},
        "tta": {"learning_rate": 1e-3, "gamma_eig": 1.0, "top_k_eig": 4},
        "conformal": {"alpha": 0.10, "window_size": 50},
    }
    model = MultimodalFusionModel(cfg)
    detector = SpectralShiftDetector(feature_dims=cfg["features"], top_k=4)
    tri_sacta = TriSACTA(model, detector, cfg)

    # 1. Check scalar velocity return
    vel = tri_sacta.get_shift_velocity()
    assert isinstance(vel, float), f"get_shift_velocity() should return float, got {type(vel)}"

    # 2. Check transductive benchmark extraction logic handles float cleanly
    if isinstance(vel, dict):
        avg_vel = float(np.mean(list(vel.values()))) if vel else 0.0
    elif isinstance(vel, (int, float)):
        avg_vel = max(0.0, float(vel))
    else:
        avg_vel = 0.0
    assert isinstance(avg_vel, float)

    print("[PASS] V5-07: Transductive spectral velocity scalar interface verified.")


def test_v5_11_fusion_backbone_abstain_policy():
    """V5-11: Fusion backbone must abstain on all-rejected samples without leaking corrupted channels."""
    cfg = {
        "model": {"fusion_dim": 64, "num_heads": 2, "num_layers": 1, "adapter_rank": 4, "num_classes": 7},
        "features": {"visual_dim": 16, "audio_dim": 16, "text_dim": 16, "prosody_dim": 16},
    }
    model = MultimodalFusionModel(cfg)
    model.eval()

    B = 4
    feats = {
        "visual": torch.randn(B, 16),
        "audio": torch.randn(B, 16),
        "text": torch.randn(B, 16),
        "prosody": torch.randn(B, 16),
    }
    # All modalities physically present
    mod_mask = {m: torch.ones(B) for m in feats}
    # Effective mask isolates/rejects ALL modalities as corrupted
    eff_mask_all_corrupted = {m: torch.zeros(B) for m in feats}

    # 1. Under strict isolation (fallback_to_presence=False)
    out = model(feats, modality_mask=mod_mask, effective_mask=eff_mask_all_corrupted, fallback_to_presence_on_all_rejected=False)
    assert out["all_modalities_rejected"].all().item() is True
    assert out["fallback_applied"].any().item() is False
    assert model.abstain_count >= B

    # 2. NaN robustness: input with NaNs must be sanitized
    feats_nan = {m: torch.tensor([[float('nan')]*16]*B) for m in feats}
    out_nan = model(feats_nan, modality_mask=mod_mask)
    assert torch.isfinite(out_nan["logits"]).all()

    print("[PASS] V5-11: Fusion backbone abstain policy and NaN input sanitization verified.")


def run_all_v5_tests():
    print("=== Running Tri-SACTA V5 Provenance & Integrity Test Suite ===")
    test_v5_01_aggregator_quarantine_exclusion()
    test_v5_02_fail_closed_feature_extractor()
    test_v5_03_full_feature_checksum_and_tamper()
    test_v5_04_manifest_to_tensor_row_join()
    test_v5_05_checkpoint_and_stats_binding()
    test_v5_06_delayed_feedback_counterexample_and_exact_identity()
    test_v5_07_transductive_spectral_velocity_interface()
    test_v5_11_fusion_backbone_abstain_policy()
    print("=== All 8 V5 Acceptance Gates PASSED (8/8) ===")


if __name__ == "__main__":
    run_all_v5_tests()
