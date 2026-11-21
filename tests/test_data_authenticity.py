"""
test_data_authenticity.py: V4 Data Provenance and Authenticity Regression Tests (R4-01 to R4-09).

Tests:
1. Manifest validator strictly rejects quarantined or synthetic fixture patterns when data_mode='real'.
2. Dataset loader fails closed (raises FileNotFoundError / ValueError) if real files are missing.
3. Spectral shift detector updates history exactly once per batch in prequential mode.
4. Authentic annotations downloaded in data/meld/raw and data/cmu_mosei/raw have authentic schemas and disjoint splits.
"""

import os
import sys
import torch
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.data.manifest import DataManifest, ManifestItem
from src.data.dataset import get_dataloaders
from src.models.spectral_detector import SpectralShiftDetector
from src.adaptation.tri_sacta import TriSACTA
from src.models.fusion_backbone import MultimodalFusionModel


def test_quarantined_fixtures_rejected_in_real_mode():
    """R4-01/R4-09: Synthetic fixture manifest patterns must fail validation in real mode."""
    items = [
        ManifestItem(
            sample_id="cmu_mosei_train_00000",
            video_id="vid_trn_000",  # Synthetic pattern
            speaker_id="spk_cmu_mosei_trn_000",  # Synthetic pattern
            split="train",
            raw_label=0,
            mapped_label=0,
        ),
        ManifestItem(
            sample_id="cmu_mosei_val_00000",
            video_id="vid_val_000",
            speaker_id="spk_cmu_mosei_val_000",
            split="val",
            raw_label=1,
            mapped_label=1,
        ),
    ]
    manifest = DataManifest(
        dataset_name="cmu_mosei",
        dataset_version="1.0-quarantined-fixture",
        feature_sources={"text": "dummy"},
        feature_dims={"text": 768},
        items=items,
    )

    # In real mode, this MUST be rejected
    report_real = manifest.validate_split_integrity(num_classes=7, data_mode="real")
    assert not report_real["is_valid"], "Real mode must reject synthetic fixture manifest!"
    assert any("Synthetic" in err or "forbidden" in err for err in report_real["errors"])

    # In synthetic mode, it does not trigger the real-mode authenticity ban
    report_synth = manifest.validate_split_integrity(num_classes=7, data_mode="synthetic")
    assert report_synth["is_valid"], "Synthetic mode should allow synthetic patterns for unit tests."
    print("V4-T01: Quarantined fixture rejection verified.")


def test_real_mode_fails_closed_without_manifest():
    """R4-02: Real data loader must raise FileNotFoundError if manifest is missing."""
    fake_config = {
        "dataset": {
            "manifest_path": "nonexistent_directory/nonexistent_manifest.json",
            "num_classes": 7,
        },
        "features": {
            "cache_dir": "nonexistent_features_cache",
        },
        "model": {"num_classes": 7},
    }
    try:
        get_dataloaders(fake_config, data_mode="real")
        assert False, "Should have raised FileNotFoundError!"
    except FileNotFoundError as e:
        assert "[FAIL-CLOSED]" in str(e)
    print("V4-T02: Real mode fail-closed without manifest verified.")


def test_spectral_detector_single_update_per_batch():
    """R4-07: Detector history must update exactly once per batch in prequential mode."""
    detector = SpectralShiftDetector(feature_dims={"text": 16, "visual": 16}, top_k=4, gamma=0.1)
    
    # Register dummy source stats
    source_feats = {
        "text": torch.randn(32, 16),
        "visual": torch.randn(32, 16),
    }
    detector.register_source_stats(source_feats, source_feats)

    config = {
        "model": {"num_classes": 7, "d_model": 64},
        "features": {"text_dim": 16, "visual_dim": 16, "audio_dim": 16, "prosody_dim": 16},
        "adaptation": {"tau": 1.0, "theta_high": 0.7, "theta_low": 0.3, "lambda_align": 0.5, "lambda_reg": 0.01, "lr": 1e-4},
    }
    model = MultimodalFusionModel(config)
    tta = TriSACTA(model, detector, config)

    batch_x = {"text": torch.randn(8, 16), "visual": torch.randn(8, 16)}
    batch_mask = {"text": torch.ones(8), "visual": torch.ones(8)}

    # Prequential Step 1: Pre-inference shift score computation
    shift_scores = detector.compute_shift_scores(batch_x, batch_mask, update_history=True)
    assert len(detector._shift_history["text"]) == 1

    # Prequential Step 2: Adaptation step passing precomputed_shift_scores
    tta.adapt_step(batch_x, batch_mask, precomputed_shift_scores=shift_scores)
    # The history must still have length 1, NOT 2!
    assert len(detector._shift_history["text"]) == 1, f"Expected 1 update, but got {len(detector._shift_history['text'])} (duplicate update!)"

    # Step 3: Next batch
    shift_scores_2 = detector.compute_shift_scores(batch_x, batch_mask, update_history=True)
    assert len(detector._shift_history["text"]) == 2
    tta.adapt_step(batch_x, batch_mask, precomputed_shift_scores=shift_scores_2)
    assert len(detector._shift_history["text"]) == 2

    # Check shift velocity: differences between batch 2 and batch 1
    vel = detector.compute_shift_velocity()
    assert isinstance(vel, float)
    print("V4-T03: Spectral detector single update per batch verified.")


def test_authentic_annotations_present_and_disjoint():
    """R4-03/R4-05: Verify authentic CSV annotations exist with genuine non-synthetic schemas."""
    import csv

    # 1. Verify MELD
    meld_train = "data/meld/raw/train_sent_emo.csv"
    assert os.path.exists(meld_train), f"MELD raw train CSV missing at {meld_train}"
    with open(meld_train, "r", encoding="utf-8", errors="ignore") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 9989
        first_row = reader[0]
        assert "Utterance" in first_row
        assert "Speaker" in first_row
        assert "Emotion" in first_row
        assert first_row["Speaker"] in {"Chandler", "Joey", "Monica", "Ross", "Rachel", "Phoebe", "The Interviewer"}

    # 2. Verify CMU-MOSEI
    mosei_train = "data/cmu_mosei/raw/train.csv"
    mosei_val = "data/cmu_mosei/raw/val.csv"
    mosei_test = "data/cmu_mosei/raw/test.csv"
    assert os.path.exists(mosei_train)
    assert os.path.exists(mosei_val)
    assert os.path.exists(mosei_test)

    def get_videos(p):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return set(r["video"] for r in csv.DictReader(f))

    v_train = get_videos(mosei_train)
    v_val = get_videos(mosei_val)
    v_test = get_videos(mosei_test)

    assert len(v_train) == 2244
    assert len(v_val) == 298
    assert len(v_test) == 675
    # Strict 100% video disjointness across splits
    assert len(v_train.intersection(v_val)) == 0, "Video overlap between Train and Val!"
    assert len(v_train.intersection(v_test)) == 0, "Video overlap between Train and Test!"
    assert len(v_val.intersection(v_test)) == 0, "Video overlap between Val and Test!"
    print("V4-T04: Authentic annotations and zero cross-split video overlap verified.")


if __name__ == "__main__":
    test_quarantined_fixtures_rejected_in_real_mode()
    test_real_mode_fails_closed_without_manifest()
    test_spectral_detector_single_update_per_batch()
    test_authentic_annotations_present_and_disjoint()
    print("\n==========================================")
    print("ALL V4 AUTHENTICITY REGRESSION TESTS PASSED.")
    print("==========================================")
