"""
Unit tests for Data Integrity, Collator, and Determinism (T19 - T23).
"""

import pytest
import torch
import numpy as np
from src.data.dataset import MultimodalDataset, MultimodalCollator


def test_real_data_fail_closed():
    """T19: Real data mode must raise FileNotFoundError if files are missing (no silent fallback)."""
    cfg = {"model": {"num_classes": 7}, "features": {}}
    with pytest.raises(FileNotFoundError):
        MultimodalDataset(
            features_dir="non_existent_directory_12345",
            split="train",
            config=cfg,
            data_mode="real",
        )


def test_collator_per_sample_missing():
    """T21: Missing modality for one sample must NOT discard it for present samples in the batch."""
    collator = MultimodalCollator(["visual", "audio"])

    # Batch of 4 samples: sample 0 has no visual, samples 1, 2, 3 have visual
    sample_vis = torch.ones(512)
    sample_aud = torch.ones(768)

    batch = [
        {"visual": torch.zeros(512), "audio": sample_aud, "visual_available": False, "label": 0},
        {"visual": sample_vis, "audio": sample_aud, "visual_available": True, "label": 1},
        {"visual": sample_vis, "audio": sample_aud, "visual_available": True, "label": 2},
        {"visual": sample_vis, "audio": sample_aud, "visual_available": True, "label": 3},
    ]

    collated = collator(batch)

    # Visual mask must be [False, True, True, True]
    assert not collated["modality_mask"]["visual"][0].item(), "Sample 0 visual should be False"
    assert collated["modality_mask"]["visual"][1].item(), "Sample 1 visual should be True"
    assert collated["modality_mask"]["visual"][2].item(), "Sample 2 visual should be True"
    assert collated["modality_mask"]["visual"][3].item(), "Sample 3 visual should be True"

    # Samples 1, 2, 3 must retain their non-zero features!
    assert (collated["features"]["visual"][1] == 1.0).all(), "Sample 1 visual was wiped out!"
    assert (collated["features"]["visual"][2] == 1.0).all(), "Sample 2 visual was wiped out!"


def test_seed_cross_process():
    """T23: Synthetic dataset generation is 100% deterministic given seed."""
    cfg = {"model": {"num_classes": 7}, "features": {}}
    ds1 = MultimodalDataset("cache", "train", cfg, num_samples=20, seed=42, data_mode="synthetic")
    ds2 = MultimodalDataset("cache", "train", cfg, num_samples=20, seed=42, data_mode="synthetic")

    for i in range(20):
        assert ds1[i]["label"] == ds2[i]["label"]
        for m in ["visual", "audio", "text", "prosody"]:
            diff = torch.abs(ds1[i][m] - ds2[i][m]).max().item()
            assert diff == 0.0, f"Synthetic generation not deterministic for sample {i}, mod {m}"
