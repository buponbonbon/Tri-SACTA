"""
Unit tests for Selective Test-Time Adaptation in Tri-SACTA (T08 - T11).
"""

import copy
import pytest
import torch
import yaml
from src.models.fusion_backbone import MultimodalFusionModel
from src.models.spectral_detector import SpectralShiftDetector
from src.adaptation.tri_sacta import TriSACTA


@pytest.fixture
def setup_tri_sacta():
    with open("configs/mosei_mvp.yaml") as f:
        cfg = yaml.safe_load(f)

    dims = {"visual": 512, "audio": 768, "text": 768, "prosody": 88}
    detector = SpectralShiftDetector(feature_dims=dims)
    torch.manual_seed(42)
    source_feats = {m: torch.randn(100, d) for m, d in dims.items()}
    detector.register_source_stats(source_feats)

    model = MultimodalFusionModel(cfg)
    tta = TriSACTA(model, detector, cfg)
    return tta, model, detector


def test_selective_parameter_delta(setup_tri_sacta):
    """T08: Inactive / anchor / corrupted parameters remain strictly UNCHANGED during adaptation."""
    tta, model, detector = setup_tri_sacta

    # Snapshot parameter values before step
    initial_params = {name: p.clone().detach() for name, p in model.named_parameters()}

    B = 8
    dims = {"visual": 512, "audio": 768, "text": 768, "prosody": 88}
    # Visual has moderate shift (adaptable), Audio is clean (anchor), Text/Prosody corrupted
    batch = {
        "visual": torch.randn(B, 512) * 1.5 + 0.8,
        "audio": torch.randn(B, 768),
        "text": torch.randn(B, 768) * 5.0,
        "prosody": torch.randn(B, 88) * 5.0,
    }
    mask = {m: torch.ones(B, dtype=torch.bool) for m in dims}

    result = tta.adapt_step(batch, mask)

    # Check which parameters changed
    for name, p in model.named_parameters():
        init_p = initial_params[name]
        diff = torch.abs(p - init_p).max().item()

        # If parameter belongs to anchor or corrupted modalities, diff MUST be 0!
        if any(a in name for a in result["anchor_set"]):
            assert diff == 0.0, f"Anchor parameter {name} was modified! diff={diff}"
        if any(c in name for c in result["corrupted_set"]):
            assert diff == 0.0, f"Corrupted parameter {name} was modified! diff={diff}"
        # Classifier and transformer layers must also remain frozen
        if "classifier" in name or "transformer_layers" in name:
            assert diff == 0.0, f"Core backbone parameter {name} was modified! diff={diff}"


def test_adaptation_skip(setup_tri_sacta):
    """T11: When no adaptable modality exists or no clean anchor exists, adaptation is skipped."""
    tta, model, detector = setup_tri_sacta
    initial_params = {name: p.clone().detach() for name, p in model.named_parameters()}

    B = 8
    dims = {"visual": 512, "audio": 768, "text": 768, "prosody": 88}
    # Everything is severely corrupted (no clean anchor)
    batch = {m: torch.randn(B, d) * 10.0 for m, d in dims.items()}
    mask = {m: torch.ones(B, dtype=torch.bool) for m in dims}

    result = tta.adapt_step(batch, mask)

    if not result["adaptable_set"]:
        assert result["adapted"] is False, "Adaptation should be skipped when adaptable_set is empty!"
        assert result["loss"] == 0.0, "Loss should be 0.0 when adaptation is skipped!"
        # All parameters must be unchanged
        for name, p in model.named_parameters():
            diff = torch.abs(p - initial_params[name]).max().item()
            assert diff == 0.0, f"Parameter {name} was modified despite adaptation skip!"
