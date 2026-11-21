"""
Unit tests for Masked Multimodal Fusion and Attention (T05 - T07).
"""

import pytest
import torch
import yaml
from src.models.fusion_backbone import MultimodalFusionModel


@pytest.fixture
def model():
    with open("configs/mosei_mvp.yaml") as f:
        cfg = yaml.safe_load(f)
    return MultimodalFusionModel(cfg)


def test_mask_invariance_attention(model):
    """T05: Changing values of a masked modality must NOT alter outputs for those samples."""
    model.eval()
    B = 4
    feats = {
        "visual": torch.randn(B, 512),
        "audio": torch.randn(B, 768),
        "text": torch.randn(B, 768),
        "prosody": torch.randn(B, 88),
    }
    # Samples 0 and 1 are missing visual modality
    mask = {
        "visual": torch.tensor([0, 0, 1, 1], dtype=torch.bool),
        "audio": torch.ones(B, dtype=torch.bool),
        "text": torch.ones(B, dtype=torch.bool),
        "prosody": torch.ones(B, dtype=torch.bool),
    }

    with torch.no_grad():
        out1 = model(feats, mask)

    # Drastically alter the missing visual modality values
    feats_corrupted = {k: v.clone() for k, v in feats.items()}
    feats_corrupted["visual"][0] = 9999.0
    feats_corrupted["visual"][1] = -8888.0

    with torch.no_grad():
        out2 = model(feats_corrupted, mask)

    # Outputs for samples 0 and 1 must be perfectly identical
    diff = torch.abs(out1["logits"][:2] - out2["logits"][:2]).max().item()
    assert diff < 1e-4, f"Mask invariance failed: diff={diff}"


def test_effective_mask_fallback(model):
    """T06: When all modalities have zero reliability weights, fallback to uniform over available."""
    model.eval()
    B = 2
    feats = {
        "visual": torch.randn(B, 512),
        "audio": torch.randn(B, 768),
        "text": torch.randn(B, 768),
        "prosody": torch.randn(B, 88),
    }
    mask = {m: torch.ones(B, dtype=torch.bool) for m in feats}

    # Pass all-zero reliability weights (simulating extreme underflow)
    zero_weights = {m: torch.zeros(B) for m in feats}

    with torch.no_grad():
        out = model(feats, mask, reliability_weights=zero_weights)

    # Output logits must NOT be all NaN or pure zero bias
    assert not torch.isnan(out["logits"]).any(), "NaN in logits upon zero-weight fallback!"
    assert out["fused"].abs().sum().item() > 0.0, "Fused representation collapsed to zero!"


def test_attention_token_axis(model):
    """T07: Cross-modal attention weights across M>1 tokens respond to input differences."""
    model.eval()
    B = 2
    feats = {
        "visual": torch.randn(B, 512),
        "audio": torch.randn(B, 768),
        "text": torch.randn(B, 768),
        "prosody": torch.randn(B, 88),
    }
    mask = {m: torch.ones(B, dtype=torch.bool) for m in feats}

    with torch.no_grad():
        out = model(feats, mask)

    # Check that per-modality features after transformer are distinct
    f_vis = out["per_modality_features"]["visual"]
    f_aud = out["per_modality_features"]["audio"]
    diff = torch.abs(f_vis - f_aud).mean().item()
    assert diff > 0.01, "Modality features are identical after attention!"
