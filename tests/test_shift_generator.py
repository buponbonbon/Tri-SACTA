"""
Unit tests for ShiftStreamGenerator and Protocol V2 (T25).
"""

import pytest
import torch
import numpy as np
from src.data.shift_generator import ShiftStreamGenerator


def test_snr_calibrated_noise():
    """T25: Calibrated audio SNR noise must produce target noise-to-signal power ratio."""
    gen = ShiftStreamGenerator(seed=42, use_protocol_v2=True)
    B = 64
    D = 768
    # Constant unit power signal
    feat = torch.ones(B, D)
    batch = {
        "features": {"audio": feat.clone(), "visual": torch.randn(B, 512)},
        "modality_mask": {"audio": torch.ones(B, dtype=torch.bool), "visual": torch.ones(B, dtype=torch.bool)},
        "labels": torch.zeros(B, dtype=torch.long),
    }

    # Regime 2 has SNR = 5 dB -> P_noise / P_signal = 10^(-5/10) = 10^(-0.5) ~ 0.3162
    corrupted = gen.apply_regime(batch, regime_idx=2)
    noise = corrupted["features"]["audio"] - feat
    noise_power = torch.mean(noise ** 2).item()
    expected_ratio = 10.0 ** (-5.0 / 10.0)

    # Within statistical tolerance for B*D = 49152 samples
    assert abs(noise_power - expected_ratio) < 0.05, f"Noise power {noise_power} deviates from target {expected_ratio}"


def test_protocol_v2_regime_sequence():
    """T25: Protocol V2 must define 6 sequential regimes with explicit R6 Recovery."""
    gen = ShiftStreamGenerator(seed=42, use_protocol_v2=True)
    regimes = gen.get_regime_sequence()
    assert len(regimes) == 6, f"Expected 6 regimes in Protocol V2, got {len(regimes)}"
    assert "Recovery" in regimes[5]["name"], f"Regime 6 must be Clean Recovery, got {regimes[5]['name']}"


def test_structured_audio_prosody_missing():
    """T25: Dropping audio must also drop derived prosody stream."""
    gen = ShiftStreamGenerator(seed=42, use_protocol_v2=True)
    B = 50
    batch = {
        "features": {
            "audio": torch.randn(B, 768),
            "prosody": torch.randn(B, 88),
            "visual": torch.randn(B, 512),
            "text": torch.randn(B, 768),
        },
        "modality_mask": {m: torch.ones(B, dtype=torch.bool) for m in ["audio", "prosody", "visual", "text"]},
        "labels": torch.zeros(B, dtype=torch.long),
    }

    # Apply regime 4 (30% missing)
    corrupted = gen.apply_regime(batch, regime_idx=4)
    mask = corrupted["modality_mask"]

    # Whenever audio is False, prosody MUST also be False
    audio_missing = ~mask["audio"]
    prosody_missing = ~mask["prosody"]
    assert (prosody_missing[audio_missing]).all(), "Audio was dropped but prosody remained present!"
