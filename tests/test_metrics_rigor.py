"""
Unit tests for Metrics Rigor, Boundary Guards, and Profiling (T26, T29).
"""

import pytest
import torch
import numpy as np
from src.utils.metrics import (
    compute_macro_f1,
    compute_weighted_f1,
    compute_nll,
    compute_brier_score,
    compute_ece,
    MetricsTracker,
    VRAMProfiler,
)


def test_macro_f1_absent_classes():
    """T26: Macro-F1 with absent classes in batch must average across all num_classes."""
    # 7 classes total (0..6), but predictions and labels only have classes 0, 1, 2
    labels = np.array([0, 0, 1, 1, 2, 2])
    preds = np.array([0, 0, 1, 1, 2, 2])  # Perfect prediction on {0, 1, 2}

    # If averaged only over present classes (3 classes), F1 would be 1.0!
    # With fixed support over all 7 classes, F1 should be 3/7 = ~0.4286
    f1_fixed = compute_macro_f1(preds, labels, num_classes=7)
    expected_f1 = 3.0 / 7.0
    assert abs(f1_fixed - expected_f1) < 1e-4, f"Macro-F1 {f1_fixed} != expected {expected_f1}"


def test_nll_fp32_precision():
    """T26: NLL calculation must convert half/AMP tensors to FP32 without overflow."""
    # Extreme logits in FP16 that would overflow if not converted
    logits_fp16 = torch.tensor([[10.0, -10.0], [-10.0, 10.0]], dtype=torch.float16)
    labels = torch.tensor([0, 1], dtype=torch.long)

    nll = compute_nll(logits_fp16, labels)
    assert not np.isnan(nll)
    assert not np.isinf(nll)
    assert nll < 0.001, f"Expected small NLL for confident correct logits, got {nll}"


def test_ece_bin_boundaries():
    """T26: ECE with zero-confidence or 1.0 confidence points must fall safely into boundary bins."""
    probs = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
    ])
    labels = np.array([0, 0, 1])

    ece = compute_ece(probs, labels, n_bins=10)
    assert not np.isnan(ece)
    assert 0.0 <= ece <= 1.0


def test_vram_profiler_footprint():
    """T29: VRAMProfiler snapshot must report structured memory dictionary."""
    profiler = VRAMProfiler()
    snap = profiler.snapshot()
    assert "allocated_mb" in snap
    assert "reserved_mb" in snap
    assert "peak_allocated_mb" in snap
    assert "peak_reserved_mb" in snap
    assert "total_vram_mb" in snap
