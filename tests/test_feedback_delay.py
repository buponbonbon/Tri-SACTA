"""
Unit tests for Conformal Feedback Delay, Queue Ordering, and Mask Isolation (T14, T18).
"""

import pytest
import torch
import numpy as np
from src.conformal.online_aci import OnlineACI


def test_delay_buffer_strict_fifo():
    """T14: Records must remain queued and not evaluated until exactly delay steps elapse."""
    delay = 3
    conf = {
        "eta_0": 0.01,
        "delay": delay,
        "score_fn": "aps",
    }
    aci = OnlineACI(num_classes=5, alpha=0.10, config=conf)

    # Step 1: predict and update (enqueue)
    logits1 = torch.randn(2, 5)
    labels1 = torch.tensor([0, 1])
    aci.predict_set(logits1, "1111")
    aci.update(logits1, labels1, "1111")

    # Coverage history should still be empty (delay = 3)
    assert len(aci._coverage_history) == 0
    assert len(aci._delay_buffer) == 1

    # Steps 2 and 3: intervening batches
    for _ in range(2):
        l = torch.randn(2, 5)
        aci.predict_set(l, "1111")
        aci.update(l, labels1, "1111")

    # After step 3, step 1 has elapsed 3-1 = 2 steps (not yet mature)
    assert len(aci._coverage_history) == 0

    # Step 4: Step 1 (t=1) now has t_current (4) - t_entry (1) = 3 >= delay, matures!
    l = torch.randn(2, 5)
    aci.predict_set(l, "1111")
    aci.update(l, labels1, "1111")

    assert len(aci._coverage_history) == 1, f"Expected 1 matured record, got {len(aci._coverage_history)}"


def test_rare_mask_fallback():
    """T18: Rare masks with support < min_mask_support fall back safely to global window."""
    conf = {
        "eta_0": 0.01,
        "min_mask_support": 10,
        "score_fn": "lac",
    }
    aci = OnlineACI(num_classes=5, alpha=0.10, config=conf)

    # Populate global score window with 20 samples from mask "1111" with confident predictions
    for _ in range(20):
        logits = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]])
        labels = torch.tensor([0])
        aci.predict_set(logits, "1111")
        aci.observe(aci._global_t, labels)

    assert len(aci.global_score_window) == 20
    assert len(aci.score_windows["0001"]) == 0  # Rare mask has 0 samples

    # Getting quantile for rare mask "0001" must NOT return cold-start 1.0; it should fall back to global!
    q_hat = aci._get_quantile("0001")
    # For confident logits, LAC score is ~0.0067, so 90th percentile is << 1.0!
    assert q_hat < 0.10, f"Expected global window quantile < 0.10, got {q_hat}"
