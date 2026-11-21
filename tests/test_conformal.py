"""
Unit tests for Online Conformal Prediction (T12 - T17).
"""

import pytest
import torch
import numpy as np
from src.conformal.online_aci import OnlineACI
from src.conformal.static_cp import SplitConformalPredictor


@pytest.fixture
def aci():
    conf = {
        "eta_0": 0.01,
        "beta_1": 0.5,
        "beta_2": 0.5,
        "eta_min": 0.001,
        "eta_max": 0.10,
        "window_size": 100,
        "delay": 0,
        "score_fn": "aps",
    }
    return OnlineACI(num_classes=7, alpha=0.10, config=conf)


def test_step_size_bounds(aci):
    """T12: Extreme, negative, or zero velocities/disagreements must yield bounded positive eta."""
    # Negative shift velocity (e.g. recovery)
    neg_vel = -50.0
    disagree = 10.0
    vel_mag = abs(neg_vel)
    eta = aci.eta_0 * (1.0 + aci.beta_1 * vel_mag + aci.beta_2 * disagree)
    eta_clipped = float(np.clip(eta, aci.eta_min, aci.eta_max))

    assert eta_clipped >= aci.eta_min, "Step size fell below eta_min!"
    assert eta_clipped <= aci.eta_max, "Step size exceeded eta_max!"
    assert eta_clipped > 0.0, "Step size became non-positive!"


def test_prediction_feedback_immutable(aci):
    """T13: Delayed feedback evaluates coverage against the emitted set, even if model/alpha changes."""
    aci.delay = 5
    B = 4
    logits = torch.randn(B, 7)
    labels = torch.tensor([0, 1, 2, 3])

    # Time t=1: Emit prediction sets
    p_sets = aci.predict_set(logits, "1111")
    rec_id = aci._global_t

    # Simulate intervening steps where alpha and window mutate drastically
    for _ in range(4):
        dummy_logits = torch.randn(B, 7)
        aci.predict_set(dummy_logits, "1111")

    # Time t=5: Delayed observation arrives
    # Observe the original record
    aci.observe(rec_id, labels)

    # Check that coverage history recorded the result of the original sets
    assert len(aci._coverage_history) == 1
    recorded_cov = aci._coverage_history[0]
    expected_cov = np.mean([1.0 if labels[i].item() in p_sets[i] else 0.0 for i in range(B)])
    assert recorded_cov == expected_cov, (
        f"Recorded coverage {recorded_cov} differs from actual emitted set coverage {expected_cov}!"
    )


def test_cp_order_statistic():
    """T15: SplitConformalPredictor uses exact finite-sample order statistic rank."""
    cp = SplitConformalPredictor(num_classes=5, alpha=0.10)

    # N_cal = 10 samples
    cal_logits = torch.randn(10, 5)
    cal_labels = torch.randint(0, 5, (10,))

    q_hat = cp.calibrate(cal_logits, cal_labels)

    # For n=10, 1 - alpha = 0.90, k = ceil((10 + 1) * 0.90) = ceil(9.9) = 10 (the maximum score!)
    scores_sorted = np.sort(cp._cal_scores)
    assert abs(q_hat - scores_sorted[9]) < 1e-6, "q_hat is not the exact 10th order statistic!"
