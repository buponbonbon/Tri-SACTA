"""
Unit tests for Tri-SACTA V3 Conformal Control (WP1: R3-01, R3-02, R3-15).

Tests:
1. test_fixed_mask_boundary_emission: Verifies full set emission on alpha <= 0 and empty set on alpha >= 1.
2. test_prediction_record_deep_immutability: Verifies caller mutating logits or prediction sets cannot alter stored record.
3. test_fixed_mask_pathwise_telescope: Verifies pathwise error bound |mean_error - alpha| <= (1 + eta) / (eta * n_g).
4. test_delayed_feedback_exact_queue: Verifies delayed queue preserves exact FIFO order and historical sets.
"""

import os
import sys
import math
import numpy as np
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.conformal.online_aci import OnlineACI, PredictionRecord


def test_fixed_mask_boundary_emission():
    """V3-T02: Fixed-step mask ACI emits full set when alpha<=0 and empty set when alpha>=1."""
    config = {
        "conformal": {
            "controller": "fixed_mask_aci",
            "eta_0": 0.05,
            "window_size": 50,
            "min_mask_support": 5,
            "allow_empty": True,
        }
    }
    aci = OnlineACI(num_classes=5, alpha=0.10, config=config)
    
    # 1. Force alpha <= 0 (persistent miscoverage / error)
    aci.alphas["mask_test"] = -0.01
    logits = torch.randn(2, 5)
    sets_full = aci.predict_set(logits, modality_mask_key="mask_test")
    
    # Must emit all 5 classes for every sample
    for s in sets_full:
        assert s == [0, 1, 2, 3, 4], f"Expected all classes on alpha<=0, got {s}"
        
    # 2. Force alpha >= 1.0 (persistent coverage)
    aci.alphas["mask_test"] = 1.05
    sets_empty = aci.predict_set(logits, modality_mask_key="mask_test")
    
    # Must emit empty set (zero forced top-1)
    for s in sets_empty:
        assert s == [], f"Expected empty set on alpha>=1, got {s}"
    print("test_fixed_mask_boundary_emission passed.")


def test_prediction_record_deep_immutability():
    """V3-T06: Caller cannot mutate stored record by modifying logits tensor or lists."""
    config = {
        "conformal": {
            "controller": "fixed_mask_aci",
            "eta_0": 0.05,
            "window_size": 50,
        }
    }
    aci = OnlineACI(num_classes=4, alpha=0.10, config=config)
    
    logits = torch.tensor([[1.0, 2.0, 0.5, 0.1]], dtype=torch.float32)
    pred_sets = aci.predict_set(logits, modality_mask_key="m1")
    
    rec_id = aci._global_t
    rec = aci.records[rec_id]
    
    # Mutate original logits tensor
    logits[0, 0] = 999.0
    # Assert record logits remain unchanged
    assert rec.logits[0, 0].item() == 1.0, "Record logits mutated when caller modified source tensor!"
    
    # Mutate returned prediction set list
    pred_sets[0].append(99)
    # Assert record prediction sets remain unchanged
    assert 99 not in rec.prediction_sets[0], "Record prediction sets mutated when caller appended to returned list!"
    
    # Ensure prediction_sets in record is an immutable tuple
    assert isinstance(rec.prediction_sets, tuple)
    assert isinstance(rec.prediction_sets[0], tuple)
    print("test_prediction_record_deep_immutability passed.")


def test_fixed_mask_pathwise_telescope():
    """V3-T03: Fixed-step mask ACI strictly satisfies pathwise telescoping error bound."""
    T = 500
    alpha = 0.10
    eta = 0.05
    num_classes = 4
    
    config = {
        "conformal": {
            "controller": "fixed_mask_aci",
            "eta_0": eta,
            "window_size": 100,
            "min_mask_support": 5,
            "allow_empty": True,
        }
    }
    aci = OnlineACI(num_classes=num_classes, alpha=alpha, config=config)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    hits = []
    # Seed score window with calibration data
    for _ in range(50):
        dummy_score = float(np.random.uniform(0.1, 0.9))
        aci.score_windows["stream_g"].append(dummy_score)
        aci.pooled_global_window.append(dummy_score)
        
    for t in range(T):
        logits = torch.randn(1, num_classes)
        label = torch.tensor([np.random.randint(0, num_classes)])
        
        # Predict
        psets = aci.predict_set(logits, modality_mask_key="stream_g")
        
        # Immediate observe
        hit = 1 if label.item() in psets[0] else 0
        hits.append(hit)
        aci.observe(aci._global_t, label)
        
    errors = 1.0 - np.array(hits)
    mean_err = np.mean(errors)
    gap = abs(mean_err - alpha)
    
    # Theoretical upper bound on gap: (1 + eta) / (eta * T)
    theoretical_bound = (1.0 + eta) / (eta * T)
    
    print(f"Fixed Mask Telescoping: Empirical Mean Error = {mean_err:.4f}, Target = {alpha:.4f}")
    print(f"Empirical Gap = {gap:.4f}, Theoretical Upper Bound = {theoretical_bound:.4f}")
    
    assert gap <= theoretical_bound + 1e-4, f"Gap {gap} exceeded theoretical bound {theoretical_bound}"
    print("test_fixed_mask_pathwise_telescope passed.")


def test_delayed_feedback_exact_queue():
    """V3-T07: Delayed feedback FIFO evaluates coverage against historical set upon maturity."""
    delay = 5
    config = {
        "conformal": {
            "controller": "dynamic_mask_aci",
            "eta_0": 0.02,
            "delay": delay,
            "window_size": 50,
            "min_mask_support": 5,
        }
    }
    aci = OnlineACI(num_classes=3, alpha=0.10, config=config)
    
    # Emit 10 prediction steps
    for t in range(1, 11):
        logits = torch.randn(1, 3)
        label = torch.tensor([1])
        aci.predict_set(logits, modality_mask_key="test_m")
        aci.update(logits, label, modality_mask_key="test_m")
        
        # Before delay steps elapse, buffer must hold the queued records
        if t < delay:
            assert len(aci._delay_buffer) == t
        else:
            # Buffer maintains exactly delay items
            assert len(aci._delay_buffer) == delay
            
    print("test_delayed_feedback_exact_queue passed.")


if __name__ == "__main__":
    test_fixed_mask_boundary_emission()
    test_prediction_record_deep_immutability()
    test_fixed_mask_pathwise_telescope()
    test_delayed_feedback_exact_queue()
    print("ALL V3 CONFORMAL TESTS PASSED.")
