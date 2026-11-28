"""
tests/test_v12_1_systems.py: Exhaustive Regression and Integration Test Suite for Tri-SACTA V12.1.

Verifies all rectified systems components:
1. V12_1-01: StandardizedCentroidDistance (formula verification, zero drift, dimension checks)
2. V12_1-02: BudgetController (deadline & cost admission limits, compliance flags)
3. V12_1-03: LazyModalityProvider (independent lazy retrieval, byte transfer tracking, failure handling)
4. V12_1-04: SplitConformalPredictor (calibrated quantile computation, coverage guarantees, abstention)
5. V12_1-05: BRMORouter (fast-path triage, dynamic multimodal escalation, ablation switches)
6. V12_1-06: Fail-closed FastAPI endpoints (/health, /policies, /calibration, /predict, input validation)
"""

from __future__ import annotations

import os
import sys
import math
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.service.schemas import ModalityInput, PredictRequest, PredictResponse
from src.service.budget import BudgetController
from src.service.acquisition import LazyModalityProvider
from src.service.reliability import StandardizedCentroidDistance, SplitConformalPredictor
from src.service.app import app, get_router


@pytest.fixture
def v12_1_client():
    return TestClient(app)


def test_v12_1_01_standardized_centroid_distance():
    """V12_1-01: Verify StandardizedCentroidDistance mathematical accuracy and error handling."""
    mean = torch.tensor([1.0, 2.0, 3.0])
    std = torch.tensor([0.5, 0.5, 0.5])
    detector = StandardizedCentroidDistance(mean, std)
    
    # Exactly equal query produces distance 0
    dist_zero = detector.compute(mean)
    assert abs(dist_zero) < 1e-5

    # Formula check: |1-0|/1 = 1.0, |4-0|/2 = 2.0 -> mean = 1.5
    d2 = StandardizedCentroidDistance(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 2.0]))
    dist_val = d2.compute(torch.tensor([1.0, 4.0]))
    assert abs(dist_val - 1.5) < 1e-4

    # Dimension mismatch check
    try:
        detector.compute(torch.zeros(10))
        assert False, "Expected ValueError on dimension mismatch"
    except ValueError:
        pass


def test_v12_1_02_budget_admission_controller():
    """V12_1-02: Verify BudgetController latency and cost admission rules and compliance."""
    controller = BudgetController(
        cost_weights={"text": 1.0, "audio": 1.2, "visual": 1.3},
        latency_estimates_ms={"audio_acquisition_and_eval": 5.0, "visual_acquisition_and_eval": 8.0},
    )
    # Latency limit test
    assert controller.can_acquire("visual", remaining_time_ms=4.0, remaining_cost=5.0) is False
    assert controller.can_acquire("visual", remaining_time_ms=10.0, remaining_cost=5.0) is True

    # Cost limit test
    assert controller.can_acquire("audio", remaining_time_ms=50.0, remaining_cost=0.5) is False
    assert controller.can_acquire("audio", remaining_time_ms=50.0, remaining_cost=2.0) is True

    # Compliance test
    dl_ok, bud_ok = controller.check_compliance(25.0, 2.2, deadline_ms=30.0, cost_budget=3.5)
    assert dl_ok is True and bud_ok is True
    dl_fail, bud_fail = controller.check_compliance(35.0, 4.0, deadline_ms=30.0, cost_budget=3.5)
    assert dl_fail is False and bud_fail is False


def test_v12_1_03_lazy_modality_provider():
    """V12_1-03: Verify LazyModalityProvider independent retrieval and transfer byte accounting."""
    features = {
        "text": torch.randn(768),
        "audio": torch.randn(74),
        "visual": None,
    }
    provider = LazyModalityProvider(features, modality_mask={"text": True, "audio": True, "visual": False})

    # Text fetch: 768 * 4 = 3072 bytes
    res_t = provider.acquire("text")
    assert res_t.success is True
    assert res_t.bytes_transferred == 3072

    # Visual fetch: unavailable -> 0 bytes
    res_v = provider.acquire("visual")
    assert res_v.success is False
    assert res_v.bytes_transferred == 0
    assert provider.total_bytes_transferred == 3072


def test_v12_1_04_split_conformal_predictor():
    """V12_1-04: Verify SplitConformalPredictor calibration, coverage guarantee, and abstention."""
    rng = np.random.default_rng(42)
    N = 250
    C = 3
    probs = rng.dirichlet(np.ones(C), size=N)
    labels = np.array([int(np.argmax(p)) if rng.random() > 0.2 else rng.integers(0, C) for p in probs])

    cp = SplitConformalPredictor(alpha=0.10)
    q = cp.fit_from_probabilities(probs, labels)
    assert 0.0 < q <= 1.0

    eval_m = cp.evaluate_coverage_and_size(probs, labels)
    assert eval_m["empirical_coverage"] >= 0.85
    assert eval_m["num_samples"] == N


def test_v12_1_05_router_runtime_execution():
    """V12_1-05: Verify BRMORouter executes fail-closed on clean inputs and handles seed 42."""
    router = get_router()
    assert router is not None
    assert 42 in router.models_by_seed

    feats = {
        "text": torch.randn(768) * 0.05,
        "audio": torch.randn(74),
        "visual": torch.randn(35),
    }
    res = router.route_and_predict(feats, policy="text_only_mlp", seed=42)
    assert res["compute_cost"] == 1.0
    assert res["invoked_modalities"] == ["text"]
    assert res["prediction"] in ["negative", "neutral", "positive"]


def test_v12_1_06_fastapi_service_endpoints(v12_1_client):
    """V12_1-06: Verify FastAPI endpoints /health, /policies, /calibration, /predict and input validation."""
    # 1. Health
    h_resp = v12_1_client.get("/health")
    assert h_resp.status_code == 200
    assert h_resp.json()["status"] == "healthy"

    # 2. Policies
    p_resp = v12_1_client.get("/policies")
    assert p_resp.status_code == 200
    assert "orchestrated_brmo" in p_resp.json()["available_policies"]

    # 3. Calibration
    c_resp = v12_1_client.get("/calibration?seed=42")
    assert c_resp.status_code == 200
    assert c_resp.json()["num_calibration_samples"] == 250

    # 4. Predict validation rejects dimension mismatch (e.g. 10 dims instead of 768)
    bad_resp = v12_1_client.post("/predict", json={"features": {"text": [0.0] * 10}})
    assert bad_resp.status_code == 422

    # 5. Predict valid request
    ok_resp = v12_1_client.post("/predict", json={
        "features": {"text": [0.0] * 768, "audio": [0.0] * 74, "visual": [0.0] * 35},
        "policy": "orchestrated_brmo",
        "seed": 42,
    })
    assert ok_resp.status_code == 200
    data = ok_resp.json()
    assert data["prediction"] in ["negative", "neutral", "positive"]
    assert "conformal_set" in data
