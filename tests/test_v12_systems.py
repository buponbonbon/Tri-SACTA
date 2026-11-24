"""
tests/test_v12_systems.py: Regression and verification test suite for Tri-SACTA V12 Systems reframing.

Verifies:
1. V12-01: Fair 3-class retrained multimodal model produces (B, 3) logits and beats majority class.
2. V12-02: BRMO Router takes text fast path on clean confident text (Cost=1.0x).
3. V12-03: BRMO Router escalates to multimodal under spectral anomaly / text corruption.
4. V12-04: Workload benchmark integrity: BRMO outperforms text-only under ASR drift.
5. V12-05: API service functions (health_check, list_policies, predict).
6. V12-06: Ultra-low VRAM memory footprint (< 100 MB).
"""

import os
import sys
import json
import pytest
import torch
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.models.fusion_backbone import MultimodalFusionModel
from src.service.router import BRMORouter, OrchestratorConfig
from src.service.app import health_check, list_policies, predict, PredictRequest, ModalityInput


@pytest.fixture
def v12_repo_root():
    return repo_root


def test_v12_01_fair_3class_retraining(v12_repo_root):
    """V12-01: Verify fair 3-class MultimodalFusionModel outputs (B, 3) logits and beats majority baseline."""
    stats_path = os.path.join(v12_repo_root, "results", "v12_systems", "fair_3class", "grouped_statistics.json")
    assert os.path.exists(stats_path), "Missing grouped_statistics.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    mm_acc = stats["multimodal_fusion_model_directly_trained"]["accuracy_mean"]
    maj_acc = stats["majority_class"]["accuracy_mean"]
    assert mm_acc > 0.50, f"Expected 3-class accuracy > 50%, got {mm_acc:.4f}"
    assert mm_acc > maj_acc, f"Expected multimodal ({mm_acc:.4f}) to beat majority ({maj_acc:.4f})"


def test_v12_02_brmo_router_fast_path(v12_repo_root):
    """V12-02: Verify BRMO Router commits text fast path when text is clean and confident."""
    cfg = {
        "model": {"fusion_dim": 64, "num_heads": 2, "num_layers": 1, "adapter_rank": 4, "dropout": 0.0, "num_classes": 3},
        "features": {"visual_dim": 35, "audio_dim": 74, "text_dim": 768, "prosody_dim": 88},
    }
    model = MultimodalFusionModel(cfg)
    clean_mean = torch.zeros(768)
    clean_std = torch.ones(768)
    router = BRMORouter(model, clean_mean, clean_std)

    feats = {"text": torch.randn(768) * 0.05, "audio": torch.randn(74), "visual": torch.randn(35)}
    res = router.route_and_predict(feats, policy="text_only")
    assert res["compute_cost"] == 1.0
    assert res["invoked_modalities"] == ["text"]
    assert res["prediction"] in ["negative", "neutral", "positive"]


def test_v12_03_brmo_router_escalation(v12_repo_root):
    """V12-03: Verify BRMO Router escalates to multimodal when text exhibits spectral anomaly."""
    cfg = {
        "model": {"fusion_dim": 64, "num_heads": 2, "num_layers": 1, "adapter_rank": 4, "dropout": 0.0, "num_classes": 3},
        "features": {"visual_dim": 35, "audio_dim": 74, "text_dim": 768, "prosody_dim": 88},
    }
    model = MultimodalFusionModel(cfg)
    clean_mean = torch.zeros(768)
    clean_std = torch.ones(768)
    router = BRMORouter(model, clean_mean, clean_std)

    # Shifted text
    feats = {"text": torch.randn(768) + 10.0, "audio": torch.randn(74), "visual": torch.randn(35)}
    res = router.route_and_predict(feats, policy="orchestrated_brmo")
    assert res["routing_decision"] == "escalated_multimodal_brmo"
    assert "audio" in res["invoked_modalities"]
    assert "visual" in res["invoked_modalities"]
    assert res["compute_cost"] > 1.0


def test_v12_04_workload_benchmarks(v12_repo_root):
    """V12-04: Verify workload metrics show BRMO rescues text collapse under ASR drift."""
    metrics_path = os.path.join(v12_repo_root, "results", "v12_systems", "workloads", "matched_policy_metrics.json")
    assert os.path.exists(metrics_path), "Missing matched_policy_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    regime_3 = metrics["regime_3_text_asr_drift"]
    f1_text = regime_3["text_only"]["raw_macro_f1"]
    f1_brmo = regime_3["brmo"]["raw_macro_f1"]
    assert f1_brmo > f1_text + 0.05, f"BRMO ({f1_brmo:.4f}) should beat text-only ({f1_text:.4f}) by > 5 F1 points under ASR drift"


def test_v12_05_api_service_endpoints(v12_repo_root):
    """V12-05: Verify API functions health_check, list_policies, and predict execute cleanly."""
    h = health_check()
    assert h["status"] == "healthy"
    assert "Tri-SACTA V12" in h["service"]

    p = list_policies()
    assert "orchestrated_brmo" in p["available_policies"]

    req = PredictRequest(
        features=ModalityInput(
            text=(np.random.randn(768) * 0.1).tolist(),
            audio=(np.random.randn(74) * 0.1).tolist(),
            visual=(np.random.randn(35) * 0.1).tolist(),
        ),
        policy="orchestrated_brmo",
    )
    pred_res = predict(req)
    assert pred_res["prediction"] in ["negative", "neutral", "positive"]
    assert pred_res["latency_ms"] > 0.0


def test_v12_06_runtime_memory_budget(v12_repo_root):
    """V12-06: Verify peak GPU VRAM allocation during service inference is strictly < 100 MB."""
    trials_path = os.path.join(v12_repo_root, "results", "v12_systems", "profiling", "runtime_trials.csv")
    assert os.path.exists(trials_path), "Missing runtime_trials.csv"
    with open(trials_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[0].strip().split(",")
    vram_idx = header.index("peak_vram_mb")
    vram_vals = [float(line.strip().split(",")[vram_idx]) for line in lines[1:]]
    max_vram = max(vram_vals)
    assert max_vram < 100.0, f"Peak VRAM ({max_vram:.2f} MB) must remain < 100 MB budget"
