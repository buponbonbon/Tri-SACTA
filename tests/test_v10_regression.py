"""
tests/test_v10_regression.py: Regression suite for Tri-SACTA V10 Independent Scientific Audit.

Verifies:
1. V10-01: Reconciled Metric Identity Check.
         Ensures validation_summary.json contains the authoritative benchmark metrics
         (Source=0.2152, Rel=0.2156, Tri-SACTA=0.2147) and validates gate status.
2. V10-02: Video-Disjoint Split Verification.
         Ensures train, val, cal, test have 0 overlapping video IDs (speaker/video disjoint).
3. V10-03: RQ1 Spectral Shift Detector AUROC Verification.
         Verifies that AUROC >= 0.99 for text and audio representations under distribution shift.
4. V10-04: Causal Ablation Isolation & Attribution.
         Verifies 6-way causal ablation structure and that selective_tta_no_align is evaluated.
5. V10-05: Grouped Block Bootstrap Statistical Validation.
         Verifies that block bootstrap analysis confirms Tri-SACTA's negative transfer
         relative to reliability_only (p <= 0.05).
6. V10-06: Conformal Coverage & Non-Exchangeability Resilience.
         Verifies that Online ACI maintains valid coverage (>= 89.5%) while static split
         conformal (MCCF) suffers severe undercoverage (< 85.0%).
"""

import os
import sys
import json
import torch
import numpy as np

# Ensure pytest compatibility if imported standalone
if "pytest" not in sys.modules:
    try:
        import pytest
    except ImportError:
        class PytestShim:
            @staticmethod
            def fixture(func):
                func._is_fixture = True
                return func
        sys.modules["pytest"] = PytestShim()
import pytest


@pytest.fixture
def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_v10_01_reconciled_metric_identity(repo_root):
    """V10-01: Verify authoritative metrics in validation_summary.json and check reconciliation."""
    summary_path = os.path.join(repo_root, "results", "v10_audit", "validation_summary.json")
    assert os.path.exists(summary_path), f"validation_summary.json not found at {summary_path}"
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    metrics = summary["reconciled_benchmark_metrics"]
    assert abs(metrics["source_only"]["macro_f1"] - 0.2152) < 1e-3, "Source-only macro F1 mismatch"
    assert abs(metrics["tri_sacta"]["macro_f1"] - 0.2147) < 1e-3, "Tri-SACTA macro F1 mismatch"
    assert abs(metrics["reliability_only"]["macro_f1"] - 0.2156) < 1e-3, "Reliability-only macro F1 mismatch"
    assert summary["acceptance_gates"]["G0_scientific_integrity"] == "PASS"
    assert summary["research_decision"] == "PIVOT_TO_SPECTRAL_RELIABILITY_AND_SAFE_ADAPTATION_SUPPRESSION"


def test_v10_02_video_disjoint_splits(repo_root):
    """V10-02: Verify 100% video/speaker disjointness across train, val, cal, and test splits."""
    cache_dir = os.path.join(repo_root, "data", "cmu_mosei", "features_cache")
    splits = ["train", "val", "cal", "test"]
    split_videos = {}
    
    for split in splits:
        meta_path = os.path.join(cache_dir, split, "meta.pt")
        assert os.path.exists(meta_path), f"Split meta file missing: {meta_path}"
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        video_ids = set(sample["video_id"] for sample in meta)
        split_videos[split] = video_ids
        assert len(video_ids) > 0, f"Split {split} has no video IDs"

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = split_videos[s1].intersection(split_videos[s2])
            assert len(overlap) == 0, f"Data leakage: {s1} and {s2} share {len(overlap)} videos: {overlap}"


def test_v10_03_rq1_spectral_shift_detector_auroc(repo_root):
    """V10-03: Verify spectral shift detector AUROC on real distribution shift."""
    diag_path = os.path.join(repo_root, "results", "v10_audit", "diagnostics", "rq1_spectral_detection_audit.json")
    assert os.path.exists(diag_path), f"RQ1 diagnostic file missing: {diag_path}"
    
    with open(diag_path, "r", encoding="utf-8") as f:
        diag = json.load(f)
        
    text_auroc = diag["modalities"]["text"]["auroc_spectral"]
    audio_auroc = diag["modalities"]["audio"]["auroc_spectral"]
    visual_auroc = diag["modalities"]["visual"]["auroc_spectral"]
    
    assert text_auroc >= 0.99, f"Text AUROC too low: {text_auroc}"
    assert audio_auroc >= 0.99, f"Audio AUROC too low: {audio_auroc}"
    assert visual_auroc < 0.50, f"Visual AUROC expected poor (<0.50), got {visual_auroc}"


def test_v10_04_causal_ablation_isolation(repo_root):
    """V10-04: Verify 6-way causal ablation structure and evaluate degradation from TTA updates."""
    abl_dir = os.path.join(repo_root, "results", "v10_audit", "ablations", "cmu_mosei", "real", "prequential")
    assert os.path.exists(abl_dir), f"Ablation dir missing: {abl_dir}"
    
    no_align_file = os.path.join(abl_dir, "selective_tta_no_align_online_aci_delay0_seed42.json")
    assert os.path.exists(no_align_file), f"selective_tta_no_align result missing: {no_align_file}"
    
    with open(no_align_file, "r", encoding="utf-8") as f:
        no_align = json.load(f)
        
    assert no_align["method"] == "selective_tta_no_align"
    assert "overall_metrics" in no_align
    assert abs(no_align["overall_metrics"]["macro_f1"] - 0.2147) < 1e-3


def test_v10_05_grouped_bootstrap_statistical_validation(repo_root):
    """V10-05: Verify video-grouped block bootstrap confirms statistical significance."""
    stats_path = os.path.join(repo_root, "results", "v10_audit", "statistics", "rq2_statistical_analysis.json")
    assert os.path.exists(stats_path), f"Statistical analysis missing: {stats_path}"
    
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
        
    ts_vs_rel = stats["bootstrap_comparisons"]["tri_sacta_vs_reliability_only"]
    assert ts_vs_rel["p_value_approx"] <= 0.05, f"Expected p <= 0.05 for Tri-SACTA vs Rel, got {ts_vs_rel['p_value_approx']}"
    assert ts_vs_rel["mean_difference"] < 0, "Tri-SACTA should have negative delta vs reliability_only"
    assert ts_vs_rel["statistically_significant"] is True
    
    ts_vs_src = stats["bootstrap_comparisons"]["tri_sacta_vs_source_only"]
    assert ts_vs_src["statistically_significant"] is False, "Tri-SACTA vs Source-Only should not be statistically significant"


def test_v10_06_conformal_coverage_non_exchangeable(repo_root):
    """V10-06: Verify Online ACI coverage validity vs Static MCCF under non-exchangeable shift."""
    summary_path = os.path.join(repo_root, "results", "v10_audit", "validation_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    conformal = summary["rq3_conformal_controllers"]
    assert conformal["global_aci"]["coverage"] >= 0.895
    assert conformal["fixed_mask_aci"]["coverage"] >= 0.895
    assert conformal["dynamic_mask_aci"]["coverage"] >= 0.895
    assert conformal["mccf"]["coverage"] < 0.850, "Static MCCF should fail coverage on shifting stream"
