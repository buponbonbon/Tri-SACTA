"""
tests/test_v11_integrity.py: Regression and integrity verification suite for Tri-SACTA V11.

Verifies:
1. V11-01: README messaging integrity and formal retraction declaration.
2. V11-02: P0-B Alignment invariance mechanism verification.
3. V11-03: P0-C Visual shift eigenvalue inversion physics.
4. V11-04: Phase B Multimodal penalty vs Text-only baseline.
5. V11-05: Phase C Raw media blocker verification (RAW_DATA_UNAVAILABLE).
6. V11-06: Phase E FREEZE_TRI_SACTA decision consistency.
"""

import os
import sys
import json
import pytest
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def v11_repo_root():
    return repo_root


def test_v11_01_readme_integrity(v11_repo_root):
    """V11-01: Verify README has purged SOTA claims and declares project freeze."""
    p = os.path.join(v11_repo_root, "README.md")
    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    assert "Target Venue: Q1 Journal" not in text
    assert "retracted" in text.lower()
    assert "FREEZE_TRI_SACTA" in text
    assert "DEVELOPMENT-EXPOSED" in text


def test_v11_02_alignment_entropy_suppression(v11_repo_root):
    """V11-02: Verify alignment loss produces non-zero param delta but zero discrete label flip."""
    no_al_path = os.path.join(
        v11_repo_root,
        "results",
        "v10_audit",
        "ablations",
        "cmu_mosei",
        "real",
        "prequential",
        "selective_tta_no_align_online_aci_delay0_seed42.json",
    )
    ts_path = os.path.join(
        v11_repo_root,
        "results",
        "v9_validation",
        "tri_sacta_online_aci_cmu_mosei_real_seed42.json",
    )
    with open(no_al_path, "r", encoding="utf-8") as f:
        no_al = json.load(f)
    with open(ts_path, "r", encoding="utf-8") as f:
        ts = json.load(f)

    diff_delta = abs(ts["adaptation_counts"]["total_param_delta"] - no_al["adaptation_counts"]["total_param_delta"])
    assert diff_delta > 0.0
    assert diff_delta < 0.005

    diff_labels = sum(a != b for a, b in zip(ts["predicted_labels"], no_al["predicted_labels"]))
    assert diff_labels == 0


def test_v11_03_visual_eigenvalue_inversion(v11_repo_root):
    """V11-03: Verify visual AUROC is inverted (< 0.20) due to low-rank covariance collapse."""
    diag_path = os.path.join(v11_repo_root, "results", "v10_audit", "diagnostics", "rq1_spectral_detection_audit.json")
    with open(diag_path, "r", encoding="utf-8") as f:
        diag = json.load(f)
    assert diag["modalities"]["visual"]["auroc_spectral"] < 0.20
    assert diag["modalities"]["visual"]["clean_mean"] > 5.0


def test_v11_04_multimodal_penalty(v11_repo_root):
    """V11-04: Verify multimodal model underperforms simple text-only Logistic Regression on 3-class."""
    pilot_path = os.path.join(v11_repo_root, "results", "v11_triage", "task_pilot_metrics.json")
    with open(pilot_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    acc_text = metrics["task_3_class"]["models"]["text_only_logistic_regression"]["accuracy"]
    acc_mm = metrics["task_3_class"]["models"]["multimodal_source_backbone_remapped"]["accuracy"]
    assert acc_text > acc_mm
    assert (acc_text - acc_mm) >= 0.10


def test_v11_05_raw_media_blocker(v11_repo_root):
    """V11-05: Verify raw media is absent and extractors are incompatible."""
    raw_path = os.path.join(v11_repo_root, "results", "v11_triage", "raw_pipeline_timings.json")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["raw_media_inventory"]["verdict"] == "RAW_DATA_UNAVAILABLE"
    assert raw["extraction_toolchain_audit"]["verdict"] == "FEATURE_INCOMPATIBLE"


def test_v11_06_freeze_decision(v11_repo_root):
    """V11-06: Verify FREEZE_TRI_SACTA decision across all gates and cancellation of V12."""
    dec_path = os.path.join(v11_repo_root, "results", "v11_triage", "go_no_go.json")
    with open(dec_path, "r", encoding="utf-8") as f:
        dec = json.load(f)
    assert dec["board_verdict"] == "FREEZE_TRI_SACTA"
    assert dec["action_items"]["v12_status"] == "CANCELLED_NO_V12"
