"""
scripts/validate_v11_evidence.py: End-to-end validator for V11 Scientific Triage and Go/No-Go Decision.

Author: Antigravity 3.8 Independent Scientific Auditor
Date: 2026-11-23
"""

import os
import sys
import json
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def validate_p0_a_readme():
    print("Checking P0-A: README Messaging Integrity...")
    readme_path = os.path.join(repo_root, "README.md")
    assert os.path.exists(readme_path), "README.md missing"
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Target Venue: Q1 Journal" not in content, "README still claims Q1 Journal target"
    assert "retracted" in content.lower(), "README does not explicitly declare retractions"
    assert "FREEZE_TRI_SACTA" in content, "README does not declare FREEZE_TRI_SACTA decision"
    assert "DEVELOPMENT-EXPOSED" in content, "README does not warn of development-exposed test set"
    print("  [PASS] P0-A: README messaging purged of SOTA claims and updated with freeze notice.")
    return True


def validate_p0_b_alignment_mechanism():
    print("Checking P0-B: Alignment Invariance Mechanism...")
    no_al_path = os.path.join(
        repo_root,
        "results",
        "v10_audit",
        "ablations",
        "cmu_mosei",
        "real",
        "prequential",
        "selective_tta_no_align_online_aci_delay0_seed42.json",
    )
    ts_path = os.path.join(
        repo_root,
        "results",
        "v9_validation",
        "tri_sacta_online_aci_cmu_mosei_real_seed42.json",
    )
    assert os.path.exists(no_al_path), "Missing selective_tta_no_align JSON"
    assert os.path.exists(ts_path), "Missing tri_sacta JSON"

    with open(no_al_path, "r", encoding="utf-8") as f:
        no_al = json.load(f)
    with open(ts_path, "r", encoding="utf-8") as f:
        ts = json.load(f)

    # Verify param delta is different but discrete labels are identical
    ts_delta = ts["adaptation_counts"]["total_param_delta"]
    no_al_delta = no_al["adaptation_counts"]["total_param_delta"]
    delta_diff = abs(ts_delta - no_al_delta)
    assert delta_diff > 0, "Parameter deltas should be slightly different (active gradient)"
    assert delta_diff < 0.005, f"Parameter delta diff too large: {delta_diff}"

    diff_labels = sum(a != b for a, b in zip(ts["predicted_labels"], no_al["predicted_labels"]))
    assert diff_labels == 0, f"Expected 0 label differences, got {diff_labels}"
    print(f"  [PASS] P0-B: Verified alignment loss is activated with zero discrete effect (diff={delta_diff:.6f}, label_diff=0).")
    return True


def validate_p0_c_visual_auroc_physics():
    print("Checking P0-C: Visual Shift AUROC Physics...")
    diag_path = os.path.join(repo_root, "results", "v10_audit", "diagnostics", "rq1_spectral_detection_audit.json")
    with open(diag_path, "r", encoding="utf-8") as f:
        diag = json.load(f)

    vis = diag["modalities"]["visual"]
    assert vis["auroc_spectral"] < 0.20, f"Visual AUROC expected < 0.20, got {vis['auroc_spectral']}"
    assert vis["clean_mean"] > 5.0, f"Clean mean visual score expected high (>5.0), got {vis['clean_mean']}"
    print(f"  [PASS] P0-C: Verified visual shift score inversion (AUROC={vis['auroc_spectral']:.4f}, Clean Mean={vis['clean_mean']:.4f}).")
    return True


def validate_phase_b_task_utility():
    print("Checking Phase B: Task Utility & Multimodal Penalty...")
    util_path = os.path.join(repo_root, "results", "v11_triage", "task_pilot_metrics.json")
    assert os.path.exists(util_path), "Missing task_pilot_metrics.json"
    with open(util_path, "r", encoding="utf-8") as f:
        util = json.load(f)

    t3 = util["task_3_class"]["models"]
    acc_text = t3["text_only_logistic_regression"]["accuracy"]
    acc_mm = t3["multimodal_source_backbone_remapped"]["accuracy"]
    assert acc_text > acc_mm, f"Text-only ({acc_text}) must outperform multimodal ({acc_mm})"
    print(f"  [PASS] Phase B: Multimodal penalty verified (Text={acc_text:.4f} vs Multimodal={acc_mm:.4f}).")
    return True


def validate_phase_c_raw_pipeline():
    print("Checking Phase C: Raw Media Feasibility...")
    raw_path = os.path.join(repo_root, "results", "v11_triage", "raw_pipeline_timings.json")
    assert os.path.exists(raw_path), "Missing raw_pipeline_timings.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    assert raw["raw_media_inventory"]["verdict"] == "RAW_DATA_UNAVAILABLE"
    assert raw["extraction_toolchain_audit"]["verdict"] == "FEATURE_INCOMPATIBLE"
    print("  [PASS] Phase C: Raw media absence and extractor incompatibility verified.")
    return True


def validate_phase_d_decision():
    print("Checking Phase D & E: Final Freeze Decision...")
    dec_path = os.path.join(repo_root, "results", "v11_triage", "go_no_go.json")
    assert os.path.exists(dec_path), "Missing go_no_go.json"
    with open(dec_path, "r", encoding="utf-8") as f:
        dec = json.load(f)

    assert dec["board_verdict"] == "FREEZE_TRI_SACTA"
    assert dec["action_items"]["v12_status"] == "CANCELLED_NO_V12"
    for cond, data in dec["go_conditions_evaluation"].items():
        assert data["status"] == "FAIL", f"Condition {cond} expected FAIL, got {data['status']}"
    print("  [PASS] Phase D & E: FREEZE_TRI_SACTA decision verified across all 5 conditions.")
    return True


def main():
    print("=====================================================================")
    print("TRI-SACTA V11 SYSTEM FEASIBILITY & SCIENTIFIC TRIAGE VALIDATOR")
    print("=====================================================================")

    checks = [
        ("P0-A", validate_p0_a_readme),
        ("P0-B", validate_p0_b_alignment_mechanism),
        ("P0-C", validate_p0_c_visual_auroc_physics),
        ("Phase B", validate_phase_b_task_utility),
        ("Phase C", validate_phase_c_raw_pipeline),
        ("Phase D/E", validate_phase_d_decision),
    ]

    all_passed = True
    for name, func in checks:
        try:
            func()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_passed = False

    print("=====================================================================")
    if all_passed:
        print("ALL V11 TRIAGE AUDIT GATES PASSED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("V11 TRIAGE AUDIT FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
