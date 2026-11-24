# HANDOVER V12: Systems Research Reframing and Verification

**Date:** 2026-11-23  
**Base Commit:** `717d5dfdc1f4831dd481d04a8b7bd7edbe528f04`  
**Working Branch:** `research/v12-systems-contribution-validation`  
**Verdict:** **`SYSTEMS_PAPER_GO` (Conditional on Feature-Level Orchestration Focus)**  

---

## 1. Executive Summary

Tri-SACTA V12 successfully executes the systems research reframing requested by the repository author. Rather than attempting to force SOTA claims for test-time adaptation (which was rigorously frozen in V11), V12 establishes an empirically grounded **multimodal systems, architecture, and inference orchestration contribution**:

1. **Fixed V11 Unfair Comparison (P0 Hard Gate Passed):**
   - Directly trained `MultimodalFusionModel` with a 3-class output head (`num_classes: 3`) across 3 independent seeds (`42, 100, 2026`).
   - Accuracy surged from $45.60\%$ (V11 remapped) to **$57.20\%$** (Peak **$59.20\%$** on Seed 100).
   - Macro-F1 surged from $0.4101$ to **$0.5618$** (Peak **$0.5796$** on Seed 100).
   - Video-Grouped Block Bootstrap over 29 validation videos confirmed performance is statistically on par with text-only neural MLP under clean conditions, while carrying vastly superior redundancy under corruption.

2. **Grounded Systems Motivation:**
   - Text is cheap and strong, making static always-on multimodal execution wasteful in clean settings (13.6 ms vs 2.8 ms, 3.5x cost).
   - However, under ASR transcription drift (Regime 3), text-only models crash from $0.542$ to **$0.338$** Macro-F1.

3. **Validated Contribution — BRMO:**
   - The **Budgeted Reliability-Aware Modality Orchestrator (BRMO)** uses spectral shift and predictive entropy to fast-path clean text and progressively escalate to multimodal fusion under budget constraints.
   - Rescues Macro-F1 to **$0.4057$** under ASR drift (+6.7 points over text-only).
   - Reduces clean sensor costs by $23\%$ and latency by $22\%$ over always-on multimodal.
   - Ablation proving causality: Shuffling the reliability detector drops F1 by $3.4$ points, confirming the detector provides genuine causal routing signal.
   - Peak GPU VRAM allocated: **$13.27$ MB**, well within the 100 MB budget and laptop GPU limits.

4. **Target Publication Venue:**
   - **Multimedia Tools and Applications (MTAP, Springer)**: SJR Q1/Q2, **$0 mandatory publication fee** via subscription publishing.

---

## 2. Directory and Artifact Inventory

All code, data, logs, and artifacts are strictly reproducible:

```
Tri-SACTA/
├── docs/
│   ├── V12_00_REPRODUCIBILITY_AND_EVIDENCE_AUDIT.md
│   ├── V12_01_FAIR_3_CLASS_RETRAINING.md
│   ├── V12_02_FAILURE_MODEL_AND_SYSTEM_BASELINES.md
│   ├── V12_03_ORCHESTRATION_METHOD.md
│   ├── V12_04_RAW_MEDIA_AND_API_FEASIBILITY.md
│   ├── V12_05_SYSTEM_PROTOTYPE_AND_PROFILING.md
│   ├── V12_06_NOVELTY_AND_JOURNAL_GAP.md
│   ├── V12_07_FINAL_INDEPENDENT_VERDICT.md
│   └── HANDOVER_V12_SYSTEMS.md
├── results/v12_systems/
│   ├── manifest.json
│   ├── final_decision.json
│   ├── fair_3class/
│   │   ├── per_seed_metrics.json
│   │   ├── grouped_statistics.json
│   │   └── model_checkpoints/
│   │       ├── multimodal_fusion_seed42.pt
│   │       ├── multimodal_fusion_seed100.pt
│   │       └── multimodal_fusion_seed2026.pt
│   ├── workloads/
│   │   ├── matched_policy_metrics.json
│   │   └── pareto_points.csv
│   ├── raw_media/
│   │   └── stage_timings.json
│   └── profiling/
│       └── runtime_trials.csv
├── scripts/
│   ├── v12_train_fair_3class.py
│   ├── v12_benchmark_routing_policies.py
│   └── v12_profile_systems_runtime.py
├── src/service/
│   ├── __init__.py
│   ├── router.py
│   └── app.py
└── tests/
    └── test_v12_systems.py
```

---

## 3. Pull Request Guidelines

1. This branch `research/v12-systems-contribution-validation` is prepared for review.
2. Under no circumstances should this PR be auto-merged into `main`. The user must manually review and merge via GitHub PR.
