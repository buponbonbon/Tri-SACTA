# V12.02: Meaningful Failure Model and Deployment System Baselines

**Author:** Antigravity 3.8 Systems Research & Verification Lead  
**Date:** 2026-11-23  
**Status:** Completed  
**Harness Script:** [scripts/v12_benchmark_routing_policies.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/scripts/v12_benchmark_routing_policies.py)  
**Artifacts:**
- Full Workload Metrics: [results/v12_systems/workloads/matched_policy_metrics.json](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v12_systems/workloads/matched_policy_metrics.json)
- Pareto Trade-Off Data: [results/v12_systems/workloads/pareto_points.csv](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v12_systems/workloads/pareto_points.csv)

---

## 1. Physical Failure Workload Taxonomy

To rigorously evaluate multimodal systems beyond synthetic white Gaussian noise, we constructed a 6-regime workload suite grounded in authentic video/audio/text operational environments:

1. **Regime 1: Clean (Baseline)**
   - All sensors operational; text, audio, and visual features intact.
2. **Regime 2: Audio Corruption (Acoustic Noise / Packet Loss)**
   - Additive acoustic noise ($\sigma=1.5$) + 50% Bernoulli coefficient dropouts, simulating low SNR and packet dropouts in transmission.
   - Text and visual intact.
3. **Regime 3: Text ASR Transcription Drift**
   - Gaussian jitter ($\sigma=0.8$) + 30% feature masking simulating speech-to-text word substitutions and out-of-vocabulary slang.
   - Audio and visual intact.
4. **Regime 4: Visual Occlusion / Detection Failure**
   - Camera lens obstructed / face detection failure: visual feature tensor masked to zero with `available_visual=False`.
   - Text and audio intact.
5. **Regime 5: Compound Fault (ASR Drift + Visual Occlusion)**
   - Text corrupted by transcription drift and visual stream lost; only acoustic audio remains reliable.
6. **Regime 6: Asymmetric Intermittent Missingness**
   - 50% requests suffer delayed or missing text (deadline expired), while remaining 50% suffer degraded acoustic channels.

---

## 2. Deployment Policy Comparison Matrix

All policies evaluate identical frozen model weights (`MultimodalFusionModel`, 3-class, Seed 42) on the 250-sample validation split:

| Policy Identifier | Routing Mechanism | Sensor Invocations | Failure Vulnerability |
| :--- | :--- | :---: | :--- |
| **`text_only`** | Static unimodal text pipeline only | 0% | Completely crashes under ASR drift (F1 drops from 0.54 to 0.33) |
| **`always_on`** | Static multimodal pipeline for all requests | 100% | High latency (13.5 ms), high cost (3.5x), suffers negative transfer from noisy audio |
| **`fixed_fallback`** | Text-first; falls back to A/V only if text missing | 0–50% | Fails silently on semantic/ASR text corruption (assumes present text is clean) |
| **`conf_routing`** | Softmax entropy/confidence threshold on text | 22–80% | Vulnerable to overconfident ASR errors |
| **`random_budget`** | Randomly escalates 35% of requests to multimodal | 34–67% | Control baseline; sub-optimal resource allocation |
| **`brmo`** | Budgeted Reliability-Aware Modality Orchestrator | Dynamic | Combines spectral anomaly score, predictive entropy, and conformal abstention |
| **`brmo_shuffled`** | BRMO with randomly permuted reliability scores | Dynamic | **Ablation Control:** Verifies causality of the spectral detector |
| **`brmo_no_conformal`**| BRMO without risk abstention | Dynamic | **Ablation Control:** Measures value of selective prediction |
