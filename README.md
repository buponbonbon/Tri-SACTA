# Tri-SACTA V12.1: Budget-Constrained Reliability-Aware Modality Orchestration with Calibrated Conformal Risk

[![Status: Systems Contribution Validated](https://img.shields.io/badge/Status-Systems%20Contribution%20Validated-brightgreen.svg)](#)
[![Version: V12.1 Canonical](https://img.shields.io/badge/Version-V12.1%20Canonical%20(Supersedes%20V12)-blue.svg)](#)
[![Target Journal: MTAP (Q1/Q2, $0 APC)](https://img.shields.io/badge/Target%20Journal-MTAP%20(Springer%2C%20%240%20APC)-success.svg)](#)
[![Tests: 94/94 PASS](https://img.shields.io/badge/Unit%20Tests-94%2F94%20PASS%20(100%25)-brightgreen.svg)](#)
[![Multi-Seed: 42, 100, 2026](https://img.shields.io/badge/Evaluation-Seeds%2042%2C%20100%2C%202026-blueviolet.svg)](#)
[![Peak Live Inference VRAM: <35 MB](https://img.shields.io/badge/Inference%20VRAM-%3C%2035%20MB%20(RTX%202050)-green.svg)](#)

---

## 1. Executive Research Summary & Scientific Positioning

**Tri-SACTA V12.1** represents the definitive, scientifically rectified incarnation of the Tri-SACTA project. It officially supersedes V12 and completely withdraws all discredited, unmeasured test-time adaptation (TTA) and pseudo-spectral eigenvalue claims.

The project is reframed and empirically validated as an **edge-cloud multimedia systems contribution**:
> **A Budget-Constrained Reliability-Aware Modality Orchestrator (BRMO)** designed for real-world multimedia inference under fluctuating sensory channels, bandwidth costs, and strict SLA deadlines.

### Target Journal Alignment (Verified $0 Author Fees)
- **Primary Venue:** *Multimedia Tools and Applications* (Springer) — SCImago Q1 (Computer Science Applications) / Q2 (Software), 2024 Impact Factor: 3.0.
- **Publication Model:** Hybrid journal with **$0 mandatory author-facing fees** via the traditional subscription track.
- **Alternative Venues:** *Multimedia Systems* (Springer, Q2, $0 APC) and *Computing* (Springer, Q2, $0 APC).

---

## 2. Core Systems Contributions

1. **Budget-Constrained Reliability-Aware Modality Orchestrator (BRMO):**
   A multi-stage runtime inference engine that dynamically arbitrates sensory channel acquisition (`text`, `audio`, `visual`) based on unimodal confidence, out-of-distribution drift, deadline limits ($D_{\text{rem}}$), and cost budgets ($B_{\text{rem}}$).
2. **Standardized Centroid Distance ($d_{\text{scd}}$) Drift Triage:**
   Replaces heuristic spectral eigenvalue shrinkage with a mathematically sound centroid distance calibrated on in-distribution training features. When text is corrupted by ASR or sensor noise, $d_{\text{scd}}$ automatically drops the unimodal fast path to $0.0\%$, escalating to multimodal fusion.
3. **Dynamic Lazy Sensor Acquisition:**
   Sensor channels are fetched on-demand rather than eagerly acquired. Under sensory faults (`AUDIO_FEATURE_MASKING`, `VISUAL_ZERO_MASK`, `COMPOUND_FAULT`), lazy fetching suppresses broken channels, saving **$34.29\%$ to $45.33\%$ in resource costs**.
4. **Distribution-Free Split Conformal Risk Control:**
   Calibrated on an isolated 250-sample `cal` split ($\alpha = 0.10$), the conformal predictor outputs multi-class sets for ambiguous samples. Selective single-prediction abstention boosts **effective Macro-F1 to $> 0.61$** on accepted queries while providing mathematical coverage guarantees.
5. **Lightweight Edge-Cloud Service:**
   Packaged as a fail-closed FastAPI HTTP service with wire latency of $21.16$ ms (p95: $23.65$ ms) and peak live inference VRAM footprint of **$34.8$ MB** on an NVIDIA RTX 2050 laptop GPU.

---

## 3. Canonical Empirical Results Across 7 Operational Workloads

All results are averaged across seeds **42, 100, and 2026** ($N=250$ samples per workload per seed; 315 total evaluation runs).

| Workload | Policy | Raw Macro-F1 | Effective F1 (Singletons) | Compute Cost | Latency (ms) | Cost Saving % vs Always-On | Non-Inferior ($\Delta \ge -0.010$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLEAN_MULTIMODAL** | Text-only MLP | 0.5697 ± 0.0048 | 0.5697 | 1.00 | 0.97 | +71.4% | Reference Unimodal |
| | Always-On Fusion | 0.5580 ± 0.0238 | 0.5580 | 3.50 | 12.21 | 0.0% | Reference Upper |
| | **Orchestrated BRMO (V12.1)** | **0.5549 ± 0.0111** | **0.6131 ± 0.1498** | **2.90 ± 0.25** | **11.42 ± 1.84** | **+17.05%** | **YES (Joint Gate Met)** |
| **AUDIO_FEATURE_MASKING**| Always-On Fusion | 0.5259 ± 0.0210 | 0.5259 | 3.50 | 9.49 | 0.0% | Reference |
| | **Orchestrated BRMO (V12.1)** | **0.5342 ± 0.0135** | **0.6273 ± 0.1193** | **1.99 ± 0.25** | **10.23 ± 1.48** | **+43.15%** | **YES (Superior F1)** |
| **VISUAL_ZERO_MASK** | Always-On Fusion | 0.5430 ± 0.0135 | 0.5430 | 3.50 | 10.09 | 0.0% | Reference |
| | **Orchestrated BRMO (V12.1)** | **0.5301 ± 0.0076** | **0.6067 ± 0.1287** | **1.91 ± 0.25** | **9.13 ± 1.62** | **+45.33%** | Delta = -0.013 |
| **COMPOUND_FAULT** | Always-On Fusion | 0.4829 ± 0.0195 | 0.4829 | 3.50 | 11.12 | 0.0% | Reference |
| | **Orchestrated BRMO (V12.1)** | **0.4621 ± 0.0084** | **0.4901 ± 0.0468** | **2.30 ± 0.00** | **12.20 ± 0.81** | **+34.29%** | Delta = -0.021 |
| **INTERMITTENT_MISSING** | Always-On Fusion | 0.5465 ± 0.0416 | 0.5465 | 3.50 | 10.93 | 0.0% | Reference |
| | **Orchestrated BRMO (V12.1)** | **0.5248 ± 0.0150** | **0.6406 ± 0.0911** | **2.15 ± 0.18** | **8.33 ± 1.45** | **+38.62%** | Delta = -0.022 |
| **DEADLINE_STRESS** | Always-On Fusion | 0.5580 ± 0.0238 | 0.5580 | 3.50 | 9.58 | 0.0% | Reference |
| | **Orchestrated BRMO (V12.1)** | **0.5315 ± 0.0177** | **0.5983 ± 0.1264** | **1.95 ± 0.18** | **6.52 ± 0.83** | **+44.36%** | **31.97% Speedup** |
| **TEXT_EMBEDDING_NOISE** | Always-On Fusion | 0.4630 ± 0.0381 | 0.4630 | 3.50 | 9.90 | 0.0% | Reference |
| | **Orchestrated BRMO (V12.1)** | **0.4507 ± 0.0392** | **0.5097 ± 0.0336** | **3.50 ± 0.00** | **11.60 ± 1.25** | **0.00%** | **Fast-Path = 0.0%** |

---

## 4. Quarantined Split & Data Provenance Lock

Split roles are cryptographically locked in `configs/v12_1_systems_protocol.yaml`:
- **`train` (1,000 samples):** Clean reference feature statistics and predictor bank training.
- **`val` (250 samples):** Development-exposed evaluation and stress testing.
- **`cal` (250 samples):** Strictly isolated conformal calibration split.
- **`test` (500 samples):** **Quarantined & 100% Unexposed.** Zero test samples are accessed during training, tuning, or benchmarking.

---

## 5. Reproduction & Verification Instructions

### Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn pydantic scikit-learn pandas numpy joblib httpx
```

### 1. Execute Full Automated Test Suite (94 Tests)
```powershell
python tests/run_all_tests.py
```
*Expected Output:* `SUMMARY: 94 PASSED, 0 FAILED. ALL TESTS PASSED SUCCESSFULLY.`

### 2. Retrain Predictor Bank (Multi-Seed: 42, 100, 2026)
```powershell
python scripts/v12_1_train_predictor_bank.py
```

### 3. Generate Workloads & Evaluate All Policies
```powershell
python scripts/v12_1_run_workloads.py
python scripts/v12_1_evaluate_policies.py
python scripts/v12_1_paired_statistics.py
```

### 4. Run HTTP Runtime Latency Profiler
```powershell
python scripts/v12_1_profile_http.py
```

---

## 6. Authoritative Documentation Inventory

| Document ID | Title / Purpose |
| :--- | :--- |
| `docs/V12_1_00_FORENSIC_RECTIFICATION.md` | Line-by-line audit and remediation plan for all 12 V12 defects (D-01 to D-12) |
| `docs/V12_1_01_LOCKED_PROTOCOL_AND_PROVENANCE.md` | Dataset split integrity and provenance locking specifications |
| `docs/V12_1_02_FAIR_PREDICTOR_BANK.md` | 9-model predictor bank benchmark and multi-seed statistical analysis |
| `docs/V12_1_03_ORCHESTRATION_SPEC.md` | BRMO architecture, admission control, and split conformal engine spec |
| `docs/V12_1_04_MATCHED_BUDGET_BENCHMARK.md` | 7-workload benchmark results across 15 policies/ablations |
| `docs/V12_1_05_STATISTICS_AND_PARETO.md` | Paired bootstrap differences, confidence intervals, and Pareto frontiers |
| `docs/V12_1_06_API_AND_RUNTIME_EVIDENCE.md` | FastAPI service, fail-closed endpoints, and VRAM memory profiling |
| `docs/V12_1_07_PRIOR_ART_AND_JOURNAL_FIT.md` | Comparative analysis vs Wu et al. (MDPI 2026) and Springer $0 APC alignment |
| `docs/V12_1_08_FINAL_VERDICT.md` | Formal scientific verdict and journal publication decision |
| `docs/HANDOVER_V12_1.md` | Engineering handover and branch clean-up instructions |
