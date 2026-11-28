# V12.1-00: Forensic Audit and Rectification of V12 Scientific Defects

**Author:** Antigravity 3.8 Scientific Systems Audit Lead  
**Date:** 2026-11-24  
**Base Commit:** `fd010fd85f9f4b70cdc537aab546c3d800ffd01d`  
**Branch:** `research/v12.1-budgeted-orchestration-rectification`  

---

## 1. Executive Forensic Summary

A comprehensive independent review of the V12 deliverables (`research/v12-systems-contribution-validation`) identified critical scientific, architectural, and methodological defects that prevented unconditional scientific acceptance. While V12 successfully rectified the severe comparison flaw of V11 (recovering +11.6% accuracy by training the multimodal model directly on 3 classes), several key claims regarding "spectral analysis", "conformal risk control", "budget enforcement", and "causality" suffered from terminology inflation and implementation shortcuts.

This document serves as the authoritative forensic audit log. Each defect is traced to its exact source file, reproduced, and assigned an actionable rectification in V12.1.

---

## 2. Forensic Defect Registry and Rectification Status

| Defect ID | Severity | File & Line Reference | Nature of Defect | V12 Finding | V12.1 Rectification Action | Status |
| :--- | :---: | :--- | :--- | :--- | :--- | :---: |
| **D-01** | P0 | `src/service/router.py`: L44-48<br>`scripts/v12_benchmark_routing_policies.py`: L115-119 | **Misnamed "Spectral" Detector:** The function `compute_text_spectral_score` computed mean absolute standardized deviation $d = \frac{1}{D}\sum \frac{\lvert x_d - \mu_d\rvert}{\sigma_d}$, which is diagonal standardized distance, not spectral analysis. | Code was labeled "spectral anomaly" throughout prose. | Renamed to `StandardizedCentroidDistance` in `src/service/reliability.py`. All outputs and docs updated to eliminate false spectral claims. | **FIXED** |
| **D-02** | P0 | `src/service/router.py`: L174-177<br>`scripts/v12_benchmark_routing_policies.py`: L234-237 | **Confidence Rejection Misnamed as Conformal Prediction:** The rule `max(p) < 0.40` was a hardcoded heuristic threshold, not conformal prediction. | Claimed "conformal risk control" without calibration or guarantees. | Implemented genuine **Split Conformal Prediction** (`src/service/reliability.py`) calibrated on held-out `cal` split ($N=250$) with nonconformity scores $s_i = 1 - p(y_i)$ and empirical quantile $\hat{q}_\alpha$. | **FIXED** |
| **D-03** | P0 | `src/service/router.py`: L32-34<br>`scripts/v12_benchmark_routing_policies.py`: L138 | **Unenforced Budget & Arbitrary Costs:** `max_latency_budget_ms` was declared in config but never checked during routing. Cost weights (1.0/1.2/1.3) were ad-hoc assumptions. | System claimed "budget-constrained" but ignored budgets during inference. | Built `src/service/budget.py` enforcing deadline admission ($t_{\text{est}} \le D_{\text{rem}}$) and cost caps ($c_{\text{est}} \le B_{\text{rem}}$), explicitly recording violations. | **FIXED** |
| **D-04** | P0 | `scripts/v12_benchmark_routing_policies.py`: L202-206 | **Weak Internal Text Head Used in Benchmark:** Policies routed to the fusion model's internal auxiliary classifier (`modality_classifiers["text"]`) rather than the independent Text MLP (0.587 F1). | The reported +6.7 F1 gain was against an un-tuned auxiliary head. | Extracted and saved the true independent Text MLP and Text LR into `results/v12_1_systems/training/` and benchmarked against both shared and independent baselines. | **FIXED** |
| **D-05** | P0 | `scripts/v12_benchmark_routing_policies.py`: L134-136<br>`src/service/router.py`: L58-60 | **Eager Loading & Joint Invocation:** All feature tensors were pre-loaded onto GPU before routing, and audio/video were always acquired together. | No true physical or logical lazy evaluation; unable to acquire audio without video. | Implemented `src/service/acquisition.py` with independent per-modality retrieval, separate transfer tracking, and explicit acquisition failure states. | **FIXED** |
| **D-06** | P0 | `scripts/v12_benchmark_routing_policies.py`: L185-194, L174 | **Flawed Ablation & Deterministic Random Baseline:** `brmo_shuffled` used uniform random noise (changing routing frequency and dropping reliability weights). `random_budget` used `i % 3 == 0`. | Ablation did not isolate score causality; random baseline was not stochastic. | Implemented true seeded score permutation at matched acquisition budget, and isolated routing decisions from reliability weighting. | **FIXED** |
| **D-07** | P0 | `scripts/v12_train_fair_3class.py`: L285-288 | **Missing Neural Grouped Bootstrap CIs:** Neural models had dummy placeholders and did not save bootstrap CIs in `grouped_statistics.json`. | Prose reported neural CIs that were not in the committed JSON artifact. | Saved exact per-sample validation predictions and implemented true video-grouped block bootstrap with paired deltas. | **FIXED** |
| **D-08** | P1 | `scripts/v12_benchmark_routing_policies.py`: L60-75 | **Feature Noise Misrepresented as Physical Noise:** Gaussian noise and dimension masking on cached embeddings were described as "ASR WER", "acoustic SNR", and "camera occlusion". | Conflated feature-space corruption with raw physical degradation. | Renamed workloads explicitly to `TEXT_EMBEDDING_NOISE`, `AUDIO_FEATURE_MASKING`, `VISUAL_ZERO_MASK`, and documented feature-space provenance. | **FIXED** |
| **D-09** | P1 | `results/v12_systems/raw_media/stage_timings.json` | **Synthetic Microbenchmark Misinterpreted:** Random OpenCV frames and FFT operations were cited as end-to-end FPS. | 704.5 ms / 1.4 FPS was a synthetic test fixture, not a raw-input proof. | Clearly labeled all raw media microbenchmarks as `SYNTHETIC_MICROBENCHMARK_ONLY` and narrowed paper scope to feature-level streaming. | **FIXED** |
| **D-10** | P0 | `src/service/app.py`: L68-80 | **Fail-Open Service Initialization:** If checkpoint was missing, `app.py` created an uninitialized model with zero/one stats, risking random predictions. | Service lacked fail-closed validation and robust input assertions. | Implemented strict fail-closed startup: refuses to serve requests if model weights or calibration stats are missing or mismatched. Added schema validation. | **FIXED** |
| **D-11** | P0 | `scripts/v12_benchmark_routing_policies.py` vs `src/service/router.py` | **Code Duplication & Semantic Drift:** Benchmark script and API service maintained separate copies of routing logic. | Risk of divergent behavior between experimental evaluation and production code. | Unified all routing, scoring, and budget logic into shared modules (`src/service/`) called identically by benchmark, CLI, and API. | **FIXED** |
| **D-12** | P1 | `docs/V12_06_NOVELTY_AND_JOURNAL_GAP.md` | **Inaccurate Literature Citations:** Wu et al. (2026) was described as "online-gradient TTA" and unverified latency/VRAM numbers were reported. | Factual mischaracterization of prior art. | Audited original publications: Wu et al. (MDPI Electronics 2026) is a static feature-weighting method. Withdrew all unverified cross-paper metric tables. | **FIXED** |

---

## 3. Protocol Rectification Verification

Every defect listed above is explicitly resolved in the V12.1 implementation. No science gate is marked as passed based on pre-populated JSON; all metrics in V12.1 are generated by fresh, reproducible execution runs on authentic CMU-MOSEI data.
