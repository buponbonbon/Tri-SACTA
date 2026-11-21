# TRI-SACTA V3 — AUDIT STATUS & ISSUE REGISTER

**Audit Date:** 2026-11-21  
**Target Venue:** Q1 Journal (Information Fusion / IEEE TMM / Neural Networks)  
**Active Git Branch:** `research/v3-scientific-validation`  
**Base Commit SHA:** `8fe100e` (upstream `1e00431666a331529c461fce23ccda387a2570df`)  
**Hardware Profile:** Laptop GPU (RTX 2050 4GB VRAM), PyTorch 2.5.1+cu121, Windows 11  

---

## 1. Executive Status Overview

| Priority | Total Issues | OPEN | IN_PROGRESS | RESOLVED | BLOCKED |
|---|---:|---:|---:|---:|---:|
| **P0 (Critical Scientific & Integrity)** | 12 | 0 | 0 | 12 | 0 |
| **P1 (Experimental Rigor & Evidence)** | 8 | 7 | 0 | 1 | 0 |
| **P2 (Manuscript & Polish)** | 1 | 1 | 0 | 0 | 0 |
| **TOTAL** | **21** | **8** | **0** | **13** | **0** |

*Note: In accordance with Q1 scientific integrity principles, no issue is marked RESOLVED without verified unit/integration tests, immutable logs, and non-vacuous code implementation.*

---

## 2. Issue Register V3 (R3-01 through R3-21)

| Issue ID | Priority | Module / Area | Description | Target Work Package | Status | Exit Criteria & Verification Evidence |
|---|---|---|---|---|---|---|
| **R3-01** | P0 | Theory / Conformal | Invalid variable-step ACI convergence theorem (dropped total variation & clipping residual). | WP1 | **RESOLVED** | Counterexample verified (`tests/test_aci_counterexample.py`); invalid Theorem 2 retracted; fixed-step telescoping proved in `docs/THEORY_V3.md`; dynamic step framed as heuristic; unit tests in `tests/test_v3_conformal.py` PASS. |
| **R3-02** | P0 | Conformal / Queue | Unsupported delay theorem & mutable event records in online ACI. | WP1 | **RESOLVED** | Deeply immutable `PredictionRecord` (frozen tuples, cloned FP32 logits); explicit queue semantics with elapsed time maturation implemented in `src/conformal/online_aci.py`; verified in `tests/test_v3_conformal.py`. |
| **R3-03** | P0 | Data Ingestion | Real feature extraction pipeline is a stub (returns empty dict). | WP2 | **RESOLVED** | Real sequential feature extraction pipeline implemented in `scripts/extract_features_sequential.py` with sequential model loading and cache clearing; verified pilot caches generated for CMU-MOSEI and MELD. |
| **R3-04** | P0 | Data Manifest | Real mode provenance and split leakage not enforced (meta optional, invent IDs). | WP2 | **RESOLVED** | Strict manifest schema validator enforced in `src/data/manifest.py` & `src/data/dataset.py:get_dataloaders()`; SHA-256 hash verified; zero speaker/dialogue cross-split leakage verified; fail-closed on missing files. |
| **R3-05** | P0 | Model / Checkpoints | Checkpoint & source stats namespace collision across datasets and data modes. | WP2 | **RESOLVED** | Checkpoints namespaced with dataset/mode/manifest_hash in `src/utils/naming.py` and `scripts/01_train_source.py`; cross-dataset and synthetic-to-real loads strictly rejected; tested in `tests/test_v3_verification.py`. |
| **R3-06** | P0 | Protocol / Stream | Protocol mismatch: transductive vs strict prequential; R5 mixes severe + recovery. | WP3 | **RESOLVED** | Explicit `--protocol [prequential\|transductive]` flags in `scripts/02_run_stream_benchmark.py`; 6 discrete regimes with explicit R6 recovery in `ShiftStreamGenerator`; prequential order verified. |
| **R3-07** | P0 | Fusion / Adaptation | Corrupted modalities leak into fusion attention key/query/value; no-anchor promotion. | WP3 | **RESOLVED** | Effective reliability mask applied before self/cross attention in `src/models/fusion_backbone.py`; JS computed on valid pair intersection; skip update when no clean anchor in `src/adaptation/tri_sacta.py`; verified in `tests/test_v3_verification.py`. |
| **R3-08** | P0 | Spectral Detector | Spectral score confounds sample size and shift ($k=32$ vs $n=8$ rank collapse). | WP3 | **RESOLVED** | Sample-size matched eigenvalue rank $k = \min(top\_k, n_{\text{valid}}-1)$; Gram matrix decomposition for $N < D$; sample-size matched null calibration grid in `src/models/spectral_detector.py`; verified in `tests/test_v3_verification.py`. |
| **R3-09** | P0 | Corruption Semantics | Latent feature noise mislabeled as physical SNR/WER/blur. | WP3 | **RESOLVED** | Honest two-tier naming: Tier A (Latent perturbation) vs Tier B (Physical signal); explicitly defined and formalized in `docs/EXPERIMENT_PROTOCOL_V3.md`. |
| **R3-10** | P0 | Matrix Runner | Matrix runner filename mismatch and inability to schedule source pretraining. | WP4 | **RESOLVED** | Unified run schema in `src/utils/naming.py`; source training scheduled automatically before benchmark in `scripts/04_run_experiment_matrix.py`; deterministic resume. |
| **R3-11** | P0 | Aggregator | Aggregator mixes distinct experiment identities and reports $\pm 0.000$ for $N=1$. | WP4 | **RESOLVED** | Schema-strict grouping in `scripts/05_aggregate_results.py`; reject duplicate seeds; report `mean (n=1)` or `NA`; separate real and synthetic tables. |
| **R3-12** | P0 | Test Rigor | Test suite passes vacuously; mock detector needed for active adapter verification. | WP4 | **RESOLVED** | Non-vacuous unit tests with failure injection in `tests/test_v3_verification.py` (9/9 PASS); full suite `tests/run_all_tests.py` passes 24/24 tests. |
| **R3-13** | P1 | Baselines | Baseline fidelity: SAR-style and READ-style need fair and accurate naming. | WP6 | OPEN | Ported baselines explicitly labeled (`sar_style`, `consistency_tta`, `tent_ln`); fair parameter and optimizer parity. |
| **R3-14** | P1 | Evaluation Factorial | Confounding online CP and TTA contributions in single comparison. | WP6 | OPEN | Factorial 2x2 predictor $\times$ controller evaluation; offline CP replay on identical immutable logits. |
| **R3-15** | P1 | Conformal / Fallback | Missing per-mask evidence and incorrect "hierarchical shrinkage" terminology. | WP1 | **RESOLVED** | Per-mask telemetry logging (15 masks) in `src/conformal/online_aci.py`; renamed fallback to "pooled global fallback"; explicit conditional vs marginal bounds in `docs/THEORY_V3.md`. |
| **R3-16** | P1 | Telemetry | Incomplete output telemetry (cannot reconstruct F1, confusion, or peak VRAM). | WP4 | OPEN | Detailed run artifact containing per-event/batch records, confusion matrix, parameter deltas, memory profiler data. |
| **R3-17** | P1 | Data Generator | Synthetic generator lacks realistic multimodal joint dependencies. | WP5 | OPEN | Synthetic clearly marked as engineering smoke only; scientific claims grounded exclusively in real benchmarks. |
| **R3-18** | P1 | Calibration | Source-model baseline calibration and unimodal performance missing. | WP5 | OPEN | Source checkpoint metrics committed with full training logs and unimodal head validation scores. |
| **R3-19** | P1 | Positioning | Literature positioning against 2024–2026 works (READ, Smoothing the Shift, MCCP). | WP8 | OPEN | Updated related works and claim boundaries; clear differentiation from recent AISTATS/ICLR/KBS papers. |
| **R3-20** | P1 | Label Budget | Label feedback assumed 100% available at test time. | WP7 | OPEN | Evaluated under label budgets (100%, 25%, 10%, 0% feedback) with explicit deployment trade-off analysis. |
| **R3-21** | P2 | Manuscript | Paper readiness and open science reproducibility package. | WP8 | OPEN | Comprehensive reproducibility appendix, verified manifests, and honest claim-evidence reporting. |

---

## 3. Work Package Execution Roadmap

- **WP0 (Setup & Audit Baseline):**
  - Verify ACI counterexample test (`tests/test_aci_counterexample.py`) -> [PASS]
  - Create `docs/AUDIT_STATUS_V3.md` and `docs/CLAIM_EVIDENCE_MATRIX.md` -> [IN_PROGRESS]
- **WP1 (Theory & Conformal Rectification):**
  - Retract flawed Theorem 2 & Proposition 1 in `docs/THEORY.md`, `README.md`, `docs/METHOD_SPEC_V2.md`.
  - Author `docs/THEORY_V3.md` with:
    1. Exact summation-by-parts identity and clipping residuals.
    2. Fixed-step per-mask ACI reference controller with boundary emissions ($\alpha \le 0 \implies \mathcal{C} = \{1,\dots,C\}$, $\alpha \ge 1 \implies \mathcal{C} = \emptyset$).
    3. Telescoping sum proof under fixed step: $|\bar{e}_{n_g} - \alpha| \le \frac{1+\eta}{\eta \cdot n_g}$.
    4. Adaptive dynamic step framed strictly as an empirical heuristic.
  - Implement fixed-step reference controller and immutable `PredictionRecord` in `src/conformal/online_aci.py`.
- **WP2 (Data Manifest, Integrity & Real Ingestion Pipeline):**
  - Implement strict schema validator in `src/data/manifest.py`.
  - Enforce fail-closed validation in `src/data/dataset.py:get_dataloaders()`.
  - Implement namespace isolation in checkpoints and source stats.
  - Rectify `scripts/extract_features_sequential.py` with actual data loader and verified feature cache pipeline.
- **WP3 (Model Isolation, Spectral Detector & Protocol V3):**
  - Hard gating: apply effective reliability mask inside attention in `src/models/fusion_backbone.py` and `src/adaptation/tri_sacta.py`.
  - Pairwise valid JS divergence in `src/adaptation/tri_sacta.py`.
  - Sample-size matched eigenvalue rank $k = \min(top\_k, n_{\text{valid}}-1)$ in `src/models/spectral_detector.py`.
  - Strict prequential protocol runner in `scripts/02_run_stream_benchmark.py`.
- **WP4 (Matrix Runner, Aggregator & Regression Tests):**
  - Align paths in `scripts/04_run_experiment_matrix.py` and schedule source pretraining.
  - Enforce schema grouping and $N=1$ handling in `scripts/05_aggregate_results.py`.
  - Build non-vacuous unit tests in `tests/test_v3_verification.py`.
- **WP5 - WP8 (Empirical Evaluation, Ablations & Manuscript Packaging).**
