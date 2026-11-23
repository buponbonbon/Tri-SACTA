# Tri-SACTA: Multimodal Streaming Reliability Analysis and Feasibility Audit

[![Status: Scientific Audit Complete / Project Frozen](https://img.shields.io/badge/Status-Scientific%20Audit%20Complete%20%2F%20Project%20Frozen-red.svg)](#)
[![Audit Version: V11](https://img.shields.io/badge/Audit%20Version-V11%20System%20Feasibility%20%26%20Go%2FNo--Go-orange.svg)](#)
[![Unit Tests: 76/76 PASS](https://img.shields.io/badge/Unit%20Tests-76%2F76%20PASS%20(Code%20Invariants)-brightgreen.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-blue.svg)](#)

---

## 1. Executive Research Summary & Scientific Integrity Notice

> [!WARNING]
> **Scientific Integrity & Retraction Notice (V10/V11):**
> Following rigorous independent audits (V9–V11) and causal ablation analyses, all historical claims of "SOTA test-time adaptation superiority" and early exploratory numbers (e.g. `0.2111 -> 0.2175`) have been **retracted**. 
> 
> Real-data evaluations on CMU-MOSEI demonstrate that **online gradient test-time adaptation (entropy minimization + anchor alignment) causes negative transfer and statistically significant performance degradation** ($p = 0.046$) compared to keeping source weights strictly frozen with spectral reliability gating (`reliability_only`). 
> 
> Furthermore, the multimodal backbone underperforms simple unimodal text baselines, and authentic raw audio/video media pipelines are blocked due to missing extraction dependencies (COVAREP, FACET). The project decision is **`FREEZE_TRI_SACTA`**.

### Verified Canonical Benchmark Metrics (CMU-MOSEI Seed 42)
*Evaluation Protocol:* 500 unique test utterances from 58 video clusters presented across 6 sequential compound shift regimes (total 3,000 stream events; development-exposed test split; nominal $1-\alpha = 90.0\%$):

| Method / Baseline | Parameter Updates | Streaming Macro-F1 $\uparrow$ | Streaming Accuracy $\uparrow$ | Empirical Coverage | Status / Audit Finding |
|---|:---:|:---:|:---:|:---:|---|
| **Source-Only (Frozen)** | 0 | 0.215190 | 23.90% | 89.80% | Unadapted baseline |
| **Reliability-Only (No TTA Updates)** | 0 | **0.215562** | **23.93%** | 89.67% | **Best overall (zero backprop FLOPs)** |
| **Tri-SACTA (Full Proposed TTA)** | 40 batches | 0.214700 | 23.83% | 89.77% | **Degrades vs Reliability-Only ($p=0.046$)** |
| **Selective TTA (No Align)** | 40 batches | 0.214700 | 23.83% | 89.77% | Identical to Tri-SACTA ($\Delta_{\text{align}} = 0$) |
| **AdapterNorm TTA (Full TTA)** | 96 batches | 0.211750 | 23.53% | 89.67% | Severe negative transfer |

*Note on Stream Events:* The 3,000 events are **not** 3,000 independent samples; they represent 500 distinct video utterances repeated under 6 synthetic regime shifts.  
*Note on Seeds:* Only `seed 42` checkpoint has been verified on disk. Multi-seed claims (seeds 100, 2026, 777, 999) are classified as **NOT RUN / UNVERIFIED**.

---

## 2. Experimental Data Categories & Scope Separation

To prevent confusion between synthetic debug tools and genuine scientific findings, the repository strictly separates four operational categories:

1. **Synthetic Debug Fixtures (`results/quarantined_v3_fixtures/`):**
   Small artificial Gaussian clusters used strictly for testing tensor shapes and unit test pipelines. These numbers (e.g. Macro-F1 ~0.4287) are quarantined and never cited as real-world evidence.
2. **Development-Exposed Cached-Feature Benchmark (`results/v10_audit/`, `results/v9_validation/`):**
   Offline pre-extracted features from CMU-MOSEI (mDeBERTa-v3 768D, COVAREP 74D, FACET 35D). The 500-sample test set has been repeatedly evaluated across iterations and is classified as `DEVELOPMENT-EXPOSED`.
3. **Task Utility Study (Train/Val Only):**
   Supervised classification comparisons on unexposed training (1,000 samples) and validation (250 samples) splits, confirming multimodal fusion suffers from negative fusion relative to text-alone linear models.
4. **Raw-Input End-to-End Pipeline:**
   Status: `BLOCKED` (`RAW_DATA_UNAVAILABLE`, `FEATURE_INCOMPATIBLE`). Authentic raw `.mp4`/`.wav` media files and legacy extraction toolchains (COVAREP, OpenFace/FACET) are not present in the runtime environment.

---

## 3. Core Diagnostic & Failure Analysis Findings

1. **Why TTA Fails on Low-Accuracy Models (P0-B):**
   On CMU-MOSEI 7-class sentiment, the source model attains ~24% accuracy (random guess is 14.3%). Consequently, unimodal anchor predictions have near-maximum entropy ($H(p_a) \approx 1.93$ out of $\ln(7) = 1.9459$). The certainty gate $1 - H(p_a)/\ln(7)$ collapses to $\sim 0.008$, effectively scaling the alignment loss down to near zero ($\approx 10^{-5}$). TTA updates are driven purely by entropy minimization on noisy pseudo-labels, reinforcing classification errors (confirmation bias).
2. **Why Visual Shift AUROC Collapsed to 0.1578 (P0-C):**
   For visual features ($D=35$), finite sample sizes ($N=32 \le D$) cause the empirical covariance matrix to be near-singular, with tail eigenvalues collapsing to $10^{-6}$. When isotropic Gaussian shift noise $\mathcal{N}(0, \sigma^2 I)$ is added, it inflates these collapsed tail eigenvalues toward $\sigma^2 \approx 1.0$, paradoxically making the noisy batch covariance appear *more* regularized and *closer* to the source anchor spectrum than clean finite-sample batches. Thus, the detector outputs lower shift scores for noisy batches than clean batches, resulting in inverted AUROC ($0.1578 < 0.50$).
3. **Multimodal Fusion vs Text-Only Baselines:**
   On 3-class sentiment (Negative / Neutral / Positive) evaluated on the clean validation split:
   - Text-only Logistic Regression: **57.2% Accuracy / 0.5425 Macro-F1**
   - Concat Logistic Regression: **56.8% Accuracy / 0.5477 Macro-F1**
   - Text-only MLP: **50.8% Accuracy / 0.4921 Macro-F1**
   - 1.37M Multimodal Model (remapped): **45.6% Accuracy / 0.4101 Macro-F1**
   - Majority Class: **40.0% Accuracy / 0.1905 Macro-F1**
   The complex multimodal fusion backbone exhibits negative fusion, underperforming simple linear text-only models by >11% in accuracy.

---

## 4. Repository Structure

```text
Tri-SACTA/
├── docs/
│   ├── V11_00_NEON_REVIEW_AND_DECISION.md       # External review evaluation & strategy audit
│   ├── V11_01_V10_INTEGRITY_DEFECTS.md          # Forensic resolution of P0-A, P0-B, P0-C, P0-D
│   ├── V11_02_SENTIMENT_TASK_UTILITY_PILOT.md   # Validation-only predictive utility benchmark
│   ├── V11_03_RAW_MEDIA_FEASIBILITY.md          # Raw video/audio pipeline feasibility analysis
│   ├── V11_04_PRIOR_ART_OVERLAP_MATRIX.md       # Overlap matrix vs RegCal-Net (2026) & Wu et al. (2026)
│   ├── V11_05_SYSTEMS_CONTRIBUTION_AND_JOURNAL_FIT.md # Journal evaluation (MTAP, SIVP, Computing)
│   ├── V11_06_FINAL_GO_NO_GO.md                 # Formal FREEZE_TRI_SACTA decision document
│   ├── HANDOVER_V11.md                          # V11 engineering handover and artifact inventory
│   ├── V10_00_REPOSITORY_FORENSIC_AUDIT.md      # V10 repository forensic audit
│   ├── V10_01_CLAIM_EVIDENCE_RECONCILIATION.md  # V10 numerical claim reconciliation
│   └── V10_05_INDEPENDENT_SCIENTIFIC_VERDICT.md # V10 independent journal verdict
├── configs/
│   ├── mosei_real.yaml                          # CMU-MOSEI feature stream configuration
│   └── shift_regimes_v2.yaml                    # 6-regime streaming shift definitions
├── src/
│   ├── adaptation/                              # TriSACTA, AdapterNormTTA, baseline adaptors
│   ├── conformal/                               # Online ACI, Global ACI, Static Split CP
│   ├── data/                                    # Manifest, dataset, shift generator
│   └── models/                                  # Multimodal fusion backbone & spectral detector
├── scripts/
│   ├── 02_run_stream_benchmark.py               # Streaming benchmark runner
│   ├── audit_rq1_spectral_detection.py          # RQ1 spectral shift AUROC evaluation
│   ├── audit_rq2_statistical_analysis.py        # RQ2 cluster block bootstrap runner
│   └── validate_v11_evidence.py                 # V11 evidence validation script
├── tests/
│   ├── run_all_tests.py                         # Unified test runner (76/76 PASS)
│   ├── test_v11_integrity.py                    # V11 regression and integrity verification
│   └── test_v10_regression.py                   # V10 regression suite
└── results/
    ├── v11_triage/                              # V11 triage reports and decision JSONs
    └── v10_audit/                               # V10 authoritative audit metrics and diagnostics
```

---

## 5. Test Suite Execution

All unit tests enforce code invariants, shape correctness, and mathematical boundaries:
```bash
python tests/run_all_tests.py
```
*Current Result:* `SUMMARY: 76 PASSED, 0 FAILED (100%)`.

---

## 6. Project Verdict

**DECISION: `FREEZE_TRI_SACTA`**  
The project is formally closed and preserved for research transparency. Reusable components (Online ACI conformal control, video-disjoint data split loaders, and spectral shift detection formulations) remain available in `src/` for future modular adoption.
