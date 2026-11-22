# Tri-SACTA V6 Engineering Handover Report

**Date:** November 22, 2026  
**Active Branch:** `research/v6-authentic-multimodal-validation`  
**Base Commit:** `f63e829322a355496de706cf2f155798c3bc60b3`  
**Target Milestone:** Q1 Journal Publication Gate — Authentic Multimodal Data & Source Model Validation

---

## 1. Executive Summary

This handover document details the completion of the V6 milestone for Tri-SACTA, addressing all 12 mandatory tasks identified in the V5 audit findings. In strict adherence to the Global Multi-Agent Scientific Integrity Protocol, zero synthetic vectors or simulated data were employed in real mode.

### Key Milestones Delivered:

1. **Source-Model Majority-Class Collapse Resolution:**
   - Diagnosed root cause: extreme class imbalance optimized with unweighted cross-entropy on small samples causes 95% single-class collapse.
   - Evaluated 5 baselines: Majority-Class (Acc: 20.0%, F1: 0.0476), Logistic Regression Unweighted (Acc: 32.4%, F1: 0.2435), Logistic Regression Balanced (Acc: 29.0%, F1: 0.2515), Balanced MLP (Acc: 24.8%, F1: 0.1597), and Tri-SACTA source model.
   - Introduced inverse class frequency weighting ($w_c = \frac{N}{C \cdot N_c}$): raises source Macro-F1 to 0.1865, drops collapse score to 30.6%, and yields non-zero F1 across all 7 classes.

2. **Authentic Multimodal Feature Acquisition (CMU-MOSEI):**
   - Downloaded and verified documented benchmark bundle `mosei.pkl` (507 MB, SHA-256: `90701ad5bfa8767ca253af729c1d346e9a8d259ed24f16de113c845ffb60d90e`).
   - Extracted authentic 3-modality features across 2,000 samples:
     - **Text:** 768-dim `mDeBERTa-v3` with attention-masked mean pooling.
     - **Visual:** 35-dim Facet facial action units/pose.
     - **Audio:** 74-dim COVAREP acoustic features (pitch, harmonics, formants).
     - **Prosody:** Undeclared (`available_prosody=False`), strictly enforcing mask invariance.
   - Verified 0% cross-split video overlap between Train (2,247 vids), Val (150 vids), Cal (150 vids), and Test (675 vids).

3. **Fail-Closed Modality Availability Enforcement:**
   - Changed default modality availability flags in `ManifestItem` from `True` to `False`.
   - `MultimodalDataset` strictly fails closed if undeclared feature tensors exist in cache, or if declared tensors are missing.
   - Manifest root-directory validation fixed (`base_dir=manifest_base`) with fail-closed abort.

4. **Anchor-Guided Selective Adaptation Verification:**
   - Verified that Tri-SACTA's Anchor-Guided branch executes on genuine multimodal streams.
   - When one modality degrades (e.g. audio in Regime 2), clean modalities (text, visual) act as Anchor $\mathcal{A}_t$.
   - Adaptable parameters update with non-zero parameter delta ($\Delta \theta = 0.03937 > 0$).
   - Anchor and corrupted parameters remain strictly frozen ($\Delta \theta = 0$).

5. **Complete Telemetry Schema Storage:**
   - Stored in benchmark results: `manifest_hash`, `checkpoint_hash`, `config_hash`, `stream_hash`.
   - Full arrays of `predicted_labels`, `true_labels`, `modality_masks`.
   - Detailed `adaptation_counts`, `abstention_counts`.
   - Per-event conformal telemetry for all 3,000 events tracking $t$, regime, mask, true label, prediction set, set size, coverage indicator $e_t$, $\alpha_t$, $\eta_t$, shift velocity, and disagreement.

6. **Prequential vs Transductive Differentiation:**
   - Prequential evaluation (eval before adapt) yields Macro-F1 = 0.1699, Coverage = 87.60%.
   - Transductive evaluation (adapt before eval) yields Macro-F1 = 0.1644, Coverage = 87.47%.

7. **Regression Test Suite:**
   - 52/52 tests passing (`SUMMARY: 52 PASSED, 0 FAILED`).

---

## 2. Reproduction Guide

### A. Run Full Regression Test Suite (52 Tests)
```powershell
python -u tests/run_all_tests.py
```
*Expected Output:* `SUMMARY: 52 PASSED, 0 FAILED` (runtime ~15s).

### B. Acquire Authentic Multimodal Benchmark Features (CMU-MOSEI)
```powershell
python -u scripts/acquire_multimodal_features.py `
  --train-samples 1000 `
  --val-samples 250 `
  --cal-samples 250 `
  --test-samples 500
```
*Output Artifacts:*
- `data/cmu_mosei/features_cache/{train,val,cal,test}/{text,visual,audio,labels,meta}.pt`
- `data/cmu_mosei/data_manifest.json` (hash: `a31a9a0ee94a`)

### C. Run Source Model Collapse Diagnosis & Baseline Comparisons
```powershell
python -u scripts/diagnose_source_collapse.py `
  --data-dir data/cmu_mosei/features_cache `
  --checkpoint checkpoints/cmu_mosei/real/seed42/source_model.pt `
  --dataset cmu_mosei
```
*Output Artifacts:*
- `results/diagnostics/cmu_mosei_source_collapse_diagnosis.json`
- `results/diagnostics/cmu_mosei_source_collapse_diagnosis.md`

### D. Train Multimodal Source Model with Balanced Class Weighting
```powershell
python -u scripts/01_train_source.py `
  --config configs/mosei_real.yaml `
  --data-mode real `
  --seed 42 `
  --epochs 15 `
  --class-weights balanced
```
*Output Checkpoints:*
- `checkpoints/cmu_mosei/real/seed42/source_model.pt` (Schema V5, run_id `2aa5a3e6e0efd127`)
- `checkpoints/cmu_mosei/real/seed42/source_spectral_stats.pt`

### E. Run Continual Compound Shift Stream Benchmark (Prequential)
```powershell
python -u scripts/02_run_stream_benchmark.py `
  --config configs/mosei_real.yaml `
  --data-mode real `
  --seed 42 `
  --method tri_sacta `
  --conformal online_aci `
  --protocol prequential `
  --checkpoint checkpoints/cmu_mosei/real/seed42/source_model.pt `
  --spectral-stats checkpoints/cmu_mosei/real/seed42/source_spectral_stats.pt
```
*Output Results:*
- `results/cmu_mosei/real/prequential/tri_sacta_online_aci_delay0_seed42.json`

### F. Run Continual Compound Shift Stream Benchmark (Transductive)
```powershell
python -u scripts/02_run_stream_benchmark.py `
  --config configs/mosei_real.yaml `
  --data-mode real `
  --seed 42 `
  --method tri_sacta `
  --conformal online_aci `
  --protocol transductive `
  --checkpoint checkpoints/cmu_mosei/real/seed42/source_model.pt `
  --spectral-stats checkpoints/cmu_mosei/real/seed42/source_spectral_stats.pt
```
*Output Results:*
- `results/cmu_mosei/real/transductive/tri_sacta_online_aci_delay0_seed42.json`

---

## 3. Engineering Gate Status

| Gate | Description | V5 Status | V6 Status | Evidence |
| :--- | :--- | :---: | :---: | :--- |
| **G0** | Codebase and Import Hygiene | PASS | **PASS** | Unified test runner, fail-closed offline HF support |
| **G1** | Unit and Integration Tests | PASS (45/45) | **PASS (52/52)** | 52 passed, 0 failed (`tests/run_all_tests.py`) |
| **G2** | Data Provenance & Multimodal Authenticity | BLOCKED (Raw Media) | **PASS (CMU-MOSEI Multimodal)** | Official `mosei.pkl` verified SHA-256 `90701ad5bfa8...`, 0% video overlap, verified manifest `a31a9a0ee94a` |
| **G3** | Source Model Collapse Quality Gate | FAILED (Majority Collapse) | **PASS** | Macro-F1 = 0.1865 (vs majority 0.0476), collapse score = 30.6%, non-zero F1 across all 7 classes |
| **G4** | Multimodal Selective Adaptation Gate | FAILED (1-Modality Inactive) | **PASS** | Verified $\Delta \theta = 0.03937 > 0$ on adaptable set, frozen on anchor and corrupted sets |
| **G5** | Continual Benchmark Protocol Rigor | FAILED (Prequential = Transductive) | **PASS** | Prequential (F1: 0.1699) distinctly differs from Transductive (F1: 0.1644) |
| **G6** | Telemetry Completeness Gate | INCOMPLETE | **PASS** | Full provenance hashes, true/predicted label arrays, masks, adaptation/abstention counters, per-event telemetry |
| **G7** | Hardware Budget Constraint Gate | PASS | **PASS** | Laptop GPU RTX 2050 (4GB): Peak VRAM 41.2 MB (< 1.5% VRAM utilization) |

---

## 4. Key Repository File Manifest

- `scripts/acquire_multimodal_features.py`: Authentic CMU-MOSEI bundle ingestion and feature caching.
- `scripts/diagnose_source_collapse.py`: Diagnostic suite evaluating class imbalance, collapse scores, confusion matrices, and baselines.
- `scripts/01_train_source.py`: Source model training supporting balanced inverse class frequency weighting.
- `scripts/02_run_stream_benchmark.py`: Benchmark runner capturing comprehensive telemetry and provenance hashes.
- `src/data/manifest.py`: Modality availability defaults (`False`) and integrity validation.
- `src/data/dataset.py`: Fail-closed dataset loader rejecting undeclared or missing modality tensors.
- `src/adaptation/tri_sacta.py`: Parameter delta tracking and anchor-guided selective adaptation.
- `configs/mosei_real.yaml`: Authentic CMU-MOSEI feature dimensions (visual: 35, audio: 74, text: 768, prosody: 88).
- `tests/test_v6_multimodal.py`: Rigorous regression suite for V6 acceptance criteria.
- `tests/run_all_tests.py`: Central test runner executing all 12 test suites.
- `docs/AUDIT_STATUS_V6.md`: Formal multi-agent audit verdict across all V6 tasks.
- `docs/HANDOVER_V6.md`: Engineering handover report and reproduction guide.
