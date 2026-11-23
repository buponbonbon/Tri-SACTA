# V10-02: DATA LEAKAGE, PROVENANCE & EVALUATION PROTOCOL AUDIT

**Repository:** `buponbonbon/Tri-SACTA`  
**Base Commit:** `d146a06b1db3a59f01d3a04b586804bdfa69c03c`  
**Date:** November 23, 2026  
**Auditor:** Independent Data Integrity & Statistical Auditor  

---

## 1. Split Independence & Speaker/Video Disjointness Audit

To verify whether feature extraction or splitting suffered from cross-split data leakage, the auditor inspected the partition files in `data/cmu_mosei/features_cache/`:

| Split | Number of Utterances | Number of Unique Videos | Class Distribution | Audio Dim | Visual Dim | Text Dim |
|---|---:|---:|---|---|---|---|
| `train` | 1,000 | 112 | [52, 173, 165, 238, 255, 94, 23] | 74 | 35 | 768 |
| `val` | 250 | 29 | [10, 39, 51, 65, 58, 17, 10] | 74 | 35 | 768 |
| `cal` | 250 | 46 | [1, 8, 34, 66, 106, 29, 6] | 74 | 35 | 768 |
| `test` | 500 | 58 | [27, 103, 83, 117, 100, 54, 16] | 74 | 35 | 768 |

### Cross-Split Overlap Matrix
- Overlap between `train` (112 videos) and `val` (29 videos): **0 videos**
- Overlap between `train` (112 videos) and `cal` (46 videos): **0 videos**
- Overlap between `train` (112 videos) and `test` (58 videos): **0 videos**
- Overlap between `val` (29 videos) and `cal` (46 videos): **0 videos**
- Overlap between `val` (29 videos) and `test` (58 videos): **0 videos**
- Overlap between `cal` (46 videos) and `test` (58 videos): **0 videos**

**Verdict:** The partitions are **100% video-disjoint and speaker-disjoint**. No cross-split sample contamination occurred at the feature-cache level.

---

## 2. Repeated Utterance Analysis in the Streaming Benchmark

### 2.1 The 3,000 Stream Events Structure
In `scripts/02_run_stream_benchmark.py:330-337`, the benchmark iterates through the test DataLoader:
```python
for regime_info in shift_gen.get_regime_sequence(): # 6 Regimes
    for step, raw_batch in enumerate(test_loader):  # 500 test samples
        batch = shift_gen.apply_regime(raw_batch, regime_idx=reg_idx, ...)
```
- Total test utterances: **500 unique utterances**.
- Number of sequential regimes: **6 regimes**.
- Total stream events: **3,000 events** ($500 \times 6$).

### 2.2 Critical Methodological Findings:
1. **The 3,000 stream events are NOT 3,000 independent samples.** They represent the exact same 500 utterances presented 6 consecutive times under different physically simulated shift conditions:
   - Regime 1: Utterances 1..500 (Clean)
   - Regime 2: Utterances 1..500 (Audio SNR ~5dB)
   - Regime 3: Utterances 1..500 (ASR WER ~35%)
   - Regime 4: Utterances 1..500 (Visual Blur/Occlusion + 30% Missing)
   - Regime 5: Utterances 1..500 (Severe Compound Shift)
   - Regime 6: Utterances 1..500 (Return to Clean Recovery)
2. **Cluster Grouping:** The 500 test utterances originate from **58 unique videos** (mean ~8.6 utterances per video). Utterances from the same video share speaker identity, recording acoustics, lighting, and conversational topic.
3. **Statistical Implication:** Treating the 3,000 stream events as independent observations produces artificially narrow confidence intervals and inflated degrees of freedom ($N=3000$). Statistical significance tests MUST use **video-grouped block bootstrap** clustering on the 58 independent video units.

---

## 3. Feedback Leakage Audit

Does prequential feedback in Regime $r$ leak labels when the same utterance reappears in Regime $r+1$?
- **Model Adaptation:**
  - Tri-SACTA, AdapterNormTTA, and TENT are **completely unsupervised**. They adapt using entropy minimization and cross-modal alignment without access to ground truth labels. No label information is backpropagated to model weights.
- **Conformal Controller:**
  - Online ACI receives ground truth $y_t$ at step $t$ (or $t + \delta$) to compute coverage error $e_t = \mathbb{I}(y_t \notin \mathcal{C}_t)$ and update the scalar significance level $\alpha_{t+1}$.
  - Online ACI does NOT maintain an utterance-level or sample-level lookup table. The scalar $\alpha$ is a global or per-mask running threshold.
  - Because 500 steps elapse between consecutive presentations of the same utterance, the feedback influence is diffused across 500 intervening scalar updates.
- **Verdict:** No direct label leakage into model weights or conformal lookup tables occurs. However, repeated presentation creates temporal autocorrelation across regimes.

---

## 4. Development Exposure Audit

- Has the test partition been exposed to iterative hyperparameter tuning?
- **Finding:** Between V5 and V9, the test split was evaluated repeatedly to diagnose majority-class collapse (V6), baseline activation (V8), and ASR drift negative transfer (V9).
- While $\tau_{\text{text}}$ was formally derived on the `val` split in `calibrate_spectral_thresholds.py`, the motivation to recalibrate was triggered by observing R3 test performance in V8.
- **Scientific Classification:** The current test partition must be classified as **DEVELOPMENT-EXPOSED**.
- For final Q1 submission, an untouched held-out evaluation partition (e.g., from full raw CMU-MOSEI or another group-disjoint video set) must be designated as the definitive confirmatory benchmark.
