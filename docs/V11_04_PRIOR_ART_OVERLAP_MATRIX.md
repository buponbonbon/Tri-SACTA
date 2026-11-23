# V11-04: PRIOR ART OVERLAP MATRIX (2024–2026 LITERATURE)

**Audit Date:** 2026-11-23  
**Auditor:** Antigravity 3.8 Independent Scientific Auditor  
**Document Code:** `TRI-SACTA-V11-04`  
**Focus:** Direct comparison with 2026 published literature on reliability-aware multimodal sentiment analysis.

---

## 1. Key 2026 Prior Art Identified

### 1.1. RegCal-Net (Multimedia Systems, September 2026)
- **Title:** *Register-augmented attention and self-calibrated fusion for robust multimodal sentiment analysis*
- **Authors:** Xu, Wang & Ding
- **Venue:** *Multimedia Systems* 32(9):562 (published online 06 Sep 2026)
- **DOI:** `10.1007/s00530-026-02653-8`
- **Core Mechanism:** Self-Calibrated Fusion (SCF) module that dynamically measures feature quality and reliability to perform **adaptive suppression of unreliable signals**. Evaluated on multimodal benchmarks with simulated noise.

### 1.2. Wu et al. (Electronics, August 2026)
- **Title:** *Reliability-Aware Multi-Modal Sentiment Analysis Under Missing and Corrupted Modalities*
- **Authors:** Wu, Zhu & Liu
- **Venue:** *Electronics* 15(16):3624 (published 14 Aug 2026)
- **DOI:** `10.3390/electronics15163624`
- **Core Mechanism:** Directly addresses tri-modal sentiment analysis (Audio, Visual, Text) under missing and corrupted modalities using reliability estimation, uncertainty quantification, and selective prediction on MOSI and MOSEI.

---

## 2. Feature-by-Feature Overlap Matrix

| Capability / Method Dimension | RegCal-Net (Xu et al., 09/2026) | Wu et al. (08/2026) | Tri-SACTA V10/V11 | Overlap Assessment |
|---|:---:|:---:|:---:|---|
| **Modality Reliability Weighting** | Yes (Self-Calibrated) | Yes (Uncertainty-based) | Yes (Spectral distance) | **Complete Conceptual Overlap** |
| **Suppression of Corrupted Channels**| Yes (Adaptive suppression)| Yes (Masking/Fallback)| Yes (`effective_mask`) | **Directly Anticipated by Prior Art** |
| **Missing Modality Handling** | Yes (Register tokens) | Yes (Robust fusion) | Yes (Zero-masking) | **Standard Industry Technique** |
| **Benchmark Datasets** | CMU-MOSI, CMU-MOSEI | CMU-MOSI, CMU-MOSEI | CMU-MOSEI (only) | Prior art has broader validation |
| **Target Task Performance** | Accuracy > 75% (Binary) | Accuracy > 78% (Binary) | Acc ~45.6% (3-class) | **Tri-SACTA is dramatically weaker** |
| **Test-Time Adaptation Updates** | No (Clean feedforward) | No (Clean feedforward) | Yes (Entropy + Align) | **Tri-SACTA TTA degrades performance ($p=0.046$)** |
| **Online Conformal Coverage** | No | No | Yes (Online ACI) | **Unique, but cannot rescue a failing classifier** |

---

## 3. Novelty & Journal Viability Analysis

1. **Failure of TTA Novelty:**
   Tri-SACTA's original proposed novelty was test-time gradient adaptation. However, as demonstrated in V10, gradient adaptation actively degrades performance on CMU-MOSEI relative to leaving weights frozen. Therefore, the TTA component cannot be claimed as a positive scientific contribution.
2. **Defeated Novelty of "Safe Suppression":**
   Attempting to pivot the paper to "safe adaptation suppression / reliability gating" runs head-on into `RegCal-Net` (September 2026) and `Wu et al.` (August 2026). Both papers already published reliability-aware suppression and quality-calibrated fusion on CMU-MOSEI.
3. **Conformal Prediction Isolation:**
   While Online ACI provides valid coverage (~89.8% under non-exchangeable shifts), Conformal Prediction is a post-processing calibration layer. Reviewers in *Multimedia Tools and Applications* or *Multimedia Systems* evaluate the primary predictive system. An accurate conformal set of size 5.0 out of 7 classes produced by a backbone with 24% accuracy will not survive peer review.
