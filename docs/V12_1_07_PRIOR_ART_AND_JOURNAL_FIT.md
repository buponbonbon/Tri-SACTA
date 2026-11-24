# Tri-SACTA V12.1: Prior Art Rectification and Target Journal Fit

**Document Version:** 1.0.0 (V12.1 Rectified)  
**Status:** Authoritative Literature & Journal Fit Analysis  
**Venue Target:** *Multimedia Tools and Applications* (MTAP, Springer) / *Multimedia Systems* / *Computing*  
**Verified Financial Policy:** **$0 Mandatory APC** under subscription publishing model

---

## 1. Prior Art Rectification & Accurate Attribution

### Forensic Rectification (Defect D-12)
In V12, prior art tables cited Wu et al. (MDPI Electronics 2026) and claimed cross-paper numerical superiority while misclassifying the method as an online backpropagation test-time adaptation technique. In V12.1, this claim is fully withdrawn and rectified based on strict scientific provenance:

### A. Wu et al. (MDPI Electronics 2026)
- **True Methodological Characterization:** Wu et al. proposed a static multi-feature fusion architecture where cross-modal attention generates inter-modality reliability weights for audio, visual, and textual tokens during forward propagation. It is **not** an online test-time adaptation algorithm, does **not** update network weights at test time, and does **not** perform dynamic sensory admission or latency budgeting.
- **Tri-SACTA V12.1 Distinction:** Tri-SACTA V12.1 does not claim algorithmic superiority over Wu et al.'s internal feature representation. Rather, Tri-SACTA V12.1 operates as an **orchestration layer** sitting *above* the neural backbone. It decides *whether* and *when* to acquire auxiliary sensor streams (audio/visual), evaluates whether the primary text stream is reliable via standardized centroid distance, and provides conformal set coverage guarantees.

### B. Withdrawal of Online Backpropagation TTA Claims
Previous project iterations (V7–V11) attempted to demonstrate that updating adapter weights online via entropy minimization (Tent, SAR) or spectral contrastive objectives (Tri-SACTA TTA) improved multimodal sentiment accuracy under domain shift.
**The definitive empirical finding of the V7–V11 audit series was that online backpropagation TTA causes severe negative transfer on CMU-MOSEI**, consistently underperforming the frozen source-only baseline across all six evaluated regimes.
Tri-SACTA V12.1 completely repudiates any claim of TTA algorithmic SOTA. All experimental comparisons are strictly framed around **system-level resource orchestration**, **latency bounding**, **sensor acquisition cost**, and **selective conformal prediction**.

---

## 2. Competitive Positioning Matrix

| Dimensional Axis | Static Unimodal (Text MLP) | Static Multimodal (Always-On) | Feature-Weighted Fusion (Wu et al. 2026) | Tri-SACTA V12.1 BRMO |
| :--- | :--- | :--- | :--- | :--- |
| **Sensor Acquisition** | Fixed Text only | Bulk All Sensors Always | Bulk All Sensors Always | **Lazy, Budget-Admissible On-Demand** |
| **Compute / Wire Cost** | Low (1.0) | High (3.5, $+250\%$) | High (3.5, $+250\%$) | **Dynamic Pareto (1.0 – 2.9, $\ge 15\%$ savings)** |
| **Inference Latency** | Ultra-low (~1 ms) | High (~13 ms, $13\times$) | High (~14 ms) | **Multi-tier (~1 ms fast, ~11 ms escalated)** |
| **ASR Drift Detection** | None (Blind pass) | Degraded (Full noise pass) | Static attention reweighting | **Standardized Centroid Triage Gating** |
| **Deadline Bounding** | None (Static) | None (Static) | None (Static) | **Dynamic Admission Control ($D_{\text{rem}}$)** |
| **Uncertainty Quantification** | Softmax argmax | Softmax argmax | Softmax argmax | **Split Conformal Prediction Sets ($1-\alpha=0.90$)** |

---

## 3. Target Journal Alignment: *Multimedia Tools and Applications* (MTAP)

### Journal Profile & Metrics
- **Publisher:** Springer Nature
- **ISSN:** 1380-7501 | **eISSN:** 1573-7721
- **SCImago Journal Rank (SJR):** Q1 (Computer Vision & Multimedia 2023–2024), Q2 (2025 transition).
- **Mandatory Author Fee:** **$0 (Zero USD)** under the traditional Subscription / Green Open Access publishing model. No mandatory page or color charges.

### Scope and Systems Fit
MTAP specifically welcomes original research addressing the practical systems engineering and algorithmic challenges of multimedia data streams:
1. *Resource-Constrained Multimedia Processing:* Tri-SACTA V12.1 directly tackles the edge-cloud bandwidth and computation bottleneck by preventing unnecessary audio/visual stream ingestion when text triage is sufficient.
2. *Robust Multimodal Systems:* V12.1 provides systematic fault tolerance under sensor missingness, frame corruption, and acoustic degradation without requiring model retraining.
3. *Calibrated Uncertainty for Human-in-the-Loop:* By integrating split conformal prediction with finite-sample coverage guarantees ($1-\alpha=0.90$), Tri-SACTA V12.1 provides dependable risk management for real-world multimedia affective computing systems.

### Alternative $0 APC Submission Venues
Should editorial timelines or review bandwidth dictate submission elsewhere, the identical manuscript framing is pre-validated for:
- *Multimedia Systems* (ACM/Springer, SJR Q2, $0 mandatory APC)
- *Computing* (Springer, SJR Q2, $0 mandatory APC)
