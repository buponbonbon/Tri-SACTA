# Tri-SACTA Empirical Evaluation Protocol V2

**Document Version:** 2.0  
**Scope:** Strict Benchmark Specifications for Continual Test-Time Adaptation and Conformal Control  

---

## 1. Evaluation Protocols: Prequential vs. Transductive

To eliminate evaluation ambiguity and ensure reproducible parity with external literature, Tri-SACTA defines two explicit protocol modes:

### 1.1. Strict Prequential Protocol (Default Deployment Setting)
In this setting, the model must commit to a prediction **before** performing any adaptation step on the current batch:
```text
For each test step t:
    1. Receive unlabeled multimodal batch (X_t, m_t)
    2. Compute spectral shift scores d_t and reliability weights r_t
    3. Forward pass in model.eval() mode -> Emit predictions and prediction sets C_t
    4. Perform anchor-guided selective adaptation on X_t (model parameters updated: θ_t -> θ_{t+1})
    5. Receive label feedback y_t (or enqueue if delay δ > 0)
    6. Evaluate empirical metrics strictly against the pre-adaptation emitted predictions
```

### 1.2. Transductive Batch TTA Protocol (Common in Visual TTA Literature)
In this setting, test-time adaptation occurs on the batch before final inference:
```text
For each test step t:
    1. Receive unlabeled batch (X_t, m_t)
    2. Perform selective adaptation on X_t -> θ_t
    3. Run clean forward pass in model.eval() -> Emit post-adaptation predictions
    4. Evaluate metrics
```
*Note:* All comparative experiments report results using the same protocol across all methods (Source-Only, TENT, SAR, READ, Tri-SACTA).

---

## 2. Continual Compound Shift Benchmark Streams

### 2.1. Six-Regime Protocol V2 Stream
The test stream consists of 6 sequential regimes:

| Regime | Description | Visual Modality | Audio Modality | Text Modality | Missing Rate |
|---|---|---|---|---|:---:|
| **R1** | Clean Baseline | Original | Clean (SNR $\infty$) | Clean transcript | 0% |
| **R2** | Audio Degradation | Original | Gaussian noise (SNR 5 dB) | Clean transcript | 0% |
| **R3** | ASR Text Drift | Original | Clean (SNR $\infty$) | Simulated WER 35% drift | 0% |
| **R4** | Visual Occlusion + Dropout | Occlusion 25% + Blur | Clean | Clean transcript | 30% |
| **R5** | Severe Compound Shift | Blur 1.5 + Occlusion 35% | Gaussian noise (SNR 0 dB) | Simulated WER 40% drift | 50% |
| **R6** | Return-to-Clean Recovery | Original | Clean | Clean | 0% |

*Key Evaluation Objectives:*
- **R1:** Source baseline retention, detector False Positive Rate (FPR $\le 5\%$).
- **R2–R4:** Modality-selective adaptation under isolated single-channel shifts.
- **R5:** Robustness under compound multi-channel failure and high missingness.
- **R6:** Catastrophic forgetting measurement upon return to the source distribution.

---

## 3. Baselines & Control Methods

All baselines operate on the exact same feature sequences, data loaders, and random seeds:

1. **Source-Only + Static Split CP:**
   - Frozen source backbone without test-time adaptation.
   - Standard Split Conformal Prediction calibrated on source validation/calibration split.
2. **TENT + Online ACI (Wang et al., ICLR 2021):**
   - Updates affine parameters of normalization layers by minimizing entropy of fused logits.
   - Conformal control via Online ACI.
3. **SAR + Online ACI (Niu et al., ICLR 2023):**
   - Sharpness-Aware Reliable TTA: filters high-entropy noisy samples and applies periodic parameter restoration toward source.
4. **READ / Multimodal-CTTA + Online ACI (Adapted from AV-CTTA, CVPR 2023):**
   - Multimodal test-time adaptation enforcing cross-modal prediction consistency across all active channels.
5. **Tri-SACTA + Online ACI (Ours):**
   - Modality-wise spectral shift detector + Anchor-guided selective low-rank adaptation + Mask-and-disagreement online conformal control.

---

## 4. Evaluation Metrics & Statistical Testing

### 4.1. Metrics Matrix
1. **Predictive Performance:**
   - Unweighted Macro-F1 with fixed full class support: $C=7$.
   - Classification Accuracy (%).
2. **Calibration & Uncertainty:**
   - Expected Calibration Error (ECE) with 15 equal-width bins.
   - Negative Log-Likelihood (NLL) computed in FP32.
   - Multi-class Brier Score in float64.
3. **Conformal Validity & Efficiency:**
   - Empirical Coverage Rate (Target $1 - \alpha = 90.0\%$).
   - Coverage Gap: $|\text{Cov} - 0.90|$.
   - Average Prediction Set Size $|\mathcal{C}_t|$.
4. **Resource Footprint:**
   - Trainable parameter count (< 2M budget).
   - Peak GPU VRAM allocated and reserved (MB) (< 1.5GB budget).
   - End-to-end latency per sample (ms/sample).

### 4.2. Statistical Significance Testing
- Fixed seed replicates: $\mathcal{S} = [42, 100, 2026, 777, 999]$.
- Paired two-tailed Student's t-test (`scipy.stats.ttest_rel`) and non-parametric Wilcoxon signed-rank test (`scipy.stats.wilcoxon`).
- Significance threshold: $p < 0.05$.
