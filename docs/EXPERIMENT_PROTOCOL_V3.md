# TRI-SACTA V3: SCIENTIFIC EXPERIMENT PROTOCOL & BENCHMARK SPECIFICATION

**Version:** 3.0-Scientific-Validation  
**Date:** 2026-11-21  
**Target Venue:** Q1 Journal (Information Fusion / IEEE TMM / Neural Networks)  
**Hardware Profile:** Laptop GPU (NVIDIA RTX 2050 4GB VRAM), PyTorch 2.5.1+cu121, FP16 AMP  

---

## 1. Protocol Foundations & Time Semantics

In online multimodal Continual Test-Time Adaptation (CTTA) and Online Conformal Prediction (CP), the temporal ordering of events governs scientific validity. We strictly delineate two evaluation protocols: **Prequential** (default, deployment-faithful) and **Transductive** (diagnostic).

```
Prequential Protocol (Streaming Deployment):
At step t:
[Arrival x_t] ──► [Predict f_{t-1}(x_t)] ──► [Emit Set C_t(x_t)] ──► [Adapt: theta_t = Adapt(theta_{t-1}, x_t)] ──► [Delayed Feedback y_t arrives at t + delay]
```

### 1.1. Strict Prequential Protocol (Primary Evaluation)
1. **Event Arrival ($t$):** Multimodal batch $B_t = \{x_{t,i}, m_{t,i}\}_{i=1}^{B}$ arrives from the input stream.
2. **Pre-Update Inference:** The current model state $\theta_{t-1}$ computes predictions and logits:
   $$\hat{p}_t(y \mid x_{t,i}) = \text{Softmax}\big(f_{\theta_{t-1}}(x_{t,i}, m_{t,i})\big)$$
3. **Conformal Set Emission:** The online conformal controller emits prediction sets $\mathcal{C}_t(x_{t,i})$ using calibration state $\alpha_{t, m_{t,i}}$ and historical non-conformity order statistics:
   $$\mathcal{C}_t(x_{t,i}) = \left\{ y \in \mathcal{Y} : s(x_{t,i}, y) \le \hat{q}_{1 - \alpha_{t, m_{t,i}}} \right\}$$
   *Note: Sets are deeply cloned and frozen in the event log before any model adaptation occurs.*
4. **Test-Time Adaptation:** Lightweight adapter parameters are updated using unsupervised objectives evaluated on $B_t$:
   $$\theta_t = \theta_{t-1} - \eta_{\text{TTA}} \nabla_{\theta} \mathcal{L}_{\text{TTA}}(B_t; \theta_{t-1})$$
5. **Delayed Feedback Arrival:** Ground-truth labels $y_{t,i}$ are queued and only exposed to the conformal controller when $t_{\text{current}} - t \ge \Delta_{\text{delay}}$. Once matured, miscoverage indicators $e_{t,i} = \mathbf{1}\{y_{t,i} \notin \mathcal{C}_t(x_{t,i})\}$ trigger online step updates.

### 1.2. Transductive Protocol (Diagnostic Reference)
Under the transductive protocol, the adapter updates on $B_t$ *prior* to evaluation:
$$\theta_t = \theta_{t-1} - \eta_{\text{TTA}} \nabla_{\theta} \mathcal{L}_{\text{TTA}}(B_t; \theta_{t-1})$$
$$\hat{p}_t(y \mid x_{t,i}) = \text{Softmax}\big(f_{\theta_t}(x_{t,i}, m_{t,i})\big)$$
*Usage:* Used strictly for diagnostic ablation to quantify the single-step adaptation margin against the prequential stream.

---

## 2. Two-Tier Corruption Specification

To eliminate the scientific discrepancy identified in R3-09 (confounding latent feature perturbations with physical sensor corruption), all shifts are categorized into two explicit tiers:

### Tier A: Latent Representation Perturbations (Default Fast Feature-Level Benchmark)
Operates directly on the pre-extracted multimodal embeddings ($h^{(m)} \in \mathbb{R}^{D_m}$):
- **Additive Representation Noise:** $\tilde{h}^{(m)} = h^{(m)} + \epsilon$, where $\epsilon \sim \mathcal{N}\left(0, \sigma^2 \frac{\|h^{(m)}\|_2^2}{D_m} I\right)$. Severity is parameterized by latent SNR:
  $$\text{SNR}_{\text{latent}} = 10 \log_{10}\left(\frac{\|h^{(m)}\|_2^2}{\|\epsilon\|_2^2}\right) \text{ dB}$$
- **Representation Dimension Masking:** Random subspace erasure zeroing a fraction $\rho \in [0, 0.8]$ of feature dimensions.
- **Dynamic Sensor Dropout:** Complete erasure of modality $m$ ($m_t^{(m)} = 0, \tilde{h}^{(m)} = \mathbf{0}$) simulating total sensor blackout.

### Tier B: Physical Signal Corruptions (Raw Media Pipeline)
Operates on the raw input signals before sequential feature extraction:
- **Audio:** Additive environmental acoustic noise (DNS Challenge noises, babble, traffic) at physical SNR from $+20\text{ dB}$ down to $-5\text{ dB}$.
- **Prosody:** Extracted via `openSMILE` eGeMAPS functionals from the corrupted audio waveform (capturing natural acoustic degradation).
- **Text:** Lexical drift simulated via phoneme confusion matrices, speech recognizer word error rates (WER 15% to 50%), or real ASR transcript streams.
- **Visual:** Spatio-temporal motion blur, optical occlusion masks (cutout), and Gaussian lens noise applied to video frames prior to ViT encoding.

---

## 3. The 6-Regime Continual Shift Stream

The stream benchmark evaluates continual adaptability, resilience to catastrophic forgetting, and recovery. In V3, the benchmark executes 6 discrete regimes over the chronological test stream:

| Regime ID | Regime Name | Primary Modality Affected | Corruption Mechanism (Tier A / Tier B) | Duration | Scientific Purpose |
|---|---|---|---|---|---|
| **R1** | **Clean Anchor** | None (All clean) | Identical to source distribution ($\sigma=0$, mask=1) | $N_{\text{batch}}$ | Verify source calibration and false-alarm rejection |
| **R2** | **Audio Degradation** | Audio, Prosody | Latent noise $\text{SNR} \in [10, 0]\text{ dB}$ / Physical acoustic noise | $N_{\text{batch}}$ | Test adaptation to auditory channel drift |
| **R3** | **Text / ASR Drift** | Text | Dimension masking $\rho \in [0.2, 0.5]$ / Simulated ASR WER drift | $N_{\text{batch}}$ | Test adaptation to linguistic transcription drift |
| **R4** | **Visual Occlusion** | Visual | Spatial dimension blackout $\rho \in [0.3, 0.7]$ / Frame blur & cutout | $N_{\text{batch}}$ | Test adaptation to optical degradation |
| **R5** | **Compound Outage** | Random Subsets | Joint corruptions + correlated sensor dropout ($m_t \in \{0,1\}^M$) | $N_{\text{batch}}$ | Stress test hard gating and non-anchor skipping |
| **R6** | **Clean Recovery** | None (All clean) | Identical to R1 clean distribution | $N_{\text{batch}}$ | **Quantify catastrophic forgetting & backward transfer** |

### Backward Transfer (BWT) & Forgetting Metric
$$\text{BWT} = \text{Metric}(R6) - \text{Metric}(R1)$$
A negative $\text{BWT} \ll 0$ indicates catastrophic forgetting induced by test-time parameter drift.

---

## 4. Factorial Experimental Matrix

To isolate the individual scientific contributions of (i) Feature Representation, (ii) TTA Adaptation, and (iii) Online Conformal Prediction (resolving R3-14), we employ a $4 \times 3$ factorial design:

### Dimension 1: Predictor / Adaptation Method
1. **Source-Only (Frozen):** Zero test-time parameter updates ($\theta_t = \theta_0$).
2. **Tent-LN:** Test-Time Entropy Minimization updating only LayerNorm affine parameters ($\gamma, \beta$) across all modalities without spectral gating.
3. **Consistency-TTA:** Multi-modal cross-attention consistency adaptation without selective anchor gating.
4. **Tri-SACTA (Ours):** Spectral shift-guided selective low-rank projection adaptation with hard attention gating.

### Dimension 2: Conformal Uncertainty Controller
1. **Static CP (Exchangeable Baseline):** Calibration thresholds $\hat{q}_{1-\alpha}$ computed on source validation split and frozen throughout the stream.
2. **Fixed-Step Mask-Conditional ACI (Reference Controller):** Independent mask queues with step $\eta_0$, unprojected state, boundary emissions ($\alpha \le 0 \implies \mathcal{Y}$, $\alpha \ge 1 \implies \emptyset$), and proven pathwise error bound:
   $$|\bar{e}_{n_g} - \alpha| \le \frac{1 + \eta_0}{\eta_0 \cdot n_g}$$
3. **Dynamic-Step Mask-Conditional ACI (Heuristic):** Adaptive step $\eta_t = \eta_0 \cdot (1 + \lambda_1 \Delta d_t + \lambda_2 \kappa_t)$ tracking distribution velocity and cross-modal disagreement.

---

## 5. Label Feedback Schedules (Resolving R3-20)

Unlike unsupervised TTA which requires zero labels, Online Conformal Prediction updates its error tracker using observed ground truth labels. We benchmark four realistic operational feedback regimes:

1. **100% Immediate Feedback ($\Delta_{\text{delay}} = 0$):** Every test label arrives before the subsequent prediction step.
2. **Delayed Feedback ($\Delta_{\text{delay}} \in \{10, 50\}$ batches):** Labels arrive after $D$ batches, testing queue maturation and stability under feedback latency.
3. **Sparse Feedback (25% and 10% Label Budget):** Labels arrive at random test steps according to an independent Bernoulli schedule $B_t \sim \text{Bernoulli}(p_{\text{budget}})$. Steps without feedback do not trigger ACI updates.
4. **Zero Test-Time Feedback (0% Budget):** Static CP operating purely from source calibration, providing the lower bound of conformal coverage under shift.

---

## 6. Statistical Rigor, Hypothesis Testing & Reporting Standards

In compliance with Q1 journal editorial standards:

1. **Unique Seed Requirement:** All multi-seed experiments are conducted across at least 5 strictly independent runs:
   $$\text{Seeds} \in \{42, 100, 2026, 777, 999\}$$
   Each seed must train its own dedicated source checkpoint from scratch. Sharing source checkpoints across test seeds is strictly prohibited.
2. **Uncertainty Reporting:**
   - Single-seed pilots ($N=1$) are reported as `mean (n=1)` or with standard deviation marked as `NA`.
   - Multi-seed runs report $\text{mean} \pm \text{std}$ across genuine independent seeds.
3. **Hypothesis Testing:**
   - Primary metric: Mean Macro-F1 across R1–R6, Worst-Regime Macro-F1, Empirical Coverage Rate $1 - \bar{e}_T$, and Mean Set Size $|\mathcal{C}|$.
   - Paired tests: For $N \ge 6$ independent seeds, report exact two-sided Wilcoxon signed-rank test. For $N = 5$ seeds, report paired differences $\Delta_i = M_{\text{Tri-SACTA}, i} - M_{\text{Baseline}, i}$ with 95% confidence intervals and acknowledge the minimum discrete two-sided p-value ($2/32 = 0.0625$).
   - Multiple comparisons: Apply Holm-Bonferroni correction across all tested regimes.

---

## 7. Hardware & Resource Budget (Laptop GPU RTX 2050 4GB)

- **Peak CTTA VRAM:** Strictly $< 1.5\text{ GB}$ (FP16 AMP enabled).
- **Sequential Feature Caching:** Stage 1 offline encoders run sequentially on CPU or single GPU with explicit cache clearing (`del model; torch.cuda.empty_cache()`).
- **Computational Latency:** Per-batch adaptation and conformal prediction time $< 50\text{ ms}$ on batch size $B=16$.
