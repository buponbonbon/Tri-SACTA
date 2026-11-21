# Tri-Modal Spectral-Aligned Continual Test-Time Adaptation with Non-Exchangeable Online Conformal Control under Compound Modality Shifts

## 1. Core Research Objective

This project addresses the problem of **continual test-time adaptation (TTA)** for multimodal sentiment and emotion recognition systems when deployed under **compound modality shifts** — scenarios where multiple input modalities (visual, audio, text, prosody) experience simultaneous and evolving distribution shifts at test time.

### Key Contributions

1. **Spectral Shift Detection**: A kernel-based spectral method that estimates per-modality distribution shift magnitude using eigenvalue decomposition of the Maximum Mean Discrepancy (MMD) operator, enabling fine-grained modality reliability scoring.

2. **Reliability-Weighted Fusion**: A dynamic cross-modal attention mechanism that down-weights shifted modalities based on spectral shift scores, preventing catastrophic error propagation across the fusion pipeline.

3. **Spectral-Aligned Adaptation**: A test-time adaptation objective that jointly minimizes confident pseudo-label loss and a spectral alignment regularizer to prevent forgetting while adapting to shifted distributions.

4. **Non-Exchangeable Online Conformal Prediction**: An online Adaptive Conformal Inference (ACI) framework that provides valid prediction sets with finite-sample marginal coverage guarantees under non-exchangeable (temporally correlated, shifting) test streams.

5. **Mask-Conditional Coverage**: Extension of conformal prediction to provide coverage guarantees conditional on the modality availability mask, ensuring calibrated uncertainty quantification across all missingness patterns.

---

## 2. Hard Constraints

| Constraint | Specification |
|---|---|
| **GPU Budget** | ≤ 1× NVIDIA A100 (40 GB) for training; ≤ 1× RTX 3090 (24 GB) for TTA |
| **TTA Latency** | ≤ 150 ms per batch (batch_size=32) on RTX 3090 |
| **TTA Parameters** | Only low-rank adapters + normalization layers updated (< 5% of total) |
| **Coverage Guarantee** | Online marginal coverage ≥ 1 − α (α = 0.1) under all shift regimes |
| **Reproducibility** | 5 random seeds; mean ± std for all metrics |
| **Precision** | Mixed precision (AMP FP16) for all forward/backward passes |

---

## 3. Mathematical Specification

### 3.1 Feature Extraction

For each sample $x$, we extract four feature vectors:

$$z^v = f_{\text{CLIP}}(x^v) \in \mathbb{R}^{512}$$
$$z^a = f_{\text{WavLM}}(x^a) \in \mathbb{R}^{768}$$
$$z^t = f_{\text{DeBERTa}}(x^t) \in \mathbb{R}^{768}$$
$$z^p = f_{\text{eGeMAPS}}(x^p) \in \mathbb{R}^{88}$$

Each is projected to a shared fusion space: $h^m = W^m z^m + b^m \in \mathbb{R}^{256}$ for $m \in \{v, a, t, p\}$.

### 3.2 Spectral Shift Detection

Given source features $\{z_i^m\}_{i=1}^{n_s}$ and target batch $\{z_j^m\}_{j=1}^{n_t}$ for modality $m$, we compute the kernel MMD operator:

$$\hat{C}^m = \frac{1}{n_t} \Phi_t^T \Phi_t - \frac{1}{n_s} \Phi_s^T \Phi_s$$

where $\Phi$ denotes the RBF kernel feature map with bandwidth $\gamma$.

The shift score for modality $m$ is:

$$d_t^m = \sum_{k=1}^{K} |\lambda_k(\hat{C}^m)|$$

where $\lambda_k$ are the top-$K$ eigenvalues of $\hat{C}^m$.

### 3.3 Reliability Weighting

Reliability weight for modality $m$ at time $t$:

$$w_t^m = \frac{\exp(-d_t^m / \tau) \cdot \mathbb{1}[m \text{ available}]}{\sum_{m'} \exp(-d_t^{m'} / \tau) \cdot \mathbb{1}[m' \text{ available}]}$$

### 3.4 Cross-Modal Fusion

Fused representation via weighted cross-attention:

$$h_{\text{fused}} = \text{CrossAttn}(\{w_t^m \cdot h^m\}_{m=1}^{M})$$

Output logits: $\hat{y} = W_c \cdot h_{\text{fused}} + b_c$.

### 3.5 TTA Objective

The test-time adaptation loss:

$$\mathcal{L}_{\text{TTA}} = \mathcal{L}_{\text{pseudo}}(\hat{y}, \tilde{y}) + \lambda_{\text{align}} \cdot \mathcal{L}_{\text{align}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}$$

where:

- **Pseudo-label loss**: $\mathcal{L}_{\text{pseudo}} = - \sum_{c} \tilde{y}_c \log \hat{y}_c$ using confident pseudo-labels ($\max_c \hat{y}_c > \theta_{\text{high}}$)
- **Spectral alignment loss**: $\mathcal{L}_{\text{align}} = \sum_m w_t^m \cdot d_t^m$ (drives adapted features toward source distribution)
- **Regularization loss**: $\mathcal{L}_{\text{reg}} = \|\theta_{\text{adapt}} - \theta_{\text{adapt}}^0\|_2^2$ (prevents catastrophic forgetting)

Only adapter parameters $\theta_{\text{adapt}}$ (low-rank matrices of rank $r = 8$) and batch normalization statistics are updated.

### 3.6 Online Adaptive Conformal Inference (ACI)

For each test sample $x_t$ with softmax output $\hat{\pi}_t$:

1. **Score function** (Adaptive Prediction Sets):

$$s(x, y) = \sum_{j: \hat{\pi}_{(j)} > \hat{\pi}_y} \hat{\pi}_{(j)} + u \cdot \hat{\pi}_y, \quad u \sim \text{Uniform}(0, 1)$$

2. **Prediction set**: $\mathcal{C}_t(x_t) = \{y : s(x_t, y) \leq \hat{q}_t\}$

3. **Online quantile update**:

$$\hat{q}_{t+1} = \hat{q}_t + \eta_t \cdot (\alpha - \mathbb{1}[y_t \notin \mathcal{C}_t(x_t)])$$

4. **Adaptive step size**:

$$\eta_{t+1} = \beta_1 \cdot \eta_t + \beta_2 \cdot |\text{err}_t - \alpha|$$

---

## 4. Datasets

### 4.1 CMU-MOSEI (Primary)

- **Task**: 7-class sentiment analysis (strongly negative → strongly positive)
- **Modalities**: Video (face crops), Audio (speech), Text (transcripts), Prosody (eGeMAPS)
- **Size**: ~23,500 utterances from 1,000 speakers
- **Splits**: Speaker-disjoint train/val/test

### 4.2 MELD (Secondary)

- **Task**: 7-class emotion recognition (anger, disgust, fear, joy, neutral, sadness, surprise)
- **Modalities**: Same 4 modalities
- **Size**: ~13,700 utterances from TV show dialogues
- **Splits**: Episode-based train/val/test

---

## 5. Baselines

### TTA Methods
| Method | Description |
|---|---|
| **Source-only** | No adaptation (frozen source model) |
| **TENT** | Entropy minimization on batch norm parameters |
| **SAR** | Sharpness-aware and reliable entropy minimization |
| **CoTTA** | Continual TTA with augmentation-averaged pseudo-labels |
| **Tri-SACTA (Ours)** | Full spectral-aligned continual TTA framework |

### Conformal Methods
| Method | Description |
|---|---|
| **Split CP** | Fixed-threshold split conformal prediction |
| **ACI** | Online ACI without modality conditioning |
| **Mask-ACI** | Mask-conditional online ACI (Ours) |

---

## 6. Evaluation Metrics

### Classification Performance
- **Accuracy** (Acc): Overall classification accuracy
- **Macro-F1** (F1): Class-balanced F1 score
- **Per-regime Accuracy**: Accuracy broken down by shift regime

### Conformal Prediction Quality
- **Marginal Coverage**: Empirical coverage rate (target ≥ 90%)
- **Conditional Coverage**: Coverage conditional on modality mask
- **Average Set Size**: Mean prediction set cardinality (smaller is better)
- **Set Size Stability**: Variance of set sizes over time

### Efficiency
- **TTA Latency**: Wall-clock time per adaptation step (ms)
- **Peak VRAM**: Maximum GPU memory usage during TTA (GB)
- **Adaptable Parameters**: Number of parameters updated at test time

---

## 7. Repository Structure

```
Tri-SACTA/
├── PROJECT_SPEC.md                 # This specification document
├── requirements.txt                # Python dependencies
├── configs/
│   ├── mosei_mvp.yaml              # CMU-MOSEI configuration
│   ├── meld.yaml                   # MELD configuration
│   └── shift_regimes.yaml          # Shift regime definitions
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # MultimodalDataset + speaker-disjoint splits
│   │   └── shift_generator.py      # ShiftStreamGenerator for TTA evaluation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fusion_backbone.py      # MultimodalFusionModel (cross-attention + adapters)
│   │   └── spectral_detector.py    # SpectralShiftDetector (kernel MMD + eigenvalues)
│   ├── adaptation/
│   │   ├── __init__.py
│   │   ├── tri_sacta.py            # TriSACTA (main adaptation algorithm)
│   │   └── baselines_tta.py        # TENT, SAR, CoTTA baselines
│   ├── conformal/
│   │   ├── __init__.py
│   │   ├── online_aci.py           # OnlineACI (adaptive conformal inference)
│   │   └── static_cp.py            # SplitConformalPredictor, MaskConditionalCP
│   └── utils/
│       ├── __init__.py
│       └── metrics.py              # MetricsTracker, VRAMProfiler, LatencyProfiler
├── scripts/
│   ├── extract_features.py         # Feature extraction pipeline
│   ├── train_source.py             # Source model training
│   ├── run_tta.py                  # TTA evaluation pipeline
│   └── run_conformal.py            # Conformal prediction evaluation
├── notebooks/
│   └── analysis.ipynb              # Results analysis and visualization
└── results/
    └── .gitkeep
```
