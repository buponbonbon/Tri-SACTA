# V10-03: ALGORITHMIC FAILURE & CAUSAL MECHANISM ANALYSIS

**Repository:** `buponbonbon/Tri-SACTA`  
**Base Commit:** `d146a06b1db3a59f01d3a04b586804bdfa69c03c`  
**Date:** November 23, 2026  
**Auditor:** Principal AI Research Scientist & Adversarial Q1 Reviewer  

---

## 1. The Central Causal Finding (RQ2)

Our 6-way causal ablation matrix on the 3,000-event streaming benchmark isolates the individual contributions of spectral gating, unsupervised entropy minimization, and certainty-weighted anchor alignment:

| Variant | Spectral Reliability Gating | Gradient Adaptation ($\Delta \theta > 0$) | Anchor Alignment ($\lambda_{\text{align}}$) | Macro-F1 | Accuracy | Adapted Batches | Mean Param $\Delta$ |
|---|---|---|---|---:|---:|---:|---:|
| **Source-Only** | No | No | No | 0.2152 | 0.2390 | 0 | 0.0000 |
| **Reliability-Only** | Yes | No | No | **0.2156** | **0.2393** | 0 | 0.0000 |
| **AdapterNormTTA** | No | Yes (All channels) | No | 0.2117 | 0.2353 | 96 | 1.3015 |
| **Selective TTA (No Align)** | Yes | Yes (Adaptable set) | No ($\lambda=0$) | 0.2147 | 0.2383 | 40 | 0.2750 |
| **Tri-SACTA Full** | Yes | Yes (Adaptable set) | Yes ($\lambda=0.5$) | 0.2147 | 0.2383 | 40 | 0.2750 |

### Statistical Contrast:
1. `Reliability-Only` vs `Source-Only`: **$\Delta = +0.0005$** (R3: **+0.0023**, R5: **+0.0095**).
2. `Tri-SACTA` vs `Reliability-Only`: **$\Delta = -0.0009$ (95% CI: [-0.0020, -0.0000], $p = 0.046$, STATISTICALLY SIGNIFICANT DEGRADATION)**.
3. `Tri-SACTA` vs `Selective TTA (No Align)`: **$\Delta = +0.0000$ (95% CI: [+0.0000, +0.0000], $p = 1.000$, ZERO CAUSAL IMPACT)**.
4. `AdapterNormTTA` vs `Source-Only`: **$\Delta = -0.0035$** (Severe unconstrained negative transfer).

---

## 2. In-Depth Failure Mode Diagnosis

### Failure Mode B & C: Confirmation Bias in Unsupervised Entropy Minimization
- **Mechanism:** On CMU-MOSEI 7-class sentiment, the top-1 accuracy of the source model is only **~24%–28%**. Consequently, for **over 70% of streaming samples**, the model's argmax class prediction is **incorrect**.
- **The Confirmation Bias Trap:**
  $$\mathcal{L}_{\text{entropy}}(\theta) = -\sum_{c=1}^{C} p_{\theta}(y=c \mid \mathbf{x}) \ln p_{\theta}(y=c \mid \mathbf{x})$$
  Gradient descent on entropy minimizes spread, pushing the predicted distribution toward a one-hot Dirac delta on the *current highest logit*. When 72% of predictions are wrong, minimizing entropy actively forces the adapter weights to become **confidently wrong**.
- **Evidence:** Unconstrained adaptation (`AdapterNormTTA`) drops Macro-F1 from 0.2152 down to 0.2117 (-0.0035).

### Failure Mode B: Weak Anchor Teacher Inversion
- **Mechanism:** In Tri-SACTA, when a primary modality (Text) drifts (as in Regime 3), non-drifting modalities (Audio and Visual) are designated as **Anchors** ($\mathcal{A}_t$). The adaptable text heads are trained to align with the anchor teacher:
  $$\mathcal{L}_{\text{align}} = \sum_{u \in \mathcal{U}_t, a \in \mathcal{A}_t} w_a D_{\text{KL}}\left( p_u \parallel \text{sg}(p_a) \right)$$
- **The Inversion Problem:**
  - On CMU-MOSEI, Text is the overwhelming carrier of sentiment signal (unimodal F1 $\approx 0.2458$).
  - Audio and Visual have unimodal Macro-F1 of only **0.125** and **0.147** respectively (barely distinguishing from random chance $1/7 = 0.1428$).
  - Even though Audio/Visual experienced zero distribution shift (so their spectral shift score is near 0), their **semantic task capacity is abysmal**.
  - While certainty weighting ($w_a = 1 - H(p_a)/\ln C$) suppresses uniform anchors, whenever an audio anchor outputs high confidence for the wrong class, it acts as a noisy teacher that corrupts the student text representations.
- **Evidence:** Comparing `Tri-SACTA Full` ($\lambda_{\text{align}}=0.5$) with `Selective TTA No Align` ($\lambda_{\text{align}}=0.0$) yields **identical Macro-F1 (0.2147)** and identical accuracy (0.2383). The anchor alignment loss contributes zero empirical benefit over entropy minimization alone.

### Failure Mode E: Low-Dimensional Spectral Shift Detection (OpenFace AUs)
- In RQ1 analysis:
  - Text ($D=768$): AUROC = **1.0000**, AUPRC = **1.0000**
  - Audio ($D=74$): AUROC = **0.9992**, AUPRC = **0.9998**
  - Visual ($D=35$): AUROC = **0.1578**, AUPRC = **0.7200**
- **Root Cause:** $D=35$ OpenFace facial action units exhibit large natural facial variance across speakers that overlaps with simulated Gaussian blur and occlusion. The log-eigenvalue covariance spectrum of clean batches varies between 4.0 and 10.0, causing the detector to fail at cleanly separating moderate visual shift from natural speaker dynamics.

### Failure Mode F: Complexity of Dynamic Conformal Heuristic
- In RQ3 analysis:
  - `GlobalACI`: Coverage = **0.9003**, Set Size = **5.04**, Coverage Gap = **0.0003**
  - `Fixed Mask ACI`: Coverage = **0.8983**, Set Size = **5.03**, Coverage Gap = **0.0017**
  - `Dynamic Mask ACI`: Coverage = **0.8977**, Set Size = **5.04**, Coverage Gap = **0.0023**
- The dynamic step-size heuristic ($\eta_t = \eta_0(1 + \beta_1 v + \beta_2 \text{dis})$) introduces two extra hyperparameters ($\beta_1, \beta_2$) but yields **no Pareto improvement** over standard fixed-step per-mask ACI or single-scalar Global ACI.

---

## 3. Why `Reliability-Only` Is the True Scientific Winner

`Reliability-Only` avoids all failure modes above:
1. **Zero Confirmation Bias:** Weights are never updated ($\Delta \theta = 0$). The pre-trained multimodal representation remains uncorrupted.
2. **Effective Channel Isolation:** Corrupted channels (where noise renders features harmful) are zeroed out from cross-modal attention via `effective_mask`.
3. **Dynamic Re-weighting:** Moderately shifted channels are down-weighted via $\exp(-\tau_m d_t^{(m)})$ in pooling.
4. **Empirical Superiority:** Outperforms Source-Only in R3 (**0.1953 vs 0.1930**) and R5 (**0.1334 vs 0.1239**), while outperforming all gradient adaptation variants overall (**0.2156 vs 0.2147 vs 0.2117**).
