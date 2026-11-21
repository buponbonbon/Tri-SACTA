# Tri-SACTA Method Specification V2

**Title:** Tri-Modal Spectral-Aligned Continual Test-Time Adaptation with Non-Exchangeable Online Conformal Control under Compound Modality Shifts  
**Target:** Q1 IEEE / Elsevier Venue (IEEE TMM / Information Fusion / Neural Networks)  
**Version:** 2.0 (Post-Audit Hardened)

---

## 1. System Overview & Problem Formulation

Let $\mathcal{M} = \{1, \dots, M\}$ denote the set of feature streams (Visual, Audio, Text, Prosody; $M=4$).
At test time $t \in \{1, \dots, T\}$, the system receives a streaming multimodal batch $(X_t, m_t)$, where:
- $X_t = \{x_t^{(m)} \in \mathbb{R}^{B \times D_m} : m \in \mathcal{M}\}$ denotes the observed features.
- $m_t \in \{0, 1\}^{B \times M}$ denotes the binary availability mask matrix ($m_t^{(b, m)} = 0$ if modality $m$ is missing for sample $b$).
- $y_t \in \{1, \dots, C\}^B$ denotes the ground-truth target labels, which are **unobserved** at prediction time.

Tri-SACTA solves two coupled challenges under continual compound modality shifts:
1. **Model Adaptation:** Continuously adapts lightweight projections of moderately shifted modalities using clean modalities as reliability anchors, while strictly isolating severely corrupted or missing modalities to prevent negative transfer.
2. **Statistical Uncertainty Quantification:** Maintains finite-sample and long-run empirical prediction set coverage $1 - \alpha$ via mask-conditional and shift-velocity-governed online conformal inference, robust to delayed label feedback $\delta \ge 0$.

---

## 2. Modality-Wise Spectral Shift Detector

Let $(\mu_s^{(m)}, \Sigma_s^{(m)})$ be the source anchor mean and covariance matrix for modality $m$ computed once on the source training set:
$$\mu_s^{(m)} = \frac{1}{N_s}\sum_{i=1}^{N_s} x_{s, i}^{(m)}, \quad \Sigma_s^{(m)} = \frac{1}{N_s - 1}\sum_{i=1}^{N_s} (x_{s, i}^{(m)} - \mu_s^{(m)})(x_{s, i}^{(m)} - \mu_s^{(m)})^\top$$

For a test batch $B_t$ at step $t$, let $\mathcal{I}_t^{(m)} = \{b \in \{1, \dots, B\} : m_t^{(b, m)} = 1\}$ be the set of valid samples where modality $m$ is present ($N_t^{(m)} = |\mathcal{I}_t^{(m)}|$). If $N_t^{(m)} \ge 2$, the empirical test mean and covariance are:
$$\mu_t^{(m)} = \frac{1}{N_t^{(m)}}\sum_{b \in \mathcal{I}_t^{(m)}} x_{t, b}^{(m)}, \quad \Sigma_t^{(m)} = \frac{1}{N_t^{(m)} - 1}\sum_{b \in \mathcal{I}_t^{(m)}} (x_{t, b}^{(m)} - \mu_t^{(m)})(x_{t, b}^{(m)} - \mu_t^{(m)})^\top$$

### 2.1. Dimension-Normalized Shift Metric
The raw spectral shift score is formulated as:
$$\tilde{d}_t^{(m)} = \frac{1}{D_m} \sum_{j=1}^{D_m} \frac{(\mu_{t, j}^{(m)} - \mu_{s, j}^{(m)})^2}{\sigma_{s, j}^{(m)2} + \epsilon} + \frac{\gamma}{k_m} \sum_{j=1}^{k_m} \big(\log \lambda_{t, j}^{(m)} - \log \lambda_{s, j}^{(m)}\big)^2$$
where:
- $\sigma_{s, j}^{(m)2} = \text{diag}(\Sigma_s^{(m)})_j$ is the source variance of feature dimension $j$.
- $\lambda_{\cdot, j}^{(m)}$ are the top-$k_m$ eigenvalues of $\Sigma_{\cdot}^{(m)}$ ($k_m = \min(top\_k, D_m)$).
- $\gamma > 0$ balances second-order covariance alignment against first-order mean shift.

### 2.2. Null-Score Calibration
To prevent small sample estimation noise from triggering false shift alarms on clean test distributions, the calibrated excess shift score is:
$$d_t^{(m)} = \max\Big(0, \; \tilde{d}_t^{(m)} - d_{\text{null}}^{(m)}\Big)$$
where $d_{\text{null}}^{(m)}$ is the expected shift score estimated over holdout source validation batches.

---

## 3. Reliability Weighting & Three-Way Modality Partition

Given $d_t^{(m)}$, the modality reliability proxy $r_t^{(m)}$ is computed as:
$$r_t^{(m)} = \begin{cases} \max\big(10^{-4}, \; \exp(-\tau d_t^{(m)})\big) & \text{if } N_t^{(m)} > 0 \\ 0.0 & \text{if } N_t^{(m)} = 0 \end{cases}$$
where $\tau > 0$ is a temperature hyperparameter.

Modality streams are dynamically partitioned into three disjoint sets:
$$\mathcal{A}_t = \{m \in \mathcal{M} : r_t^{(m)} \ge \theta_{\text{high}}\} \quad \text{(Clean Anchor Set)}$$
$$\mathcal{U}_t = \{m \in \mathcal{M} : \theta_{\text{low}} \le r_t^{(m)} < \theta_{\text{high}}\} \quad \text{(Adaptable Set)}$$
$$\mathcal{D}_t = \{m \in \mathcal{M} : r_t^{(m)} < \theta_{\text{low}} \text{ or } m_t^{(m)} = 0\} \quad \text{(Corrupted / Dropped Set)}$$

---

## 4. Masked Modality Transformer Fusion Backbone

The fusion backbone projects each available modality into a shared latent space $\mathbb{R}^D$:
$$h_t^{(b, m)} = f_{\text{proj}}^{(m)}(x_{t, b}^{(m)}) \in \mathbb{R}^D$$

Representations across modalities form a token sequence of shape $(B, M, D)$.
Cross-modal interaction is modeled via `MaskedModalityTransformerLayer` with attention mask $K \in \mathbb{R}^{B \times 1 \times 1 \times M}$:
$$K_{b, 1, 1, m} = \begin{cases} 0.0 & \text{if } m_t^{(b, m)} = 1 \\ -10^4 & \text{if } m_t^{(b, m)} = 0 \end{cases}$$

$$\text{Attention}(Q, K_v, V) = \text{Softmax}\left(\frac{Q K_v^\top}{\sqrt{D}} + K\right) V$$

The refined modality tokens $\tilde{h}_t^{(b, m)}$ are pooled using normalized reliability weights:
$$\bar{w}_t^{(b, m)} = \frac{r_t^{(m)} \cdot m_t^{(b, m)}}{\sum_{m'=1}^M r_t^{(m')} \cdot m_t^{(b, m')} + \epsilon}$$
$$z_t^{(b)} = \sum_{m=1}^M \bar{w}_t^{(b, m)} \tilde{h}_t^{(b, m)}$$
Final classification logits are generated by the fused classifier: $\hat{y}_t^{(b)} = g_{\text{fused}}(z_t^{(b)})$.

---

## 5. Anchor-Guided Selective Adaptation (TTA)

Adaptation runs if and only if $\mathcal{U}_t \ne \emptyset$ and $\mathcal{A}_t \ne \emptyset$.
If either condition fails (no adaptable stream, or no clean anchor to supervise adaptation), the adaptation step is safely skipped to prevent drift.

### 5.1. Selective Gradient Gating
All parameters of the foundation encoders, core transformer layers, fused classifier, anchor modality heads ($\mathcal{A}_t$), and corrupted modality heads ($\mathcal{D}_t$) are strictly frozen:
$$\forall \theta \notin \Theta_{\mathcal{U}_t}: \quad \text{requires\_grad} = \text{False}$$
Only the low-rank adapters and LayerNorm parameters of modalities in $\mathcal{U}_t$ are updated.

### 5.2. Adaptation Objective Function
$$\mathcal{L}_{\text{TTA}} = \mathcal{L}_{\text{ent}}(p_t) + \lambda_{\text{align}} \sum_{u \in \mathcal{U}_t, a \in \mathcal{A}_t} \mathcal{L}_{\text{KL}}\Big(p_t^{(u)} \,\Big\|\, \text{sg}\big(p_t^{(a)}\big)\Big) + \lambda_{\text{reg}} \sum_{\theta \in \Theta_{\mathcal{U}_t}} \|\theta - \theta_0\|_2^2$$
where:
- $\mathcal{L}_{\text{ent}}(p_t) = -\sum_{c=1}^C p_t(c) \log p_t(c)$ minimizes entropy on the fused output.
- $\text{sg}(\cdot)$ is the stop-gradient operator applied to anchor teachers.
- $\theta_0$ is the source checkpoint parameter snapshot, penalizing deviation to prevent catastrophic forgetting.

---

## 6. Mask-and-Disagreement Online Conformal Control (Online ACI)

Nonconformity scores are computed via Adaptive Prediction Sets (APS):
$$s(x, y) = \sum_{c=1}^C \hat{p}(c \mid x) \cdot \mathbb{I}\{\hat{p}(c \mid x) \ge \hat{p}(y \mid x)\}$$

Prediction sets are emitted as:
$$\mathcal{C}_t(x_t) = \{y \in \{1, \dots, C\} : s(x_t, y) \le \hat{q}_{t, m_t}\}$$
where $\hat{q}_{t, m_t}$ is the exact finite-sample order statistic rank:
$$k = \left\lceil (n_{m_t} + 1)(1 - \alpha_{t, m_t}) \right\rceil$$

### 6.1. Dynamic Step Size Governed by Velocity & Disagreement
$$\eta_t = \text{clip}\Big(\eta_0 \cdot \big(1 + \beta_1 |\Delta d_t| + \beta_2 \max(0, \kappa_t)\big), \; \eta_{\text{min}}, \; \eta_{\text{max}}\Big)$$
where:
- $|\Delta d_t| = \frac{1}{M}\sum_{m=1}^M |d_t^{(m)} - d_{t-1}^{(m)}|$ is the spectral shift velocity.
- $\kappa_t = \frac{1}{|\mathcal{A}_t \cup \mathcal{U}_t|}\sum_{m} D_{\text{JS}}\big(p_t^{(m)} \,\|\, \bar{p}_t\big)$ is the cross-modal predictive disagreement.

### 6.2. Delayed Feedback Update
When true labels $y_{t}$ arrive at step $t + \delta$ ($\delta \ge 0$), the update rule evaluates miscoverage against the **frozen emitted prediction set**:
$$\alpha_{t+\delta+1, m_t} = \text{clip}\Big(\alpha_{t+\delta, m_t} + \eta_t \cdot \big(\alpha - \text{Miss}_t\big), \; 0.01, \; 0.99\Big)$$
where $\text{Miss}_t = 1 - \frac{1}{B}\sum_{b=1}^B \mathbb{I}\{y_{t, b} \in \mathcal{C}_t(x_{t, b})\}$.
