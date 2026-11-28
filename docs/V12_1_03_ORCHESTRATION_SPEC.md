# Tri-SACTA V12.1: Unified Systems Orchestration Engine Specification

**Document Version:** 1.0.0 (V12.1 Rectified)  
**Status:** Authoritative Systems Specification  
**Venue Target:** *Multimedia Tools and Applications* (MTAP) / *Multimedia Systems* / *Computing* (SCImago Q1/Q2, $0 APC)  
**Primary Engine:** `src/service/router.py`, `src/service/budget.py`, `src/service/acquisition.py`, `src/service/reliability.py`, `src/service/app.py`

---

## 1. Systems Architecture Overview

Tri-SACTA V12.1 is reframed as a **multimodal systems instrument** designed for latency-sensitive, bandwidth-constrained edge/cloud sentiment processing. In contrast to speculative test-time adaptation (which failed empirical verification in V7–V11), V12.1 focuses strictly on **deterministic multi-tier resource orchestration**:

```mermaid
flowchart TD
    Req[Incoming Query: text, audio, visual] --> Prov[LazyModalityProvider\nIndependent Sensor Acquisition]
    Prov --> T1[Stage 1: Fast Unimodal Triage\nText-only MLP 64D]
    
    T1 --> Rel[Reliability & Quality Assessment\n1. StandardizedCentroidDistance d_std\n2. Normalized Entropy H_norm\n3. Winning Softmax Confidence c_t]
    
    Rel --> Decision{Triage Passed?\nd_std <= 1.15 &\nH_norm <= 0.65 &\nc_t >= 0.60}
    
    Decision -- Yes --> TextFast[Accept Text Fast-Path\nCost: 1.0 | Latency: ~2.5 ms]
    Decision -- No --> Adm[Stage 2: Budget Admission Control\nRemaining Deadline: D_rem\nRemaining Cost: B_rem]
    
    Adm --> AcqPlan[Sensor Acquisition Plan\nAcquire Audio if admissible\nAcquire Visual if admissible]
    AcqPlan --> Fusion[Multimodal Transformer Fusion 1.07M\nReliability Weighting: w_t = 0.20 if d_std > 1.15]
    
    TextFast --> ConfRisk[Stage 3: Split Conformal Risk Assessment\nCalibrated on CMU-MOSEI 'cal' split N=250]
    Fusion --> ConfRisk
    
    ConfRisk --> ConfSet{Conformal Set Cardinality\n|C(x)| == 1?}
    ConfSet -- Yes --> ReturnPred[Emit Verified Prediction]
    ConfSet -- No --> Abstain[Abstain / Flag Ambiguous Query]
```

---

## 2. Sensor Acquisition Model & Cost Accounting

In real multimedia systems, sensor data is not pre-materialized in shared RAM for free. Tri-SACTA V12.1 replaces bulk in-memory tensor access with `LazyModalityProvider` (`src/service/acquisition.py`):

| Modality | Dimension | Transferred Bytes (FP32) | Unit Cost Weight ($c_m$) | Latency Estimate ($t_{\text{est}}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Text** | 768 | 3,072 bytes (3.0 KB) | 1.0 | 2.5 ms |
| **Audio** | 74 | 296 bytes (0.3 KB) | 1.2 | 5.0 ms |
| **Visual** | 35 | 140 bytes (0.14 KB) | 1.3 | 8.0 ms |
| **Multimodal Fusion** | 877 (Total) | 3,508 bytes (3.5 KB) | 3.5 (Cumulative) | 12.0 ms |

### Budget Invariants
1. If the fast text path is accepted, audio and visual acquisition is completely bypassed. Zero transfer bytes are incurred, reducing compute and network cost by **$71.4\%$** ($1.0 / 3.5$).
2. Modality acquisition is strictly independent: an audio stream failure or visual camera occlusion does not stall the text triage path.

---

## 3. Reliability Engine: Standardized Centroid Distance

### Forensic Rectification (Defect D-01)
In V12, the metric $d(x) = \frac{1}{D}\sum \frac{|x_d - \mu_d|}{\sigma_d}$ was erroneously termed a "spectral anomaly score". In V12.1, this terminology is completely eliminated. The metric is strictly defined and audited as **Standardized Centroid Distance**:

$$d_{\text{std}}(x) = \frac{1}{D}\sum_{d=1}^{D} \frac{|x_d - \mu_d|}{\sigma_d + \epsilon}$$

Where:
- $\mu \in \mathbb{R}^{768}$ is the empirical coordinate-wise mean of clean training embeddings.
- $\sigma \in \mathbb{R}^{768}$ is the empirical standard deviation of clean training embeddings.
- $\epsilon = 10^{-6}$ is a numerical stabilizer.

### Triage Gating Conditions
A text embedding passes unimodal fast-path triage if and only if all three criteria hold:
1. **Distributional In-Bounds:** $d_{\text{std}}(x) \le \tau_{\text{dist}} = 1.15$
2. **Normalized Information Entropy:**
   $$H_{\text{norm}}(p) = -\frac{\sum_{c=1}^3 p_c \ln(p_c + 10^{-12})}{\ln(3)} \le \tau_{\text{ent}} = 0.65$$
3. **Softmax Margin Confidence:**
   $$c_{\text{top}} = \max_{c \in \{0, 1, 2\}} p_c \ge \tau_{\text{conf}} = 0.60$$

---

## 4. True Split Conformal Prediction Engine

### Forensic Rectification (Defect D-02)
V12 implemented confidence thresholding (`max(p) < 0.40`) and claimed conformal calibration. V12.1 implements **True Inductive / Split Conformal Prediction** calibrated on the isolated 250-sample `cal` split:

1. **Nonconformity Measure:**
   $$s_i = 1 - p(y_i)$$
   Where $y_i \in \{0, 1, 2\}$ is the ground-truth sentiment label in `cal`, and $p(y_i)$ is the model's assigned softmax probability.
2. **Empirical Quantile Calibration:**
   $$\hat{q} = \text{Quantile}_{\left\lceil \frac{(N_{\text{cal}} + 1)(1 - \alpha)}{N_{\text{cal}}} \right\rceil} \left(\{s_1, \dots, s_{N_{\text{cal}}}\}\right)$$
   For target coverage $1 - \alpha = 0.90$ ($\alpha = 0.10$) over $N_{\text{cal}} = 250$ samples:
   $$\text{Quantile Level} = \min\left(1.0, \frac{\lceil 251 \times 0.90 \rceil}{250}\right) = \frac{226}{250} = 0.904$$
3. **Prediction Set Formulation:**
   $$C(x) = \left\{ y \in \{0, 1, 2\} : 1 - p(y) \le \hat{q} \right\}$$
4. **Selective Prediction & Abstention Rule:**
   - **Singleton Set ($|C(x)| = 1$):** High confidence; emit single predicted class.
   - **Ambiguous Set ($|C(x)| > 1$):** Model uncertainty exceeds conformal threshold; flag for human review or fallback.
   - **Empty Set ($|C(x)| = 0$):** Extreme nonconformity / out-of-distribution; trigger explicit abstention.

---

## 5. Fail-Closed Service Architecture

### Forensic Rectification (Defect D-11)
`src/service/app.py` enforces fail-closed initialization:
1. If model checkpoints (`text_only_mlp_seed*.pt`, `multimodal_fusion_seed*.pt`) are missing, the service raises `HTTP 503 Service Unavailable` with explicit root-cause logging. It **never** falls back to uninitialized random weights.
2. If the calibration split (`data/cmu_mosei/features_cache/cal`) is missing or uncalibrated, service startup is rejected.
3. Every incoming request is validated via Pydantic (`src/service/schemas.py`) against NaN, Inf, dimension mismatch, and non-positive deadlines.

---

## 6. Scientific Non-Inferiority Invariant

To substantiate a Q1/Q2 systems paper without relying on TTA superiority claims, Tri-SACTA V12.1 adheres to the **Budgeted Systems Non-Inferiority Thesis**:

$$\text{Macro-F1}(\text{BRMO}) \ge \text{Macro-F1}(\text{Always-On Multimodal}) - 0.01$$
$$\text{Cost}(\text{BRMO}) \le 0.85 \times \text{Cost}(\text{Always-On Multimodal}) \quad (\ge 15\% \text{ Cost Savings})$$

If BRMO achieves comparable classification quality to always-on multimodal fusion while reducing computational cost and sensor transmission by $\ge 15\%$, the systems trade-off is empirically established.
