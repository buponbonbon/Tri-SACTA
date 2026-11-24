# Tri-SACTA V13.1 — Policy-Matched Conformal Prediction & Limitations

**Document Version:** 1.0.0  
**Timestamp (UTC):** 2026-11-24T15:52:00Z  
**Branch:** `research/v13.1-conditional-utility-independent-validation`  
**Artifact Reference:** `results/v13_1/conformal/conformal_metrics.json`  

---

## 1. Conformal Protocol & Exact Finite-Sample Calibration (Stage E)

To provide calibrated uncertainty quantification without inflated safety claims:
- **Calibration Split:** `cal` (250 utterances, 46 unique video clusters disjoint from `train`, `val`, and `test`).
- **Policy Evaluated:** Deployed sequential policy (`SequentialConfidenceRouter`).
- **Nonconformity Score:** $s_i = 1 - \hat{P}(Y = y_i \mid x_{S_i})$.
- **Order Statistic:** Exact finite-sample quantile $k = \lceil (n + 1)(1 - \alpha) \rceil = \lceil 251 \times 0.90 \rceil = 226$ (out of 250 samples).
- **Calibrated Nonconformity Quantile:** $\hat{q} = 0.8908 \pm 0.0019$ across seeds.
- **Prediction Set Formation:** $C(x) = \{c \in \{0, 1, 2\} : \hat{P}(Y = c \mid x_S) \ge 1 - \hat{q}\}$.

---

## 2. Empirical Coverage & Efficiency Across Stress Workloads

Evaluated on 250 validation utterances $\times$ 3 training seeds ($N = 750$ instances per workload):

| Workload Scenario | Mean Marginal Coverage | Mean Set Size (out of 3) | Singleton Rate ($|C(x)| = 1$) | Empty Set Rate ($|C(x)| = 0$) |
|---|---|---|---|---|
| `CLEAN_MULTIMODAL` | **90.5%** | 2.27 | 12.0% | 0.0% |
| `INTERMITTENT_MISSINGNESS` | **90.8%** | 2.31 | 11.5% | 0.0% |
| `HETEROGENEOUS_DEADLINE_STRESS` | **90.5%** | 2.27 | 12.0% | 0.0% |
| `VISUAL_ZERO_MASK` | **90.7%** | 2.29 | 11.9% | 0.0% |
| `AUDIO_FEATURE_MASKING` | **87.5%** | 2.21 | 14.1% | 0.0% |
| `TEXT_EMBEDDING_NOISE` | **80.0%** | 2.00 | 20.1% | 0.0% |
| `COMPOUND_FAULT` | **75.9%** | 1.97 | 21.9% | 0.0% |

---

## 3. Scientific Limitations & Disclosure of Boundaries

1. **Marginal vs. Conditional Validity:**
   - Marginal coverage strictly meets the 90% target on exchangeable data (`CLEAN_MULTIMODAL`: 90.5%).
   - However, standard split conformal prediction does **NOT** guarantee conditional coverage under arbitrary distribution shift. When text embeddings suffer Gaussian noise or compound channel failure, coverage drops from 90.5% to 80.0% and 75.9%.
   - In accordance with Gate E, Tri-SACTA V13.1 explicitly rejects universal, distribution-free safety guarantees under unknown corruptions.

2. **Prediction Set Inefficiency:**
   - In a 3-class sentiment task, an average set size of $2.27$ indicates that the model frequently outputs 2 or all 3 classes to maintain 90% coverage.
   - The singleton rate (predictions where the set narrows to a single confident class) is only $12.0\%$.
   - Conformal prediction is therefore retained as a **secondary diagnostic tool for conservative decision boundaries**, and is not claimed as a primary SOTA breakthrough.
