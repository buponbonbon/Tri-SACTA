# Tri-SACTA V13.2: Frozen Confirmatory Protocol & Pre-Holdout Manifest

**Manifest Version:** 13.2  
**Date Locked:** 2026-11-24 UTC  
**Active Branch:** `research/v13.2-final-systems-validation-release-candidate`  
**Base Commit:** `e71147fc511c5d50c218ded1fdad96953ae15b90`  
**Manifest Path:** [freeze_manifest.json](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v13_2/protocol/freeze_manifest.json)  
**Manifest SHA-256 Digest:** `54ef6b48f9aede4c0e119e0e1bda29bbfcc16e0c4c943241101a6e2e98fbe0f8`

---

## 1. Frozen Primary Hypothesis ($H_1$)

The primary hypothesis evaluated in this research program is formally frozen as follows:
> **Primary Systems Hypothesis ($H_1$):**  
> Under identical task-matched predictor banks and heterogeneous workload conditions, the sequential confidence acquisition runtime (`SequentialConfidence_0.55_0.45`) reduces application-protocol resource costs by $\ge 15.0\%$ (lower bound of 95% Clustered CI $\ge 15.0\%$) with non-inferior predictive 3-class Macro-F1 (lower bound of 95% Clustered CI $\ge -0.010$) compared with `AlwaysOnFusion` and `Static_TA`, while maintaining useful on-time response rate $\ge 95.0\%$ and acceptance rate $\ge 98.0\%$.

### Predefined Margins & Acceptance Guardrails
- **Non-Inferiority Margin ($\Delta_{\text{NI}}$):** $-0.010$ (two-sided 95% Clustered CI lower bound must be $\ge -0.010$).
- **Resource Savings Threshold ($\Delta_{\text{Cost}}$):** $+15.0\%$ (two-sided 95% Clustered CI lower bound must be $\ge 15.0\%$).
- **On-Time Rate Guardrail:** $\ge 95.0\%$ on time.
- **Acceptance Rate Guardrail:** $\ge 98.0\%$ accepted.
- **Budget Compliance Guardrail:** $0.0\%$ hard budget violations.

---

## 2. Canonical Estimand Definitions

1. **Primary Endpoint:** Unweighted arithmetic mean of per-(workload, seed) 3-class Macro-F1:
   $$\text{Macro-F1}_{\text{primary}} = \frac{1}{|\mathcal{W}| \cdot |\mathcal{S}|} \sum_{w \in \mathcal{W}} \sum_{s \in \mathcal{S}} \text{Macro-F1}(y^{(w, s)}, \hat{y}^{(w, s)})$$
   Label set is fixed as $\mathcal{Y} = \{0, 1, 2\}$. Rejections ($\hat{y} = -1$) count as False Negatives for the true class.
2. **Secondary Endpoint:** Pooled confusion matrix Macro-F1 across all queries.

---

## 3. Workload Suite and Comparator Grid

- **Workload Families (7):**
  1. `CLEAN_MULTIMODAL`
  2. `TEXT_EMBEDDING_NOISE`
  3. `AUDIO_FEATURE_MASKING`
  4. `VISUAL_ZERO_MASK`
  5. `COMPOUND_FAULT`
  6. `INTERMITTENT_MISSINGNESS`
  7. `HETEROGENEOUS_DEADLINE_STRESS`
- **Predictor Bank Checkpoints (21):**
  - Seven modality subsets: `T`, `A`, `V`, `TA`, `TV`, `AV`, `TAV`.
  - Three training seeds: `42`, `100`, `2026`.
- **Primary Comparators:** `AlwaysOnFusion` (unrestricted), `Static_TA` (best static multi-modal subset).
- **Secondary Comparators:** `BudgetAdmittedMaxSubset`, `ConfidenceRouting_0.55`, `MatchedBudgetRandom_1.45`, `Static_T` (Fixed Fallback).

---

## 4. Fresh Unexposed Holdout Population

- **Source Asset:** `data/cmu_mosei/raw/mosei.pkl` (`raw_test` partition).
- **Population Size:** Exactly 4,131 utterances spanning 617 video clusters.
- **Independence Guarantee:** Exactly 0 video overlap with `train`, `val`, `cal`, or historical `test`.
- **Phase I Boundary Lock:**
  - **Status:** `READY_FOR_HOLDOUT_REVIEW`.
  - **Inference Executed:** `False` (zero model predictions or feature extractions executed on this holdout in Phase I).
  - **Phase II Execution:** Strictly gated upon independent author review and explicit authorization.
