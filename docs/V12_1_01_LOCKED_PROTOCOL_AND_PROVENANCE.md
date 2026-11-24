# V12.1-01: Locked Scientific Protocol and Data Provenance

**Author:** Antigravity 3.8 Scientific Systems Audit Lead  
**Date:** 2026-11-24  
**Protocol Configuration:** [configs/v12_1_systems_protocol.yaml](file:///c:/Users/ASUS/Desktop/Tri-SACTA/configs/v12_1_systems_protocol.yaml)  
**Base Commit:** `fd010fd85f9f4b70cdc537aab546c3d800ffd01d`  

---

## 1. Experimental Integrity & Quarantine Rules

In strict adherence to the scientific directives of V12.1:
1. **Quarantine of Development-Exposed Splits:**
   - The 500-sample `test` split was exposed during historical exploratory runs (V7–V9). It remains strictly **QUARANTINED** and is not used for model selection, threshold sweeps, or policy comparisons.
   - The 250-sample `val` split is designated as **development-exposed** (used for early stopping of models and baseline policy benchmarking).
   - The 250-sample `cal` split is reserved strictly for **conformal quantile calibration** and has never been used for model weight updates or threshold tuning.
2. **Label Protocol:**
   - Primary 3-class sentiment mapping is predeclared and immutable:
     $$\text{class}(s) = \begin{cases} 0 \ (\text{Negative}), & s < 0.0 \\ 1 \ (\text{Neutral}), & s = 0.0 \\ 2 \ (\text{Positive}), & s > 0.0 \end{cases}$$
   - No post-hoc logit threshold adjustments or label binning alterations are permitted.
3. **Multi-Seed Replicates:**
   - All neural models are evaluated across 3 independent training seeds: `42, 100, 2026`.
   - Workload failure realizations are independently seeded and executed identically across all evaluated policies.

---

## 2. Statistical Analysis and Non-Inferiority Margins

To ensure scientific honesty and prevent post-hoc metric cherry-picking, the decision thresholds are pre-registered:
- **Primary Hypothesis:** The Budgeted Reliability-Aware Modality Orchestrator (BRMO) achieves **Pareto superiority** over static deployment policies by satisfying:
  1. **Non-Inferiority in Clean Settings:** The paired difference in Macro-F1 $\Delta = \text{F1}_{\text{BRMO}} - \text{F1}_{\text{Text-MLP}} \ge -0.01$ (with a paired one-sided 95% lower bootstrap bound above $-0.01$) alongside $\ge 15\%$ reduction in normalized sensor cost.
  2. **Robustness in Fault Settings:** Under text corruption (ASR drift emulation), BRMO achieves an absolute gain $\Delta \ge +0.05$ Macro-F1 over text-only inference.
- **Resampling Method:** Paired Video-Grouped Block Bootstrap (1,000 resamples grouped by `video_id` across the 29 validation videos) to capture cluster-level variance.
