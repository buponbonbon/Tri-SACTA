# Tri-SACTA V13: Forensic Audit of V12.1 and Utility-Driven Protocol

**Document ID:** `V13_00_FORENSIC_AUDIT_AND_PROTOCOL`  
**Status:** Canonical & Audited  
**Base Commit:** `44bc28ceb4306fe51893110996841315e967a1ff`  
**Branch:** `research/v13-utility-driven-orchestration`  
**Protocol Reference:** `configs/v13_utility_protocol.yaml`  

---

## 1. De-authorization of V12.1 Verdict

The V12.1 verdict `PROCEED_SYSTEMS_PAPER` is **not scientifically established** and is hereby de-authorized. An independent audit revealed four fundamental scientific defects in V12.1:

### Defect 1: Systematic Underperformance Against Confidence Routing
In `results/v12_1_systems/statistics/paired_deltas.json`, the proposed BRMO policy was outperformed by simple Confidence Routing across **all seven workloads** in raw Macro-F1:
- `CLEAN_MULTIMODAL`: Confidence Routing **0.5776** vs BRMO **0.5549** ($\Delta = -0.0228$)
- `TEXT_EMBEDDING_NOISE`: Confidence Routing **0.4841** vs BRMO **0.4507** ($\Delta = -0.0334$)
- `AUDIO_FEATURE_MASKING`: Confidence Routing **0.5377** vs BRMO **0.5342** ($\Delta = -0.0034$)
- `VISUAL_ZERO_MASK`: Confidence Routing **0.5482** vs BRMO **0.5301** ($\Delta = -0.0181$)
- `COMPOUND_FAULT`: Confidence Routing **0.4670** vs BRMO **0.4621** ($\Delta = -0.0049$)
- `INTERMITTENT_MISSINGNESS`: Confidence Routing **0.5483** vs BRMO **0.5248** ($\Delta = -0.0235$)
- `HETEROGENEOUS_DEADLINE_STRESS`: Confidence Routing **0.5776** vs BRMO **0.5315** ($\Delta = -0.0461$)

*Root Cause:* V12.1 BRMO used hard, uncalibrated triage thresholds ($d_{\text{scd}} \le 1.2$, $H \le 0.7$, $p \ge 0.60$) and an all-or-nothing escalation (either text alone or full fusion). It lacked intermediate subset predictors ($\{T, A\}$, $\{T, V\}$, $\{A, V\}$) and lacked any quantitative estimate of whether acquiring an extra modality actually reduces predictive uncertainty.

### Defect 2: Absence of Video-Grouped Block Bootstrap
In `scripts/v12_1_paired_statistics.py`, paired deltas were computed merely by subtracting seed-level summary means across 3 seeds (`len(common_seeds) == 3`). No sample-level or video-level clustered bootstrap was executed. In CMU-MOSEI, utterances from the same video share speaker identity, lighting, recording acoustics, and vocabulary. Evaluating standard errors without clustering by `video_id` severely underestimates variance and produces spurious confidence intervals.

### Defect 3: Conformal Calibration Distribution Mismatch
In `src/service/app.py`, the Split Conformal Predictor was calibrated by evaluating the *full multimodal fusion model* on the calibration split (`cal`, $N=250$). However, during inference, the deployed BRMO router routed queries between text-only MLP, fallback models, and fusion. Evaluating text-only probabilities with a quantile calibrated on multimodal fusion violates conformal exchangeability ($S_i \sim \mathcal{D}_{\text{cal}} \neq \mathcal{D}_{\text{route}}$).

### Defect 4: Simulation of Acquisition Costs Conflated with Systems Savings
In V12.1, resource savings were calculated analytically using static scalar multipliers ($C_T = 1.0, C_A = 1.2, C_V = 1.3$). While analytical cost modeling is valid for relative compute weighting, it was erroneously framed as measured network bandwidth and energy savings without socket-level byte transfer and distributed edge-cloud latency measurements.

---

## 2. V13 Scientific Mandate & Research Questions

V13 investigates whether **adaptive modality acquisition** has genuine theoretical headroom and can be orchestrated via an **Expected-Utility** decision framework that provably outperforms strong confidence-based baselines under clustered video bootstrap.

### Research Questions
- **RQ1 (Oracle Headroom):** What is the upper bound on predictive fidelity and cost efficiency if an Oracle has access to all 7 modality subsets ($\{T\}$, $\{A\}$, $\{V\}$, $\{T, A\}$, $\{T, V\}$, $\{A, V\}$, $\{T, A, V\}$)? Does adaptive acquisition have meaningful headroom ($\Delta \text{Macro-F1} > 0.05$) over the single best static subset?
- **RQ2 (Subset Complementarity):** Which sensory combinations provide orthogonal, error-correcting information rather than redundant noise?
- **RQ3 (Expected-Utility Acquisition):** Can a sequential acquisition policy that estimates marginal risk reduction $\mathbb{E}[\Delta R(m \mid S)]$ per unit cost $\Delta C(m)$ beat standard confidence-threshold routing in Pareto efficiency?
- **RQ4 (Policy-Matched Conformal Calibration):** Does calibrating conformal quantiles on the exact trajectory of the deployed policy restore valid distribution-free coverage under distribution shift?
- **RQ5 (Video-Clustered Statistical Rigor):** Do the observed performance and cost gains hold under genuine paired video-grouped block bootstrap ($B=1,000$ clusters)?

---

## 3. Experimental Architecture & Provenance Matrix

### Table 1: Dataset Splits and Video Clustering

| Split Name | Sample Count | Unique Video Clusters | Video Overlap with Others | Assigned Role |
| :--- | :---: | :---: | :---: | :--- |
| **`train`** | 1,000 | 112 | 0 | Training all 7-subset task-matched predictors |
| **`val`** | 250 | 29 | 0 | Oracle headroom analysis, expected utility tuning |
| **`cal`** | 250 | 46 | 0 | Policy-matched split conformal calibration |
| **`test`** | 500 | 58 | 0 | Independent confirmatory evaluation |

### Table 2: 7 Modality Subsets & Cost Model

| Subset ID | Modalities Included | Feature Dimension | Acquisition Cost ($\Delta C$) | Latency Estimate (ms) |
| :---: | :--- | :---: | :---: | :---: |
| **T** | Text | 768 | 1.00 | 1.0 |
| **A** | Audio | 74 | 1.20 | 5.0 |
| **V** | Visual | 35 | 1.30 | 8.0 |
| **TA** | Text + Audio | 842 | 2.20 | 6.0 |
| **TV** | Text + Visual | 803 | 2.30 | 9.0 |
| **AV** | Audio + Visual | 109 | 2.50 | 13.0 |
| **TAV** | Text + Audio + Visual | 877 | 3.50 | 14.0 |

---

## 4. Acceptance Criteria & Go/No-Go Decision Gates

1. **Gate 1 (Oracle Headroom):** If $\text{Headroom}_{\text{F1}} \le 0.02$, stop immediately: adaptive acquisition has negligible potential on CMU-MOSEI features. If $\text{Headroom}_{\text{F1}} \ge 0.05$, proceed to policy development.
2. **Gate 2 (Confidence Routing Non-Inferiority):** The proposed Expected-Utility policy must NOT lose to Confidence Routing in Pareto efficiency (must achieve higher F1 at matched cost, or lower cost at non-inferior F1).
3. **Gate 3 (Video-Grouped Bootstrap Significance):** 95% bootstrap confidence interval of $\Delta \text{Macro-F1}$ vs Always-On Fusion must be $\ge -0.010$ with $\ge 15\%$ cost reduction.
4. **Gate 4 (Policy-Matched Conformal Validity):** Marginal coverage on held-out test split must satisfy $\ge 1 - \alpha - \epsilon$ with matched calibration.
