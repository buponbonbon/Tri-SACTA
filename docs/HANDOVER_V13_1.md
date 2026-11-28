# Tri-SACTA V13.1 — Complete Handover & Scientific Milestone Report

**Milestone:** Tri-SACTA V13.1 (Conditional Utility Validation & Systems Evidence)  
**Date:** 2026-11-24  
**Working Branch:** `research/v13.1-conditional-utility-independent-validation`  
**Base Commit SHA:** `a95c375bb04df2ac7fa67d54ecc8f58c419de136`  
**Historical Branches Preserved:** `research/v13-utility-driven-orchestration`, `research/v12.1-budgeted-systems-contribution-rectified`, `research/v12-systems-contribution-validation`, `research/v11-system-feasibility-go-no-go` (All intact, unedited, and preserved for scientific reproducibility).

---

## 1. Executive Verdict & Core Findings

Tri-SACTA V13.1 was executed to resolve foundational mathematical, provenance, and systems questions identified during independent auditing of V13.

### What Worked:
1. **Measured Real Systems Bandwidth & Latency Savings (Stage G):**
   - Implemented independent TCP socket modality providers (`Audio` on port 8765, `Visual` on port 8766) and an Edge Coordinator.
   - Evaluated 250 real distributed queries without mock delays.
   - Sequential Confidence Routing achieved **61.64% wire bytes reduction** (from 133,030 bytes to 51,029 bytes) and reduced end-to-end latency by **53.3%** (from 2.40 ms to 1.12 ms) while improving Macro-F1 from 0.5686 to 0.5992.
2. **Reachable Sequential Oracle Headroom (Stage A):**
   - By enforcing sunk costs and compulsory initial text, the Reachable Sequential Oracle established **+0.1524 Macro-F1 headroom** over the best static baseline ($TA$), with a 95% clustered bootstrap interval of `[+0.1254, +0.1846]`.
3. **Data Provenance & Discovery of 4,131 Unexposed Utterances (Stage F):**
   - Identified that the historical 500-sample test set was exposed across V7–V9, officially quarantining it as an exposed baseline.
   - Discovered **4,131 untouched, unexposed CMU-MOSEI test utterances** across 617 video clusters in `mosei.pkl` with zero cross-split leakage, reserving them for prospective freeze.

### What Failed:
1. **V13 Utility Rule is Mathematically Confidence Routing:**
   - Proved analytically and verified numerically (0 mismatches across 10,001 points) that V13 utility is an affine transformation of confidence: $\tau = \frac{\gamma - \lambda \Delta C}{\gamma + \nu}$.
   - Proved that under V13 priors, Visual utility was strictly inferior to Audio utility for all $p \in [0, 1]$, rendering Visual unselectable at state $T$.
2. **Conditional Utility Meta-Model (Candidate M2) Lost to Simple Baseline:**
   - When trained on state features, `ConditionalUtility_0.05` achieved Macro-F1 of $0.5321 \pm 0.0542$ across 7 workloads $\times$ 3 seeds, underperforming the clean `SequentialConfidence_0.55_0.45` baseline ($0.5447 \pm 0.0576$).
   - Per protocol Section 13, the project formally adopts **`SIMPLIFY_TO_STRONG_BASELINE`**.

---

## 2. Comprehensive Benchmark Comparison Table

Evaluated across 7 stress workloads $\times$ 3 training seeds ($N = 21$ runs per policy, 94,500 total evaluation instances on the validation split):

| Policy Name | Category | Macro-F1 (Mean $\pm$ Std) | Accuracy (Mean $\pm$ Std) | Mean Cost | Latency (ms) | On-Time Rate |
|---|---|---|---|---|---|---|
| **SequentialConfidence_0.55_0.45** | **Primary Fair Baseline** | **$0.5447 \pm 0.0576$** | **$0.5604 \pm 0.0514$** | **1.45** | **3.2** | **100.0%** |
| **DerivedThreshold_V13** | Derived Structural Equiv. | $0.5425 \pm 0.0561$ | $0.5592 \pm 0.0505$ | 1.54 | 3.5 | 100.0% |
| **UtilityRouter_V13** | Frozen V13 Reference | $0.5425 \pm 0.0561$ | $0.5592 \pm 0.0505$ | 1.54 | 3.5 | 100.0% |
| **ConfidenceRouting_0.55** | One-Shot Confidence | $0.5416 \pm 0.0561$ | $0.5577 \pm 0.0505$ | 1.62 | 4.3 | 100.0% |
| **Static_TA** | Static Modality Subset | $0.5384 \pm 0.0505$ | $0.5577 \pm 0.0471$ | 1.80 | 4.4 | 100.0% |
| **SequentialConfidence_0.75_0.60** | Fair Sequential Baseline | $0.5369 \pm 0.0560$ | $0.5522 \pm 0.0503$ | 2.10 | 6.7 | 100.0% |
| **ConfidenceRouting_0.65** | One-Shot Confidence | $0.5335 \pm 0.0553$ | $0.5484 \pm 0.0494$ | 1.98 | 6.2 | 100.0% |
| **FixedFallback / Static_T** | Static Modality Subset | $0.5331 \pm 0.0503$ | $0.5528 \pm 0.0465$ | 1.00 | 1.0 | 100.0% |
| **ConditionalUtility_0.05** | Candidate M2 | $0.5321 \pm 0.0542$ | $0.5510 \pm 0.0492$ | 1.42 | 3.6 | 100.0% |
| **MatchedBudgetRandom** | Random Baseline | $0.5318 \pm 0.0519$ | $0.5516 \pm 0.0475$ | 1.65 | 4.5 | 100.0% |
| **DerivedThreshold_Corrected_M1** | Corrected Priors Baseline | $0.5310 \pm 0.0487$ | $0.5516 \pm 0.0456$ | 1.02 | 1.1 | 100.0% |
| **ConditionalUtility_0.08** | Candidate M2 | $0.5293 \pm 0.0537$ | $0.5495 \pm 0.0489$ | 1.12 | 1.8 | 100.0% |
| **AlwaysOnFusion / Static_TAV** | Full Fusion Reference | $0.5291 \pm 0.0547$ | $0.5434 \pm 0.0488$ | 2.84 | 10.8 | 100.0% |
| **ConditionalUtility_0.02** | Candidate M2 | $0.5201 \pm 0.0496$ | $0.5369 \pm 0.0441$ | 1.68 | 5.2 | 100.0% |
| **Static_TV** | Static Modality Subset | $0.5182 \pm 0.0525$ | $0.5307 \pm 0.0477$ | 2.03 | 7.4 | 100.0% |

---

## 3. Paired Clustered Video Bootstrap ($B = 2,000$ Resamples)

Comparisons against the primary policy `SequentialConfidence_0.55_0.45`:

| Comparator | $\Delta$ Macro-F1 | 95% Clustered CI | Nominal Cost Reduction (%) | 95% Cost CI | Verdict |
|---|---|---|---|---|---|
| **vs AlwaysOnFusion** | **+0.0147** | `[-0.0013, +0.0332]` | **+48.9%** | `[+47.2%, +50.7%]` | **Non-Inferior & Cost-Superior** |
| **vs Static_TA** | **+0.0056** | `[-0.0053, +0.0165]` | **+19.6%** | `[+16.8%, +22.3%]` | **Non-Inferior & Cost-Superior** |
| **vs ConfidenceRouting_0.55** | **+0.0030** | `[-0.0067, +0.0136]` | **+10.6%** | `[+9.6%, +11.6%]` | **Non-Inferior & Cost-Superior** |
| **vs UtilityRouter_V13** | **+0.0019** | `[-0.0037, +0.0073]` | **+5.6%** | `[+4.3%, +7.1%]` | **Non-Inferior & Cost-Superior** |
| **vs ConditionalUtility_0.05** | **+0.0123** | `[-0.0034, +0.0286]` | **-2.4%** | `[-3.9%, -0.8%]` | **Baseline Wins on Quality** |

---

## 4. Real Distributed Socket Telemetry (Stage G)

Measured over real TCP sockets (Localhost, ports 8765 and 8766):
- **Always-On Full Transmission:** 133,030 wire bytes total (532.1 B/req), 1.57 ms network RTT, 2.40 ms wall-clock time, 0.5686 F1.
- **Sequential Confidence Routing:** 51,029 wire bytes total (204.1 B/req), 0.58 ms network RTT, 1.12 ms wall-clock time, 0.5992 F1.
- **Empirical Bandwidth Reduction:** **61.64% wire bytes saved**.
- **Empirical Latency Reduction:** **53.3% faster wall-clock response time**.

---

## 5. Artifact Directory & Reproduction Instructions

### 5.1 Primary Artifact Locations
- Audit Report: `results/v13_1/provenance/provenance_audit.json`
- Priors & Thresholds: `results/v13_1/priors/empirical_transition_priors.json`, `derived_thresholds.json`
- Reachable Oracle: `results/v13_1/oracle/reachable_oracle_summary.json`, `oracle_frontier.csv`
- Policy Benchmark: `results/v13_1/evaluation/policy_metrics.json`, `matched_budget_frontiers.csv`
- Bootstrap Statistics: `results/v13_1/statistics/paired_bootstrap_summary.json`
- Conformal Uncertainty: `results/v13_1/conformal/conformal_metrics.json`
- Distributed Systems Telemetry: `results/v13_1/systems/distributed_summary.json`, `distributed_benchmark.csv`
- Final Gate Decision: `results/v13_1/evaluation/final_decision.json`

### 5.2 Step-by-Step Reproduction Commands
```powershell
# 1. Audit provenance and verify unexposed pool
python scripts/v13_1_audit_provenance.py

# 2. Derive thresholds and compute transition risks
python scripts/v13_1_fit_transition_estimators.py

# 3. Run reachable sequential oracle analysis
python scripts/v13_1_reachable_oracle.py

# 4. Fit conditional utility models
python scripts/v13_1_fit_conditional_utility.py

# 5. Benchmark fair baselines across 7 stress workloads
python scripts/v13_1_evaluate_policies.py

# 6. Run 2,000 paired video-clustered bootstrap resamples
python scripts/v13_1_cluster_statistics.py

# 7. Calibrate policy-matched split conformal prediction
python scripts/v13_1_calibrate_policies.py

# 8. Run physical distributed TCP socket benchmark
python scripts/v13_1_benchmark_distributed.py

# 9. Verify automated decision gates G0-G8
python scripts/v13_1_verify_gates.py

# 10. Execute complete test suite
python run_all_tests.py
```
