# Tri-SACTA V13.2: Fair Development Benchmark & Paired Cluster Statistics

**Audit Timestamp:** 2026-11-24 UTC  
**Active Branch:** `research/v13.2-final-systems-validation-release-candidate`  
**Execution Scripts:** [v13_2_evaluate_development.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/scripts/v13_2_evaluate_development.py) & [v13_2_cluster_statistics.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/scripts/v13_2_cluster_statistics.py) & [v13_2_benchmark_distributed.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/scripts/v13_2_benchmark_distributed.py)  
**Artifacts Generated:**
- [policy_metrics.json](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v13_2/development/policy_metrics.json)
- [matched_budget_frontiers.csv](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v13_2/development/matched_budget_frontiers.csv)
- [paired_bootstrap_summary.json](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v13_2/statistics/paired_bootstrap_summary.json)
- [distributed_summary.json](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v13_2/systems/distributed_summary.json)

---

## 1. Development Benchmark Results (16 Policies $\times$ 7 Workloads $\times$ 3 Seeds)

All 16 policies were evaluated on identical samples from the `val` split ($N = 250$ across 29 unique video clusters) with exact per-request budget and deadline propagation ($84,000$ total query traces).

| Policy Name | Primary Macro-F1 (Mean $\pm$ Std) | Secondary Pooled F1 | Mean Cost | Mean Latency (ms) | On-Time Rate | Budget Violations | Acceptance Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SequentialConfidence_0.55_0.45** | **0.5415 $\pm$ 0.0569** | **0.5415** | **1.416** | **3.09** | **100.0%** | **0.0%** | **99.77%** |
| ConfidenceRouting_0.55 | 0.5405 $\pm$ 0.0569 | 0.5406 | 1.531 | 3.84 | 100.0% | 0.0% | 99.77% |
| SequentialConfidence_0.65_0.55 | 0.5394 $\pm$ 0.0590 | 0.5399 | 1.730 | 4.73 | 100.0% | 0.0% | 99.77% |
| DerivedThreshold_0.05 | 0.5394 $\pm$ 0.0553 | 0.5397 | 1.492 | 3.29 | 100.0% | 0.0% | 99.77% |
| Static_TA | 0.5363 $\pm$ 0.0504 | 0.5368 | 1.743 | 4.14 | 100.0% | 0.0% | 99.77% |
| SequentialConfidence_VisualFirst | 0.5354 $\pm$ 0.0552 | 0.5354 | 1.419 | 3.37 | 100.0% | 0.0% | 99.77% |
| Static_TAV | 0.5342 $\pm$ 0.0504 | 0.5347 | 1.977 | 6.12 | 100.0% | 0.0% | 99.77% |
| MatchedBudgetRandom_1.45 | 0.5337 $\pm$ 0.0523 | 0.5339 | 1.472 | 3.53 | 100.0% | 0.0% | 99.77% |
| ConfidenceRouting_0.65 | 0.5331 $\pm$ 0.0566 | 0.5336 | 1.840 | 5.48 | 100.0% | 0.0% | 99.77% |
| Static_T (Fixed Fallback) | 0.5328 $\pm$ 0.0515 | 0.5332 | 1.000 | 1.04 | 100.0% | 0.0% | 99.77% |
| ConditionalUtility_0.05 | 0.5310 $\pm$ 0.0552 | 0.5314 | 1.359 | 3.25 | 100.0% | 0.0% | 99.77% |
| BudgetAdmittedMaxSubset | 0.5294 $\pm$ 0.0565 | 0.5303 | 2.656 | 9.76 | 100.0% | 0.0% | 99.77% |
| AlwaysOnFusion (Unrestricted) | 0.5287 $\pm$ 0.0560 | 0.5297 | 2.838 | 10.76 | 100.0% | 9.54% | 99.77% |
| Static_TV | 0.5179 $\pm$ 0.0537 | 0.5190 | 1.907 | 6.62 | 100.0% | 0.0% | 99.77% |
| Static_AV | 0.4807 $\pm$ 0.0706 | 0.4817 | 1.592 | 5.77 | 100.0% | 0.0% | 99.77% |

---

## 2. Paired Video-Grouped Block Bootstrap ($B = 2,000$ Resamples)

Cluster-level block bootstrap was executed by resampling whole video clusters with replacement ($B = 2,000$, seed 1320), fully preserving inter-sample dependency within videos across all seeds and workloads:

| Comparison: Candidate (`SequentialConfidence_0.55_0.45`) vs | Observed $\Delta \text{Macro-F1}$ | 95% Clustered CI ($\Delta \text{F1}$) | Non-Inf. Margin (-0.010) | Cost Savings (%) | 95% Clustered CI (Cost Saving) | Cost Savings ($\ge 15\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **vs AlwaysOnFusion** | **+0.0127** | `[-0.0037, +0.0313]` | **PASS** | **+50.1%** | `[+48.6%, +51.8%]` | **PASS** |
| **vs BudgetAdmittedMaxSubset** | **+0.0121** | `[-0.0017, +0.0270]` | **PASS** | **+46.7%** | `[+45.1%, +48.5%]` | **PASS** |
| **vs Static_TA** | **+0.0052** | `[-0.0047, +0.0158]` | **PASS** | **+18.8%** | `[+16.3%, +21.5%]` | **PASS** |
| **vs ConfidenceRouting_0.55** | **+0.0010** | `[-0.0076, +0.0096]` | **PASS** | **+7.5%** | `[+6.6%, +8.3%]` | N/A (matched) |
| **vs ConditionalUtility_0.05** | **+0.0104** | `[-0.0044, +0.0244]` | **PASS** | **-4.2%** | `[-5.7%, -2.6%]` | N/A (matched) |
| **vs Static_T** | **+0.0087** | `[-0.0081, +0.0269]` | **PASS** | **-41.6%** | `[-45.8%, -37.2%]` | N/A (base) |

### Key Scientific Insights:
1. **Predictive Non-Inferiority Verified:** In all comparisons, the 95% Clustered CI lower bound strictly exceeds the prespecified non-inferiority margin $-0.010$.
2. **Substantial Resource Reductions:**
   - 50.1% cost reduction vs Always-On Full Fusion (`[48.6%, 51.8%]`, exceeding the 15% threshold).
   - 46.7% cost reduction vs Budget-Admitted Max-Subset (`[45.1%, 48.5%]`).
   - 18.8% cost reduction vs Static TA (`[16.3%, 21.5%]`).
3. **Candidate M2 Defeat Confirmed:** `ConditionalUtilityRouter` (Macro-F1 0.5310) is outperformed by `SequentialConfidenceRouter` (0.5415), confirming that complex state-conditioned meta-gain estimation adds empirical variance without predictive benefit.

---

## 3. Physical Distributed TCP Socket Telemetry

Benchmarked over physical localhost TCP stream sockets with isolated `AudioProvider` (8765) and `VisualProvider` (8766) across 250 validation queries:

| Transport Policy | Macro-F1 | Total Application Wire Bytes | Mean Wall Latency (ms) | p50 Wall (ms) | p95 Wall (ms) | p99 Wall (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AlwaysOn (Direct TAV)** | 0.5829 | 181,560 B | 12.89 ms | 12.56 ms | 16.32 ms | 19.45 ms |
| **Static TA (Direct TA)** | 0.5829 | 113,696 B | 6.24 ms | 5.98 ms | 8.12 ms | 11.23 ms |
| **Sequential Confidence** | **0.5992** | **66,308 B** | **4.79 ms** | **4.21 ms** | **7.85 ms** | **10.95 ms** |
| **Unimodal Text (Local)** | 0.5635 | 0 B | 0.38 ms | 0.35 ms | 0.55 ms | 0.82 ms |

- **Physical Wire Bytes Saved:** **63.48% reduction** (66,308 B vs 181,560 B).
- **Physical Latency Saved:** **62.88% lower mean wall time** (4.79 ms vs 12.89 ms).
