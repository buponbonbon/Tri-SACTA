# Tri-SACTA V13.2: Fair Development Benchmark, Dual-Endpoint Bootstrap & TCP Telemetry

**Generated from canonical artifacts on:** `2026-11-25T06:14:21.827323+00:00`  
**Sources:** [policy_metrics.json](../results/v13_2/development/policy_metrics.json), [paired_bootstrap_summary.json](../results/v13_2/statistics/paired_bootstrap_summary.json), [distributed_summary.json](../results/v13_2/systems/distributed_summary.json)

## 1. Offline Development Replay (84,000 Queries, Boundary Guardrails Passed: `True`)

| Policy | SQ Macro-F1 (Mean ± Std) | Pooled F1 | Mean Nominal Cost | Mean Latency (ms) | Useful On-Time | Budget Viol. | Acceptance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SequentialConfidence_0.55_0.45` | 0.5415 ± 0.0569 | 0.5415 | 1.416 | 3.09 | 99.77% | 0.00% | 99.77% |
| `ConfidenceRouting_0.55` | 0.5405 ± 0.0569 | 0.5406 | 1.531 | 3.84 | 99.77% | 0.00% | 99.77% |
| `SequentialConfidence_0.65_0.55` | 0.5394 ± 0.0590 | 0.5399 | 1.730 | 4.73 | 99.77% | 0.00% | 99.77% |
| `DerivedThreshold_0.05` | 0.5394 ± 0.0553 | 0.5397 | 1.492 | 3.29 | 99.77% | 0.00% | 99.77% |
| `Static_TA` | 0.5363 ± 0.0504 | 0.5368 | 1.743 | 4.14 | 99.77% | 0.00% | 99.77% |
| `SequentialConfidence_VisualFirst_0.55_0.45` | 0.5354 ± 0.0552 | 0.5354 | 1.419 | 3.37 | 99.77% | 0.00% | 99.77% |
| `Static_TAV` | 0.5342 ± 0.0504 | 0.5347 | 1.977 | 6.12 | 99.77% | 0.00% | 99.77% |
| `MatchedBudgetRandom_1.45` | 0.5337 ± 0.0523 | 0.5339 | 1.472 | 3.53 | 99.77% | 0.00% | 99.77% |
| `ConfidenceRouting_0.65` | 0.5331 ± 0.0566 | 0.5336 | 1.840 | 5.48 | 99.77% | 0.00% | 99.77% |
| `Static_T` | 0.5328 ± 0.0515 | 0.5332 | 1.000 | 1.04 | 99.77% | 0.00% | 99.77% |
| `FixedFallback` | 0.5328 ± 0.0515 | 0.5332 | 1.000 | 1.04 | 99.77% | 0.00% | 99.77% |
| `ConditionalUtility_0.05` | 0.5310 ± 0.0552 | 0.5314 | 1.359 | 3.25 | 99.77% | 0.00% | 99.77% |
| `BudgetAdmittedMaxSubset` | 0.5294 ± 0.0565 | 0.5303 | 2.656 | 9.76 | 99.77% | 0.00% | 99.77% |
| `AlwaysOnFusion` | 0.5287 ± 0.0560 | 0.5297 | 2.838 | 10.76 | 99.77% | 9.54% | 99.77% |
| `Static_TV` | 0.5179 ± 0.0537 | 0.5190 | 1.907 | 6.62 | 99.77% | 0.00% | 99.77% |
| `Static_AV` | 0.4807 ± 0.0706 | 0.4817 | 1.592 | 5.77 | 99.77% | 0.00% | 99.77% |

## 2. Primary Paired Cluster Bootstrap (Measured Application-Protocol Bytes & SQ-F1)

| Comparator | Observed $\Delta$ SQ-F1 | 95% CI | NI (-0.010) | AppBytes Saved | 95% CI | Abs Diff | $\ge 15\%$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AlwaysOn_Direct_Serial` | +0.0127 | `[-0.0037, +0.0313]` | PASS | +74.7% | `[+72.1%, +77.5%]` | -392.5 B | PASS |
| `AlwaysOn_Direct_Parallel` | +0.0127 | `[-0.0037, +0.0313]` | PASS | +74.7% | `[+72.1%, +77.5%]` | -392.4 B | PASS |
| `Static_TA_Direct` | +0.0052 | `[-0.0047, +0.0158]` | PASS | +51.8% | `[+46.9%, +57.3%]` | -143.2 B | PASS |
| `BudgetAdmittedMaxSubset` | +0.0121 | `[-0.0017, +0.0270]` | PASS | +72.1% | `[+69.3%, +75.2%]` | -344.2 B | PASS |
| `OneShot_ConfidenceRouting_0.55` | +0.0010 | `[-0.0076, +0.0096]` | PASS | +14.3% | `[+12.5%, +15.8%]` | -22.1 B | FAIL |
| `MatchedBudgetRandom_1.45` | +0.0041 | `[-0.0104, +0.0203]` | FAIL | +4.8% | `[-9.7%, +17.9%]` | -6.7 B | FAIL |
| `Unimodal_Text` | +0.0087 | `[-0.0081, +0.0269]` | PASS | -3522.5% | `[-5414.9%, -2617.4%]` | +129.4 B | FAIL |

## 3. Physical Loopback TCP Telemetry Summary (42,000 Persistent Queries)

| Policy | SQ Macro-F1 | Total App Bytes | Mean App Bytes/q | Mean Wall (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Useful On-Time | Budget Viol. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AlwaysOn_Direct_Serial` | 0.5287 | 2,759,404 | 525.6 | 0.87 | 0.87 | 1.32 | 1.59 | 99.77% | 9.54% |
| `AlwaysOn_Direct_Parallel` | 0.5287 | 2,758,767 | 525.5 | 0.92 | 0.99 | 1.46 | 1.74 | 99.77% | 9.54% |
| `Static_TA_Direct` | 0.5363 | 1,450,536 | 276.3 | 0.51 | 0.56 | 0.86 | 1.12 | 99.77% | 0.00% |
| `BudgetAdmittedMaxSubset` | 0.5294 | 2,505,399 | 477.2 | 0.86 | 0.74 | 1.46 | 1.83 | 99.77% | 0.00% |
| `OneShot_ConfidenceRouting_0.55` | 0.5405 | 814,665 | 155.2 | 0.54 | 0.22 | 1.62 | 2.00 | 99.77% | 0.00% |
| `SequentialConfidence_0.55_0.45` | 0.5415 | 698,556 | 133.1 | 0.51 | 0.23 | 1.31 | 1.80 | 99.77% | 0.00% |
| `Unimodal_Text` | 0.5328 | 19,284 | 3.7 | 0.22 | 0.18 | 0.33 | 0.68 | 99.77% | 0.00% |
| `MatchedBudgetRandom_1.45` | 0.5373 | 733,941 | 139.8 | 0.45 | 0.24 | 1.13 | 1.64 | 99.77% | 0.00% |

* Sensitivity runs (`CLEAN_MULTIMODAL`, seed 42, persistent 3-rep, non-persistent TCP connect, concurrency=4) are stored in [distributed_summary.json](../results/v13_2/systems/distributed_summary.json) and raw rows in `distributed_sensitivity.csv`.
* Controlled fault-injection results (`36` queries across 9 fault/boundary scenarios) are stored in [distributed_fault_benchmark.csv](../results/v13_2/systems/distributed_fault_benchmark.csv).
