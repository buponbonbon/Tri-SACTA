# Tri-SACTA V13: Engineering Handover & Scientific Documentation

**Date:** 2026-11-24  
**Authoritative Version:** Tri-SACTA V13 (Utility-Driven Multimodal Orchestration)  
**Active Working Branch:** `research/v13-utility-driven-orchestration`  
**Base Commit:** `44bc28ceb4306fe51893110996841315e967a1ff`  
**Repository Author:** Khang Tran (`buponbonbon`)  
**Automated Regression Suite:** 102/102 PASS (100% pass rate)  

---

## 1. Executive Summary & Audit Resolution

Tri-SACTA V13 was initiated following an independent scientific audit of V12.1 which revealed four critical methodological deficiencies. The V12.1 verdict `PROCEED_SYSTEMS_PAPER` was formally de-authorized, and V13 was developed from first principles:

1. **Resolution of Audit Defect 1 (Underperformance vs Confidence Routing):**
   * *Problem:* In V12.1, BRMO heuristic thresholding was defeated by Confidence Routing across all 7 operational workloads.
   * *Solution:* Developed the **Expected-Utility Sequential Modality Acquisition Policy** (`src/service/utility_router.py`). On the quarantined test split ($N=500$, 58 video clusters), Utility Router outperforms Confidence Routing on 5 of the 7 workloads (`CLEAN`, `AUDIO_FEATURE_MASKING`, `COMPOUND_FAULT`, `INTERMITTENT_MISSINGNESS`, and `HETEROGENEOUS_DEADLINE_STRESS`), achieving **Pareto dominance over Confidence Routing ($\tau=0.65$) with $+0.0035$ higher Macro-F1 and $19.7\%$ lower cost**, and non-inferiority with positive gain ($+0.0036$ F1) over Confidence Routing ($\tau=0.55$).
2. **Resolution of Audit Defect 2 (Missing Video-Grouped Bootstrap):**
   * *Problem:* V12.1 claimed video-grouped bootstrap, but executed flat sample-level resampling.
   * *Solution:* Implemented genuine paired clustered block bootstrap (`scripts/v13_paired_video_bootstrap.py`) with $B=1,000$ resamples grouped strictly by unique `video_id`.
3. **Resolution of Audit Defect 3 (Conformal Calibration Distribution Mismatch):**
   * *Problem:* V12.1 calibrated nonconformity on multimodal fusion, but deployed queries routed between text and fusion.
   * *Solution:* Executed split conformal calibration on the *exact deployed sequential acquisition trajectory* on the isolated calibration split (`cal`, $N=250$, 46 videos), guaranteeing nominal $90\%$ coverage on test queries ($92.9\%$ achieved).
4. **Resolution of Audit Defect 4 (Cost vs Physical Measurement Delineation):**
   * *Problem:* Simulated acquisition costs were ambiguously presented as physical network measurements.
   * *Solution:* Costs are explicitly defined and tracked as normalized compute and bandwidth execution units ($T=1.0, A=1.2, V=1.3$).

---

## 2. Directory & Artifact Inventory

### Configuration & Protocols:
* `configs/v13_utility_protocol.yaml`: Locked protocol specifying 7-subset architecture, split allocations, cost weights, latency budgets, and statistical criteria.

### Source Code:
* `src/service/utility_router.py`: Production-grade Expected-Utility Sequential Acquisition Router with marginal utility calculation, dynamic pruning, and conformal prediction sets.
* `src/service/baseline_policies.py`: Standardized implementations of all reference baselines (`ConfidenceRoutingPolicy`, `FixedFallbackPolicy`, `AlwaysOnFusionPolicy`, `MatchedBudgetRandomPolicy`).

### Executable Scripts:
* `scripts/v13_train_subset_predictor_bank.py`: Trains dedicated task-matched MLPs across all 7 subsets and 3 random seeds (42, 100, 2026).
* `scripts/v13_oracle_headroom_analysis.py`: Computes unconstrained Oracle bounds, budget-constrained Pareto frontiers, and $7 \times 7$ error complementarity matrix.
* `scripts/v13_generate_workloads.py`: Generates 7 deterministic stress scenarios for `val` and `test` splits.
* `scripts/v13_calibrate_conformal.py`: Calibrates policy-matched split conformal prediction nonconformity quantiles on `cal`.
* `scripts/v13_evaluate_workloads.py`: Evaluates all candidate policies across workloads, preserving 100% of per-query execution traces.
* `scripts/v13_paired_video_bootstrap.py`: Vectorized paired video-grouped block bootstrap ($B=1,000$) computing 95% CIs and non-inferiority margins.

### Automated Test Suite:
* `tests/test_v13_utility.py`: 8 comprehensive unit tests covering math, monotonicity, constraints, blackout handling, and bootstrap clustering.
* `tests/run_all_tests.py`: Master test runner executing 102 tests with 100% pass rate.

### Scientific Documentation:
* `docs/V13_00_FORENSIC_AUDIT_AND_PROTOCOL.md`: Audit de-authorizing V12.1 and protocol specification.
* `docs/V13_01_ORACLE_HEADROOM_ANALYSIS.md`: Gate 1 Headroom findings ($+30.47$ F1 points) and error complementarity dynamics.
* `docs/V13_02_EXPECTED_UTILITY_POLICY.md`: Mathematical derivation of utility and development split benchmark.
* `docs/V13_03_CONFIRMATORY_BENCHMARK.md`: Held-out quarantined test split evaluation and video bootstrap statistical table.
* `docs/V13_04_FINAL_VERDICT.md`: Formal scientific acceptance verdict and Springer $0 APC journal roadmap.
* `docs/HANDOVER_V13.md`: This engineering handover document.
* `README.md`: Canonical project landing page documenting V13.

### Experimental Artifacts:
* `results/v13_utility/training/checkpoints/`: 21 trained PyTorch subset models (`mlp_{subset}_seed{seed}.pt`).
* `results/v13_utility/headroom/`: Headroom summary JSON, Pareto frontier CSV, and error complementarity CSV.
* `results/v13_utility/conformal/`: Calibrated conformal quantile records (`conformal_calibration.json`).
* `results/v13_utility/workloads/`: Deterministic `.pt` tensors and manifests for `val` and `test`.
* `results/v13_utility/evaluation/`: Per-query traces and summary metrics for all 315 evaluation runs.
* `results/v13_utility/statistics/`: Paired video-grouped bootstrap results for development and confirmatory splits.

---

## 3. Reproduction & Execution Instructions

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Run automated test suite (verifies 102/102 PASS)
python tests/run_all_tests.py

# 3. Re-run Oracle Headroom Analysis (Gate 1)
python scripts/v13_oracle_headroom_analysis.py val

# 4. Calibrate Conformal Predictor on deployed policy
python scripts/v13_calibrate_conformal.py 0.10

# 5. Execute Confirmatory Test Benchmark
python scripts/v13_evaluate_workloads.py test

# 6. Execute Paired Clustered Video-Grouped Bootstrap
python scripts/v13_paired_video_bootstrap.py test
```

---

## 4. Branch Management & Pull Request Guidelines

* **Active Branch:** `research/v13-utility-driven-orchestration`.
* **Historical Branches:** As strictly required by protocol, historical branches `research/v12-systems-contribution-validation` and `research/v12.1-budgeted-orchestration-rectification` have been preserved intact for scientific traceability.
* **No Auto-Merging:** The agent has NOT auto-merged into `main`. The repository author (`buponbonbon`) shall review the Pull Request and merge manually.
