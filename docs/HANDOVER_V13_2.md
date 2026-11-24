# Tri-SACTA V13.2: Phase I Handover & Release Candidate Protocol Report

**Repository:** https://github.com/buponbonbon/Tri-SACTA  
**Active Branch:** `research/v13.2-final-systems-validation-release-candidate`  
**Base Commit:** `e71147fc511c5d50c218ded1fdad96953ae15b90`  
**Phase Executed:** **Phase I (Audit, Evaluator Repair, Fair Baselines, Systems Telemetry, Freeze Package)**  
**Phase II Status:** **PENDING_AUTHOR_APPROVAL (Gated prior to holdout inference)**  
**Target Publication Venue:** *Multimedia Tools and Applications* (Springer, SJR Q2, $0 mandatory APC via subscription track) or *Multimedia Systems* (Springer).

---

## 1. Executive Summary & Decoupled Verdicts

| Verdict Field | Value | Operational Meaning |
| :--- | :---: | :--- |
| **`merge_recommendation`** | **`REVIEWABLE_DEVELOPMENT_PROTOTYPE`** | All engineering bugs, evaluator budget overrides, socket modulo indexing, and verifier hardcodings are resolved. Code is clean and regression-tested. |
| **`scientific_status`** | **`READY_FOR_HOLDOUT_REVIEW`** | Developmental evidence confirms predictive non-inferiority and >45% cost savings. The protocol and candidate holdout pool (4,131 samples) are frozen pending author approval. |

---

## 2. All P0 Defects & Exact Code Rectifications

1. **Stale Hash & Markdown Discrepancies (Section 2.1):**
   - *Defect:* Markdown reported `mosei.pkl` hash `950662d...` and label counts `245/167/588`.
   - *Fix:* Reconciled against physical files via [v13_2_audit_provenance.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/scripts/v13_2_audit_provenance.py). Actual on-disk hash is `90701ad5bfa8767ca253af729c1d346e9a8d259ed24f16de113c845ffb60d90e` and true 3-class distribution is `390/238/372` for train, `100/65/85` for val, `43/66/141` for cal, `213/117/170` for historical test, `1132/900/2099` for unexposed test pool.
2. **Video-Derived Speaker Proxy Disclosure (Section 2.2):**
   - *Defect:* Speaker IDs were derived via `spk_id = f"mosei_spk_{vid[:8]}"`, making "0 shared speakers" a trivial video artifact.
   - *Fix:* Updated schema to `video_group_id`, `speaker_id_verified: null`, `speaker_proxy`, `speaker_overlap_status: UNKNOWN`. Statistical independence is strictly claimed at the video group level.
3. **Unconditional Budget Bug (Section 3.1):**
   - *Defect:* Evaluator passed `cost_budget=3.5` unconditionally inside the loop.
   - *Fix:* In [v13_2_evaluate_development.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/scripts/v13_2_evaluate_development.py), exact per-query `w_budgets[i]` and `w_deadlines[i]` are passed. Infeasible initial budgets (< 1.0) and blackouts return explicit rejection (`prediction = -1`, `spent_cost = 0.0`).
4. **Masked Feature Access Trap (Section 3.1):**
   - *Fix:* Implemented `MaskedFeatureAccessViolation` in [baseline_policies.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/src/service/baseline_policies.py) to prevent unacquired feature reading.
5. **Baseline Transport Fairness (Section 5.1):**
   - *Defect:* Always-On and Static TA executed redundant 1st Text forward passes.
   - *Fix:* Implemented direct TAV and direct TA paths in [distributed_coordinator.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/src/service/distributed_coordinator.py). Sequential routing pays intermediate forward passes and sequential branching.
6. **Socket Modulo Indexing Defect (Section 5.2):**
   - *Defect:* `sample_idx % total_samples` silently served incorrect samples on bad IDs.
   - *Fix:* Implemented strict bounds checking in [distributed_providers.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/src/service/distributed_providers.py) returning `400 Bad Request`.
7. **Verifier Cosmetic Passes (Section 8):**
   - *Defect:* V13.1 hardcoded `g0_pass = True`, excluded G5, and printed `ALL GATES PASS`.
   - *Fix:* [v13_2_verify_gates.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/scripts/v13_2_verify_gates.py) recomputes metrics from raw traces and passes 7 adversarial corruption tests ([test_v13_2_verifier_adversarial.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/tests/test_v13_2_verifier_adversarial.py)).

---

## 3. Developmental Benchmark & Statistical Headroom

Across 84,000 developmental queries (16 policies $\times$ 7 workloads $\times$ 3 seeds) with exact per-request budget enforcement:

| Policy | Primary Macro-F1 | Mean Cost | Mean Latency | Budget Violations | Acceptance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SequentialConfidence_0.55_0.45** | **0.5415 $\pm$ 0.0569** | **1.416** | **3.09 ms** | **0.0%** | **99.77%** |
| ConfidenceRouting_0.55 | 0.5405 $\pm$ 0.0569 | 1.531 | 3.84 ms | 0.0% | 99.77% |
| Static_TA | 0.5363 $\pm$ 0.0504 | 1.743 | 4.14 ms | 0.0% | 99.77% |
| ConditionalUtility_0.05 | 0.5310 $\pm$ 0.0552 | 1.359 | 3.25 ms | 0.0% | 99.77% |
| BudgetAdmittedMaxSubset | 0.5294 $\pm$ 0.0565 | 2.656 | 9.76 ms | 0.0% | 99.77% |
| AlwaysOnFusion (Unrestricted) | 0.5287 $\pm$ 0.0560 | 2.838 | 10.76 ms | 9.54% | 99.77% |

### Paired Video-Grouped Bootstrap ($B = 2,000$ Resamples)
- **vs AlwaysOnFusion:** $\Delta \text{Macro-F1} = \mathbf{+0.0127}$ `[-0.0037, +0.0313]`, Cost Saved = $\mathbf{+50.1\%}$ `[+48.6%, +51.8%]` (**PASS** on non-inferiority and $>15\%$ savings).
- **vs BudgetAdmittedMaxSubset:** $\Delta \text{Macro-F1} = \mathbf{+0.0121}$ `[-0.0017, +0.0270]`, Cost Saved = $\mathbf{+46.7\%}$ `[+45.1%, +48.5%]` (**PASS**).
- **vs Static_TA:** $\Delta \text{Macro-F1} = \mathbf{+0.0052}$ `[-0.0047, +0.0158]`, Cost Saved = $\mathbf{+18.8\%}$ `[+16.3%, +21.5%]` (**PASS**).

---

## 4. Physical Distributed TCP Socket Telemetry

Benchmarked over physical localhost TCP stream sockets with isolated providers (`AudioProvider`: 8765, `VisualProvider`: 8766):
- **Application Wire Bytes Reduction:** **63.48%** (66,308 B for Sequential vs 181,560 B for Always-On).
- **End-to-End Latency Reduction:** **62.88%** (4.79 ms mean wall time vs 12.89 ms).
- **Predictive Quality on Socket Run:** Macro-F1 = **0.5992** (Sequential) vs **0.5829** (Always-On).

---

## 5. Frozen Confirmatory Protocol & Manifest

- **Locked Manifest:** [freeze_manifest.json](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v13_2/protocol/freeze_manifest.json)
- **Manifest Content Digest (SHA-256):** `54ef6b48f9aede4c0e119e0e1bda29bbfcc16e0c4c943241101a6e2e98fbe0f8`
- **Candidate Pool:** 4,131 unexposed utterances across 617 unique video clusters (0 video overlap with historical data).
- **Phase I Boundary:** Confirmatory evaluation is deliberately **NOT_RUN** until the author reviews and approves the exact freeze commit.
