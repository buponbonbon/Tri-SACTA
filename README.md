# Tri-SACTA V13.2: Phase I Protocol Freeze & Systems Evaluation

- **Branch:** `research/v13.2-final-systems-validation-release-candidate`
- **Scientific Source SHA:** `f51a9edc819623b10339c9b0c8232a85b09643a9`
- **Freeze Manifest SHA-256:** `b5e004f17c75dcfeff65bdb9f51841ef2c1deecdc427c036f904a8db3e34d008`
- **Eligible Holdout Manifest SHA-256:** `146157febd1f2c775e4195f694293d6e8701fcf4e8537884e4d15ad55251502e`
- **Engineering Recommendation:** `RELEASE_CANDIDATE_REVIEWABLE`
- **Scientific Readiness Status:** `READY_FOR_HOLDOUT_REVIEW`
- **Development Effect Status:** `SUPPORTED`
- **Novelty Status:** `OVERLAP_IDENTIFIED`
- **Confirmatory Execution Status:** `NOT_RUN` (`PENDING_AUTHOR_APPROVAL`)

---

## 1. Scope, Candidate Policy & Protocol Amendment

- **Fixed Candidate:** `SequentialConfidence_0.55_0.45` (audio-first, stages `T -> TA -> TAV`, $\tau_1 = 0.55, \tau_2 = 0.45$), backed by 21 fixed MLP checkpoints (`T, A, V, TA, TV, AV, TAV` $\times$ seeds `42, 100, 2026`).
- **Amended Primary Resource Endpoint (Section 3.5):** Measured **application-protocol bytes per submitted request** over localhost TCP loopback sockets across the complete **7-workload $\times$ 3-seed** validation population (`5,250` queries/policy, persistent keep-alive TCP), evaluated jointly with **Service-Quality Macro-F1** (rejections & timeouts counted as FN for the true class) against BOTH `AlwaysOnFusion` (`AlwaysOn_Direct_Serial`, `AlwaysOn_Direct_Parallel`) and `Static_TA_Direct`. Nominal acquisition cost (`1.0 / 1.2 / 1.3`) is reported separately as a secondary simulation endpoint.
- **Holdout Isolation (Strict Phase I Boundary):** The candidate holdout pool (`4,131` utterances across `617` video clusters in `mosei.pkl`) has **zero feature extractions and zero model predictions** executed in Phase I (`results/v13_2/provenance/holdout_access_ledger.json`).

---

## 2. Recomputed Distributed TCP Benchmark (7 Workloads $\times$ 3 Seeds, Persistent Loopback TCP)

Source: [results/v13_2/systems/distributed_summary.json](results/v13_2/systems/distributed_summary.json) & [results/v13_2/systems/distributed_benchmark.csv](results/v13_2/systems/distributed_benchmark.csv)

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

* **Client/Server Byte Agreement:** Client = `11,740,552` B vs Server = `11,741,673` B (diff = `1121` B across 42,000 queries; caused by 3 timeout requests where server responded post-timeout).
* **Important Tail-Latency & Baseline Trade-Offs (Section 4.4 / 7):**
  - In persistent TCP, `Static_TA_Direct` achieves lower p95/p99 tail latency (`0.86 / 1.12 ms`) than `SequentialConfidence_0.55_0.45` (`1.31 / 1.80 ms`).
  - In non-persistent per-request TCP sensitivity (`CLEAN_MULTIMODAL`, seed 42), `OneShot_ConfidenceRouting_0.55` achieves lower mean/p95/p99 wall latency than `SequentialConfidence_0.55_0.45`, and across all 7 workloads `SequentialConfidence` reduces app bytes by `+14.3%` vs `OneShot_ConfidenceRouting_0.55` (`95% CI [+12.5%, +15.8%]`), which does not meet a 15% lower bound vs OneShot.
  - Same-host loopback socket measurements do **not** measure WAN latency, multi-device wireless links, device energy, or raw media ingestion.

---

## 3. Paired Video-Grouped Cluster Bootstrap ($B = 2,000$, Seed `1320`, `29` Video Clusters)

Source: [results/v13_2/statistics/paired_bootstrap_summary.json](results/v13_2/statistics/paired_bootstrap_summary.json)

| Comparator vs Candidate | Observed $\Delta$ SQ-F1 | 95% Clustered CI | NI (-0.010) | AppBytes Saved | 95% CI | Abs Diff | $\ge 15\%$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AlwaysOn_Direct_Serial` | +0.0127 | `[-0.0037, +0.0313]` | PASS | +74.7% | `[+72.1%, +77.5%]` | -392.5 B | PASS |
| `AlwaysOn_Direct_Parallel` | +0.0127 | `[-0.0037, +0.0313]` | PASS | +74.7% | `[+72.1%, +77.5%]` | -392.4 B | PASS |
| `Static_TA_Direct` | +0.0052 | `[-0.0047, +0.0158]` | PASS | +51.8% | `[+46.9%, +57.3%]` | -143.2 B | PASS |
| `BudgetAdmittedMaxSubset` | +0.0121 | `[-0.0017, +0.0270]` | PASS | +72.1% | `[+69.3%, +75.2%]` | -344.2 B | PASS |
| `OneShot_ConfidenceRouting_0.55` | +0.0010 | `[-0.0076, +0.0096]` | PASS | +14.3% | `[+12.5%, +15.8%]` | -22.1 B | FAIL |
| `MatchedBudgetRandom_1.45` | +0.0041 | `[-0.0104, +0.0203]` | FAIL | +4.8% | `[-9.7%, +17.9%]` | -6.7 B | FAIL |
| `Unimodal_Text` | +0.0087 | `[-0.0081, +0.0269]` | PASS | -3522.5% | `[-5414.9%, -2617.4%]` | +129.4 B | FAIL |

---

## 4. Verification & Reproduction Commands

```powershell
# 1. Metadata-only provenance & checkpoint/environment lock (no holdout inference)
python scripts/v13_2_audit_provenance.py

# 2. Fair development offline benchmark (84,000 queries) & boundary guardrails
python scripts/v13_2_evaluate_development.py

# 3. Real TCP socket benchmark (42,000 primary queries + sensitivity + fault injection)
python scripts/v13_2_benchmark_distributed.py

# 4. Dual-endpoint paired video-grouped block bootstrap (B=2,000)
python scripts/v13_2_cluster_statistics.py

# 5. Automated fail-closed gate verifier
python scripts/v13_2_verify_gates.py

# 6. Full adversarial & regression tests (174/174 passing)
python -m unittest -v tests/test_v13_2_evaluator_regression.py tests/test_v13_2_runtime_safety.py tests/test_v13_2_verifier_adversarial.py
```

---

## 5. Authoritative Artifact & Document Links

* Blocker Matrix & Resolution: [docs/V13_2_PHASE_I_BLOCKER_RESOLUTION.md](docs/V13_2_PHASE_I_BLOCKER_RESOLUTION.md)
* Prior Art & Novelty Matrix: [docs/V13_2_05_PRIOR_ART_AND_CLAIM_EVIDENCE.md](docs/V13_2_05_PRIOR_ART_AND_CLAIM_EVIDENCE.md)
* Frozen Executable Protocol: [results/v13_2/protocol/freeze_manifest.json](results/v13_2/protocol/freeze_manifest.json)
* Eligible Holdout Manifest: [results/v13_2/provenance/eligible_holdout_manifest.json](results/v13_2/provenance/eligible_holdout_manifest.json)
* Machine Gate Verdict: [results/v13_2/validation/final_decision.json](results/v13_2/validation/final_decision.json)
* Handover Report: [docs/HANDOVER_V13_2.md](docs/HANDOVER_V13_2.md)

---

## 6. Historical Evolution & Scientific Integrity Notice

* **V11 Formal Retraction:** All preliminary V11 exploratory test-time adaptation metrics and streaming SOTA claims were formally **retracted** under decision `FREEZE_TRI_SACTA`; the historical 500-utterance `test` split is permanently quarantined as `DEVELOPMENT-EXPOSED`.
* **V12 / V12.1 / V13 / V13.1 Lineage:** Transitioned from test-time adaptation to distributed systems orchestration (`V12`–`V13`), rectified expected-utility equivalence to sequential confidence thresholding (`V13.1`), and eliminated remaining protocol/verifier blockers in `V13.2`.
