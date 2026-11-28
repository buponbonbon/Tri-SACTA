# Tri-SACTA V13.2 — Resource-Aware Multimodal Feature Acquisition Under Service Constraints

**Tri-SACTA** is a research system and empirical evaluation framework for resource-aware multimodal feature acquisition under explicit per-request cost budgets, latency deadlines, and modality-fault workloads.

> [!NOTE]
> **Current Scientific Status:**
> - **Phase II Confirmatory Status:** `PHASE_II_COMPLETED` (`CONFIRMATORY_SUPPORTED` on `4,131` video-disjoint CMU-MOSEI holdout utterances across `617` video clusters).
> - **Phase III Supplementary Status:** `COMPLETED` (controlled application-layer TCP loopback network emulation on `val` split, `4,224` measured queries).
> - **Novelty Status:** `OVERLAP_IDENTIFIED` (positioned as an **Empirical Systems Evaluation**, not a novel ML algorithm).

---

## 1. Phase II Confirmatory Summary (`N = 4,131`, `617` Video Clusters)
Evaluated across `7` workloads × `3` seeds (`1,388,016` offline traces, `694,008` measured TCP traces, `B = 2,000` paired video-cluster bootstrap):
- **`SequentialConfidence_0.55_0.45`:** Service-Quality Macro-F1 = `0.4895`, `134.6` application-protocol bytes/request, `99.85%` useful on-time rate, `99.85%` acceptance rate, `0.00%` budget violation rate.
  - vs `AlwaysOn_Direct_Serial` (`0.4852`, `528.2 B`): `ΔF1 = +0.0042` (`95% CI [-0.0014, +0.0103]`), **`74.52%`** byte reduction (`[73.78%, 75.22%]`).
  - vs `AlwaysOn_Direct_Parallel` (`0.4852`, `528.2 B`): `ΔF1 = +0.0042` (`95% CI [-0.0014, +0.0103]`), **`74.52%`** byte reduction (`[73.78%, 75.22%]`).
  - vs `Static_TA_Direct` (`0.4841`, `276.7 B`): `ΔF1 = +0.0054` (`95% CI [+0.0019, +0.0088]`), **`51.35%`** byte reduction (`[49.93%, 52.76%]`).
- **Strongest Competing Baseline:** `OneShot_ConfidenceRouting_0.55` (`0.4897` SQ-F1, `156.2 B/req`; candidate `ΔF1 = -0.0002`, `13.84%` byte saving).
- **Text-Only Baseline:** `Unimodal_Text` (`0.4834` SQ-F1, `5.0 B/req`).

---

## 2. Phase III Supplementary Network-Aware Evaluation
- **Regime A (Original `15/30/50 ms` Deadlines) & Regime B (Relaxed `250 ms` Deadline):** `0, 10, 30, 80 ms` target RTT, `10/1 Mbps` bandwidth, and `±5 ms` jitter over TCP loopback (`4,224` queries on `val`).
- **Analytical Latency Model (`T_p = H_p + R_p * r + B_p,critical / b`):** `MAE = 4.74 ms` across unseen RTTs.
- **Details:** See [docs/V13_2_PHASE3_NETWORK_EVALUATION.md](docs/V13_2_PHASE3_NETWORK_EVALUATION.md) and [docs/V13_2_PUBLICATION_SCIENTIFIC_AUDIT_REPORT.md](docs/V13_2_PUBLICATION_SCIENTIFIC_AUDIT_REPORT.md).

---

## 3. Reproduction Instructions
1. **Environment Setup:** `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-ci.txt`
2. **Synthetic Regression Tests:** `python -m unittest -v tests/test_v13_2_evaluator_regression.py tests/test_v13_2_runtime_safety.py tests/test_v13_2_verifier_adversarial.py tests/test_v13_2_phase2_pipeline.py tests/test_phase3_audit_integrity.py tests/test_phase3_network_emulation.py`
3. **Read-Only Phase II Artifact Verification:** `python scripts/publication_audit/verify_phase2_integrity.py`
4. **Supplementary Network Experiment (Dev Only):** `python scripts/phase3_network/run_network_benchmark.py`

*(Note: Phase II confirmatory evaluation is a completed single-shot run; do not rerun holdout inference.)*

---

## 4. Explicit Limitations
- No novel ML confidence-routing algorithm (`novelty_status = OVERLAP_IDENTIFIED` with DynMM, A2MT, SPINN, SeMARC).
- Single-host TCP loopback and controlled application-layer delay emulation (not physical WAN deployment or hardware energy measurement).
- Cached feature evaluation on CMU-MOSEI only; video-group disjointness verified, speaker disjointness `UNKNOWN`.
- `OneShot_ConfidenceRouting_0.55` slightly exceeds `SequentialConfidence_0.55_0.45` in overall Macro-F1 (`0.4897` vs `0.4895`), and `TEXT_EMBEDDING_NOISE` exposes overconfident Stage-1 early-exit failures.
