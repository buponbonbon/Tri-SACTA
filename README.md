# Tri-SACTA — Resource-Aware Multimodal Feature Acquisition Under Service Constraints

**Tri-SACTA** is a research system and empirical evaluation framework for resource-aware multimodal feature acquisition under explicit per-request cost budgets, latency deadlines, and modality-fault workloads.

> [!NOTE]
> **Current Scientific Status:**
> - **V13.2 Experimental Foundation:** `V13_2_FOUNDATION_USABLE_WITH_DOCUMENTED_LIMITATIONS` ([docs/V13_2_FOUNDATION_CLOSURE.md](docs/V13_2_FOUNDATION_CLOSURE.md))
>   - **Phase II Confirmatory (Frozen `SequentialConfidence_0.55_0.45`):** `PHASE_II_COMPLETED` (`CONFIRMATORY_SUPPORTED` on `4,131` CMU-MOSEI utterances / `617` videos; permanently exposed and retired from V14/V15 confirmation).
>   - **Phase III Supplementary Network Emulation:** Historical pilot (`pilot_v1/`, `4,224` requests, post-hoc delay) preserved separately from the rectified benchmark (`rectified/`, `26,136` requests across `99` dev utterances / `12` videos with in-acquisition delay & event tracing).
>   - **V13.2 Novelty Status:** `OVERLAP_IDENTIFIED` (Empirical Systems Evaluation).
> - **V14 Research Stage (Active Branch `research/v14-decision-theoretic-acquisition-research`):** Research-gap discovery, formal decision-theoretic formulation, and development-only identifiability/oracle diagnostics (`V14_METHOD_GAP_SUPPORTED_READY_FOR_V15`). *V14 is a design & feasibility stage only — V15 is not yet implemented or validated.*

---

## 1. V13.2 Phase II Confirmatory Summary (`N = 4,131`, `617` Video Clusters)
- **`SequentialConfidence_0.55_0.45`:** Service-Quality Macro-F1 = `0.4895`, `134.6` application-protocol bytes/request, `99.85%` useful on-time rate, `99.85%` acceptance rate, `0.00%` budget violation rate.
  - vs `AlwaysOn_Direct_Serial` (`0.4852`, `528.2 B`): `ΔF1 = +0.0042` (`[-0.0014, +0.0103]`), **`74.52%`** byte reduction (`[73.78%, 75.22%]`).
  - vs `AlwaysOn_Direct_Parallel` (`0.4852`, `528.2 B`): `ΔF1 = +0.0042` (`[-0.0014, +0.0103]`), **`74.52%`** byte reduction (`[73.78%, 75.22%]`).
  - vs `Static_TA_Direct` (`0.4841`, `276.7 B`): `ΔF1 = +0.0054` (`[+0.0019, +0.0088]`), **`51.35%`** byte reduction (`[49.93%, 52.76%]`).
- **Strong Competing Baselines:** `OneShot_ConfidenceRouting_0.55` (`0.4897` SQ-F1, `156.2 B/req`), `Unimodal_Text` (`0.4834` SQ-F1, `5.0 B/req`).

---

## 2. V14 Research Direction & Dev-Only Diagnostics (`val`, `N = 250`)
- **Primary Candidate (Design Spec for V15):** *Joint Sequential-Parallel Option-Value Policy (`JOV-Acquisition`)* over `{STOP, ACQ_A, ACQ_V, ACQ_AV_PAR}` with asymmetric rescue/harm risk (`docs/v14/V14_08_PRIMARY_METHOD_DESIGN.md`).
- **Dev-Only Oracle Headroom (`val`):** `+0.1886` to `+0.3863` Macro-F1 reachable headroom over `Static_TAV`; `T -> TV` exhibits `158` rescues vs `246` harms across 7 workloads.
- **5-Fold `GroupKFold` Learnability (`val`):** Observable state improves out-of-group AUROC over `max(p_T)` on `T->TA` rescue (`0.7659` vs `0.7278`), `T->TA` harm (`0.7671` vs `0.7179`), and `T->TAV` harm (`0.7115` vs `0.6561`).
- **Full V14 Package:** [docs/v14/V14_00_RESEARCH_FOUNDATION.md](docs/v14/V14_00_RESEARCH_FOUNDATION.md) through [docs/v14/V14_11_RESEARCH_DECISION.md](docs/v14/V14_11_RESEARCH_DECISION.md).

---

## 3. Reproduction Instructions (No Protected Holdout Rerun)
1. **CI / Synthetic Regression Tests:** `python -m unittest -v tests/test_v13_2_evaluator_regression.py tests/test_v13_2_runtime_safety.py tests/test_v13_2_verifier_adversarial.py tests/test_v13_2_phase2_pipeline.py tests/test_phase3_audit_integrity.py tests/test_phase3_network_emulation.py tests/test_phase3_rectified_transport.py tests/test_v14_diagnostics.py`
2. **Read-Only Phase II Integrity Verification:** `python scripts/publication_audit/verify_phase2_integrity.py`
3. **V14 Development-Only Diagnostics (`val`):** `python scripts/v14_research/run_v14_diagnostics.py`
