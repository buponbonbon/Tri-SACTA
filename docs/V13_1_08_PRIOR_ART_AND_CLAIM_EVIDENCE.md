# Tri-SACTA V13.1 — Prior Art Matrix & Claim-Evidence Ledger

**Document Version:** 1.0.0  
**Timestamp (UTC):** 2026-11-24T15:53:00Z  
**Branch:** `research/v13.1-conditional-utility-independent-validation`  
**Target Venue:** *Multimedia Tools and Applications* (Springer, SCImago Q1/Q2, $0 mandatory APC).

---

## 1. Nearest-Neighbor Prior Art Comparison Matrix

| Dimension | One-Shot Confidence Routing (e.g. Gao et al. 2021) | Evidential Multimodal Fusion (Wu et al. 2026 / RegCal-Net) | Expected Utility / VOI (Historical Tri-SACTA V13) | **Tri-SACTA V13.1 (Systems Architecture)** |
|---|---|---|---|---|
| **Action Space** | One-shot binary branching ($T$ vs $TAV$) | Static fusion with learned modal discounting | Sequential greedy acquisition ($T \to TA \to TAV$) | **Budget-bounded sequential state machine ($T \to TA \to TAV$ or $TV$)** |
| **Feedback Mechanism** | Initial confidence threshold | Evidential Dirichlet variance | Expected utility (affine confidence threshold) | **Per-stage confidence re-evaluation after physical retrieval** |
| **Decision Parameterization** | Heuristic scalar threshold | Evidence concentration | Hardcoded cross-modal rescue priors | **Empirically validated derived thresholds + simplified strong baseline** |
| **Wire & Transport Telemetry** | Assumed zero or synthetic | Offline simulated tensors | Assigned nominal costs (1.0, 1.2, 1.3) | **Measured TCP socket transmission (raw payload + framing bytes)** |
| **Network RTT & Latency** | Synthetic constants | Neglected | Assigned constant delays (1, 5, 8 ms) | **High-resolution monotonic timer measurements (`time.perf_counter`)** |
| **Uncertainty Calibration** | None / Softmax only | Dirichlet variance | Reused offline conformal calibration | **Policy-matched split conformal sets on disjoint calibration split** |
| **Experimental Headroom** | Not analyzed | Not analyzed | Unconstrained, non-sequential label-aware oracle | **Reachable sequential oracle with sunk cost accounting (+0.1524 F1)** |
| **Reproducibility & Lineage** | Ad-hoc splits | Standard benchmarks | Quarantined test set mislabeled as fresh | **Full cryptographic SHA-256 audit, 4,131 unexposed pool identified** |

---

## 2. Claim-Evidence-Counterevidence Ledger

Each claim is formally categorized in accordance with Section 11 of the research protocol:

### Claim C1 (Algorithmic Expressivity):
- **Statement:** The sequential modality acquisition policy can be conditioned on observed state distributions to differentiate queries with identical confidence.
- **Evidence:** `tests/test_v13_1_utility.py::test_conditional_utility_counterfactual_expressivity` demonstrates that two queries with identical $p_{\max} = 0.55$ yield distinct estimated gains ($0.1012$ vs $0.0791$) and distinct routing actions.
- **Counterevidence:** Empirical evaluation across 7 stress workloads (`results/v13_1/evaluation/policy_metrics.json`) reveals that Candidate M2 (`ConditionalUtilityRouter`) achieves $0.5321$ Macro-F1, losing by $-0.0126$ to the simpler `SequentialConfidenceRouter` ($0.5447$).
- **Status:** `SUPPORTED_EXPLORATORY` (Expressive, but scientifically inferior to strong baseline).

### Claim C2 (Bandwidth & Operational Efficiency):
- **Statement:** Sequential modality acquisition achieves substantial real wire bandwidth savings and latency reduction over real network sockets compared to full multimodal transmission.
- **Evidence:** `results/v13_1/systems/distributed_summary.json` demonstrates an empirically measured **61.64% reduction in total wire bytes transmitted** (51,029 B vs 133,030 B) and a **53.3% reduction in end-to-end response latency** (1.12 ms vs 2.40 ms) over TCP sockets.
- **Counterevidence:** Unimodal text alone transmits 0 bytes, but achieves lower Macro-F1 ($0.5636$ vs $0.5992$).
- **Status:** `SUPPORTED_SYSTEMS_EVIDENCE` (Gate G7 PASS).

### Claim C3 (Headroom & Oracle Feasibility):
- **Statement:** Significant predictive headroom exists for reachable sequential modality acquisition over static unimodal or bimodal baselines.
- **Evidence:** `results/v13_1/oracle/reachable_oracle_summary.json` demonstrates a Reachable Sequential Oracle Macro-F1 of **0.7288 $\pm$ 0.0048** (Headroom **+0.1524** over static $TA$, 95% Clustered CI: `[+0.1254, +0.1846]`).
- **Counterevidence:** Deployable policies capture a fraction of this headroom due to prediction uncertainty.
- **Status:** `SUPPORTED_DIAGNOSTIC` (Gate G3 PASS).

### Claim C4 (Statistical Pareto Dominance over Full Fusion):
- **Statement:** Sequential confidence routing is non-inferior in predictive performance and statistically superior in cost reduction relative to Always-On Fusion.
- **Evidence:** `results/v13_1/statistics/paired_bootstrap_summary.json` ($B = 2,000$ paired video-grouped cluster bootstrap):
  - $\Delta \text{Macro-F1} = \mathbf{+0.0147}$, 95% CI: `[-0.0013, +0.0332]` (Non-inferiority margin $-0.01$ passed: lower bound $-0.0013 > -0.0100$).
  - Nominal Cost Saved: **+48.9%**, 95% CI: `[+47.2%, +50.7%]` (Exceeds $\ge 15\%$ criterion).
- **Status:** `SUPPORTED_EXPLORATORY` (Gate G4 PASS).

---

## 3. Publication Positioning: *Multimedia Tools and Applications*

1. **Aims & Scope Alignment:**
   - *Multimedia Tools and Applications* (Springer) explicitly seeks papers addressing "multimedia systems, networked and distributed multimedia, and resource allocation in multimedia applications."
   - Framing Tri-SACTA as a **distributed, resource-aware multimodal orchestration architecture** directly fits the journal's core systems focus.
2. **Subscription Model ($0 Author Fees):**
   - Springer offers traditional subscription publishing with no mandatory open-access APC, satisfying the user requirement.
3. **Journal Quality Verification:**
   - SCImago Journal Rank (SJR): Q1/Q2 in Computer Science (Software, Information Systems, Media Technology).
