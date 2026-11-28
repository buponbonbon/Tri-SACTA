# Tri-SACTA V13.2 — Publication Scientific Audit Report

- **Working Branch:** `research/v13.2-publication-scientific-audit`
- **Authorized Freeze Commit:** `c9177414a2eadd45a6df99dd832b3512a460e3d2`
- **Phase II Results Commit:** `f559b7d82973bcd2b1d5237da36a105d2ac0c5de` (with `.gitignore` update `6675f11`)
- **Freeze Manifest SHA-256:** `fb23d9e16f8557e6aaf807d893fd5d8bf756d96e792e9c835934046e960cbdb3`
- **Audit Mode:** Publication scientific audit & contribution lock (zero retraining, zero threshold tuning, zero holdout re-execution).

---

## 1. Independent Evidence Integrity (Priority 1)

All primary and secondary metrics were independently recomputed directly from `results/v13_2/confirmatory/distributed_benchmark.csv` and `results/v13_2/confirmatory/query_traces.json.gz`:
- **Offline Compressed Traces (`query_traces.json.gz`):** `1,388,016` rows (`4,131` samples × `7` workloads × `3` seeds × `16` policies), `14,813,847` bytes, SHA-256 `38d852b72ec0768e64371654069b6228b2eb2412c42421286251bc745e86c8a1`.
- **Measured TCP Traces (`distributed_benchmark.csv`):** `694,008` rows (`4,131` samples × `7` workloads × `3` seeds × `8` policies), `694,008` unique `(sample_idx, seed, workload, policy)` keys, `4,131` unique utterances, `617` unique video clusters.
- **GitHub Blob vs Local Size Discrepancy:**
  - Local `distributed_benchmark.csv`: `94,272,383` bytes (`\r\n` CRLF).
  - Git blob (`391f1990789d5faa01d369881f13663f2c897310`): `93,578,374` bytes (`\n` LF).
  - Exact difference: `94,272,383 - 93,578,374 = 694,009` bytes = `694,008` data rows + `1` header row. This confirms pure **CRLF-to-LF line-ending normalization** with zero data corruption.
- **Primary Confirmatory Endpoint (`SequentialConfidence_0.55_0.45`, SQ-F1=`0.4895`, `134.6 B/req`, `99.85%` useful on-time, `99.85%` acceptance, `0.00%` budget violation):**
  - vs `AlwaysOn_Direct_Serial` (`0.4852`, `528.2 B`): `ΔF1 = +0.0042` (`[-0.0014, +0.0103]`), byte saving = `74.52%` (`[73.78%, 75.22%]`) -> **PASS**.
  - vs `AlwaysOn_Direct_Parallel` (`0.4852`, `528.2 B`): `ΔF1 = +0.0042` (`[-0.0014, +0.0103]`), byte saving = `74.52%` (`[73.78%, 75.22%]`) -> **PASS**.
  - vs `Static_TA_Direct` (`0.4841`, `276.7 B`): `ΔF1 = +0.0054` (`[+0.0019, +0.0088]`), byte saving = `51.35%` (`[49.93%, 52.76%]`) -> **PASS**.

---

## 2. Hostile Prior-Art Review (Priority 2)

| Prior Work | Core Mechanism | What It Already Establishes | What Tri-SACTA V13.2 Adds | Novelty Risk |
|---|---|---|---|---|
| **DynMM** (Xue & Marculescu, CVPRW 2023) | Coarse/fine gating for dynamic multimodal fusion | Input-dependent skipping of redundant modalities/fusion on CMU-MOSEI | Explicit per-request deadline/budget enforcement + measured TCP wire-byte accounting | High overlap on dynamic modality skipping |
| **A2MT** (Kossen et al., TMLR) | Sequential active feature acquisition over temporal modalities | Cost-aware sequential acquisition policies | Confidence-cascade runtime with fail-closed socket fault tolerance | High overlap on sequential acquisition |
| **SPINN** (Laskaridis et al., MobiCom 2020) | Synergistic progressive inference + device-cloud early-exit scheduling | Deadline- and network-aware progressive inference | Multimodal subset-bank cascade + paired cluster-bootstrap evaluation | High overlap on deadline-aware progressive inference |
| **SeMARC** (NeurIPS) | Sequential modality aggregator + adaptive runtime controller | Incremental multimodal acquisition under runtime constraints | Reproducible frozen-protocol TCP loopback benchmark on 7 stress workloads | High overlap on adaptive runtime control |

**Novelty Verdict:** `novelty_status = OVERLAP_IDENTIFIED` is **preserved**. Combining confidence cascades with TCP socket providers does **not** constitute a novel ML algorithm.

---

## 3. Scientific Findings & Break-Even Analysis (Priority 3)

- **When Sequential Acquisition Wins:**
  - On `CLEAN_MULTIMODAL`, `SequentialConfidence_0.55_0.45` achieves the highest SQ-F1 (`0.5238`) at `188.7 B` (`0.53 ms` mean) vs `OneShot` (`0.5208`, `273.4 B`, `0.72 ms`) and `AlwaysOn_Direct_Parallel` (`0.5151`, `730.9 B`, `1.27 ms`).
  - On `HETEROGENEOUS_DEADLINE_STRESS`, `SequentialConfidence_0.55_0.45` achieves `0.5201` SQ-F1 with `0.00%` budget violations, whereas `AlwaysOn_Direct_Parallel` suffers `66.67%` budget violations.
- **Negative Findings (Reported Transparently):**
  - On `TEXT_EMBEDDING_NOISE`, `AlwaysOn_Direct_Parallel` (`0.4417`) and `OneShot_ConfidenceRouting_0.55` (`0.4367`) outperform `SequentialConfidence_0.55_0.45` (`0.4290`) because noisy text embeddings can still yield overconfident Stage-1 softmax scores (`>= 0.55`), causing premature early exit.
  - Across all 7 workloads, `OneShot_ConfidenceRouting_0.55` (`0.4897`) marginally exceeds `SequentialConfidence_0.55_0.45` (`0.4895`) by `+0.0002` (`95% CI [-0.0027, +0.0022]`), while `SequentialConfidence` saves `13.84%` bytes (`134.6 B` vs `156.2 B`).
- **Communication-Latency Break-Even Model (Exploratory):**
  - Sequential acquisition beats parallel always-on in mean wall latency (`0.51 ms` vs `1.07 ms`) whenever the Stage-1 early-exit rate `p_1` avoids both audio and visual TCP round-trips (`L_seq < L_par` when `(1 - p_1) * (RTT_A + (1 - p_2) * RTT_V) < RTT_par + T_inf_overhead`). In loopback TCP, `p_1` is high enough that `SequentialConfidence` is faster than `AlwaysOn_Direct_Parallel` on all 7 workloads, though this is a **systems-engineering trade-off**, not a novel theoretical theorem.

---

## 4. Locked Contributions & Framing (Priority 4)

- **Recommended Framing:** **Empirical Systems Evaluation** (stronger and more honest than "novel ML algorithm", and broader than a pure replication study).
- **Primary Contribution (C1 — Empirical Systems Evaluation):**
  - *RQ:* Under strict per-request cost/deadline budgets and 7 fault/noise workloads, does two-stage confidence-cascaded modality acquisition reduce measured application-protocol bytes by `>= 15%` with non-inferior Service-Quality Macro-F1 (`>= -0.010`) vs `AlwaysOn` and `Static_TA`?
  - *Gap:* Prior multimodal dynamic fusion works (DynMM, A2MT) report simulated FLOPs/nominal costs rather than measured application-protocol bytes, tail latencies, and budget-violation rates under a unified Service-Quality estimand.
  - *Falsifiable Claim & Direct Evidence:* Confirmed on `4,131` video-disjoint CMU-MOSEI holdout utterances (`694,008` TCP traces, `B=2,000` cluster bootstrap): `51.35%–74.52%` byte savings with `ΔF1 >= +0.0042`.
  - *Counterevidence & Strongest Objection:* `OneShot_ConfidenceRouting_0.55` achieves `+0.0002` higher Macro-F1, and `TEXT_EMBEDDING_NOISE` exposes overconfident early-exit failure; single-host loopback TCP does not measure WAN jitter or physical edge-cloud energy.
- **Supporting Contribution (C2 — Reproducible Frozen-Protocol Evaluation Harness):**
  - *RQ:* Can a two-phase frozen-protocol harness prevent post-hoc threshold tuning and silent leakage while auditing `1.388M` offline + `694k` TCP traces?
  - *Evidence:* `174/174` unit/adversarial tests + `holdout_access_ledger.json` + exact CRLF/LF-verified trace hashes.

---

## 5. Minimal Additional Evidence Plan (Priority 5 — Not Executed)

1. **Simulated WAN/Edge-Cloud Latency Injection (CPU-only, `0` retraining):** Replay `SequentialConfidence_0.55_0.45` vs `OneShot_ConfidenceRouting_0.55` and `AlwaysOn_Direct_Parallel` under controlled `10 ms / 30 ms / 80 ms` provider RTTs to empirically map the sequential vs parallel break-even curve.
2. **Second Dataset Validation (e.g., CMU-MOSI or IEMOCAP, CPU-only):** Verify whether the overconfident-text early-exit failure under `TEXT_EMBEDDING_NOISE` generalizes beyond CMU-MOSEI.

*(Neither experiment is executed in this audit; both require separate author approval.)*

---

## 6. Journal-Fit Assessment & Final Publication Decision (Priority 6)

- **Multimedia Tools and Applications (Springer):** SJR `~0.798`, **Q1** (*Media Technology*) / **Q2** (*Computer Networks*, *Software*), hybrid (no mandatory APC). **Best realistic fit** if framed as an *Empirical Systems Evaluation* with the 1-2 minimal additional experiments.
- **Multimedia Systems (Springer):** SJR `~0.669`, **Q1** (*Media Technology*) / **Q2** (*Information Systems*, *Software*), hybrid (no mandatory APC). Strong systems-oriented alternative.
- **Final Publication Decision:** **`CONDITIONAL_SUBMIT_AS_EMPIRICAL_SYSTEMS_PAPER`** — Phase II confirmatory evidence is solid and reproducible, but `OVERLAP_IDENTIFIED` precludes claiming algorithmic novelty; adding the simulated WAN RTT break-even experiment is recommended before Q1/Q2 submission.
