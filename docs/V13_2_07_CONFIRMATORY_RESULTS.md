# Tri-SACTA V13.2 — Phase II Confirmatory Results

- **Authorized Freeze Package Commit:** `c9177414a2eadd45a6df99dd832b3512a460e3d2`
- **Scientific Source Commit:** `d59c6f5ba6834494078d9cece2241e7838723df6`
- **Freeze Manifest SHA-256:** `fb23d9e16f8557e6aaf807d893fd5d8bf756d96e792e9c835934046e960cbdb3`
- **Eligible Holdout Manifest SHA-256:** `146157febd1f2c775e4195f694293d6e8701fcf4e8537884e4d15ad55251502e`
- **Execution Status:** `PHASE_II_COMPLETED` (exit code `0`)
- **Evaluated Population:** `4,131` eligible holdout utterances across `617` video clusters (`7` workloads × `3` seeds: `42, 100, 2026`).
- **Trace Completeness:** `1,388,016` offline traces (`16` policies) and `694,008` measured TCP traces (`8` policies, `694,008` unique composite keys).

## Primary Confirmatory Endpoint (Measured Application-Protocol Bytes & Service-Quality Macro-F1)

Candidate `SequentialConfidence_0.55_0.45` achieved Service-Quality Macro-F1 = `0.4895`, mean `134.6` application-protocol bytes/request, `99.85%` useful on-time rate, `99.85%` acceptance rate, and `0.00%` budget violation rate.

| Primary Comparator | Δ Service-Quality Macro-F1 (95% CI) | Measured Byte Reduction (95% CI) | Joint Pass |
|---|---|---|---|
| `AlwaysOn_Direct_Serial` (`0.4852`, `528.2` B) | `+0.0042` (`[-0.0014, +0.0103]`) | `74.52%` (`[73.78%, 75.22%]`) | **PASS** |
| `AlwaysOn_Direct_Parallel` (`0.4852`, `528.2` B) | `+0.0042` (`[-0.0014, +0.0103]`) | `74.52%` (`[73.78%, 75.22%]`) | **PASS** |
| `Static_TA_Direct` (`0.4841`, `276.7` B) | `+0.0054` (`[+0.0019, +0.0088]`) | `51.35%` (`[49.93%, 52.76%]`) | **PASS** |

- **Strongest Competing Baseline:** `OneShot_ConfidenceRouting_0.55` (`0.4897` Macro-F1, `156.2` bytes/request; candidate ΔF1 = `-0.0002`, `95% CI [-0.0027, +0.0022]`, `13.84%` byte saving).
- **Confirmatory Hypothesis Verdict:** `CONFIRMATORY_SUPPORTED` (all three primary comparators and TCP operational guardrails satisfied).
- **Novelty & Scope Disclosure:** `novelty_status = OVERLAP_IDENTIFIED` preserved. Results reflect single-host TCP loopback feature acquisition on video-group-disjoint CMU-MOSEI utterances; speaker disjointness remains `UNKNOWN`.
