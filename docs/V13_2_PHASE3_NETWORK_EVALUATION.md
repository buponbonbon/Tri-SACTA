# Tri-SACTA V13.2 — Phase III Rectified Network-Aware Systems Evaluation

- **Rectified Benchmark (`results/v13_2/phase3_network/rectified/`):** `26,136` measured queries (`99` utterances across `12` seeded whole-video clusters from `val`, `4` workloads, `6` policies, `11` controlled TCP loopback emulation conditions). **Zero** Phase II holdout access.
- **Historical Pilot (`results/v13_2/phase3_network/pilot_v1/`):** Retained untouched (`4,224` queries, `16` utterances, post-hoc delay injection, `MAE = 4.74 ms`, F7 single-query `29.49 s` OS/socket stall at `sample_idx=9` in `RTT30_BW10_B`).
- **Rectified Instrumentation:** `RectifiedNetworkCoordinator` injects RTT/bandwidth/jitter **inside** `try_fetch_remote_modality()` before returning to routing logic, and records monotonic per-request `CommunicationEvent` traces (`provider_events.jsonl.gz`).
- **Analytical Model Validation (Rectified):** `MAE = 5.06 ms` across `RTT10_UNRES_B`, `RTT30_UNRES_B`, and `RTT80_UNRES_B` (`analytical_model_validation.json`).
