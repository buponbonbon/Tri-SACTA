# Tri-SACTA V13.2 — Phase III Supplementary Network-Aware Systems Evaluation

- **Scope:** Post-confirmatory supplementary evaluation on development (`val`) data only (`5` sorted video clusters, `16` utterances, `4` workloads, `11` controlled TCP loopback emulation conditions = `4,224` measured queries). **Zero** Phase II holdout access.
- **Emulation Disclosure:** Controlled application-layer communication-delay and bandwidth-throttling emulation over TCP loopback (NOT physical WAN deployment).

## 1. Experimental Regimes
- **Regime A (Original Deadlines `15/30/50 ms`):** Tests practical degradation when injected RTT (`0, 10, 30, 80 ms`) approaches or exceeds service deadlines. At `80 ms` RTT, all remote-acquisition policies (`SequentialConfidence`, `OneShot`, `AlwaysOn`, `Static_TA`) violate the `15–50 ms` deadlines, leaving `Unimodal_Text` (`0` remote rounds) as the only deadline-compliant policy.
- **Regime B (Relaxed `250 ms` Deadline):** Isolates RTT/bandwidth scaling and analytical break-even behavior across `0, 10, 30, 80 ms` RTT, `10 Mbps` / `1 Mbps` bandwidth, and `±5 ms` jitter.

## 2. Analytical Latency Model Validation
- Model: `T_p = H_p + R_p * r + B_p,critical / b`, fitted on `RTT0_UNRES_B` and evaluated on `RTT10_UNRES_B`, `RTT30_UNRES_B`, and `RTT80_UNRES_B`.
- **Validation Error:** `MAE = 4.74 ms` across all 6 policies and 3 unseen RTTs (`analytical_model_validation.json`).
- **Break-Even Insight:** Because `SequentialConfidence_0.55_0.45` exits at Stage 1 (`0` remote rounds) on a large fraction of queries, its mean critical-path rounds `R_seq < 1.0`, allowing it to remain faster on average than `AlwaysOn_Direct_Parallel` (`R_par = 1.0`) even at `80 ms` RTT, while `Unimodal_Text` (`R_T = 0`) dominates strict-deadline regimes.

## 3. Artifacts
- Raw traces: `results/v13_2/phase3_network/network_benchmark_traces.csv.gz`
- Aggregated metrics: `results/v13_2/phase3_network/network_aggregated_metrics.csv`
- Analytical validation: `results/v13_2/phase3_network/analytical_model_validation.json`
- Operating regions: `results/v13_2/phase3_network/operating_regions.json`
- Publication figures: `results/v13_2/publication_audit/figures/` (`figB`, `figC`, `figE` in `.svg` and `.png`).
