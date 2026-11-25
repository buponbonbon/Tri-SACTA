# Tri-SACTA V13.2: Runtime Architecture, Sensitivity Studies & Fault Stress Evidence

**Generated from:** [distributed_summary.json](../results/v13_2/systems/distributed_summary.json) & [distributed_fault_benchmark.csv](../results/v13_2/systems/distributed_fault_benchmark.csv)  
**Scope:** `Single-host localhost TCP loopback sockets (ports 8765, 8766); threaded providers; NOT multi-device WAN or link-layer energy/TCP-IP packet capture.`  
**Quantile Convention:** `numpy.quantile(method='linear')`

## 1. Non-Persistent TCP Sensitivity (Per-Request Connect, `CLEAN_MULTIMODAL`, Seed 42, N=250)

| Policy | Mean App Bytes/q | Mean Wall (ms) | p50 (ms) | p95 (ms) | p99 (ms) | On-Time Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `AlwaysOn_Direct_Serial` | 726.2 | 18.66 | 16.81 | 32.77 | 39.16 | 100.0% |
| `AlwaysOn_Direct_Parallel` | 726.2 | 6.18 | 2.99 | 18.41 | 25.47 | 100.0% |
| `Static_TA_Direct` | 440.1 | 8.70 | 3.42 | 21.52 | 26.86 | 100.0% |
| `OneShot_ConfidenceRouting_0.55` | 398.0 | 3.57 | 2.52 | 16.72 | 25.48 | 100.0% |
| `SequentialConfidence_0.55_0.45` | 265.2 | 5.31 | 2.18 | 16.77 | 32.05 | 100.0% |
| `Unimodal_Text` | 0.0 | 0.20 | 0.18 | 0.30 | 0.41 | 100.0% |

## 2. Concurrency = 4 Workers (`CLEAN_MULTIMODAL`, Persistent TCP)

| Policy | Mean Wall (ms) | p95 (ms) | p99 (ms) | On-Time |
| :--- | :---: | :---: | :---: | :---: |
| `AlwaysOn_Direct_Parallel` | 5.68 | 7.46 | 13.05 | 100.0% |
| `Static_TA_Direct` | 3.69 | 5.41 | 6.16 | 100.0% |
| `OneShot_ConfidenceRouting_0.55` | 4.49 | 8.24 | 12.87 | 100.0% |
| `SequentialConfidence_0.55_0.45` | 3.51 | 7.85 | 9.96 | 100.0% |
