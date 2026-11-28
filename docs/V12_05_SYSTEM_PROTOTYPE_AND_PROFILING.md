# V12.05: Minimal Reproducible API Service Prototype and Profiling

**Author:** Antigravity 3.8 Systems Research & Verification Lead  
**Date:** 2026-11-23  
**Status:** Completed & Validated  
**Service Implementation:**
- App: [src/service/app.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/src/service/app.py)
- Router: [src/service/router.py](file:///c:/Users/ASUS/Desktop/Tri-SACTA/src/service/router.py)
**Profiling Results:**
- Runtime Trials CSV: [results/v12_systems/profiling/runtime_trials.csv](file:///c:/Users/ASUS/Desktop/Tri-SACTA/results/v12_systems/profiling/runtime_trials.csv)

---

## 1. Prototype Architecture

To demonstrate the real-world utility of the **Budgeted Reliability-Aware Modality Orchestrator (BRMO)** without building extraneous cosmetic UI, a minimal typed REST API was implemented in FastAPI:

- `GET /health`: Reports service status, active device, and parameter count (1,066,397 parameters).
- `GET /policies`: Enumerates supported routing policies (`text_only`, `always_on`, `fixed_fallback`, `orchestrated_brmo`).
- `POST /predict`: Accepts a multimodal payload (text, audio, visual feature vectors and availability masks), executes the BRMO tiered routing decision, and returns the classification, predicted class, confidence, abstention flag, routing path, invoked modalities, and exact latency and compute cost.

---

## 2. Empirical Profiling Across 100 Trials

Profiling was executed on an NVIDIA GeForce RTX 2050 Laptop GPU (PyTorch 2.5.1+cu121) across 100 requests simulating realistic request streams (70% clean, 15% text noise, 15% missing text):

| Metric | Measured Value | Constraint / Target | Status |
| :--- | :---: | :---: | :---: |
| **Cold Start Latency** | 67.09 ms | $< 250$ ms | **PASS** |
| **Warm Mean Latency** | 13.03 ms | $< 30$ ms | **PASS** |
| **p50 Latency (Median)**| 12.86 ms | $< 25$ ms | **PASS** |
| **p95 Latency** | 16.59 ms | $< 50$ ms | **PASS** |
| **p99 Latency** | 20.67 ms | $< 100$ ms | **PASS** |
| **Peak VRAM Allocated** | **13.27 MB** | $< 100$ MB Budget (4GB Hardware) | **PASS** |
| **Process RAM Footprint** | ~580 MB | $< 4.0$ GB | **PASS** |

### Findings
1. **Ultra-Low Memory Footprint:** At only 13.27 MB of GPU VRAM, the entire BRMO orchestrator and 1.07M parameter cross-modal fusion model operates well inside the 100 MB budget, ensuring frictionless deployment on memory-constrained edge hardware.
2. **Stable Tail Latency:** The p95 latency of 16.59 ms confirms that dynamic routing introduces minimal tail variance compared to static always-on multimodal inference (which consumes 13–17 ms on every single request regardless of need).
